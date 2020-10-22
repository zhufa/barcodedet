import math
import time
from itertools import product
import torch
import numpy as np


def str2bool(s):
    return s.lower() in ('true', '1')


class Timer:
    def __init__(self):
        self.clock = {}

    def start(self, key="default"):
        self.clock[key] = time.time()

    def end(self, key="default"):
        if key not in self.clock:
            raise Exception(f"{key} is not in the clock.")
        interval = time.time() - self.clock[key]
        del self.clock[key]
        return interval


def save_checkpoint(epoch, net_state_dict, optimizer_state_dict, best_score, checkpoint_path, model_path):
    torch.save({
        'epoch': epoch,
        'model': net_state_dict,
        'optimizer': optimizer_state_dict,
        'best_score': best_score
    }, checkpoint_path)
    torch.save(net_state_dict, model_path)


def load_checkpoint(checkpoint_path):
    return torch.load(checkpoint_path)


def freeze_net_layers(net):
    for param in net.parameters():
        param.requires_grad = False


def store_labels(path, labels):
    with open(path, "w") as f:
        f.write("\n".join(labels))


def generate_ssd_priors(specs, image_size, clamp=True):
    """
    生成 priors 作为锚框，训练和测试都会用到。
    若各尺度(n个尺度)的先验框尺寸为p1,p2...pn,则各个尺度生成的先验框为
    (pn×pn),(pn*√宽高比 × pn/√宽高比),(√(pn * p(n+1)) × √(pn * p(n+1)))
    pn 和 p(n+1) 在下面代码中对应 spec.box_sizes.min * spec.box_sizes.max

    Return:
        tensor([[center_x, center_y, w, h], [], ...])
    """
    priors = []
    for spec in specs:
        scale = image_size / spec.shrinkage
        for j, i in product(range(spec.feature_map_size), repeat=2):
            x_center = (i + 0.5) / scale
            y_center = (j + 0.5) / scale

            size = spec.box_sizes.min
            h = w = size / image_size
            priors.append([x_center, y_center, w, h])

            size = math.sqrt(spec.box_sizes.max * spec.box_sizes.min)
            h = w = size / image_size
            priors.append([x_center, y_center, w, h])

            size = spec.box_sizes.min
            h = w = size / image_size
            for ratio in spec.aspect_ratios:
                ratio = math.sqrt(ratio)
                priors.append([x_center, y_center, w * ratio, h / ratio])

    priors = torch.tensor(priors)
    if clamp:
        torch.clamp(priors, 0.0, 1.0, out=priors)

    return priors


def hard_negative_mining(loss, labels, neg_pos_ratio):
    """
    难例挖掘，用于平衡正负样本，负/正比例为neg_pos_ratio:1
    :return ：bool型的列表，里面为True的数量等于 正样本数量×(neg_pos_ratio+1),以列表中值为True的位置进行取样

    :param loss: (B, 8732)
    :param labels: (B, 8732)
    """
    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio

    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg

    return pos_mask | neg_mask


def center2corner(locations):
    """
    中心坐标形式转换为角点坐标形式，如：
    (x_center, y_center, w, h)  -->  (x_lefttop, y_lefttop, x_rightbottom, y_rightbottom)
    """
    return torch.cat([locations[..., :2] - locations[..., 2:] / 2,
                      locations[..., :2] + locations[..., 2:] / 2], locations.dim() - 1)


def corner2center(boxes):
    """
    角点坐标形式转换为中心坐标形式，如：
    (x_lefttop, y_lefttop, x_rightbottom, y_rightbottom) --> (x_center, y_center, w, h)
    """
    return torch.cat([(boxes[..., :2] + boxes[..., 2:]) / 2,
                      boxes[..., 2:] - boxes[..., :2]], boxes.dim() - 1)


def area_of(left_top, right_bottom):
    # 设置了区间最小为0.0，那么小于0的值都会强制等于边界0
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """
    计算两个框之间的 IoU

    Return intersection-over-union (Jaccard index) of boxes
    Args:
        boxes0: (1, num_target, 4)
        boxes1: (num_priors, 1, 4)
        eps: 防止两区域重合，分母为0的情况
    Return:
        ious: (num_priors, num_target)
    """
    # 相交区域的左上角坐标等于两个区域左上角坐标更大的坐标，右下角同理
    # 连续多个：可以用...表示，：2 表示 0：2 因为0可省略
    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])  # (num_priors, num_target, 2)
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])

    return overlap_area / (area0 + area1 - overlap_area + eps)


def assign_priors(gt_boxes, gt_labels, corner_form_priors, iou_threshold):
    """
    将真实框分别分配至给生成的 priors，将这些真实框与 priors 的偏差值作为监督标签训练模型
    Assign ground truth boxes and targets to priors

    Args:
        gt_boxes: tensor (num_target, 4)
        gt_labels: tensor (num_target, )
        corner_form_priors: tensor (num_priors, 4)
    Return:
        boxes: (num_priors, 4), 每个 prior 对应的 gt_box
        labels: (num_priors)
    """
    ious = iou_of(gt_boxes.unsqueeze(0), corner_form_priors.unsqueeze(1))  # (num_priors, num_target)

    # max(1)表示按行找最大，max(0)按列,此处可想象一个列向表示priors，横向表示目标类别class的一个列表
    best_target_per_prior, best_target_per_prior_index = ious.max(1)  # (num_priors)
    best_prior_per_target, best_prior_per_target_index = ious.max(0)  # (num_target)

    # 保证每个 target 都有 prior 分配
    # 比如目标1最大IOU在第3个prior位置，但第3个prior的最大IOU对应目标2，则强行将第3个prior的最大IOU对应目标1
    for target_index, prior_index in enumerate(best_prior_per_target_index):
        best_target_per_prior_index[prior_index] = target_index

    # best_target_per_prior装了每一个prior中最大的IOU值，下方代码将目标对应最大IOU所在的prior位置的IOU设置为一个较大的值(2或其他)
    # 可防止某一目标最大IOU小于iou_threshold，所在prior位置的label被下面代码设置为0，即被当为背景的情况
    best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)

    # 下面两行代码会将类别与prior对应上，且将IOU值少于阈值的label设置为0，0代表背景
    # 这样就会出现一个目标可能对应多个prior，为什么不是只保留最大IOU的prior同时将其他的prior的label置为0？
    # 答：后面会用到非极大值抑制选择最佳prior
    labels = gt_labels[best_target_per_prior_index]  # (num_priors)
    labels[best_target_per_prior < iou_threshold] = 0

    boxes = gt_boxes[best_target_per_prior_index]  # (num_priors, 4)

    return boxes, labels


def boxes2locations(center_form_boxes, center_form_priors, center_variance, size_variance):
    """
    真实框与 priors 的偏差值计算，监督训练时采用此值计算 loss
    """
    if center_form_priors.dim() + 1 == center_form_boxes.dim():
        center_form_priors = center_form_priors.unsqueeze(0)

    return torch.cat([
        (center_form_boxes[..., :2] - center_form_priors[..., :2]) / center_form_priors[..., 2:] / center_variance,
        torch.log(center_form_boxes[..., 2:] / center_form_priors[..., 2:]) / size_variance
    ], center_form_boxes.dim() - 1)


def locations2boxes(locations, priors, center_variance, size_variance):
    """
    参见上一函数定义，与上一函数相反操作
    因为训练时我们拿“真实框与 priors 的偏差值”来计算loss，测试时我们需要将得到的“偏差值”转化成预测框的坐标
    Convert regressional location results of SSD into boxes in the form of (center_x, center_y, h, w)
    Args:
        locations (batch_size, num_priors, 4): the regression output of SSD. It will contain the outputs as well.
        priors (num_priors, 4) or (batch_size/1, num_priors, 4): prior boxes.
        center_variance: a float used to change the scale of center.
        size_variance: a float used to change of scale of size.
    Returns:
        boxes:  priors: [[center_x, center_y, h, w]]. All the values
            are relative to the image size.
    """
    if priors.dim() + 1 == locations.dim():
        priors = priors.unsqueeze(0)

    return torch.cat([
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        torch.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    ], locations.dim() - 1)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    hard 极大值抑制，用于去除多余的预测框
    Args:
        box_scores: (N, 5), corner-form
        top_k: keep top_k result, if <= 0, keep all the results
    """
    scores = box_scores[:, -1]  # (N,)
    boxes = box_scores[:, :-1]  # (N, 4)
    picked = []
    _, indexes = scores.sort(descending=True)
    indexes = indexes[:candidate_size]
    while len(indexes) > 0:
        current = indexes[0]
        picked.append(current.item())
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            current_box.unsqueeze(0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


def soft_nms(box_scores, score_threshold, sigma=0.5, top_k=-1):
    """
    soft 极大值抑制，去除多余预测框同时可以优化同类目标重叠预测的问题
    Soft NMS implementation.

    References:
        https://arxiv.org/abs/1704.04503
        https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/cython_nms.pyx

    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        score_threshold: boxes with scores less than value are not considered.
        sigma: the parameter in score re-computation.
            scores[i] = scores[i] * exp(-(iou_i)^2 / simga)
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
         picked_box_scores (K, 5): results of NMS.
    """
    picked_box_scores = []
    while box_scores.size(0) > 0:
        max_score_index = torch.argmax(box_scores[:, 4])
        cur_box_prob = torch.tensor(box_scores[max_score_index, :])
        picked_box_scores.append(cur_box_prob)
        if len(picked_box_scores) == top_k > 0 or box_scores.size(0) == 1:
            break
        cur_box = cur_box_prob[:-1]
        box_scores[max_score_index, :] = box_scores[-1, :]
        box_scores = box_scores[:-1, :]
        ious = iou_of(cur_box.unsqueeze(0), box_scores[:, :-1])
        box_scores[:, -1] = box_scores[:, -1] * torch.exp(-(ious * ious) / sigma)
        box_scores = box_scores[box_scores[:, -1] > score_threshold, :]
    if len(picked_box_scores) > 0:
        return torch.stack(picked_box_scores)
    else:
        return torch.tensor([])


def nms(box_scores, nms_method=None, score_threshold=None, iou_threshold=None,
        sigma=0.5, top_k=-1, candidate_size=200):
    """
    极大值抑制算法
    """
    if nms_method == 'soft':
        return soft_nms(box_scores, score_threshold, sigma, top_k)
    else:
        return hard_nms(box_scores, iou_threshold, top_k, candidate_size=candidate_size)


def compute_average_precision(precision, recall):
    """
    计算平均准确率
    ""
    It computes average precision based on the definition of Pascal Competition. It computes the under curve area
    of precision and recall. Recall follows the normal definition. Precision is a variant.
    pascal_precision[i] = typical_precision[i:].max()
    """
    # identical but faster version of new_precision[i] = old_precision[i:].max()
    precision = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])

    # find the index where the value changes
    recall = np.concatenate([[0.0], recall, [1.0]])
    changing_points = np.where(recall[1:] != recall[:-1])[0]

    # compute under curve area
    areas = (recall[changing_points + 1] - recall[changing_points]) * precision[changing_points + 1]
    return areas.sum()


def compute_voc2007_average_precision(precision, recall):
    """
    计算平均准确率
    """
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap = ap + p / 11.
    return ap

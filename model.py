import numpy as np
from collections import namedtuple
import torch.nn as nn
import torch
import torch.nn.functional as F

from utils import center2corner, corner2center, boxes2locations, locations2boxes
from utils import assign_priors, nms, hard_negative_mining
from utils import Timer
from dataset.preprocess import PredictionTransform
import config as C


class SSD(nn.Module):

    def __init__(self, num_classes, base_net, source_layer_indexes,
                 extras, classification_headers, regression_headers,
                 is_test=False, config=None, device=None):

        """
        SSD 检测网络的搭建
        """
        super(SSD, self).__init__()

        self.num_classes = num_classes
        self.base_net = base_net
        self.source_layer_indexes = source_layer_indexes
        self.extras = extras
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.is_test = is_test
        self.config = config

        # self.source_layer_add_ons = nn.ModuleList([t[1] for t in source_layer_indexes
        #                                            if isinstance(t, tuple) and not isinstance(t, GraphPath)])

        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if is_test:
            self.config = config
            self.priors = config.priors.to(self.device)

    def forward(self, x):
        """
        网络前向计算

        Return:
            train phase:
                confidences: (B,H*W*n_a,n_cls)
                locations: (B,H*W*n_a,4)
        """
        confidences = []
        locations = []
        start_layer_index = 0
        header_index = 0
        # for end_layer_index in self.source_layer_indexes:
        #     if isinstance(end_layer_index, GraphPath):
        #         path = end_layer_index
        #         end_layer_index = end_layer_index.s0
        #         added_layer = None
        #     elif isinstance(end_layer_index, tuple):
        #         added_layer = end_layer_index[1]
        #         end_layer_index = end_layer_index[0]
        #         path = None
        #     else:
        #         added_layer = None
        #         path = None
        #     for layer in self.base_net[start_layer_index:end_layer_index]:
        #         x = layer(x)
        #     if added_layer:
        #         y = added_layer(x)
        #     else:
        #         y = x
        #     if path:
        #         sub = getattr(self.base_net[end_layer_index], path.name)
        #         for layer in sub[:path.s1]:
        #             x = layer(x)
        #         y = x
        #         for layer in sub[path.s1:]:
        #             x = layer(x)
        #         end_layer_index += 1

        #     start_layer_index = end_layer_index
        #     confidence, location = self.compute_header(header_index, y)
        #     header_index += 1

        #     confidences.append(confidence)
        #     locations.append(location)

        ##### source_layer_indexes first
        end_layer_index = self.source_layer_indexes[0]
        for layer in self.base_net[start_layer_index:end_layer_index]:
            x = layer(x)
        y = x

        confidence, location = self.compute_header(header_index, y)
        confidences.append(confidence)
        locations.append(location)

        header_index += 1
        ##### source_layer_indexes second
        second_layer_index = self.source_layer_indexes[1]
        for layer in self.base_net[end_layer_index:second_layer_index]:
            x = layer(x)

        ##### extra layers
        for layer in self.extras:
            x = layer(x)
            confidence, location = self.compute_header(header_index, x)
            header_index += 1

            confidences.append(confidence)
            locations.append(location)

        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)

        if self.is_test:
            confidences = F.softmax(confidences, dim=2)
            boxes = locations2boxes(locations, self.priors,
                                    self.config.center_variance,
                                    self.config.size_variance)
            boxes = center2corner(boxes)

            return confidences, boxes
        else:
            return confidences, locations

    def compute_header(self, i, x):
        """
        每种预测尺度的两个分支，分别为类别预测分支和位置回归预测分支
        """
        # classification headers
        confidence = self.classification_headers[i](x)
        # permute：按维度重新排列，contiguous：类似深复制，view：类似resize
        confidence = confidence.permute(0, 2, 3, 1).contiguous()  # (B,H,W,n_a*n_cls)
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)  # (B,H*W*n_a,n_cls)

        # bbox location headers
        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)  # (B,H*W*n_a,4)

        return confidence, location

    def init(self):
        """
        模型的权重随机初始化，在创建模型时进行
        """
        self.base_net.apply(_xavier_init)
        # self.source_layer_add_ons.apply(_xavier_init)
        self.extras.apply(_xavier_init)
        self.classification_headers.apply(_xavier_init)
        self.regression_headers.apply(_xavier_init)

    def init_from_base_net(self, model):
        """
        模型的权重随机初始化，在创建模型时进行
        """
        # no model to load
        self.base_net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage), strict=True)
        self.source_layer_add_ons.apply(_xavier_init)
        self.extras.apply(_xavier_init)
        self.classification_headers.apply(_xavier_init)
        self.regression_headers.apply(_xavier_init)

    def init_from_pretrained_ssd(self, model):
        """
        模型权重初始化，从与训练完成的权重进行初始化
        """
        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        state_dict = {k: v for k, v in state_dict.items() if
                      not (k.startswith("classification_headers") or k.startswith("regression_headers"))}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        self.classification_headers.apply(_xavier_init)
        self.regression_headers.apply(_xavier_init)

    def load(self, model):
        """
        加载训练完成的模型
        """
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def save(self, model_path):
        """
        存储训练完成的模型
        """
        torch.save(self.state_dict(), model_path)


def _xavier_init(m):
    """
    初始化方法，上述初始化函数会调用
    """
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        # if has bias
        nn.init.constant_(m.bias, 0.0)


class MatchPrior(object):

    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
        """
        训练时将 priors 根据真实框的 iou 进行标签分配
        生成训练需要的 priors
        priors 的生成可参考 utils.py 中的 generate_ssd_priors 函数
        """
        self.center_form_priors = center_form_priors
        # tensor (n_priors, 4)
        self.corner_form_priors = center2corner(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):
        """
        Args:
            gt_boxes: np.ndarray, (n_targets, 4)
            gt_labels: np.ndarray, (n_targets, )
        """
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)

        boxes, labels = assign_priors(gt_boxes, gt_labels, self.corner_form_priors, self.iou_threshold)
        boxes = corner2center(boxes)
        locations = boxes2locations(boxes, self.center_form_priors, self.center_variance, self.size_variance)

        return locations, labels


class Predictor:

    def __init__(self, net, size, mean=0.0, cvt2gray=False, std=1.0, nms_method=None,
                 iou_threshold=0.45, filter_threshold=0.01,
                 candidate_size=200, sigma=0.5, device=None):

        """
        预测阶段会用到，预测类
        """
        self.net = net
        self.transform = PredictionTransform(size, mean, std, cvt2gray)
        self.iou_threshold = iou_threshold
        self.filter_threshold = filter_threshold
        self.candidate_size = candidate_size
        self.nms_method = nms_method
        self.sigma = sigma
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net.to(self.device)
        self.net.eval()

        self.timer = Timer()

    def predict(self, image, top_k=-1, prob_threshold=None):
        """
        输入图片，输出预测的结果(待检测目标位置及类别)
        """
        cpu_device = torch.device('cpu')
        height, width, _ = image.shape
        image = self.transform(image)
        images = image.unsqueeze(0)
        images = images.to(self.device)
        with torch.no_grad():
            self.timer.start()
            scores, boxes = self.net.forward(images)
            print(f'Inference time: {self.timer.end()}')

        boxes = boxes[0]
        scores = scores[0]

        if not prob_threshold:
            prob_threshold = self.filter_threshold

        boxes = boxes.to(cpu_device)
        scores = scores.to(cpu_device)
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, scores.size(1)):
            probs = scores[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            subset_boxes = boxes[mask, :]

            box_probs = torch.cat([subset_boxes,
                                   probs.reshape(-1, 1)], dim=1)
            box_probs = nms(box_probs, self.nms_method,
                            score_threshold=prob_threshold,
                            iou_threshold=self.iou_threshold,
                            sigma=self.sigma,
                            top_k=top_k,
                            candidate_size=self.candidate_size)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.size(0))

        if not picked_box_probs:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])

        picked_box_probs = torch.cat(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 3] *= height

        # (N, 4) (N,) (N,)
        return picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]


class MultiboxLoss(nn.Module):

    def __init__(self, priors, iou_threshold, neg_pos_ratio,
                 center_variance, size_variance, device):
        """
        loss 计算类，用于计算类别和位置的 loss

        Implement SSD multibox loss
        classification loss + smooth l1 loss
        """
        super(MultiboxLoss, self).__init__()

        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = priors
        self.priors.to(device)

    def forward(self, confidence, pred_locations, labels, gt_locations):
        """
        Args:
            confidence: (B, 8732, num_classes), class prediction
            pred_locations: (B, 8732, 4), predicted locations
            labels: (B, 8732), real labels of all the priors
            gt_locations: (B, 8732, 4), real boxes corresponding all the priors
        """
        num_classes = confidence.size(2)
        with torch.no_grad():
            # 为什么先求log_softmax在去hard_negative_mining
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]  # (B, num_priors)
            mask = hard_negative_mining(loss, labels, self.neg_pos_ratio)

        confidence = confidence[mask, :]
        classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask], reduction='sum')

        # 只计算正样本的位置LOSS
        pos_mask = labels > 0
        pred_locations = pred_locations[pos_mask, :].reshape(-1, 4)
        gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
        # pred_locations是预测的(和prior的)位置偏差，gt_locations是真实框的(和prior的)位置偏差，他们并不是预测位置和真实框位置
        # 可追溯gt_locations的来源来理解
        smooth_l1_loss = F.smooth_l1_loss(pred_locations, gt_locations, reduction='sum')

        num_pos = gt_locations.size(0)
        return smooth_l1_loss / num_pos, classification_loss / num_pos


"""
Create model
"""


def tiny(cfg, batch_norm=False, channels=3):
    """
    主干网络，区别于用于目标检测的卷积分支
    """
    layers = []
    in_channels = channels  # 转灰度去训练
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]  # floor(H/2)
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            if batch_norm:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)  # H
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]

            in_channels = v

    return layers


def create_tiny_ssd(num_classes, is_test=False, input_imgChannel=3):
    """
    创建模型结构

    assume input image is: (N, 3, 260, 260)
    """
    tiny_config = [32, 'M',
                   64, 'M',
                   64, 'C',
                   64, 'C']  # [N, 64, 33, 33] exclude 'C' conv_4_feats

    base_net = nn.ModuleList(tiny(tiny_config, batch_norm=True, channels=input_imgChannel))

    # source_layer_indexes = [
    #     (11, nn.BatchNorm2d(64)),
    #     len(base_net)
    # ]
    source_layer_indexes = [15, len(base_net)]

    # 主体网络之后会接再接一些卷积层，进一步抽象高级语义特征，同时基于这部分进行 ssd 的多尺度分支预测
    extras = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()  # (N, 128, 17, 17) conv_5_feats
            # nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            # nn.ReLU()
        ),
        nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()  # (N, 128, 9, 9) conv_6_feats
            # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            # nn.ReLU()
        ),
        nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()  # (N, 64, 5, 5) conv_7_feats
            # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            # nn.ReLU()
        ),
        nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU()  # (N, 64, 3, 3) conv_8_feats
            # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            # nn.ReLU()
        )
    ])

    regression_headers = nn.ModuleList([
        nn.Conv2d(in_channels=64, out_channels=4 * 4, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=128, out_channels=6 * 4, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=128, out_channels=6 * 4, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=64, out_channels=6 * 4, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=64, out_channels=4 * 4, kernel_size=3, padding=1)
    ])

    classification_headers = nn.ModuleList([
        nn.Conv2d(in_channels=64, out_channels=4 * num_classes, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=128, out_channels=6 * num_classes, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=128, out_channels=6 * num_classes, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=64, out_channels=6 * num_classes, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=64, out_channels=4 * num_classes, kernel_size=3, padding=1)
    ])

    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers,
               is_test=is_test, config=C)


def create_predictor(net, cvt2gray=False, candidate_size=200, nms_method=None, sigma=.5, device=None):
    """
    创建预测实例，在训练完成后，部署检测时使用
    """
    predictor = Predictor(net, C.image_size, C.image_mean, cvt2gray,
                          nms_method=nms_method,
                          iou_threshold=C.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)

    return predictor

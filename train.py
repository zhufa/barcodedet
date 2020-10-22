# -*- coding: utf-8 -*-
import argparse
import os
import sys
from itertools import chain
from pprint import pprint
import pdb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime
from model import create_tiny_ssd, MatchPrior, MultiboxLoss
import config as C
from dataset.preprocess import TrainAugmentation, TestTransform
from dataset.barcode_dataset import BarCodeDataset
from utils import Timer, str2bool
import warnings
import cv2
# 忽略运行时系统输出的warning，以免影响观看程序的正常输出
warnings.filterwarnings("ignore", category=Warning)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def args_parser():
    parser = argparse.ArgumentParser(description='SSD train phase')
    parser.add_argument('-d', '--train_dataset', help='Train dataset directory path')
    parser.add_argument('-v', '--val_dataset', help='Val dataset directory path')
    parser.add_argument('--checkpoint_folder', default='models/', help='Directory for saving checkpoint models')

    # only tiny architecture for speed on gree chip
    # tiny/sq-ssd-lite
    parser.add_argument('-n', '--net', default='tiny', type=str, help='The network architecture')
    # param for SGD
    parser.add_argument('--lr', default=1e-3, type=float, help='Initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
    parser.add_argument('--base_net_lr', default=None, type=float, help='Initial lr for base net')
    parser.add_argument('--extra_layers_lr', default=None, type=float,
                        help='Initial lr for layers not in base net and pred heads')

    # scheduler
    parser.add_argument('--scheduler', default='multi-step', type=str, help='Scheduler for SGD, multi-step or cosine')
    parser.add_argument('--milestones', default='80,100', type=str, help='Milestones for MultiStepLR')
    # param for Cosine Annealing
    parser.add_argument('--t_max', default=120, type=float, help='T_max value for cosine annealing scheduler')

    # training params
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for training')
    parser.add_argument('--num_epochs', default=120, type=int, help='Num epochs')
    parser.add_argument('--num_workers', default=4, type=int, help='Num workers in dataloading')
    parser.add_argument('--validation_epochs', default=5, type=int, help='Validation num epochs')
    parser.add_argument('--debug_steps', default=10, type=int, help='Debug log output frequency')
    # parser.add_argument('--use_cuda', default=True, type=str2bool, help='use cuda to train model')
    # 分布式训练
    parser.add_argument('--gpu', type=int, nargs='+', default=None, help='Multi GPU train')
    parser.add_argument('--dist_url', default='localhost:12356', type=str)
    parser.add_argument('--dist_backend', default='nccl', type=str)
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--distributed', default=True, type=str2bool)

    args = parser.parse_args()

    return args


def train(loader, net, criterion, optimizer, device, debug_steps, epoch=0):
    """
    模型训练
    :param loader: 数据加载器
    :param net: 网络
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param device: 设备(GPU or CPU)
    :param debug_steps: 每逢多少步输出训练情况
    :param epoch: 训练轮数,注意本项目代码中轮数的循环放在了本方法外面，方法里面只执行batch的循环
    :return: NONE
    """

    # 将模块设置为训练模式
    # 对于一些含有BatchNorm，Dropout等层的模型，在训练时使用的forward和验证时使用的forward在计算上不太一样。因此需指定train(True/False)
    net.train(True)
    # loader按batch size分好数据了,loader_total其实就是batch,等于(图像张数/batch_size)向上取整
    loader_total = len(loader)

    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0

    # 得到的枚举值从1开始
    for i, data in enumerate(loader, 1):
        # (B, 3, 300, 300) (B, 8732, 4) (B, 8732)
        images, boxes, labels = data[0].to(device), data[1].to(device), data[2].to(device)

        # 将所有参数重置为0
        optimizer.zero_grad()
        # 前向传播，输入images，经过网络输出类别概率以及位置
        confidence, locations = net(images)
        '''
        pprint("locations: " + str(locations.size()))
        pprint("confidence: " + str(confidence.size()))
        pprint("images: " + str(images.size()))
        pprint("boxes: " + str(boxes.size()))
        pprint("labels: " + str(labels.size()))
        '''
        # 计算损失：位置回归损失以及分类损失
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
        # 总损失
        loss = regression_loss + classification_loss
        # 进行反向传播，更新参数
        loss.backward()
        optimizer.step()
        # 输出训练情况
        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_regression_loss = running_regression_loss / debug_steps
            avg_classification_loss = running_classification_loss / debug_steps

            print(datetime.datetime.now().strftime('%H:%M:%S') +
                  f' Epoch:{epoch:3d}|{args.num_epochs}, ' +
                  f'Step:{i:4d}|{loader_total}, ' +
                  f'AVG Loss {avg_loss:.4f}, ' +
                  f'AVG reg_loss {avg_regression_loss:.4f}, ' +
                  f'AVG cls_loss {avg_classification_loss:.4f}')

            running_loss = 0.0
            running_classification_loss = 0.0
            running_regression_loss = 0.0


def test(loader, net, criterion, device):
    """
    模型测试，根据训练的模型在测试集上的 loss 情况判断训练过程是否正常
    """
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        # 下面3句没涉及反向传播，即没有进行梯度计算，为何此处要声明不进行梯度计算
        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()

    return running_loss / num, running_regression_loss / num, running_classification_loss / num


def main(args):
    if args.net == 'tiny':
        create_net = create_tiny_ssd
        config = C
    else:
        print('Not supported network architecture')
        sys.exit(0)

    print('Prepare transform')
    # 进入网络前图像/训练数据的增强操作，真实框的坐标会相应改变
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    # 进入网络前各个先验框priors根据IUO值与真实框匹配，各个先验框priors得到/分配到/匹配到标签值与“真实框”(“真实框”其实是真实框与prior的偏差)
    target_transform = MatchPrior(config.priors, config.center_variance, config.size_variance, 0.5)
    # 测试时同样需要进行数据增强操作
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    print('Prepare training dataset')
    # 定义”数据集“类，执行上述数据操作，得到最终数据集
    train_dataset = BarCodeDataset(args.train_dataset, transform=train_transform, target_transform=target_transform)
    print(f'Train dataset size: {len(train_dataset)}')
    # 定义“数据加载”类，用于管理数据集，比如按batch_size分成[len(train_dataset)/batch_size]个batch, 是否打乱顺序(shuffle=True?)等
    train_loader = DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers, shuffle=True)

    print('Prepare validation dataset')
    val_dataset = BarCodeDataset(args.val_dataset,
                                 transform=test_transform,
                                 target_transform=target_transform,
                                 is_test=True)
    val_loader = DataLoader(val_dataset,
                            args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False)
    print(f'Validation dataset size: {len(val_dataset)}')

    print('Build network')
    num_classes = len(train_dataset.class_names)
    # 创建网络
    net = create_net(num_classes)
    # 网络参数量
    print("net have {} paramerters in total".format(sum(x.numel() for x in net.parameters())))
    # print(net)
    last_epoch = -1
    # 学习率设置
    base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
    extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr
    pprint(vars(args), indent=4)

    # 整个网络由base_net，extras，regression_headers，classification_headers4个部分组成，将各个部分的训练参数打包送往optimizer
    params = [
        {'params': net.base_net.parameters(), 'lr': base_net_lr},
        {'params': chain(
            net.extras.parameters()
        ), 'lr': extra_layers_lr},
        {'params': chain(
            net.regression_headers.parameters(),
            net.classification_headers.parameters()
        )}
    ]
    net.to(DEVICE)
    # 定义损失函数和优化器
    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # 使用分步式学习率，比如milestones=[30，50]，lr=0.5，gamma=0.1，则学习率lr会在30轮处变为0.5×0.1，在50轮变为0.5×0.1×0.1
    # 使用CosineAnnealingLR即使用cosine变化的学习率
    if args.scheduler == 'multi-step':
        print('Uses MultiStepLR scheduler')
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones,
                                gamma=0.1, last_epoch=last_epoch)
    elif args.scheduler == 'cosine':
        print('Uses CosineAnnealingLR scheduler')
        scheduler = CosineAnnealingLR(optimizer, args.t_max, last_epoch=last_epoch)
    else:
        print(f'Unsupported Scheduler: {args.scheduler}')
        sys.exit(1)

    print(f'Start training from epoch {last_epoch + 1}')
    # 开始训练，跑num_epochs轮
    for epoch in range(last_epoch + 1, args.num_epochs):
        train(train_loader, net, criterion, optimizer, device=DEVICE, debug_steps=args.debug_steps, epoch=epoch)
        # scheduler用于更新学习率
        scheduler.step()

        # 通过测试集测试训练情况,保存训练好的模型
        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
            val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE)
            print(datetime.datetime.now().strftime('%H:%M:%S') + f' Epoch:{epoch:3d}|{args.num_epochs}, ' +
                  f'Val Loss {val_loss:.4f}, ' +
                  f'Val reg_loss {val_regression_loss:.4f}, ' +
                  f'Val cls_loss {val_classification_loss:.4f}')

            model_path = os.path.join(args.checkpoint_folder, f"{args.net}-Epoch-{epoch}-Loss-{val_loss:.4f}.pth")
            net.save(model_path)
            print(f'Saved model {model_path}')


if __name__ == '__main__':
    args = args_parser()
    main(args)

import numpy as np
from collections import namedtuple

from utils import generate_ssd_priors


image_size = 260  # 训练输入图片大小
image_mean = np.array([123, 117, 104])  # 预处理，图片输入网络训练前在 rgb 三个通道上减去均值
image_std = 1.0  # 预处理，图片除以标准差 img/std

class_names = ['background', 'bar_code', 'qr_code']

iou_threshold = 0.45  # iou 的阈值，预测时根据此阈值进行筛选，进行 priors 标签分配时使用的 iou 阈值为 0.5
center_variance = 0.1  # 预测 x, y 偏差除以此方差，放大作用，原理参见 faster rcnn 相关知识
size_variance = 0.2  # 预测 w, h 偏差除以此方差，放大作用，原理参见 faster rcnn 相关知识

SSDBoxSizes = namedtuple('SSDBoxSizes', ['min', 'max'])
SSDSpec = namedtuple('SSDSpec', ['feature_map_size', 'shrinkage', 'box_sizes', 'aspect_ratios'])

# 'feature_map_size': 卷积后的特征图尺寸
# 'shrinkage': 对应原图缩小的程度，每次池化操作会将图像尺寸减半，下面5个尺度对应原图缩小的倍数为8,16,32...(结合网络去理解)
# 'box_sizes': 先验框的尺寸(在原图中)，如下面值为26时，会生成1:1(26×26)，以及2:1(26*根号2 × 26/根号2),1:2(同前)，外加一个1:1的
#              根号下(26*52)的先验框，其中26,52,98...分别对应原图(260×260)的0.1,0.2,0.375...
# 'aspect_ratios: 先验框的宽/高比例
# 从5个尺度去预测，下面给出了各个尺度对应的特征图大小，缩放倍数，映射在原图中的先验框尺寸，先验框的宽/高比例，可以看出最大的特征尺度使用了最小的
# 预测框，这样有利于小物体的检测
specs = [
    SSDSpec(33, 8, SSDBoxSizes(26, 52), [2, 0.5]),
    SSDSpec(17, 16, SSDBoxSizes(52, 98), [2, 0.5, 3, 0.333]),
    SSDSpec(9, 32, SSDBoxSizes(98, 143), [2, 0.5, 3, 0.333]),
    SSDSpec(5, 64, SSDBoxSizes(143, 192), [2, 0.5, 3, 0.333]),
    SSDSpec(3, 104, SSDBoxSizes(192, 300), [2, 0.5])
]

priors = generate_ssd_priors(specs, image_size)

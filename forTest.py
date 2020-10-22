# -*- coding: utf-8 -*-
# @Time : 2020/9/24 下午2:17
# @Author : zhufa
# @Software: PyCharm
# @description:
import random

from dataset.barcode_dataset import BarCodeDataset
from torch.utils.data import DataLoader
import torch
from dataset.preprocess import TrainAugmentation, TestTransform
import config
from model import MatchPrior
import torch.nn.functional as F
from utils import assign_priors, nms, hard_negative_mining
import math
from itertools import product

t = torch.tensor(0)
tt = t.item()
color = lambda: random.randint(0,255)
c1 = color()
c3 = color()
c2 = color()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
prodct = product(range(3), repeat=2)
f = torch.randn(4,1,3)
f1 = torch.randn(4,3,3)
f2 = torch.max(f,f1)

d = torch.randn(3,3)
dm0 = d.max(0)
dm1 = d.max(1)
d1 = torch.randn(3,3)
d4 = torch.randn(3,3,2)
e = d[True, :]
e1 = d[False, :]
e2 = d >0
d1[e2] = -math.inf
d3 = d[e2]
d5 = d4[e2,:]
e3 = e2.long().sum(dim=1, keepdim=True)
c = -F.softmax(d, dim=0).numpy()

m1 = torch.randn(10,10)
m2 = torch.randn(10,10)
m3 = d[1:,2]
m4 = d4[0:2,0]
mask = hard_negative_mining(m1, m2, 2)

print('Prepare transform')
train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
target_transform = MatchPrior(config.priors, config.center_variance, config.size_variance, 0.5)
test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)
print('Prepare training datasets')
train_dataset = BarCodeDataset("barcode_dat",transform=train_transform, target_transform=target_transform)
print(f'Train dataset size: {len(train_dataset)}')
train_loader = DataLoader(train_dataset, 32, num_workers=2, shuffle=True)
loader_total = len(train_loader)
for i, data in enumerate(train_loader, 1):
    # (B, 3, 300, 300) (B, 8732, 4) (B, 8732)
    images, boxes, labels = data[0].to(device), data[1].to(device), data[2].to(device)
    ln = labels.numpy()
    loss = -F.log_softmax(boxes, dim=2)[:, :, 0]  # (B, num_priors)
    mask = hard_negative_mining(loss, labels, 3)
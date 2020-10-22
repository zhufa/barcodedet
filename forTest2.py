# -*- coding: utf-8 -*-
# @Time : 2020/9/28 下午4:05
# @Author : zhufa
# @Software: PyCharm
# @description:
import torch

ious = torch.tensor([[0.7],[0.3],[0.1]])  # (num_priors, num_target)
gt_labels = torch.tensor([[0]])
best_target_per_prior, best_target_per_prior_index = ious.max(1)  # (num_priors)
best_prior_per_target, best_prior_per_target_index = ious.max(0)  # (num_target)

# 保证每个 target 都有 prior 分配

labels = gt_labels[best_target_per_prior_index]  # (num_priors)
labels[best_target_per_prior < 0.5] = 0
# boxes = gt_boxes[best_target_per_prior_index]  # (num_priors, 4)

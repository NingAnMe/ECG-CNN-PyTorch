#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020-12-14 10:48
# @Author  : NingAnMe <ninganme@qq.com>
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss, _Loss


# 多分类
class FocalLoss(torch.nn.Module):
    def __init__(self, reduction='mean', alpha=None, gamma=0, OHEM_percent=0.6, smooth_eps=0, class_num=2):
        super(FocalLoss, self).__init__()
        self.reduction = reduction
        self.gamma = gamma
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        else:
            self.alpha = alpha
        self.OHEM_percent = OHEM_percent
        self.smooth_eps = smooth_eps
        self.class_num = class_num

    def forward(self, logits, label):
        # logits:[b,c,h,w] label:[b,c,h,w]
        pred = logits.softmax(dim=1)
        if pred.dim() > 2:
            pred = pred.view(pred.size(0), pred.size(1), -1)  # b,c,h,w => b,c,h*w
            pred = pred.transpose(1, 2)  # b,c,h*w => b,h*w,c
            pred = pred.contiguous().view(-1, pred.size(2))  # b,h*w,c => b*h*w,c
            label = label.argmax(dim=1)
            label = label.view(-1, 1)  # b*h*w,1

        pt = pred.gather(1, label).view(-1)  # b*h*w
        diff = (1 - pt) ** self.gamma

        if self.alpha:
            self.alpha = self.alpha.type_as(pred.data)
            alpha_t = self.alpha.gather(0, label.view(-1))  # b*h*w
            FL = -1 * alpha_t * diff * pt.log()
        else:
            FL = -1 * diff * pt.log()

        # OHEM = FL.topk(k=int(self.OHEM_percent * FL.size(0)), dim=0)
        if self.smooth_eps > 0:
            lce = -1 * torch.sum(pred.log(), dim=1) / self.class_num
            loss = (1 - self.eps) * FL + self.eps * lce
        else:
            loss = FL

        if self.reduction == 'mean':
            return loss.mean()  # or OHEM.mean()
        else:
            return loss.sum()  # + OHEM.sum()


# 二分类
class BCEFocalLoss(torch.nn.Module):
    def __init__(self, reduction='mean', alpha=None, gamma=0):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, label):
        # logits:[b,h,w] label:[b,h,w]
        pred = logits.sigmoid()
        pred = pred.view(-1)  # b*h*w
        label = label.view(-1)

        pt = pred * label + (1 - pred) * (1 - label)
        diff = (1 - pt) ** self.gamma

        if self.alpha:
            self.alpha = self.alpha.type_as(pred.data)
            alpha_t = self.alpha * label + (1 - self.alpha) * (1 - label)  # b*h*w
            FL = -1 * alpha_t * diff * pt.log()
        else:
            FL = -1 * diff * pt.log()

        if self.reduction == 'mean':
            return FL.mean()
        else:
            return FL.sum()


class LabelSmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                .fill_(smoothing / (n_classes - 1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = LabelSmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
                                                              self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss


class BiTemperedLogisticLoss(_Loss):
    def __init__(self, reduction='mean', t1=1, t2=1, label_smoothing=0.0, num_iters=5):
        super().__init__(reduction=reduction)
        self.t1 = t1
        self.t2 = t2
        self.label_smoothing = label_smoothing
        self.num_iters = num_iters

    @classmethod
    def log_t(cls, u, t):
        """Compute log_t for `u`."""

        if t == 1.0:
            return torch.log(u)
        else:
            return (u ** (1.0 - t) - 1.0) / (1.0 - t)

    @classmethod
    def exp_t(cls, u, t):
        """Compute exp_t for `u`."""

        if t == 1.0:
            return torch.exp(u)
        else:
            return torch.relu(1.0 + (1.0 - t) * u) ** (1.0 / (1.0 - t))

    @classmethod
    def compute_normalization_fixed_point(cls, activations, t, num_iters=5):
        """Returns the normalization value for each example (t > 1.0).
        Args:
        activations: A multi-dimensional tensor with last dimension `num_classes`.
        t: Temperature 2 (> 1.0 for tail heaviness).
        num_iters: Number of iterations to run the method.
        Return: A tensor of same rank as activation with the last dimension being 1.
        """

        mu = torch.max(activations, dim=-1).values.view(-1, 1)
        normalized_activations_step_0 = activations - mu

        normalized_activations = normalized_activations_step_0
        i = 0
        while i < num_iters:
            i += 1
            logt_partition = torch.sum(cls.exp_t(normalized_activations, t), dim=-1).view(-1, 1)
            normalized_activations = normalized_activations_step_0 * (logt_partition ** (1.0 - t))

        logt_partition = torch.sum(cls.exp_t(normalized_activations, t), dim=-1).view(-1, 1)

        return -cls.log_t(1.0 / logt_partition, t) + mu

    @classmethod
    def compute_normalization(cls, activations, t, num_iters=5):
        """Returns the normalization value for each example.
        Args:
        activations: A multi-dimensional tensor with last dimension `num_classes`.
        t: Temperature 2 (< 1.0 for finite support, > 1.0 for tail heaviness).
        num_iters: Number of iterations to run the method.
        Return: A tensor of same rank as activation with the last dimension being 1.
        """

        if t < 1.0:
            return None  # not implemented as these values do not occur in the authors experiments...
        else:
            return cls.compute_normalization_fixed_point(activations, t, num_iters)

    @classmethod
    def tempered_softmax(cls, activations, t, num_iters=5):
        """Tempered softmax function.
        Args:
        activations: A multi-dimensional tensor with last dimension `num_classes`.
        t: Temperature tensor > 0.0.
        num_iters: Number of iterations to run the method.
        Returns:
        A probabilities tensor.
        """
        if t == 1.0:
            normalization_constants = torch.log(torch.sum(torch.exp(activations), dim=-1))
        else:
            normalization_constants = cls.compute_normalization(activations, t, num_iters)

        return cls.exp_t(activations - normalization_constants, t)

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                .fill_(smoothing / (n_classes - 1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
        return targets

    def forward(self, inputs, targets):
        """Bi-Tempered Logistic Loss with custom gradient.
        Args:
        activations: A multi-dimensional tensor with last dimension `num_classes`.
        labels: A tensor with shape and dtype as activations.
        t1: Temperature 1 (< 1.0 for boundedness).
        t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
        label_smoothing: Label smoothing parameter between [0, 1).
        num_iters: Number of iterations to run the method.
        Returns:
        A loss tensor.
        """
        if self.label_smoothing > 0.0:
            targets = BiTemperedLogisticLoss._smooth_one_hot(targets, inputs.size(-1), self.label_smoothing)

        probabilities = self.tempered_softmax(inputs, self.t2, self.num_iters)

        temp1 = (self.log_t(targets + 1e-10, self.t1) - self.log_t(probabilities, self.t1)) * targets
        temp2 = (1 / (2 - self.t1)) * (torch.pow(targets, 2 - self.t1) - torch.pow(probabilities, 2 - self.t1))
        loss = temp1 - temp2

        loss = loss.sum(dim=-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss

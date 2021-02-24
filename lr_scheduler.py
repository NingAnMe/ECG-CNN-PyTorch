#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021-02-05 9:31
# @Author  : NingAnMe <ninganme@qq.com>
import math
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, StepLR


class ConcatLR(_LRScheduler):
    def __init__(self, optimizer, scheduler1, scheduler2, step_before, last_epoch=-1):
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.step_before = step_before
        super(ConcatLR, self).__init__(optimizer, last_epoch)

    def step(self):
        if self.last_epoch < self.step_before:
            self.scheduler1.step()
        else:
            self.scheduler2.step()
        super().step()

    def get_lr(self):
        if self.last_epoch < self.step_before:
            return self.scheduler1.get_last_lr()
        else:
            return self.scheduler2.get_last_lr()


def DelayedCosineAnnealingLR(optimizer, total_steps, pct_start=0.72):
    step_constant = math.floor(float(total_steps * pct_start) - 2)
    step_cosine = total_steps - step_constant
    print("step_constant : {}".format(step_constant))
    print("step_cosine : {}".format(step_cosine))
    constant = StepLR(optimizer, step_constant + 1)
    cosine = CosineAnnealingLR(optimizer, step_cosine, eta_min=0)
    scheduler = ConcatLR(optimizer, constant, cosine, step_constant)
    return scheduler

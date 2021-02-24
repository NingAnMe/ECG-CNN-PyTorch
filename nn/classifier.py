#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021-02-22 15:04
# @Author  : NingAnMe <ninganme@qq.com>

import numpy as np

import torch
import torch.nn.functional as F

import pytorch_lightning as pl

from loss import LabelSmoothCrossEntropyLoss, BiTemperedLogisticLoss, FocalLoss
from optimizer import Ranger, Ranger2000, Novograd, RangerLars
from lr_scheduler import DelayedCosineAnnealingLR
from logger import train_log


class Classifier(pl.LightningModule):
    def __init__(self, net,
                 loss_function='CrossEntropyLoss',
                 optim='Adam',
                 lr_scheduler=None,
                 lr=1e-3,
                 local=False,
                 epoch=0,
                 fmix=None):
        """
        :param net_name (str):  网络名称
        :param train_layer (str): 从某一层开始恢复训练
        :param loss_function (str):  损失函数名称
        :param optim (str):  优化器名称
        :param lr (float):  学习率
        :param local (bool):  是否本地训练
        """
        super(Classifier, self).__init__()
        self.m_net = net
        self.m_loss_function = self.configure_loss_function(loss_function)
        self.m_optim = optim
        self.m_lr = lr
        self.m_lr_scheduler = lr_scheduler
        self.m_accuracy_train = pl.metrics.Accuracy()
        self.m_accuracy_val = pl.metrics.Accuracy()
        if fmix:
            self.fmix = fmix
        else:
            self.fmix = None

        self.local = local
        self.m_epoch = epoch

        self.l_loss_train = None
        self.l_loss_val = None
        self.l_acc_train = None
        self.l_acc_val = None

    def forward(self, x):
        return self.m_net(x)

    def predict(self, x: np.ndarray):
        self.eval()
        self.freeze()
        x = torch.from_numpy(x).float().to(self.device)
        y = self.forward(x)
        sm = torch.nn.functional.softmax(y, dim=1)
        r = torch.argmax(sm, dim=1).to('cpu')
        return r.data

    @staticmethod
    def configure_loss_function(loss_function):
        # 损失函数
        if loss_function == 'CrossEntropyLoss':
            loss_func = torch.nn.CrossEntropyLoss()
        elif loss_function == 'LabelSmoothCrossEntropyLoss':
            loss_func = LabelSmoothCrossEntropyLoss(smoothing=0.2)
        elif loss_function == 'BiTemperedLogisticLoss':
            loss_func = BiTemperedLogisticLoss(reduction='sum', t1=0.8, t2=1.4, label_smoothing=0.2)
        elif loss_function == 'FocalLoss':
            loss_func = FocalLoss(reduction='sum', alpha=0.25, gamma=2, smooth_eps=0.2, class_num=5)
        else:
            raise ValueError(loss_function)
        return loss_func

    def configure_optimizers(self):
        config = {
            'frequency': 1,
            'strict': True,
        }

        if self.m_optim == 'Adam':
            optim = torch.optim.Adam(self.parameters(), lr=self.m_lr)
        elif self.m_optim == 'Ranger':
            optim = Ranger(self.parameters(), lr=self.m_lr)
        elif self.m_optim == 'Ranger2020':
            optim = Ranger2000(self.parameters(), lr=self.m_lr)
        elif self.m_optim == 'Novograd':
            optim = Novograd(self.parameters(), lr=self.m_lr)
        elif self.m_optim == 'RangerLars':
            optim = RangerLars(self.parameters(), lr=self.m_lr)
        else:
            raise ValueError(self.m_optim)
        config['optimizer'] = optim

        # 继续训练的时候，需要基于恢复的optimizer重定义lr_scheduler
        if self.m_lr_scheduler == 'CosineAnnealingLR':
            MAX_STEP = int(10e4)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, MAX_STEP, eta_min=1e-5)
            config['lr_scheduler'] = lr_scheduler
            config['interval'] = 'step'
        elif self.m_lr_scheduler == 'CosineAnnealingWarmRestarts':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optim, T_0=15, T_mult=1, eta_min=1e-6, last_epoch=-1)
            config['lr_scheduler'] = lr_scheduler
            config['interval'] = 'epoch'
        elif self.m_lr_scheduler == 'ReduceLROnPlateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, patience=1)
            config['lr_scheduler'] = lr_scheduler
            config['interval'] = 'epoch'
            config['monitor'] = 'loss_val'
        elif self.m_lr_scheduler == 'DelayedCosineAnnealingLR':
            lr_scheduler = DelayedCosineAnnealingLR(optim, self.m_epoch, pct_start=0.72)
            config['lr_scheduler'] = lr_scheduler
            config['interval'] = 'epoch'

        return config

    def training_step(self, batch, batch_idx):
        # print('lr : {}\n'.format(self.optimizers().__dict__['param_groups'][0]['lr']))

        x, y = batch

        if self.fmix:
            x = self.fmix(x)

        pred = self.forward(x)

        if self.fmix:
            loss = self.fmix.loss(pred, y, loss_function=self.m_loss_function)
        else:
            loss = self.m_loss_function(pred, y)
        self.m_accuracy_train.update(pred, y)

        return loss

    def training_epoch_end(self, outputs):
        print("training_epoch_end")
        nums = len(outputs)
        self.l_loss_train = None
        for i in outputs:
            self.l_loss_train = self.l_loss_train + i['loss'] if self.l_loss_train is not None else i['loss']
        self.l_loss_train /= nums
        self.l_acc_train = self.m_accuracy_train.compute()
        self.m_accuracy_train.reset()

        self.log('loss_train', self.l_loss_train, on_epoch=True)
        self.log('acc_train', self.l_acc_train, on_epoch=True)

        train_log(loss_train=self.l_loss_train,
                  acc_train=self.l_acc_train,
                  loss_val=self.l_loss_val,
                  acc_val=self.l_acc_val)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = self.m_loss_function(pred, y)
        self.m_accuracy_val.update(pred, y)

        return loss

    def validation_epoch_end(self, outputs):
        print("validation_epoch_end")
        nums = len(outputs)
        self.l_loss_val = None
        for i in outputs:
            self.l_loss_val = self.l_loss_val + i if self.l_loss_val is not None else i
        self.l_loss_val /= nums

        self.l_acc_val = self.m_accuracy_val.compute()
        self.m_accuracy_val.reset()

        self.log('loss_val', self.l_loss_val, on_epoch=True)
        self.log('acc_val', self.l_acc_val, on_epoch=True)

    def test_step(self, batch, batch_idx):
        """
        model.eval() and torch.no_grad() are called automatically for testing.
        The test loop will not be used until you call: trainer.test()
        .test() loads the best checkpoint automatically
        """
        x, y = batch
        pred = self.forward(x)
        loss = self.m_loss_function(pred, y)
        return loss

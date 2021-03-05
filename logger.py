#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020-12-14 17:24
# @Author  : NingAnMe <ninganme@qq.com>


def train_log(loss_train=None, acc_train=None, loss_val=None, acc_val=None):

    if loss_train is not None and acc_train is not None:
        print("Log > acc_train:", '{:0.4f}'.format(acc_train), "- loss_train:", '{:0.4f} >>>>>>'.format(loss_train))

    if loss_val is not None and acc_val is not None:
        print("Log > acc_val  :", '{:0.4f}'.format(acc_val), "- loss_val  :", '{:0.4f} ######'.format(loss_val))

    # loss_train = loss_train if loss_train is not None else 0.0
    # acc_train = acc_train if acc_train is not None else 0.0
    # loss_val = loss_val if loss_val is not None else 0.0
    # acc_val = acc_val if acc_val is not None else 0.0
    # print("Log > acc_train:", '{:0.4f}'.format(acc_train), "- acc_val:", '{:0.4f}'.format(acc_val),
    #       "- loss_train:", '{:0.4f}'.format(loss_train), "- loss_val:", '{:0.4f}'.format(loss_val))

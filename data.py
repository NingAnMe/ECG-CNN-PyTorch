#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020-12-23 22:27
# @Author  : NingAnMe <ninganme@qq.com>
import os

import pandas as pd

from sklearn.model_selection import StratifiedKFold

from path import DATA_PATH

train_csv = os.path.join(DATA_PATH, 'train.csv')

image_dir = os.path.join(DATA_PATH, 'train_images')


def get_full_path_image(image):
    return os.path.join(image_dir, image)


def get_train_data(n_splits=5, shuffle=True, random_state=None):
    data_df = pd.read_csv(train_csv, index_col=None)
    # data_df = data_df.iloc[:20]
    print(data_df.head())
    data_df['image'] = data_df['image_id'].map(get_full_path_image)
    print(data_df.head())

    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    x = data_df.image
    y = data_df.label
    for train_idx, val_idx in skf.split(x, y):
        yield x[train_idx], y[train_idx], x[val_idx], y[val_idx]


if __name__ == '__main__':
    next(get_train_data())

#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020-12-14 0:45
# @Author  : NingAnMe <ninganme@qq.com>

from torch.utils.data import Dataset

import os

import numpy as np
from PIL import Image
from PIL import ImageFile
# import cv2
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import torch

from path import DATA_PATH

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ClassifierDatasetTest(torch.utils.data.Dataset):
    def __init__(self, images, image_size=256, transform=None):
        super().__init__()
        self.images = images
        self.transform = transform

        if self.transform:
            self.tx = self.transform
        else:
            self.tx = A.Compose([
                # A.CenterCrop(image_size, image_size, p=1.),
                A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR, always_apply=True),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True),
                ToTensorV2(),
            ])

    def get_x(self, img_path: str):
        image = Image.open(img_path)
        image = np.array(image)
        return self.tx(image=image)['image']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_id = self.images[idx]
        x = self.get_x(os.path.join(DATA_PATH, img_id))
        return x, img_id


class ClassifierDatasetVal(torch.utils.data.Dataset):
    """Class for getting individual transformations and data
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
        transformI = Input Images transformation (default: None)
        transformM = Input Labels transformation (default: None)
    Output:
        tx = Transformed images
        lx = Transformed labels"""

    def __init__(self, images, labels, image_size: int = 256, transform=None):
        super(ClassifierDatasetVal, self).__init__()
        self.images = images
        self.labels = labels
        self.transform = transform  # return torch.Tensor

        if self.transform:
            self.tx = self.transform
        else:
            self.tx = A.Compose([
                # A.CenterCrop(image_size, image_size, p=1.0),
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                ToTensorV2(),
            ])

    def get_x(self, img_path: str):
        image = Image.open(img_path)
        image = np.array(image)
        return self.tx(image=image)['image']

    @staticmethod
    def get_y(label: str):

        return int(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        x = self.get_x(self.images.iloc[idx])
        y = self.get_y(self.labels.iloc[idx])
        return x, y


class ClassifierDatasetTrain(torch.utils.data.Dataset):
    """Class for getting individual transformations and data
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
        transformI = Input Images transformation (default: None)
        transformM = Input Labels transformation (default: None)
    Output:
        tx = Transformed images
        lx = Transformed labels"""

    def __init__(self, datas, labels):
        super(ClassifierDatasetTrain, self).__init__()
        self.datas = torch.from_numpy(datas).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx: int):
        x = self.datas[idx]
        y = self.labels[idx]
        return x.reshape(1, -1), y

# -*- coding: utf-8 -*-
import os
import json
from collections import Counter

from flyai.data_helper import DataHelper
from flyai.framework import FlyAI

import pandas as pd
import numpy as np

import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgbm
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN


from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

# from data import get_train_data
from dataset import ClassifierDatasetTrain

from path import MODEL_PATH, DATA_PATH

from nn.classifier import Classifier
from nn.msresnet1d import MSResNet

'''
此项目为FlyAI2.0新版本框架，数据读取，评估方式与之前不同
2.0框架不再限制数据如何读取
样例代码仅供参考学习，可以自己修改实现逻辑。
模版项目下载支持 PyTorch、Tensorflow、Keras、MXNET、scikit-learn等机器学习框架
第一次使用请看项目中的：FlyAI2.0竞赛框架使用说明.html
使用FlyAI提供的预训练模型可查看：https://www.flyai.com/models
学习资料可查看文档中心：https://doc.flyai.com/
常见问题：https://doc.flyai.com/question.html
遇到问题不要着急，添加小姐姐微信，扫描项目里面的：FlyAI小助手二维码-小姐姐在线解答您的问题.png
'''
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# 所有标签列表
all_label_list = ['N', 'F', 'Q', 'S', 'V']
label_encoder = LabelEncoder()
label_encoder.fit(all_label_list)


class Main(FlyAI):
    """
    项目中必须继承FlyAI类，否则线上运行会报错。
    """
    X = None
    y = None

    def download_data(self):
        # 根据数据ID下载训练数据
        data_helper = DataHelper()
        data_helper.download_from_ids("ArrhythmiaClassification")

    def deal_with_data(self):
        """
        处理数据，没有可不写。
        :return:
        """
        # with open(os.path.join(DATA_PATH, 'ArrhythmiaClassification', 'train.csv'), 'r') as f:
        #     for line in f.readlines():
        #         print(line.strip())
        if not os.path.isfile(os.path.join(DATA_PATH, 'ArrhythmiaClassification', 'train.csv')):
            self.download_data()
        df = pd.read_csv(os.path.join(DATA_PATH, 'ArrhythmiaClassification', 'train.csv'))

        print(df['label'].value_counts())
        y = label_encoder.transform(df['label'])
        print(Counter(y))

        X = np.zeros((len(df), 187), dtype=np.float)
        for idx, data in enumerate(df.values):
            temp_input = json.loads(data[0])
            X[idx] = temp_input

        # 欠采样
        sampling_strategy = {0: 603, 1: 6000, 2: 6000, 3: 2579, 4: 6000}
        # sampling_strategy = {0: 603, 1: 2800, 2: 2700, 3: 2579, 4: 2650}
        rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
        X_res, y_res = rus.fit_resample(X, y)
        print(Counter(y_res))

        # 过采样
        # sampler = SMOTEENN(random_state=0)
        # X_res, y_res = sampler.fit_resample(X_res, y_res)
        # print(Counter(y_res))

        # self.X = X_res[:, np.newaxis, :]
        # self.y = y_res
        self.X = X_res
        self.y = y_res

    def data2csv(self):
        df = pd.DataFrame(self.X)
        # df['label'] = self.y
        df.to_csv('train.csv', index=False)

    def train_xgboot(self, X_train, y_train):
        model = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0)
        model.fit(X_train, y_train)
        return model

    def train_lightgbm(self, X_train, y_train):
        model = lgbm.LGBMClassifier()
        model.fit(X_train, y_train)
        return model

    def train_nn(self, X_train, y_train, X_test, y_test):
        CFG = {
            'fold_num': 5,
            'seed': 42,
            'num_classes': 5,  # 类别数量
            'net_name': 'cnn',
            # tf_efficientnet_b4_ns, resnest50d, gluon_seresnext50_32x4d, tf_efficientnet_b5_ns
            'pretrained': True,
            'loss_name': 'BiTemperedLogisticLoss',  # CrossEntropyLoss  LabelSmoothCrossEntropyLoss  BiTemperedLogisticLoss
            'optim_name': 'Adam',  # Adam  Ranger Ranger2020  Novograd  RangerLars
            'lr_scheduler_name': 'DelayedCosineAnnealingLR',
            # CosineAnnealingWarmRestarts ReduceLROnPlateau DelayedCosineAnnealingLR
            'fmix': False,  # 是否开启FMix
            'es_patience': 5,  # 提前停止可忍受的epoch轮次
            'img_size': 512,  # 图片缩放到的尺寸
            'epochs': 100,  # 最大训练轮次
            'train_bs': 1024,  # 训练batch大小
            'valid_bs': 1024,  # 预测batch大小
            'T_0': 10,
            'lr': 1e-3,
            'min_lr': 1e-6,
            'weight_decay': 1e-6,
            'num_workers': 8,
            'accum_iter': 1,  # suppoprt to do batch accumulation for backprop with effectively larger batch size
            'verbose_step': 1,
            'device': 'cuda:0',
        }

        # net = AnomalyClassifier(1, 5)

        net = MSResNet(input_channel=1, num_classes=5)
        # print(net)
        # for name, parameters in net.named_parameters():
        #     print(name, ':', parameters.size())
        # exit()
        model = Classifier(net,
                           loss_function=CFG['loss_name'],
                           optim=CFG['optim_name'],
                           lr=CFG['lr'],
                           lr_scheduler=CFG['lr_scheduler_name'],
                           epoch=CFG['epochs'],
                           fmix=None)

        filename = '{net_name}-'.format(net_name=CFG["net_name"]) + '{epoch}-{step}-{acc_val:.4f}-{acc_train:.4f}'
        checkpoint_callback = ModelCheckpoint(monitor='acc_val', mode='max', filename=filename)
        # early_stop = EarlyStopping(monitor='acc_val', patience=CFG['es_patience'], mode='max')

        trainer = pl.Trainer(gpus=1,
                             default_root_dir=MODEL_PATH,
                             # callbacks=[checkpoint_callback, early_stop],
                             callbacks=[checkpoint_callback],
                             max_epochs=CFG['epochs'],
                             accumulate_grad_batches=CFG['accum_iter'])

        print(Counter(y_train))
        print(Counter(y_test))
        dataset_train = ClassifierDatasetTrain(X_train, y_train)
        dataset_val = ClassifierDatasetTrain(X_test, y_test)

        train_loader = DataLoader(dataset_train, batch_size=CFG['train_bs'], num_workers=CFG['num_workers'])
        val_loader = DataLoader(dataset_val, batch_size=CFG['valid_bs'], num_workers=CFG['num_workers'])
        trainer.fit(model, train_loader, val_loader)
        return model

    def train(self):
        """
        训练模型，必须实现此方法
        :return:
        """
        main.deal_with_data()
        # exit()
        X = self.X
        y = self.y
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
        scores = list()
        cms = list()
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # 上采样
            sampling_strategy = {0: 4800, 3: 4800}
            ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
            X_train, y_train = ros.fit_resample(X_train, y_train)
            print(Counter(y_train))
            X_train = X_train[:, np.newaxis, :]
            X_test = X_test[:, np.newaxis, :]
            # model = self.train_xgboot(X_train, y_train)
            # model = self.train_lightgbm(X_train, y_train)
            model = self.train_nn(X_train, y_train, X_test, y_test)
            y_hat = model.predict(X_test)
            score = accuracy_score(y_test, y_hat)
            cm = confusion_matrix(y_test, y_hat)
            scores.append(score)
            cms.append(cm)
            print(Counter(y_test))
            print('score: {}'.format(score))
            print(cm)
            print(confusion_matrix(y_test, y_hat, normalize='true'))
            # model_save = os.path.join(MODEL_PATH, 'model.pkl')
            # joblib.dump(model, model_save)
        print(scores, np.mean(scores))


if __name__ == '__main__':
    main = Main()
    main.train()

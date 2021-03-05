# -*- coding: utf-8 -*
import os
from collections import Counter
from flyai.framework import FlyAI

import torch

import joblib
from path import MODEL_PATH
import numpy as np
from main import label_encoder, Main
from nn.classifier import Classifier
from nn.msresnet1d import MSResNet


class Prediction(FlyAI):
    def load_model(self):
        """
        模型初始化，必须在此方法中加载模型
        """
        # CNN
        self.models = list()
        for i in range(0, 5):
            checkpoint_dir = os.path.join(MODEL_PATH, f'lightning_logs/version_{i}/checkpoints')
            if not os.path.isdir(checkpoint_dir):
                continue
            device = torch.device('cuda:0')
            # net = AnomalyClassifier(1, 5)
            net = MSResNet(input_channel=1, num_classes=5)
            checkpoint = os.path.join(checkpoint_dir, os.listdir(checkpoint_dir)[0])
            model = Classifier.load_from_checkpoint(checkpoint, net=net)
            model.to(device)
            self.models.append(model)

        # xgboost
        # self.model = joblib.load(os.path.join(MODEL_PATH, 'model.pkl'))

    def predict(self, data):
        """
        模型预测返回结果
        :param data: 评估传入样例 data 为list类型的一条数据信息
        :return: 模型预测成功中, 直接返回预测的类别
        """
        labels = list()
        data = np.array(data)
        data = data.reshape(1, -1)

        # CNN
        data = data[:, np.newaxis, :]
        for model in self.models:
            y_hat = model.predict(data)
            label = label_encoder.inverse_transform(y_hat)
            labels.append(label.tolist()[0])
        c = Counter(labels)
        return c.most_common(1)[0][0]

        # xgboost
        # y_hat = self.model.predict(data)
        # label = label_encoder.inverse_transform(y_hat)
        # return label[0]


if __name__ == '__main__':
    main = Main()
    main.deal_with_data()
    prediction = Prediction()
    prediction.load_model()
    for i in range(85, 91):
        label = prediction.predict(main.X[i])
        print(label, main.y[i])

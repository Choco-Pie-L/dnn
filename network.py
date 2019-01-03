# coding: utf-8
import numpy as np
from layers import Affine, Relu, Dropout, SoftmaxWithLoss, Sigmoid


class OriginalNet:

    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # 重みとバイアスの初期化
        W1 = 0.01 * np.random.randn(I, H)
        b1 = np.zeros(H)
        W2 = 0.01 * np.random.randn(H, O)
        b2 = np.zeros(O)

        # ネットワーク層追加
        self.layers = [
            Affine(W1, b1),
            Relu(),
            Affine(W2, b2),
            Sigmoid()
        ]

        # 損失関数
        self.loss_layer = SoftmaxWithLoss()

        # 全層の重みと勾配をリストにまとめる。
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

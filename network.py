# coding: utf-8
import numpy as np

class OriginalNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size
        
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)
        
        self.layers = [
            Affine(W1, b1),
            Relu(),
            Affine(W2, b2),
            Relu(),
            Dropout(0.1)
        ]
        self.loss_layer = SoftmaxWithLoss()
        
        # 全層の重みと勾配をリストにまとめる。
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layers.grads
        
        def predict(self, x):
            for layer in self.layers:
                x = layers.forward(x)
            return x
        def forward(self, x, t):
            score = self.predict(x)
            loss = self.loss_layer.forward(score, t)
            return loss
        
        def backward(self, dout=1):
            dout = self.loss_layer.forward(dout)
            for layer in reversed(self.layers):
                dout = layer.backward(dout)
            return dout
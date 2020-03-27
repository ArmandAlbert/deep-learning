import numpy as np
from needs import Relu, Sigmoid, Softmax_With_Loss, Affine
from collections import OrderedDict


class TwoLayerNet_Pro:  # 真正意义上的网络
    def __init__(self, input_size, hidden_size, output_size, w=0.01):

        self.params = {}  # 随机生成初始值，W1和W1权重值使用高斯随机分布生成，偏置b1和b2全为0
        self.params['W1'] = w * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = w * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 创建 生成层
        self.layers = OrderedDict()
        # 第一仿射层，a = W * x + b
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        # 第一激活层，Relu层
        self.layers['Relu1'] = Relu()
        # 第二仿射层
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        # 神经网络最后一层
        self.lastLayer = Softmax_With_Loss()

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def predict(self, x):
        for layer in self.layers:  # 分别计算Afffine1、Relu1、Affine2层
            x = layer.forward(x)
        return x  # x = a2(z2 = softmax(a2))

    def cal_accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)  # t若不是1维，进行正规化
        return np.sum(y == t) / float(len(t))

    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()  # 列表反向

        for layer in layers:  # 反向计算各层的导数
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads

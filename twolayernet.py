from needs import relu, sigmoid, softmax, cross_entrppy_error, cal_gradient, sigmoid_grad
from needs import Relu, Affine, Softmax_With_Loss, Sigmoid
from collections import OrderedDict
import numpy as np


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
        # # 第一激活层，Relu层
        self.layers['Relu1'] = Relu()
        # self.layers['Sigmoid1'] = Sigmoid()
        # 第二仿射层
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        # 神经网络最后一层
        self.lastLayer = Softmax_With_Loss()

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def predict(self, x):
        for layer in self.layers.values():  # 分别计算Afffine1、Relu1、Affine2层
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


class TwoLayerNet():

    def __init__(self, input_size, hidden_size, output_size, w=0.01):
        self.params = {}  # 随机生成初始值，W1和W1权重值使用高斯随机分布生成，偏置b1和b2全为0
        self.params['W1'] = w * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = w * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):  # 检测函数，返回检测结果
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, W1) + b1
        # z1 = sigmoid(a1)
        z1 = relu(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y

    def loss(self, x, t):  # 计算损失函数的值
        y = self.predict(x)
        return cross_entrppy_error(y, t)

    def numerical_differential(self, x, t):  # 数值微分方法计算梯度
        def loss_W(W): return self.loss(x, t)
        grads = {}
        grads['W1'] = cal_gradient(loss_W, self.params['W1'])
        grads['b1'] = cal_gradient(loss_W, self.params['b1'])
        grads['W2'] = cal_gradient(loss_W, self.params['W2'])
        grads['b2'] = cal_gradient(loss_W, self.params['b2'])
        return grads

    def gradient(self, x, t):  # 误差反向传播算法高速计算梯度
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

    def cal_accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        return np.sum(y == t) / float(len(t))


class TwoLayerNet_Pro2:
    def __init__(self, input_size, hidden_size1, hidden_size2,hidden_size3, output_size, w=0.01):

        self.params = {}  # 随机生成初始值，W1和W1权重值使用高斯随机分布生成，偏置b1和b2全为0
        self.params['W1'] = w * np.random.randn(input_size, hidden_size1)
        self.params['b1'] = np.zeros(hidden_size1)
        self.params['W2'] = w * np.random.randn(hidden_size1, hidden_size2)
        self.params['b2'] = np.zeros(hidden_size2)
        self.params['W3'] = w * np.random.randn(hidden_size2, hidden_size3)
        self.params['b3'] = np.zeros(hidden_size3)
        self.params['W4'] = w * np.random.randn(hidden_size3, output_size)
        self.params['b4'] = np.zeros(output_size)

        # 创建 生成层
        self.layers = OrderedDict()
        # 第一仿射层，a = W * x + b
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        # # 第一激活层，Relu层
        self.layers['Relu1'] = Relu()
        # self.layers['Sigmoid1'] = Sigmoid()
        # 第二仿射层
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['Relu3'] = Relu()
        self.layers['Affine4'] = Affine(self.params['W4'], self.params['b4'])
        # 神经网络最后一层
        self.lastLayer = Softmax_With_Loss()

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def predict(self, x):
        for layer in self.layers.values():  # 分别计算
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

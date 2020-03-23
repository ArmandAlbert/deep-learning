from functions import sigmoid, softmax, cross_entrppy_error, cal_gradient, sigmoid_grad
import numpy as np


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
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y

    def loss(self, x, t):  # 计算损失函数的值
        y = self.predict(x)
        return cross_entrppy_error(y, t)

    def numerical_gradient(self, x, t):  # 数值微分方法计算梯度
        def loss_W(W): return self.loss(x, t)
        grads = {}
        grads['W1'] = cal_gradient(loss_W, self.params['W1'])
        grads['b1'] = cal_gradient(loss_W, self.params['b1'])
        grads['W2'] = cal_gradient(loss_W, self.params['W2'])
        grads['b1'] = cal_gradient(loss_W, self.params['b2'])
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
        y = np.argmax(y)
        return np.sum(y == t) / float(len(t))

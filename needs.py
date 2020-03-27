import numpy as np


def relu(x):
    mask = (x <= 0)
    out = x.copy()
    out[mask] = 0
    return out


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):  # new
    return (1.0 - sigmoid(x)) * sigmoid(x)


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


def cross_entrppy_error(y, t):  # 交叉熵误差
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def cal_gradient(f, x):  # 梯度下降，计算梯度
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    for i in range(x.shape[0]):
        tmp = x[i]
        x[i] = tmp + h
        fxh1 = f(x)
        x[i] = tmp - h
        fxh2 = f(x)
        grad[i] = (fxh1 - fxh2) / (2 * h)
        x[i] = tmp
    return grad


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):  # 前向传播
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):  # 反向传播
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid:
    def __init__(self):
        self.out = None  # 正向输入

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out


class Affine:
    def __init__(self, W, b):
        self.b = b
        self.W = W
        self.x = None
        self.db = None
        self.dW = None

    def forward(self, x):  # 正向，矩阵乘积运算
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):  # 反向，计算反向误差..
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


class Softmax_With_Loss:
    def __init__(self):
        self.loss = None
        self.t = None
        self.y = None

    def forward(self, x, t):  # 正向，计算损失函数
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entrppy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size  # 反向，除以批大小，返回的是单个数据的误差
        return dx

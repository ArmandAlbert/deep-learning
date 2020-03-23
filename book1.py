# 使用 sample_pickle 识别

from source.dataset.mnist import load_mnist
from functions import sigmoid, softmax, cross_entrppy_error
import numpy as np
import pickle
import sys
import os
sys.path.append(os.pardir)


def init_network():
    with open("sample_weight.pkl", 'rb') as op:
        return pickle.load(op)


def predict(x, net):  # 使用sample_pickle计算并返回识别结果(0-9)
    a1 = np.dot(x, net['W1'])+net['b1']
    z1 = sigmoid(a1)
    a2 = np.dot(z1, net['W2'])+net['b2']
    z2 = sigmoid(a2)
    a3 = np.dot(z2, net['W3'])+net['b3']
    z3 = softmax(a3)
    return z3


def batch_verify(x, t, net):  # 100 批处理
    batch_step = 100
    test_acc = 0
    for i in range(0, len(t), batch_step):
        test_acc += np.sum(t[i:i+batch_step] ==
                           predict(x[i:i+batch_step], net))
    return test_acc / len(t)


def verify(x, t, net):  # 直接计算准确度
    # return np.sum((t == predict(x, net)) == True) / len(t)
    test_acc = 0
    for i in range(len(x)):
        y = predict(x[i], net)
        p = np.argmax(y)
        if p == t[i]:
            test_acc += 1
    return test_acc / len(x)


def test_cross_entropy_error(x, t, net):
    y_ = predict(x[0], net)
    print(y_, t[0])
    print(len(t[0]))
    return -np.sum(t*np.log(y_))


def main():
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, flatten=True, one_hot_label=True)
    # print(batch_verify(x_test, t_test, init_network()))
    # print(verify(x_test, t_test, init_network()))
    print(test_cross_entropy_error(x_test, t_test, init_network()))


main()
# init_network()

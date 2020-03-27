# 使用twolayernet来进行学习
from twolayernet import TwoLayerNet, TwoLayerNet_Pro, TwoLayerNet_Pro2
import numpy as np
import matplotlib.pyplot as plt
from source.dataset.mnist import load_mnist
import sys
import os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定


class instant():
    def __init__(self):
        (self.x_train, self.t_train), (x_test, t_test) = load_mnist(
            normalize=True, one_hot_label=True)
        self.iter_num = 10000  # 学习次数
        self.learn_rate = 0.3  # 学习率
        self.batch_size = 100  # 批处理数量
        self.train_size = self.x_train.shape[0]  # 总训练数量

        self.network = TwoLayerNet_Pro(  # 初始化2层网络
            input_size=784, hidden_size=50, output_size=10)

        # self.network = TwoLayerNet_Pro2(  # 初始化3层网络
        #     input_size=784, hidden_size1=100, hidden_size2=50, output_size=10)

        # self.network = TwoLayerNet_Pro2(  # 初始化4层网络
        #     input_size=784, hidden_size1=200, hidden_size2=100, hidden_size3=50, output_size=10)

        self.train_loss = list()
        self.acc_list = list()

    def draw(self):
        iters = list()

        # for i in range(int(self.iter_num / 5)):
        #     iters.append(i + 1)

        for i in range(int(self.iter_num)):
            iters.append(i + 1)

        # plt.plot(iters, self.train_loss)
        plt.plot(iters, self.acc_list)
        plt.show()

    def learn(self):
        for i in range(self.iter_num):
            # 随机选择100个数据
            batches = np.random.choice(self.train_size, self.batch_size)
            x_bat = self.x_train[batches]
            t_bat = self.t_train[batches]

            # 计算梯度
            # grad = self.network.numerical_differential(
            # x_bat, t_bat)  # 数值微分法，慢速
            grad = self.network.gradient(x_bat, t_bat)  # 误差反向传播法（快速）
            # print(grad['W1'].shape)
            # print(grad['W2'].shape)
            # print(grad['b1'].shape)
            # print(grad['b1'].shape)
            # 梯度下降，更新权值和偏置参数
            for j in ['W1', 'W2', 'b1', 'b2']:
                self.network.params[j] -= self.learn_rate * grad[j]
            # self.train_loss.append(self.network.loss(x_bat, t_bat))  # 计算损失函数
            print('第' + str(i) + '个完成')

            self.acc_list.append(
                self.network.cal_accuracy(x_bat, t_bat))  # 每个采集

            # if i % 5 == 0:  # 每五个采集一次
            #     self.acc_list.append(self.network.cal_accuracy(x_bat, t_bat))


# try:
#     A = instant()
#     A.learn()
#     A.draw()
# except:
#     del(A)
#     print("error accured")
A = instant()
A.learn()
A.draw()

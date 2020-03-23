import numpy as np
from functions import numerical_descent


# 实现梯度下降的计算，寻找函数最小值（的近似值），学习率0.01,学习次数100
def gradient_descent(f, init_x, lr=0.01, nums=100):
    x = init_x
    for i in range(nums):
        x -= lr*numerical_descent(f, x)
    return x


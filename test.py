from source.dataset.mnist import load_mnist
import numpy as np

(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=True, one_hot_label=True)

batch_size = 100
batch_mask = np.random.choice(10000, batch_size)
x_bat = x_train[batch_mask]
t_bat = t_train[batch_mask]
print(batch_mask)
print(t_bat)

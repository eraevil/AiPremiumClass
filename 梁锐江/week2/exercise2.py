import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 数据集
X, y = make_classification(n_features=20)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 权重参数
theta = np.random.randn(1, 20)
bias = 0

# 超参数
# 学习率
lr = 1e-3
# 迭代次数
epoch = 10000


# 前项计算
def foward(x):
    z = np.dot(theta, x.T) + bias
    y_hat = 1 / (1 + np.exp(-z))
    return y_hat


# 损失函数
def loss(y, y_hat):
    e = 1e-8
    return -y * np.log(y_hat + e) - (1 - y) * np.log(1 - y_hat + e)


# 计算梯度
def calc_gradient(x, y, y_hat):
    # 特征向量的数量
    m = x.shape[-1]

    # 每次变化的大小  np.dot((70,) * (70,20))
    delta_theta = np.dot(y_hat - y, x) / m
    delta_bias = np.mean(y_hat - y)

    return delta_theta, delta_bias


if __name__ == '__main__':

    # 保存训练好的参数
    np.savez('model_test_data.npz', x_test=X_test, y_test=y_test)

    for i in range(epoch):
        y_hat = foward(X_train)

        delta_theta, delta_bias = calc_gradient(X_train, y_train, y_hat)

        theta = theta - lr * delta_theta
        bias = bias - lr * delta_bias

        if i % 1000 == 0:
            acc = np.mean(np.round(y_hat) == y_train)
            print('epoch:', i, 'loss:', np.mean(loss(y_train, y_hat)), 'acc:', acc)

    # 保存训练好的参数
    np.savez('model_parameters.npz', theta=theta, bias=bias)

    data = np.load('model_parameters.npz')
    print(data['theta'])
    print(data['bias'])

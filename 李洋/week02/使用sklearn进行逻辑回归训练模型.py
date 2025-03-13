# -*- coding: utf-8 -*-
# @Date    : 2025/3/5 9:31
# @Author  : Lee
from sklearn.datasets import load_iris #数据集引用
from sklearn.model_selection import train_test_split #对数据集引用进行拆分
import numpy as np
import pickle
# 进行数据sigmoid处理
def forward(w, x, b):
    z = np.dot(w, x.T) + b
    sigmoid = 1 / (1 + np.exp(-z))
    return sigmoid


# 计算损失
def loss(y_hat, y):
    e = 1e-8
    return -y * np.log(y_hat + e) - (1 - y) * np.log(1 - (y_hat + e))


# 计算梯度
def clce_gradient(x, y, y_hat):
    m = X_train.shape[1]
    delta_w = np.dot((y_hat - y), x) / m
    delta_b = np.mean((y_hat - y))
    return delta_w, delta_b


# 进行训练
def train(X_train, y_train, weights, bias, lr, epochs):
    for i in range(epochs):
        y_hat = forward(weights, X_train, bias)
        loss_val = loss(y_hat, y_train)
        w, b = clce_gradient(X_train, y_train,y_hat)
        weights -= lr * w
        bias -= lr * b

        if i % 500 == 0:
            acc = np.mean(np.round(y_hat) == y_train)
            print("Epoch: %d, loss: %.4f, acc: %.4f" % (i, np.mean(loss_val), acc))
    return weights, bias


if __name__ == "__main__":
    # 数据导入
    X, y = load_iris(return_X_y=True)
    X = X[:100, :2]
    y = y[:100]
    # 数据集拆分验证集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # 定义超参数
    weights = np.zeros(2)
    # 定义偏执参数b
    bias = 0
    # 定义学习率
    lr = 0.001
    # 定义训练次数
    epochs = 5000
    #得到权重和偏执参数
    weights,bias=train(X_train,y_train,weights,bias,lr,epochs)
    #保存模型
    with open('model.pkl','wb') as f:
        pickle.dump((weights,bias),f)



# encoding: utf-8
# @File  : numpy_logistic_train.py
# @Author: GUIFEI
# @Desc :  numpy 实现逻辑回归示例训练
# @Date  :  2025/03/04

import numpy as np
import matplotlib.pyplot as plt

def forward(theta, x, bias):
    """
    定义模型运算函数，将线性方程计算结果转换为概率模型
    :param theta: 系数矩阵
    :param x: 自变量
    :param bias: 截距
    :return: 预测值
    """
    # 定义线性方程组
    Z = np.dot(theta, x.T) + bias
    # 使用sigmoid 函数将线性方程计算结果转化为概率模型
    y_hat = 1 / (1 + np.exp(-Z))
    return y_hat

def loss_fun(y_hat, y):
    """
    定义损失函数，注意需要添加一个极小值，避免出现log 0 的情形
    :param y_hat: 预测值
    :param y: 真实值
    :return: 返回本轮计算损失函数的损失
    """
    e = 1e-8
    # 定义损失函数，此处加 e 是为了避免log 0 的情况出现，因为在log 0是没有意义的
    return -y * np.log(y_hat + e) - (1 - y) * np.log(1 - y_hat + e)


def calc_gradient(x, y, y_hat):
    """
    计算梯度（权重）及 截距（bias）
    :param x: 自变量
    :param y: 因变量
    :param y_hat: 预测值
    :return: 返回计算得到的系数矩阵、截距
    """
    m = x.shape[-1]
    delta_theta = np.dot((y_hat - y), x) / m
    delta_bias = np.mean(y_hat - y)
    return delta_theta, delta_bias

def train(theta, bias, x, y, lr, iter_num):
    """
    训练函数
    :param theta: 系数矩阵
    :param bias: 截距
    :param x: 训练集自变量
    :param y: 训练集应变量
    :param iter_num: 迭代次数
    :return:
    """
    iter = 0
    loss_history = []
    while iter < iter_num:
        # 2. 模型运算
        y_hat = forward(theta, x, bias)
        # 3. 计算损失的均值
        loss = np.mean(loss_fun(y_hat, y))
        loss_history.append(loss)
        # 4. 计算梯度
        delta_theta, delta_bias = calc_gradient(x, y, y_hat)
        # 5. 更新参数
        theta = theta - lr * delta_theta
        bias = bias - lr * delta_bias
        if iter % 100 == 0:
            accuracy = np.mean(np.round(y_hat) == y)
            print(f"iter: {iter}, loss: {np.mean(loss)}, acc:{accuracy}")
        iter += 1
    # 绘制损失随迭代次数的变化
    plot_x = np.arange(0, iter, 1)
    plt.plot(plot_x, loss_history)
    plt.show()



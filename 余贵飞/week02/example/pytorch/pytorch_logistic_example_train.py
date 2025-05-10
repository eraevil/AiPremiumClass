# encoding: utf-8
# @File  : pytorch_logistic_example_train.py
# @Author: GUIFEI
# @Desc : 使用pytorch实现逻辑回归
# @Date  :  2025/03/05
import torch
import numpy as np


if __name__ == '__main__':
    # 初始化参数
    # 初始化权总张量，且设置自动跟踪梯度
    theta = torch.randn(1, 20, requires_grad=True, dtype=torch.float)
    # 初始化截距张量，设置自动跟踪梯度
    bias = torch.randn(1, requires_grad=True, dtype=torch.float)
    # 学习率
    lr = 1e-3
    # 读取训练数据
    X = np.load("../../train_X.npy")
    y = np.load("../../train_y.npy")
    train_X = torch.tensor(X, dtype=torch.float)
    train_y = torch.tensor(y, dtype=torch.float)
    print(train_X.shape, train_y.shape)

    iter_num = 10000
    for i in range(iter_num):
        # 向前计算
        # 定义线性方程
        z = torch.nn.functional.linear(train_X, theta, bias)
        # 使用sigmoid函数转换线性回归为逻辑回归的概率模型
        z = torch.sigmoid(z)

        # 计算损失
        loss = torch.nn.functional.binary_cross_entropy(z.squeeze(1), train_y, reduction='mean')

        # 计算梯度
        loss.backward()

        # 更新参数
        # 关闭梯度计算跟踪
        with torch.autograd.no_grad():
            # 更新权重梯度值
            theta = theta - lr * theta.grad
            # 清空本次计算的梯度（因为梯度是累加计算，不清空就累加, 详情参考pytorch 计算图自动微分的原理)
            # theta.grad.zero_()
            # 更新偏置项梯度
            bias = bias - lr * bias.grad
            # 清空本次计算的梯度（因为梯度是累加计算，不清空就累加, 详情参考pytorch 计算图自动微分的原理)
            # bias.grad.zero_()
        print("train loss: {loss.item():.4f}")


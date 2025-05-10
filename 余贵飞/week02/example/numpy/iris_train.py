# encoding: utf-8
# @File  : iris_train.py
# @Author: GUIFEI
# @Desc : 鸢尾花数据集训练
# @Date  :  2025/03/05
import numpy as np
import numpy_logistic_train as nlt

if __name__ == '__main__':
    # 定义模型初始参数
    # 定义theta 其实参数符合标准正太分布 （均值0，标准差1）的随机数
    theta = np.random.randn(1, 4)
    # 截距
    bias = 0
    # 学习率
    lr = 0.01
    # 模型训练轮数
    iter_num = 1e4
    # 加载数据
    train_X = np.load("../../iris_train_X.npy")
    train_y = np.load("../../iris_train_y.npy")
    # 开始训练
    nlt.train(theta, bias, train_X, train_y, lr, iter_num)
    # 保存训练后得到的权重参数、截距参数
    np.save("iris_numpy_theta.npy", theta)
    np.save("iris_numpy_bias.npy", bias)
    print("训练结束，模型参数保存完毕")

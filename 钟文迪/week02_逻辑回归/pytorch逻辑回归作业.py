from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os


# 模型运算
def forward(x, theta, bias):
    # linear
    z = np.dot(theta, x.T) + bias
    # sigmoid
    y_hat = 1 / (1 + np.exp(-z))
    return y_hat

# 计算损失
def loss_function(y, y_hat):
    e = 1e-8 #防止y_hat计算值为0，添加极小值epsilon
    return -y * np.log(y_hat + e) - (1 - y) * np.log(1 - y_hat + e) 

# 计算梯度
def calc_gradient(x, y , y_hat):
    m = x.shape[0]
    # print('m: ', m) # 80
    delta_w = np.dot(y_hat - y, x) / m
    delta_b = np.mean(y_hat - y)
    return delta_w, delta_b

if __name__ == '__main__':

    # 数据准备，参数初始化
    # 样本数默认为100（n_samples=100），特征数默认为20（n_features=20）
    X, y = make_classification(n_samples=200, n_features = 30)
    # print('X Shape: ', X.shape) # (200, 30)
    # print('y: ', y)   
    # print('y size: ', y.size) # 200

    # 训练集和测试集
    # test_size=0.2 取20%作为测试集
    train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size = 0.2, shuffle = True, random_state=123)
    # print('train_X Shape: ', train_X.shape) # (160, 30)
    # print('train_Y: ', train_Y)
    # print('test_X Shape: ', test_X.shape) #  (40, 30)
    # print('test_Y: ', test_Y)

    # 权重参数
    theta = np.random.randn(1, 30)
    bias = 0
    # print('theta: ', theta)
    # print('theta Shape: ', theta.shape) # (1, 10)

    # 超参数
    lr = 1e-2 # 学习率
    epoch = 10000   # 训练轮次
    # print('lr: ', lr)

    for i in range(epoch):
        # 正向
        y_hat = forward(train_X, theta, bias)
        # print('y_hat shape:', y_hat.shape)

        # 计算损失
        loss = np.mean(loss_function(train_Y, y_hat))
        # print('loss: ',loss)
        # if i % 100 == 0:
        #     print('step: ', 1, 'loss: ', loss)

        # 梯度下降
        dw, db = calc_gradient(train_X, train_Y, y_hat)
        # print('dw: ',dw)
        # print('db: ',db)

        # 更新参数
        theta = theta - lr * dw
        bias = bias - lr * db

    # print('theta: ', theta)
    # print('bias: ', bias)

    # 保存参数到文件中
    # 文件只创建一次
    if not os.path.isfile("train_result.csv"):
        df = pd.DataFrame(columns=['train_X', 'train_Y', 'test_X', 'test_Y', 'theta', 'bias'])
        df.to_csv("train_result.csv", index=False)
    # 将测试集和超参数写入文件
    data = pd.DataFrame([[train_X.tolist(), train_Y.tolist(), test_X.tolist(), test_Y.tolist(), theta.tolist(), bias]])
    data.to_csv("train_result.csv", mode='a', header=False, index=False)

    print('本次训练完成')

    
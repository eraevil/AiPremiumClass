import pandas as pd
import numpy as np
import os


# 模型运算
def forward(x, theta, bias):
    # linear
    z = np.dot(theta, x.T) + bias
    # sigmoid
    y_hat = 1 / (1 + np.exp(-z))
    return y_hat

if __name__ == '__main__':

    # 测试集和超参数的编号
    num = 1 # 每次训练完的结果会追加到文件中，可以切换num得到对应的训练集、测试集和超参数
    test_epoch = 5000   # 测试轮次
    accuracy = 0    # 本次测试成功的次数

    # 从文件中读取测试集和超参数，用于测试模型
    if not os.path.isfile("train_result.csv"):
        print('暂无训练结果')
        exit()

    data = pd.read_csv('train_result.csv')
    max_num = data['train_X'].size
    if num > (max_num - 1):
        print(f'暂无编号为{num}的训练结果, 最大训练结果编号为: {max_num - 1}')
        exit()
        
    train_X = np.array(eval(data['train_X'][num]))
    train_Y = np.array(eval(data['train_Y'][num]))
    test_X = np.array(eval(data['test_X'][num]))
    test_Y = np.array(eval(data['test_Y'][num]))
    theta = np.array(eval(data['theta'][num]))
    bias = data['bias'][num]
    # print('train_X: ', train_X)
    # print('train_Y: ', train_Y)
    # print('test_X: ', test_X)
    # print('test_Y: ', test_Y)
    # print('theta: ', theta)
    # print('bias: ', bias)

    # 开始进行测试
    for i in range(test_epoch):
        # 测试模型
        idx = np.random.randint(len(test_X))
        x = test_X[idx]
        y = test_Y[idx]

        pred = forward(x, theta, bias)
        if pred > 0.5:
            accuracy = accuracy + 1
    
    print(f'选择的结果集编号为: {num}')
    print(f'预测正确的次数: {accuracy} / {test_epoch}')
    print(f'测试正确率: {accuracy / test_epoch * 100:.2f}%')
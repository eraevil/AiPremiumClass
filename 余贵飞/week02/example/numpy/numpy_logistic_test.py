# encoding: utf-8
# @File  : numpy_logistic_test.py
# @Author: GUIFEI
# @Desc : numpy 实现逻辑回归示例测试
# @Date  :  2025/03/05
import numpy as np
import numpy_logistic_train as npl

def predict(theta, bias, x):
    """
    预测函数
    :param x: 测试集自变量
    :param theta: 训练好的模型权重矩阵
    :param bias: 训练好的模型截距矩阵
    :return: 预测结果
    """
    y_hat = npl.forward(theta, x, bias)
    return y_hat

def test(theta, bias, x, y):
    """
    测试方法
    :param theta: 训练好的模型权重矩阵
    :param bias: 训练好的模型截距矩阵
    :param x: 测试集自变量
    :param y: 测试集因变量
    :return:
    """
    y_hat = predict(theta, bias, x)
    accuracy = np.mean(np.round(y_hat) == y)
    print("accuracy:", accuracy)


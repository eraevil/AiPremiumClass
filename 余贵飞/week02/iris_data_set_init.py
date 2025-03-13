# encoding: utf-8
# @File  : iris_data_set_init.py.py
# @Author: GUIFEI
# @Desc : 鸢尾花数据集初始化
# @Date  :  2025/03/05
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def init():
    """
    鸢尾花数据集初始化
    :return:
    """
    iris = load_iris()
    data, target = iris.data, iris.target
    print(data.shape, target.shape)
    print(data)
    print(target)
    # 顺序取出前100个样本
    X = data[0:100,]
    y = target[0:100]
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)
    # 保存训练及测试数据
    np.save("iris_train_X.npy", train_X)
    np.save("iris_test_X.npy", test_X)
    np.save("iris_train_y.npy", train_y)
    np.save("iris_test_y.npy", test_y)

if __name__ == '__main__':
    # 数据集初始化
    init()

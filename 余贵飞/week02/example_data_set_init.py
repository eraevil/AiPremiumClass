# encoding: utf-8
# @File  : example_data_set_init.py
# @Author: GUIFEI
# @Desc : 数据准备
# @Date  :  2025/03/05

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

'''
定义数据集准备方法
'''
def dataInit():
    # 1. 数据准备，使用scikit_learn 包生成用于分类的数据
    # 获取一个拥有150 样本, 20 个特征的分类数据集
    X, y = make_classification(n_samples=150, n_features=20)
    # 分别查询 X 、y 的形状
    print(X.shape)
    print(y.shape)
    # 拆分数据集为训练集和测试集
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, shuffle=True)
    # 将生成的数据集保存在本地，在模型训练时，直接加载数据集，避免每次训练的数据都不一样
    np.save("train_X.npy", train_X)
    np.save("test_X.npy", test_X)
    np.save("train_y.npy", train_y)
    np.save("test_y.npy", test_y)

if __name__ == '__main__':
    # 初始化数据集
    dataInit()

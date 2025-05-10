# encoding: utf-8
# @File  : example_test.py
# @Author: GUIFEI
# @Desc : 课程样例测试
# @Date  :  2025/03/05
import numpy as np
import numpy_logistic_test as nltt

if __name__ == '__main__':
    # 加载训练好的模型参数
    theta = np.load("iris_numpy_theta.npy")
    print(theta)
    bias = np.load("iris_numpy_bias.npy")
    print(bias)
    test_y = np.load("../../iris_test_y.npy")
    test_X = np.load("../../iris_test_X.npy")
    nltt.test(theta, bias, test_X, test_y)

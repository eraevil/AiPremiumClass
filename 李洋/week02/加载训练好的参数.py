# -*- coding: utf-8 -*-
# @Date    : 2025/3/5 10:57
# @Author  : Lee
# -*- coding: utf-8 -*-
# @Date    : 2025/3/5 10:57
# @Author  : Lee
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pickle

#数据加载
X,y = load_iris(return_X_y=True)
x=X[:100,:2]
y=y[:100]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

#模型导入
def forward(weights,X,bias):
    z=np.dot(weights,X.T)+bias
    sigmoid=1/(1+np.exp(-z))
    return sigmoid

if __name__ == "__main__":
    #加载参数
    with open("model.pkl",'rb') as f:
        weights,bias =pickle.load(f)

    #查看模型结果
    idx=np.random.randint(len(x_test))
    x_test=x_test[idx]
    y_test=y_test[idx]
    perdict=np.round(forward(weights,x_test,bias))
    print("真实值: %d , 预估值: %d "%(y_test,np.mean(perdict)))




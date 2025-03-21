#逻辑回归：应用：1 意图识别：行为预测 2 情感分析：积极或消极 3 金融交易：涨或跌   二分类 判断问题
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

#作业 使用sklearn数据集训练逻辑回归模型  用datasets中的iris数据集合 使用numpy进行模型运算

#1 数据生成
X,y = load_iris(return_X_y = True) #只返回X y
#X有花萼长 花萼宽 花瓣长 花瓣宽 只取前100个
#y 只取前面100个 x y对齐 符合逻辑回归模型
print(X,y)
X = X[:100]
y = y[:100]
#数据切分 防止过拟合
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)

#2 模型参数设定
#   权重参数 模型的运算过程会用到
theta = np.random.randn(1,4) #按照正态分布随机生成初始参数
bias = 0 #偏差
#   超参数 模型的训练整个过程会用到
lr = 1e-2 #学习率eta
epochs = 3000 #训练轮数 也需要调整 会影响训练效果

#3 模型的运算 逻辑回归模型的运算 两步得到预测值y_hat
def forward(X,theta,bias):
    #线性运算
    z = np.dot(theta,X.T) + bias
    #转换为概率 用sigmoid
    y_hat = 1 / (1 + np.exp(-z))
    return y_hat

#4 loss函数 二分类问题 用伯努利分布的log形式 取负数 计算预测与真实值的误差 参数是y_hat y
def loss(y_hat,y):
    #有log 设极小参数 防止取值无穷大
    e = 1e-8
    return -y * np.log(y_hat + e) - (1-y) * np.log(1 - y_hat + e)

#5 计算梯度 对loss的权重参数求导 得到的求导结果包含X,y,y_hat 所以 参数有X,y,y_hat
def calc_derivative(X,y,y_hat): #不熟悉 再看看
    m = X.shape[0] #70
    delta_theta = np.dot(y_hat - y,X) / m
    delta_bias = np.mean(y_hat - y)
    return delta_theta,delta_bias

#6 模型训练 更新参数
for i in range(epochs):
    #前向计算
    y_hat = forward(X_train,theta,bias)
    #计算损失
    loss_val = loss(y_hat,y_train) #70组同时求出loss
    #计算梯度
    delta_theta,delta_bias = calc_derivative(X_train,y_train,y_hat)
    #更新参数 GD
    theta = theta - lr * delta_theta
    bias = bias - lr * delta_bias
    #计算准确率 跟踪训练效果
    if i % 100 == 0:
        acc = np.mean([np.round(y_hat) == y_train]) #mean拉平一维序列运算
        print(f"epoch:{i},loss:{np.mean(loss_val)}，acc:{acc}")

#模型训练完毕 运行后检验模型效果 根据训练效果情况保存模型
np.savez('homework_model.npz',theta,bias)




#调整学习率 样本拆分比率 观察训练结果
#把模型训练好的参数保存到文件 在另一个代码中加载参数实现预测功能

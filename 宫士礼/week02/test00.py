#导入iris数据集，进行训练
from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
#1、数据处理
iris = load_iris()
X = iris.data[:100, :]  # 前 100 个样本，所有 4 个属性
y = iris.target[:100]   # 前 100 个样本的标签
##数据拆分为训练数据和测试数据
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.6)
# print(X_test.shape)查看训练数据分类后的形状
# print(X_train.shape)
##设置初始权重参数，超参数，训练轮次
#权
theta = np.random.randn(1,4)#生成形状为（1，10）的正态分布数组
bias = 0
#超
lr = 0.01
#轮
epochs = 4000

##第二步，模型运算(前向运算)

def forward(x,theta,bias):
    #线性运算 可以理解为z = theta*X + bias，下方是矩阵形式，X是一个样本所有的特征值之和
    z = np.dot(theta,x.T) + bias#theta是（1，10），训练数据x是（80，10），矩阵乘法原因，需x转置之后才能运算
    #sigmoid,激活函数，将y_hat转换为对应概率值
    y_hat = 1 / (1 + np.exp(-z))
    return y_hat

##第三步，计算损失，且损失函数公式一般都由模型直接给出,本周学习用的是伯努利二分步对数损失函数
def loss(y , y_hat) :
    eposion = 1e-8
    return - y * np.log( y_hat + eposion ) - ( 1 - y) * np.log( 1 -y_hat )

## 第四步，计算梯度，求导，公式经由损失函数推出
def calc_gradient(x,y,y_hat) :
    m = x.shape[-1]#要用中括号
    delta_theta = np.dot((y_hat - y), x ) / m
    delta_bias = np.mean(y_hat - y)
    return delta_theta,delta_bias
## 第五步，更新参数，重复训练（需要用到学习率）
for i in range(epochs):
    # 前向计算
    y_hat = forward(X_train, theta, bias)
    # 计算损失
    loss_val = loss(y_train, y_hat)
    # 计算梯度
    delta_theta, delta_bias = calc_gradient(X_train, y_train, y_hat)
    # 更新参数
    theta = theta - lr * delta_theta
    bias = bias - lr * delta_bias

    if i % 500 == 0:
        # 计算准确率
        acc = np.mean(np.round(y_hat) == y_train)  # [False,True,...,False] -> [0,1,...,0]
        print(f"epoch: {i}, loss: {np.mean(loss_val)}, acc: {acc}")
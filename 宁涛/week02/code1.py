from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as 


X,y = load_iris(return_X_y=True) #生成训练数据
X = X[:100] #取前100个标签
y = y[:100]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) #划分数据
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4) #划分数据（调整数据拆分比例）

#权重参数
theta = np.random.randn(1,4) 
bias = 0 
#超参数
lr = 0.01
lr = 0.008
#lr = 0.006
epochs = 20 #训练次数
#epochs = 50 #训练次数
#epochs = 80 #训练次数

def forward(x, theta, bias):
    #线性运算
    z = np.dot(theta,x.T) + bias
    #sigmoid
    y_hat = 1 / (1 + np.exp(-z))
    return y_hat 
#计算损失函数
def loss(y, y_hat):
    e = 1e-8;
    return - y * np.log(y_hat + e) + (1 - y) * np.log(1 - y_hat + e)
#计算梯度
def calc_gradient(x, y, y_hat):
    m = x.shape[-1]
    delta_theta = np.dot((y_hat - y), x) / m
    delta_bias = np.mean(y_hat - y)
    return delta_theta, delta_bias

#模型训练
for i in range(epochs):
    #前向计算
    y_hat = forward(X_train, theta, bias)
    #计算损失
    loss_val = loss(y_train, y_hat)
    #计算梯度
    delta_theta, delta_bias = calc_gradient(X_train, y_train, y_hat)
    #更新参数
    theta -= lr * delta_theta
    bias -= lr*delta_bias

    if i % 2 == 0:
        acc = np.mean(np.round(y_hat) == y_train)
        print(f"epoch: {i}, loss: {np.mean(loss_val)}, acc: {acc}")


#模型推理
idx = np.random.randint(len(X_test))
x = X_test[idx]
y = y_test[idx]

predict = np.round(forward(x, theta, bias))
print(f"y: {y}, predict: {predict}")

#保存参数
arr = theta
num = bias
np.savez("data.npz", array = arr, number = num)

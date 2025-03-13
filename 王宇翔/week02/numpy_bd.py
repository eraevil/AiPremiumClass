from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import make_classification
from sklearn.datasets import load_iris

X,y = load_iris(return_X_y=True)
# y = β0 + β1x + ϵ
# # ⽣成观测值数据
# X, y = make_classification(n_features=10)
# 拆分训练和测试集
train_X, test_X, train_y, test_y = train_test_split(X[:100], y[:100], test_size=0.3, shuffle=True)
# 初始化参数模型参数
theta = np.random.randn(1, 4)
bias = 0
# 学习率
lr = 1e-2
# 模型训练的轮数
epoch = 5000
# 前向计算


def forward(x, theta, bias):
    # linear
    z = np.dot(theta, x.T) + bias
    # z = np.dot(theta.T,x.T) + bias
    # sigmoid
    y_hat = 1 / (1 + np.exp(-z))
    return y_hat


# 损失函数
def loss_function(y, y_hat):
    e = 1e-8  # 防⽌y_hat计算值为0，添加的极⼩值epsilon
    return - y * np.log(y_hat + e) - (1 - y) * np.log(1 - y_hat + e)


# 计算梯度
def calc_gradient(x, y, y_hat):
    m = x.shape[-1]
    delta_w = np.dot(y_hat - y, x) / m
    delta_b = np.mean(y_hat - y)
    return delta_w, delta_b


for i in range(epoch):
    # 正向
    y_hat = forward(train_X, theta, bias)
    # 计算损失
    loss = np.mean(loss_function(train_y, y_hat))
    if i % 100 == 0:
        print('step:', i, 'loss:', loss)
    # 梯度下降
    dw, db = calc_gradient(train_X, train_y, y_hat)
    # 更新参数
    theta -= lr * dw
    bias -= lr * db
# 测试模型
idx = np.random.randint(len(test_X))
x = test_X[idx]
y = test_y[idx]

def predict(x):
     pred = forward(x,theta,bias)[0]
     if pred > 0.5:
       return 1
     else:
       return 0

pred = predict(x)

print(f'预测值：{pred} 真实值：{y}')
# 保存模型参数
np.savez('model_params.npz', theta=theta, bias=bias)
print("模型参数已保存至model_params.npz")
# 使用sklearn数据集训练逻辑回归模型
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# 加载经典的鸢尾花（Iris）数据集
X,y = load_iris(return_X_y=True) # return_X_y=True 表示直接返回特征矩阵 X 和目标标签 y

X = X[:100]  # 取前100个数据
y = y[:100]  # 取前100个标签(0,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# 权重参数
theta = np.random.randn(1,4)  # shape (1, 4)
bias = 0
# 超参数
lr = 0.1 # 学习率
epochs = 3000  # 训练次数

# 模型函数计算
def forward(x, theta, bias):
    # 线性运算
    z = np.dot(theta, x.T) + bias # shape (70,4)
    # sigmoid
    y_hat = 1 / (1 + np.exp(-z))  # shape (70,4)
    return y_hat

# 计算损失函数
def loss(y, y_hat):
    e = 1e-8
    return - y * np.log(y_hat + e) - (1 - y) * np.log(1 - y_hat + e)

# 计算梯度
def calc_gradient(x,y,y_hat):
    # 计算梯度
    m = x.shape[-1] # 4
    # print(m)
    # theta梯度计算
    delta_theta = np.dot((y_hat - y), x) / m
    # bias梯度计算
    delta_bias = np.mean(y_hat - y)
    # 返回梯度
    return delta_theta, delta_bias

# 模型训练
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

    if i % 150 == 0:
        # 计算准确率
        acc = np.mean(np.round(y_hat) == y_train)  # [False,True,...,False] -> [0,1,...,0]
        print(f"epoch: {i}, loss: {np.mean(loss_val)}, acc: {acc}")
print()
print(theta)
print(bias)

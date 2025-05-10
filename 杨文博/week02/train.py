import numpy as np
import matplotlib
from sklearn.model_selection import train_test_split
import json
matplotlib.use('TkAgg')  # 或者 'Qt5Agg', 'Agg' 等
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# 梯度下降
plot_x = np.linspace(-1, 6, 100)
plot_y = (plot_x - 2.5) ** 2 - 1

def loss(x):
    try:
        return (x-2.5)**2 -1
    except:
        return float("inf")
def derivate(x):
    return 2*(x -2.5)

eta = 0.1
theta = 0

theta_history = []
while True:
    last_theta = theta
    theta -= eta * derivate(theta)
    loss_v = loss(theta)
    theta_history.append(theta)
    if abs(loss_v - loss(last_theta)) < 1e-9:
        break



plt.plot(plot_x, plot_y)
plt.plot(theta_history,loss(np.array(theta_history)),color='r',marker="+")
plt.show(block=False)

# 数据生成
# make_classification
X,y = make_classification(n_samples=100,n_features=10)
"""X 是特征矩阵（也称为输入数据），形状为 (n_samples, n_features)，其中：

n_samples 是样本数量。

n_features 是每个样本的特征数量。

y 是目标标签（也称为输出数据），形状为 (n_samples,)，表示每个样本的类别标签（通常是 0 或 1，表示二分类问题）。"""
print(X)
print(y)

# 数据拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train.shape)
print(y_train.shape)

# 权重参数
theta = np.random.randn(1,10)
bias = 0
# 超参数
lr = 0.01
epochs = 3000

# 模型运算
# 计算运算函数
def forward(X, theta, bias):
    # 线性运算
    z = np.dot(theta, X.T) + bias
    # sigmoid
    y_hat = 1 / (1 + np.exp(-z))
    return y_hat
# 计算损失函数
def loss(y_hat, y):
    e = 1e-8
    return -y * np.log(y_hat+e) - (1 - y) * np.log(1 - y_hat+e)
# 计算梯度
def gradient(x,y,y_hat):
    m = x.shape[-1]
    delta_theta =np.dot((y_hat - y), x) / m
    delta_bias = np.mean(y_hat - y)
    return delta_theta, delta_bias
# 模型训练
for epoch in range(epochs):
    # 前向传播
    y_hat = forward(X_train, theta, bias)
    # 损失计算
    current_loss = loss(y_hat, y_train)
    # 梯度计算
    delta_theta, delta_bias = gradient(X_train, y_train, y_hat)
    # 参数更新
    theta = theta - lr * delta_theta
    bias = bias - lr * delta_bias
    accuracy = np.mean(np.round(y_hat) == y_train)
    if epoch % 100 == 0:
        print(accuracy)

# 模型推理
idx = np.random.randint(len(X_test))
x = X_test[idx]
y = y_test[idx]
prediction = np.round(forward(x, theta, bias))
print(prediction,y)

"""作业
使用sklearn数据集训练逻辑回归模型
调整学习率，观察训练结果
把模型训练的好的参数进行保存在另一个文件中打开来实现预测功能"""
# sklearn中iri数据集
from sklearn.datasets import load_iris
X,y = load_iris(return_X_y=True)
print(X.shape,y.shape)
print(x[:100])


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

iris_df = pd.DataFrame(data=X, columns=iris.feature_names)
iris_df['target'] = y
print(iris_df.head())
print(iris.feature_names)
print(iris.target_names)
# 筛选target为0和1的样本
iris_filtered = iris_df[iris_df['target'].isin([0, 1])]
print(iris_filtered.shape)
X_filtered = iris_filtered.drop(columns = ["target"])
y_filtered = iris_filtered['target']
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)
# 在X中添加一列全为1的数据，用于计算偏置项
X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]  # 添加偏置项的列
X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta):
    m = y.size
    h = sigmoid(X.dot(theta))
    cost = -1/m * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

def compute_gradients(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    gradients = 1/m * X.T.dot(h - y)  # 梯度
    return gradients

# 5. 实现梯度下降
def gradient_descent(X, y, theta, learning_rate, iterations):
    cost_history = []

    for i in range(iterations):
        gradients = compute_gradients(X, y, theta)  # 计算梯度
        theta = theta - learning_rate * gradients  # 更新theta
        cost = compute_cost(X, y, theta)  # 计算损失
        cost_history.append(cost)  # 保存损失历史

    return theta, cost_history

theta_initial = np.random.randn(X_train_b.shape[1], 1)
learning_rate = 0.1
iterations = 1000
theta, cost_history = gradient_descent(X_train_b, y_train.values.reshape(-1, 1), theta_initial, learning_rate, iterations)
print(theta.dtype)
print(f"最终损失: {cost_history[-1]}")

# # 可视化损失曲线
# plt.plot(range(iterations), cost_history, label="Cost (Log Loss)")
# plt.xlabel('Iterations')
# plt.ylabel('Cost')
# plt.title('Gradient Descent Optimization')
# plt.legend()
# plt.show()
theta_optimal = theta.copy()
def predict(X, theta):
    probabilities = sigmoid(X.dot(theta))
    return (probabilities >= 0.5).astype(int)

y_pred = predict(X_test_b, theta_optimal)

accuracy = accuracy_score(y_test, y_pred)
print(f"模型在测试集上的准确率: {accuracy}")
theta_list = theta_optimal.tolist()
with open("theta.json", "w") as f:
    json.dump(theta_list, f)
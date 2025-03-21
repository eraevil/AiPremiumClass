import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X,y = load_iris(return_X_y=True) #生成训练数据
X = X[:100] #取前100个标签
y = y[:100]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#加载模型
data = np.load('data.npz')
theta = data['array']
bias = data['number']

def forward(x, theta, bias):
    #线性运算
    z = np.dot(theta,x.T) + bias
    #sigmoid
    y_hat = 1 / (1 + np.exp(-z))
    return y_hat 

#模型推理
idx = np.random.randint(len(X_test))
x = X_test[idx]
y = y_test[idx]

predict = np.round(forward(x, theta, bias))
print(f"y: {y}, predict: {predict}")


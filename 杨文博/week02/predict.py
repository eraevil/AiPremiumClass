import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

with open("theta.json", "r") as f:
    theta_list = json.load(f)  # 解析 JSON
    theta_optimal = np.array(theta_list)  # 显示 JSON 文件的内容
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, theta):
    probabilities = sigmoid(X.dot(theta))
    return (probabilities >= 0.5).astype(int)

iris = load_iris()
X = iris.data
y = iris.target
iris_df = pd.DataFrame(data=X, columns=iris.feature_names)
iris_df['target'] = y
iris_filtered = iris_df[iris_df['target'].isin([0, 1])]
X_filtered = iris_filtered.drop(columns = ["target"])
y_filtered = iris_filtered['target']
X_b = np.c_[np.ones((X_filtered.shape[0], 1)), X_filtered]  # 添加偏置项的列
y_pred = predict(X_b, theta_optimal)
accuracy = accuracy_score(y_filtered, y_pred)
print(f"模型在测试集上的准确率: {accuracy}")
from sklearn.datasets import make_classification
# from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot  as plt

X, y = make_classification()

bias = 0


def forward(X, y, theta):
    z = np.dot(theta, X.T) + bias
    y_hat = 1 / (1 + np.exp(-z))
    return y_hat


idx = np.random.randint(len(X))  

x = X[idx]
y = y[idx]

theta = np.load('theta.npy')

predict = np.round(forward(x, y, theta))
print(f"y: {y}, predict: {predict}")

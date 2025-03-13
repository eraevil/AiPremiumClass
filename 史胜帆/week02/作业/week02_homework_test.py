#week_homework_test
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

X,y = load_iris(return_X_y = True)
X  = X[:100]
y = y[:100]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)

def forward(X,theta,bias):
    z = np.dot(theta,X.T) + bias
    y_hat = 1 / (1 + np.log(-z))
    return y_hat

#模型推理
weights = np.load('homework_model.npz')
#print(weights['arr_0'])
idx = np.random.randint(len(X_test))
x = X_test[idx]
y = y_test[idx]
predict = np.round(forward(x,weights['arr_0'],weights['arr_1']))
print(f'y:{y},predict:{predict}')
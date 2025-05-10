import numpy as np
import logistic_regression_train as tx_train

print()
print(tx_train.theta)
print(tx_train.bias)
print()

y_hat = tx_train.forward(tx_train.X_test, tx_train.theta, tx_train.bias)

print(y_hat)
print()
print(np.round(y_hat))
print(tx_train.y_test)
print()
acc = np.mean(np.round(y_hat) == tx_train.y_test)
print(acc) #准确率

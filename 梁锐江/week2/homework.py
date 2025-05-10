import numpy as np


def load_model_param():
    data = np.load('model_parameters.npz')
    theta = data['theta']
    bias = data['bias']
    return theta, bias


def predict(x, theta, bias):
    z = np.dot(theta, x.T) + bias

    return 1 / (1 + np.exp(-z))


if __name__ == '__main__':
    theta, bias = load_model_param()

    test_data = np.load('model_test_data.npz')

    predict_y = predict(test_data['x_test'], theta, bias)

    acc = np.mean(np.round(predict_y) == test_data['y_test'])
    print('acc:', acc)

import numpy as np
import matplotlib.pyplot as plt

# 斜率
theta = 0.
# 学习率
eta = 1.005
# 截距
bias = 0
# 最小值
epsilon = 1e-8


def get_x_train():
    x_train = np.linspace(-1, 6, 140)
    return x_train


def get_y_train(plot_x):
    plot_y = (plot_x - 2.5) ** 2 - 1
    return plot_y


def show_photo(plot_x, plot_y, theta_history):
    plt.plot(plot_x, plot_y)
    plt.plot(np.array(theta_history), loss(np.array(theta_history)), color="r", marker="+")
    plt.show()


def loss(theta):
    # 损失函数
    return (theta - 2.5)**2 - 1


def calc_gradient(theta):
    # 计算梯度
    return 2 * (theta - 2.5)


if __name__ == '__main__':
    # 样本数据
    plot_x = get_x_train()
    plot_y = get_y_train(plot_x)
    theta_history = [theta]
    for i in range(10):
        gradient = calc_gradient(theta)

        last_theta = theta

        theta = theta - eta * gradient

        theta_history.append(theta)

        if abs(loss(theta) - loss(last_theta)) < epsilon:
            break

    print(theta)
    print(loss(theta))
    show_photo(plot_x, plot_y, theta_history)

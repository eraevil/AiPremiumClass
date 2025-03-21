import numpy as np
for i in range(1,10):
    print(i/10, np.log(i / 10))
  1e-2
# 梯度下降
import numpy as np
import matplotlib.pyplot as plt
plot_x = np.linspace(-1, 9, 150)
print(plot_x)
plot_y = (plot_x - 2.5) ** 2 
plt.plot(plot_x, plot_y)
plt.show()
# 计算导数
def derivate(theta):
    return 2 * (theta - 2.5)

# 计算损失函数
def loss(theta):
    try:
        return (theta - 2.5) ** 2 - 1
    except:
        return float('inf') # 返回一个无穷大的数
# 梯度下降
def gradient_descent(init_theta, eta, epsilon=1e-8):
    # 收集theata的值
    theta_history = [init_theta]
    theta = init_theta
    # 记录循环次数
    iter_i = 0

    while iter_i < 8:
        gradient = derivate(theta)  # 参数值对应的梯度
        # 记录上一次theta的值
        last_theta = theta
        # 更新参数值
        theta = theta - eta * gradient  
        # 记录theta
        theta_history.append(theta)
        # 损失函数值
        loss_v = loss(theta)
        # 中断循环条件 （连续损失函数差小于一个阈值）
        if abs(loss_v - loss(last_theta)) < epsilon:
            break

        iter_i += 1

    return theta_history
  def plot_theta_history(theta_history):
    plt.plot(plot_x, plot_y)
    plt.plot(theta_history, loss(np.array(theta_history)), color='r', marker='+')
    plt.show()
    theta = 0.0  # 参数初始值
eta = 0.1  # 学习率
epsilon = 1e-8  # 精度
theta_history[-1]
  

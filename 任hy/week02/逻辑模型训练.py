from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np 

# 生成一个二分类数据集
# X是特征矩阵（150，10）表示150个样本，每个样本10个特征  
# y是标签向量（150，） 表示每个样本的类别标签（0或1）
X,y = make_classification(n_samples=150,n_features=10) # 默认生成二分类数据集
# 数据集分为训练集和测试集
# 测试集占30%即45个样本。训练集占105个
# X_train 训练集特征矩阵，形状为 (105, 10)。  X_test：测试集特征矩阵，形状为 (45, 10)。
# y_train：训练集标签向量，形状为 (105,)。  y_test：测试集标签向量，形状为 (45,)。
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

#? 也许是不知道每个样本的权重是多少？（测试别的分布）
# 决定每个参数的贡献度，每个元素对应输入特征的权重。乘积是每个样本的预测值
# 随机还能为梯度下降提供初始化方向。后续theta会梯度下降更新
theta = np.random.randn(1,10) # (标准正态分布随机数)模型权重参数，表示每个特征对应的权重
bias = 0 # 模型的截距(偏置)，允许模型在特征全0时也能输出一个非零值。使得预测结果适应不经过原点的数据分布
lr = 0.01 # 学习率
epochs = 3000 #训练次数

# 2 模型计算函数
def forward(x,theta,bias):
    # 计算每个样本的预测值
    z = np.dot(theta,x.T)+bias #bias广播到所有样本
    # sigmoid # ？也许是01分布？
    y_hat = 1/(1+np.exp(-z))
    return y_hat

# 3 计算损失函数  对数似然函数
def loss(y,y_hat):
    e = 1e-8 # 防止log出现0
    return -y * np.log(y_hat + e) - (1-y) * np.log(1 - y_hat + e)

# 4 计算梯度
def calc_gradient(x,y,y_hat):
    # 计算梯度
    m = x.shape[-1]
    # theta梯度计算 某一个权重参数对损失函数的影响
    delta_theta = np.dot((y_hat - y), x) / m
    # bias梯度计算
    delta_bias = np.mean(y_hat - y)
    # 返回梯度
    return delta_theta,delta_bias

# 5 模型训练
for i in range(epochs):
    # 前向计算
    y_hat = forward(X_train,theta,bias)
    # 计算损失
    loss_val = loss(y_train, y_hat)
    # 计算梯度
    delta_theta, delta_bias = calc_gradient(X_train, y_train, y_hat)
    # 更新参数
    theta = theta - lr * delta_theta
    bias = bias - lr * delta_bias
    if i % 100 == 0:
        # 计算准确率  round四舍五入
        acc = np.mean(np.round(y_hat) == y_train)  # mean这里会计算准确率
        print(f"epoch: {i}, loss: {np.mean(loss_val)}, acc: {acc}")


# 模型推理
idx = np.random.randint(len(X_test))
x = X_test[idx]
y = y_test[idx]

predict = np.round(forward(x,theta,bias))
print(f"y:{y},predict:{predict}")
    



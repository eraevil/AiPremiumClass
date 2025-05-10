# %%
#导入包
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_iris

# %% [markdown]
# 接下来建设各个模块
# 首先导入Iris数据集,load_iris 使用return_X_y=true可以直接指定返回特征矩阵和目标向量，切片操作取100个之后再进行train_test_split操作

# %%
X,y=load_iris(return_X_y=True)
X_real,y_real=X[:100],y[:100]
X_train,X_test,y_train,y_test=train_test_split(X_real,y_real,test_size=0.3)
print(y_train.shape)

# %% [markdown]
# 定义参数theis和偏置量bias以及学习率lr,和训练轮数epochs,注意theis维数与iris数据一致
# 

# %%
theta=np.random.randn(1,4)
bias=0
lr=0.0001
epochs=2000

# %% [markdown]
# 定义前向传播，损失函数，以及计算梯度和模型训练

# %%
#前向传播,首先计算z,再利用sigmod
def forward(x,theta,bias):
    z=np.dot(theta,x.T)+bias
    #sigmod
    y_hat=1/(1+np.exp(-z))
    return y_hat
#损失函数，对照线性回归公式
def loss(y,y_hat):
    e=1e-8 
    #利用广播机制内积乘
    return -y*np.log(y_hat+e)-(1-y)*np.log(1-y_hat+e)
def cal_gradient(x,y,y_hat):
     # 计算梯度
    m=x.shape[-1]
    # theta梯度计算
    delta_theta=np.dot((y_hat-y),x)/m
    # bias梯度计算
    delta_bias =np.mean(y_hat-y)
    # 返回梯度
    return delta_theta, delta_bias
#真实模型训练
for i in range(epochs):
    #前向计算
    y_hat=forward(X_train,theta,bias)
    #损失计算
    loss_val=loss(y_train,y_hat)
    #梯度计算
    dw,db=cal_gradient(X_train,y_train,y_hat)
    #梯度更新参数
    theta=theta-lr*dw
    bias=bias-lr*db
    #每100轮评估一次,目前测试结果iris数据集极易过拟合
    if i% 100 ==0:
        #四舍五入y_hat之后与y_train进行比较
        acc =np.mean(np.round(y_hat)==y_train)
        print(f"epoch:{i},loss_val:{np.mean(loss_val)},acc:{acc}")

# %% [markdown]
# 模型推理测试,此处选择评估单个样本和整个测试集的准确率

# %%
idx = np.random.randint(len(X_test))
x=X_test[idx]
y=y_test[idx]
predict=np.round(forward(x,theta,bias))
print(f"y:{y},predict:{predict}")
accc=np.mean(np.round(forward(X_test,theta,bias))==y_test)
print(f"acc:{accc}")

# %% [markdown]
# 保存参数方法使用的是np.save保存在一个文件里

# %%
np.savez('hyperparameters_linear.npz', theta=theta, bias=bias, lr=lr, dw=dw, db=db)
print(f"theta:{theta},bias:{bias},lr:{lr},dw:{dw},db:{db}")

# %%




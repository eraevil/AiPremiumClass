# %% [markdown]
# 本程序为加载参数在预测

# %%
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_iris

# %%
X,y=load_iris(return_X_y=True)
X_real,y_real=X[:100],y[:100]
X_train,X_test,y_train,y_test=train_test_split(X_real,y_real,test_size=0.3)
print(y_train.shape)

# %% [markdown]
# 加载参数

# %%
loaded = np.load('hyperparameters_linear.npz')
theta_loaded = loaded['theta']
bias_loaded = loaded['bias']
lr_loaded = loaded['lr']
dw_loaded = loaded['dw']
db_loaded = loaded['db']
print(f"theta:{theta_loaded},bias:{bias_loaded},lr:{lr_loaded},dw:{dw_loaded},db:{db_loaded}")

# %% [markdown]
# 预测模型

# %%
def forward(x,theta,bias):
    z=np.dot(theta,x.T)+bias
    #sigmod
    y_hat=1/(1+np.exp(-z))
    return y_hat

# %%
y_hat=forward(X_train,theta_loaded,bias_loaded)
acc =np.mean(np.round(y_hat)==y_train)
print(f"acc:{acc}")



import torch#tensor的创建
data=torch.tensor([[2,3],[6,7]])
data
import numpy as np
data=torch.tensor([[2,3],[6,7]])
test = np.array(data)#张量-->numpy数组
test
data1 = torch.from_numpy(test)#numpy数组-->张量
data1
data2 = torch.ones_like(data1) # 保留data的属性
print(f"Ones Tensor: \n {data2} \n")
shape = (3,2)#张量维度的元组
rand_tensor = torch.rand(shape)#3行2列随机值的张量
ones_tensor = torch.ones(shape)#全一
zeros_tensor = torch.zeros(shape)#全0
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# 基于现有tensor构建，但使⽤新值填充
m = torch.ones(5,3, dtype=torch.double)
n = torch.rand_like(m, dtype=torch.float)
# 获取tensor的⼤⼩
print(m.size()) # torch.Size([5,3])
# 均匀分布
torch.rand(5,3)
# 标准正态分布
torch.randn(5,3)
# 离散正态分布
torch.normal(mean=.0,std=1.0,size=(5,3))
# 线性间隔向量(返回⼀个1维张量，包含在区间start和end上均匀间隔的steps个点)
torch.linspace(start=1,end=10,steps=20)#开始，结束，多少个
tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")#维度
print(f"Datatype of tensor: {tensor.dtype}")#数据类型
print(f"Device tensor is stored on: {tensor.device}")#模型训练所依靠的设备
#怎么创建张量在不同位置上？不支持....
#检查pytorch是否支持gpu
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     tensor = tensor.to(device)
# print(tensor)
# print(tensor.device)
print(torch.cuda.is_available())

tensor = torch.tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[...,-1])
tensor[:,1] = 0
print(tensor)
t1 = torch.cat([tensor, tensor, tensor], dim=1)##向量拼接,dim=0是在行的维度上拼接，为1是在列的维度上拼接
print(t1)
print(t1.shape)
t2 = torch.stack(tensors=[tensor,tensor,tensor])#sack默认在行的维度进行拼接，cat则是列的维度
print(t2)
print(t2.shape)
import torch#可按照矩阵的乘法记忆，几个方法都可以做到
tensor = torch.arange(1,10,dtype=torch.float32).reshape(3,3)
# # 计算两个张量之间矩阵乘法的⼏种⽅式。 y1, y2, y3 最后的值是⼀样的 dot
# y1 = tensor @ tensor.T
# y2 = tensor.matmul(tensor.T)#.matmul()也是一种矩阵乘法的方法
# y3 = torch.rand_like(tensor)
# torch.matmul(tensor, tensor.T, out=y3)
# # 计算张量逐元素相乘的⼏种⽅法。 z1, z2, z3 最后的值是⼀样的。out后面是存放结果输出的变量
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(z1)
print(z3)

import torch#tensor的创建
data=torch.tensor([[2,3],[6,7]])
data
import numpy as np
data=torch.tensor([[2,3],[6,7]])
test = np.array(data)#张量-->numpy数组
test
data1 = torch.from_numpy(test)#numpy数组-->张量
data1
data2 = torch.ones_like(data1) # 保留data的属性
print(f"Ones Tensor: \n {data2} \n")
shape = (3,2)#张量维度的元组
rand_tensor = torch.rand(shape)#3行2列随机值的张量
ones_tensor = torch.ones(shape)#全一
zeros_tensor = torch.zeros(shape)#全0
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# 基于现有tensor构建，但使⽤新值填充
m = torch.ones(5,3, dtype=torch.double)
n = torch.rand_like(m, dtype=torch.float)
# 获取tensor的⼤⼩
print(m.size()) # torch.Size([5,3])
# 均匀分布
torch.rand(5,3)
# 标准正态分布
torch.randn(5,3)
# 离散正态分布
torch.normal(mean=.0,std=1.0,size=(5,3))
# 线性间隔向量(返回⼀个1维张量，包含在区间start和end上均匀间隔的steps个点)
torch.linspace(start=1,end=10,steps=20)#开始，结束，多少个
tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")#维度
print(f"Datatype of tensor: {tensor.dtype}")#数据类型
print(f"Device tensor is stored on: {tensor.device}")#模型训练所依靠的设备
#怎么创建张量在不同位置上？不支持....
#检查pytorch是否支持gpu
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     tensor = tensor.to(device)
# print(tensor)
# print(tensor.device)
print(torch.cuda.is_available())

tensor = torch.tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[...,-1])
tensor[:,1] = 0
print(tensor)
t1 = torch.cat([tensor, tensor, tensor], dim=1)##向量拼接,dim=0是在行的维度上拼接，为1是在列的维度上拼接
print(t1)
print(t1.shape)
t2 = torch.stack(tensors=[tensor,tensor,tensor])#sack默认在行的维度进行拼接，cat则是列的维度
print(t2)
print(t2.shape)
import torch#可按照矩阵的乘法记忆，几个方法都可以做到
tensor = torch.arange(1,10,dtype=torch.float32).reshape(3,3)
# # 计算两个张量之间矩阵乘法的⼏种⽅式。 y1, y2, y3 最后的值是⼀样的 dot
# y1 = tensor @ tensor.T
# y2 = tensor.matmul(tensor.T)#.matmul()也是一种矩阵乘法的方法
# y3 = torch.rand_like(tensor)
# torch.matmul(tensor, tensor.T, out=y3)
# # 计算张量逐元素相乘的⼏种⽅法。 z1, z2, z3 最后的值是⼀样的。out后面是存放结果输出的变量
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(z1)
print(z3)

tensor = torch.arange(2,11,dtype=torch.float32).reshape(3,3)
agg = tensor.sum()#agg为数组
print(agg)
agg_item = agg.item()
print(agg_item, type(agg_item))#item()可将pytorch中的数据重新输出为原始数据类型，agg为54.0 <class 'float'>
np_array = z1.numpy()#将张量转化为numpy数组
np_array
tensor = torch.arange(1,10,dtype=torch.float32).reshape(3,3)#in_place-practice
print(tensor, "\n")
tensor.add_(5)#直接就地内部更改，比较危险
print(tensor)
import torch
from torchviz import make_dot
# 定义矩阵 A，向量 b 和常数 c
A = torch.randn(10, 10,requires_grad=True)
b = torch.randn(10,requires_grad=True)
c = torch.randn(1,requires_grad=True)
x = torch.randn(10, requires_grad=True)
# 计算 x^T * A  b * x + c
result = torch.matmul(A, x.T) + torch.matmul(b, x) + c
# ⽣成计算图节点
dot = make_dot(result, params={'A': A, 'b': b, 'c': c, 'x': x})
# 绘制计算图
dot.render('test_image', format='png', cleanup=True, view=False)
from sklearn.datasets import load_iris#预习内容
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
# 读取iris数据集
X,y = load_iris(return_X_y=True)
# 截取2分类数据
X = X[:100]
y = y[:100]
# 拆分训练和测试集
train_X,test_X, train_y,test_y = train_test_split(X,y, test_size=0.3, shuffle=True) 
print(train_X)
print(train_y)
# 初始化参数模型参数
# theta = np.zeros((1,4))
theta = np.random.randn(1, 4)##随机正态分布
# theta = np.random.randn(4,1)
bias = 0
# 学习率
lr = 1e-3
# 模型训练的轮数
epoch = 5000
# 前向计算
def forward(x, theta, bias):
 # linear
 z = np.dot(theta,x.T) + bias
 # z = np.dot(theta.T,x.T  bias
 # sigmoid
 y_hat = 1 / (1 + np.exp(-z))
 return y_hat
# 损失函数
def loss_function(y, y_hat):
 e = 1e-8 # 防⽌y_hat计算值为0，添加的极⼩值epsilon
 return - y * np.log(y_hat + e) - (1 - y) * np.log(1 - y_hat + e)
# 计算梯度
def calc_gradient(x,y,y_hat):
 m = x.shape[-1]
 delta_w = np.dot(y_hat-y,x)/m
 delta_b = np.mean(y_hat-y)
 return delta_w, delta_b
for i in range(epoch):
 #正向
  y_hat = forward(train_X,theta,bias)
 #计算损失
  loss = np.mean(loss_function(train_y,y_hat))
  if i % 100 == 0:
    print('step:',i,'loss:',loss)
 #梯度下降
    dw,db = calc_gradient(train_X,train_y,y_hat)
 #更新参数
 # theta -= lr * dw.T
    theta -= lr * dw
    bias -= lr * db
# 测试模型
idx = np.random.randint(len(test_X))
x = test_X[idx]
y = test_y[idx]
def predict(x):
 pred = forward(x,theta,bias)[0]
 if pred > 0.5:
    return 1
 else: 
    return 0
pred = predict(x)
print(f'预测值：{pred} 真实值：{y}')

tensor = torch.arange(2,11,dtype=torch.float32).reshape(3,3)
agg = tensor.sum()#agg为数组
print(agg)
agg_item = agg.item()
print(agg_item, type(agg_item))#item()可将pytorch中的数据重新输出为原始数据类型，agg为54.0 <class 'float'>
np_array = z1.numpy()#将张量转化为numpy数组
np_array
tensor = torch.arange(1,10,dtype=torch.float32).reshape(3,3)#in_place-practice
print(tensor, "\n")
tensor.add_(5)#直接就地内部更改，比较危险
print(tensor)
import torch
from torchviz import make_dot
# 定义矩阵 A，向量 b 和常数 c
A = torch.randn(10, 10,requires_grad=True)
b = torch.randn(10,requires_grad=True)
c = torch.randn(1,requires_grad=True)
x = torch.randn(10, requires_grad=True)
# 计算 x^T * A  b * x + c
result = torch.matmul(A, x.T) + torch.matmul(b, x) + c
# ⽣成计算图节点
dot = make_dot(result, params={'A': A, 'b': b, 'c': c, 'x': x})
# 绘制计算图
dot.render('test_image', format='png', cleanup=True, view=False)

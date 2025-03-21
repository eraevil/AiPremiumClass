import numpy as np

a = np.zeros((2,3),np.float16)
a2 = np.ones((2,3),np.int16)
a3 = np.random.random(5)
a4 = np.random.random(2)
print(a3)
mu = 0
sigma = 0.1
np.random.normal(mu,sigma,5) #模型参数初始化手段
a = np.array([(1,2),(3,4),(5,6)])
print(a)
print(a[:,1])
print(a[1:])
print(a[:,:1])
print(a.ndim)
print(a.shape)
print(a.size)
print(a.dtype)
a = np.array([(1,2,3),(4,5,6),(7,8,9)])
a = a.transpose() #a = a.T
a = np.array([(1,2),(3,4),(5,6)])
print(a.shape)
a = a[:,np.newaxis,:] #添加维度
a.shape
a.argmax() #最大值索引
a.argmin() #最小值索引
np.ceil(a) #向上取整
np.floor(a) #向下取整
np.rint(a) #四舍五入
a9 = np.array([[1,2],[4,5]])
b9 = np.array([[4,5],[7,8]])
res_dot = np.dot(a9,b9)
res_at = a9@b9
print(res_dot)
print(res_at)

import torch
import numpy as np
print(torch.cuda.is_available())

data = torch.tensor([[1,2],[3,4]])
a = np.array([[1,2],[3,4]])
b = torch.from_numpy(a)
shape = (2,3,)
rand_ten = torch.rand(shape)
ones_ten = torch.ones(shape)
print(torch.rand(5,3))
print(torch.randn(5,3))
print(torch.normal(mean=.0,std=1.0,size=(5,3)))
print(torch.linspace(start=1,end=10,steps=10)) #steps：为拆分成多少份
tensor = torch.ones(3,3)
print(tensor[0]) #第一行
print(tensor[:,0]) #第一列
print(tensor[...,-1]) #最后一列
tensor[:,1] = 0
print(tensor)

a = torch.tensor([[1,2,3],[4,5,6]])
b = torch.tensor([[6,7,8],[7,8,9]])
c = torch.cat([a,b],dim=1) #拼接
#点积
tensor = torch.arange(1,10).reshape(3,3)
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
print(y1)
print(y2)
y3 = torch.rand_like(tensor,dtype=torch.float32)
print(y3)

#乘积
z1 = tensor*tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor,dtype=torch.float32)
torch.mul(tensor,tensor,out=z3)
print(z3)

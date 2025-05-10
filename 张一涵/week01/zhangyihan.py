import  numpy as np
a = np.array([1,2,3])
print(a)
a1 = np.array([1,2,3],float)
print(a1)
a2 = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a2)
a3 = np.zeros((2,3),dtype=float)
print(a3)
a4 = np.ones_like(a3)
print(a4)
#创建等差数列
a5 = np.arange(1,10,0.5)
print(a5)
#创建单位矩阵
a6 = np.eye(5)
print(a6)
#生成指定长度，在[0,1）之间平均分布的随机数组
a7 = np.random.random((3,5))
print(a7)
#生成指定长度符合正态分布的数据
a8 = np.random.normal(0,0.1,6)
print(a8)
print("-------------------------------------------------")
#numpy数组的随机访问
b = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(b[1])
print(b[2][0])
print(b[:,1])
print(b[:])
print(b[-2:])
print("====================================")
#numpy数组的遍历
for i in b:
    for j in i:
        print(j,end=",")
print()
for i,j,k in b:
    print(i,j,k)
#numpy数组的常用属性
print("ndim",b.ndim)
print("shape",b.shape)
print("size",b.size)
print("dtype",b.dtype)
print("================================")
#numpy数组的基本操作
print(3 in b)
print(5 in b)
#数组的重排列
b1 = np.array([1,2,3,4,5,6])
b2 = b1.reshape((3,2))
print(b2)

b3 = np.array([[1,2,3],[7,8,9]])
b4 = b3.flatten()
print(b4)
#numpy的数学运算
a = np.array([[1,2],[3,4]])
b = np.array([[2,2],[3,4]])
print(a+b)
print(a-b)
print(a*b)
print(a/b)
print(a.sum())
print(a.prod())
print(a.mean())
print(a.var())
print(a.std())
print(a.max())
print(a.min())
print(a.argmax())
print(a.argmin())

result = np.dot(a,b)
print(result)

import torch
a = [[1,2],[3,4]]
data = torch.tensor(a)
print(data)

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(rand_tensor)
print(ones_tensor)
print(zeros_tensor)

m = torch.ones(5,3,dtype=float)
n = torch.rand_like(m,dtype=torch.double)
print(m.size())
print(m)
print(n)
print("===========================")
print(torch.rand(5,4))
print(torch.randn(5,4))
print(torch.normal(0,1,size=(5,4)))
print(torch.linspace(1,10,20))
print("=========================================")
tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)
print(y1)
print(y2)
print(y3)
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(z1)
print(z2)
print(z3)

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

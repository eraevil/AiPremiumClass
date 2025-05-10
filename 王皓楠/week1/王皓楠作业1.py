import numpy as np 
arr=np.array([1,2,3,4,5,9,10,15,25],int)
print(arr)
print(arr.size)

In [ ]: 
a=np.array([[1,4,7],[2,5,8],[3,6,9]])
print(a.shape)
a=a.T
print(a)

In [ ]: 
a2=np.zeros(5,6,dtype=np.double64)
a2

In [ ]: 
import math
b=np.arange(1,10,math.e)
b

In [ ]: 
b4=np.eye(4)
print(b4)
c4=np.eye(10)
c4

In [ ]: 
a4=np.random.random(6)
a4
In [ ]: 
a=np.random.normal(0,1,10)
a
In [ ]: a=np.array([[1,3],[1,5],[2,4]])
print(a)
print(a[:,1])
print(a[2,:])
    In [ ]: a = np.array([(1,4,7), (2,5,8), (3,6,9)])
print("ndim:", a.ndim)
print("shape:", a.shape)
print("size", a.size)
print("dtype", a.dtype)
In [ ]:
print((5,6,8)in a)

In [ ]: 
a=np.arange(2,20,2)
print(a)
#变维度
a=a.reshape(3,3)
print(a)
print(a.shape)
#转置
print(a.T)
#平展
a=a.flatten()
print(a)

In [ ]: a8=np.array([(1,2,3),(3,5,6),(7,20,59)])
for i,j,k in a8:
    print(i,j,k)
a8=a8[:,np.newaxis,:]
print(a8.shape)
print(a8)

In [ ]:
a=np.array([[1,2],[3,4]])
b=np.zeros([2,2])
b=b-1.1
print(b)
print(a+b)
print(a-b)
print(a*b)
#除法不同
print(a/b)
print(a//b)

In [ ]: #和
print(a.sum())
#乘积
print(a.prod())
#平均数
print(a.mean())
#方差
print(a.var())
#标准差
print(a.std())

In [ ]: a=np.array([1.8,3.49,4.51])
#最大索引
print(a.argmax())
#最小索引
print(a.argmin())
#向上取整
print(np.ceil(a))
#向下取整
print(np.floor(a))
#四舍五入
print(np.rint(a))

In [ ]: a1=np.arange(3,30,3)
a1=a1.reshape(3,3)
print(a1)
b1=np.array([[1,2,3],[4,5,6],[7,8,9]])
c1=np.array([1,4,7])
print(b1)
print(c1)
#矩阵乘法与元素乘法
print("元素乘法:",a1*b1)
print("矩阵乘法:",np.dot(a1,b1))
#广播
print("广播加法:",a1+c1)
print("广播减法",a1-c1)
print("广播乘法",a1*c1)

In [ ]: 
result=np.save("numpy_test1.npy",np.dot(a1,c1))
result2=np.load("numpy_test1.npy")
result2


import torch
data=torch.tensor([[1.1,2.2],[3.3,4.4]],dtype=torch.float32)
data
arr =np.array([1,2,3,4,5],int) print(arr)   

In [ ]: 
import numpy as np
data1=np.array([[1,2],[3,4]])
data2=torch.from_numpy(data1)
print(data2.dtype)
data2

In [ ]: data2=data2.to(torch.int)
print(data2.dtype)
#创建维度相同张量，但是内部变量随机0-1
data3=torch.rand_like(data2,dtype=torch.double)
data3

In [ ]: 
shape=(3,4)
print("随机张量:",torch.rand(shape))
print("0张量:",torch.ones(shape))
print("1张量:",torch.zeros(shape))

In [ ]: #均匀分布
print(torch.rand(5,3))
#标准正态
print(torch.randn(5,3))
#离散分布正态
print(torch.normal(mean=0,std=1.0,size=(5,3)))
#构成范围内一维向量
print(torch.linspace(start=1,end=10,steps=50))

In [ ]: 
tensor = torch.rand(5,6)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

In [ ]: 
import torch
tensor
if torch.cuda.is_available():
    device = torch.device("cuda")
    tensor = tensor.to(device)
print(tensor)
print(tensor.device)

In [ ]: 
tensor = torch.tensor([[1,2,3],[4,5,6],[7,7,8]])
print(tensor[0])
print(tensor[:,0])
print(tensor[...,-1])

In [ ]: #dim指定按哪个维度合并
t1=torch.cat([tensor,tensor,tensor],dim=0)
print(t1-1)
print(t1.shape)

In [ ]: 
import torch
tensor = torch.arange(2,20, 2,dtype=torch.float32).reshape(3, 3)
print(tensor)
# 计算两个张量之间矩阵乘法的几种方式。 y1, y2, y3 最后的值是一样的 dot
ans1= tensor @tensor.T
ans2=tensor.matmul(tensor.T)
ans3=tensor 
print(ans1)
print(ans2)

ans3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=ans3)
print(ans3)

# 计算张量逐元素相乘的几种方法。 z1, z2, z3 最后的值是一样的。
z1 = tensor * tensor
z2 = tensor.mul(tensor)
#下面一行生成同维随机矩阵
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(z1)
print(z3)

In [ ]: agg=tensor.sum()
agg_item=agg.item()
print(agg_item,type(agg_item))

    
In [ ]: np_arr=z1.numpy()
np_arr

In [ ]: print(tensor,"\n")
tensor.add_(10)
print(tensor)

In [ ]: 
import torch
from torchviz import make_dot

# 定义矩阵 A，向量 b 和常数 c
A = torch.randn(10, 10,requires_grad=True)  # requires_grad=True 表示我们要对 A 求导
b = torch.randn(10,requires_grad=True)
c = torch.randn(1,requires_grad=True)
x = torch.randn(10, requires_grad=True)


# 计算 x^T * A + b * x + c
result = torch.matmul(A, x.T) + torch.matmul(b, x) + c

# 生成计算图节点
dot = make_dot(result, params={'A': A, 'b': b, 'c': c, 'x': x})
# 绘制计算图
dot.render('expression', format='png', cleanup=True, view=False)

In [ ]: 
import os
print(os.getcwd())


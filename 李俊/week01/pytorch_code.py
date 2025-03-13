import torch as t
import numpy as np

#张量
tensor=t.arange(1,13,dtype=t.float32).reshape(3,4)
print(tensor.shape)
#矩阵乘积
t1 = t.tensor([[2,3],[4,5],[1,2]])
t2 = t1.T
print(t1,t2)
print(t1@t2)
print(t1.matmul(t2))

#逐个元素相乘
t4=tensor*tensor
t5=tensor.mul(tensor)
t7 = t.rand_like(tensor)
print(t.mul(t4,t5,out=t7))
print(t4==t5)

#张量创建 numpy转换
data1=np.array([1,2,3])
print(t.from_numpy(data1))
data2=t.rand_like(t.from_numpy(data1),dtype=t.float32)
print(data2)
#
shape = (2,6,)
rand_tensor = t.rand(shape)
ones_tensor = t.ones(shape)
zeros_tensor = t.zeros(shape)
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
#创建全1tensor
data3=t.ones((3,3))
print(data3)
data4=t.rand_like(data3,dtype=t.float32)
print(data4)
# 获取tensor的大小
print(data4.size()) # torch.Size([5,3])
print(data4.shape)
print(data4.dtype)
print(data4.device)
# 均匀分布
print(t.rand(6,3))
# 标准正态分布
print(t.randn(6,3))
# 离散正态分布
print(t.normal(mean=.0,std=1.0,size=(6,3)))
# 线性间隔向量(返回一个1维张量，包含在区间start和end上均匀间隔的steps个点)
print(t.linspace(start=1,end=10,steps=21))


# 检查pytorch是否支持GPU
if t.cuda.is_available():
    device = t.device("cuda")
    tensor = tensor.to(device)

print(tensor)
print(tensor.device)

# mac上没有GPU，使用M系列芯片
if t.backends.mps.is_available():
    device = t.device("mps")
    tensor = tensor.to(device)

print(tensor)
print(tensor.device)


tensor = t.ones(3, 3)
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])
tensor[:,1] = 0
print(tensor)
#拼接
t1 = t.cat([tensor, tensor, tensor], dim=0)
print(t1 * 3)
print(t1.shape)
#求和
sum = t1.sum()
sum_item = sum.item()
print(sum_item)
#转换成numpy
print(t1.numpy())

#加法
t2 = t.tensor([[1,2],[3,4]])
print(t2.add_(5))
print(t2.add(5))
from torchviz import make_dot
#计算图
A = t.randn(10,10,requires_grad=True)
# print(A)
b=t.randn(10,requires_grad=True)
# print(b)
c = t.randn(1,requires_grad=True)
# print(c)
x = t.randn(10, requires_grad=True)
print(x)
result =t.matmul(A,x.T)+t.matmul(b,x)+c
print(result.shape)


dot=make_dot(result,params={'A': A, 'b': b, 'c': c, 'x': x})
# 绘制计算图
dot.render('expression', format='png', cleanup=True, view=False)
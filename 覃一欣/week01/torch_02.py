import torch

data = torch.tensor([[1,2,9],[3,4,5]], dtype=torch.float32)
data
import numpy as np

np_array = np.array([[1,2,4,8],[3,4,6,45]])
data2 = torch.from_numpy(np_array)
data2
data2.dtype
# 通过已知张量维度，创建新张量
data3 = torch.rand_like(data2, dtype=torch.float)
data3
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
# 基于现有tensor构建，但使用新值填充
m = torch.ones(5,3, dtype=torch.double)
n = torch.rand_like(m, dtype=torch.float)

# 获取tensor的大小
print(m.size()) # torch.Size([5,3])

# 均匀分布
print(torch.rand(5,3))
# 标准正态分布
print(torch.randn(5,3))
# 离散正态分布
print(torch.normal(mean=.0,std=1.0,size=(3,3)))
# 线性间隔向量(返回一个1维张量，包含在区间start和end上均匀间隔的steps个点)
print(torch.linspace(start=1,end=10,steps=1))
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
# 检查pytorch是否支持GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    tensor = tensor.to(device)

print(tensor)
print(tensor.device)

# mac上没有GPU，使用M系列芯片
if torch.backends.mps.is_available():
    device = torch.device("mps")
    tensor = tensor.to(device)

print(tensor)
print(tensor.device)
tensor = torch.ones(5, 5)
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])
tensor[:,1] = 0
print(tensor)
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1 * 3)
print(t1.shape)
import torch
tensor = torch.arange(1,17, dtype=torch.float32).reshape(4, 4)

# 计算两个张量之间矩阵乘法的几种方式。 y1, y2, y3 最后的值是一样的 dot
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

# print(y1)
# print(y2)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)
# print(y3)


# 计算张量逐元素相乘的几种方法。 z1, z2, z3 最后的值是一样的。
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

print(z1)
print(z3)
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))
np_arr = z1.numpy()
np_arr
print(tensor, "\n")
tensor.add_(5)
# tensor = tensor + 5
# tensor += 5
print(tensor)
tensor
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

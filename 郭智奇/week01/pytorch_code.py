# %%
# 数据中创建张量
import torch

data = torch.tensor([[1,2],[3,4]],dtype=torch.float)
data  

# %%
# 从 NumPy 数组中创建张量
import numpy as np

np_array = np.array([[1,2],[3,4]])
data2 = torch.from_numpy(np_array)
data2

# %%
 # 通过已知张量维度，创建新的张量,维度保持不变，只是改变值
x_one = torch.zeros_like(data)
x_one

# %%
 # 通过已知张量维度，创建随机新的张量,维度保持不变，只是改变值
x_two = torch.rand_like(data,dtype=torch.float)
x_two


# %%
# shape 是张量维度的元组。在下⾯的函数中，它决定了输出张量的维度
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# %%
# 基于现有tensor构建，但使⽤新值填充
m = torch.ones(5,3, dtype=torch.double)
n = torch.rand_like(m, dtype=torch.float)
# 获取tensor的⼤⼩
print(m.size()) # torch.Size([5,3])
# 均匀分布
print(torch.rand(5,3))
# 标准正态分布
print(torch.randn(5,3))
# 离散正态分布
print(torch.normal(mean=.0,std=1.0,size=(5,3)))
# 线性间隔向量(返回⼀个1维张量，包含在区间start和end上均匀间隔的steps个点)
print(torch.linspace(start=1,end=10,steps=20))

# %%
# 张量的属性
tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# %%
# 检查pytorch是否支持GPU
print(torch.cuda.is_available())

# %%
# 张量的索引和切片
tensor = torch.ones(4, 4)
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])
tensor[:,1] = 0
print(tensor)

# %%
# 张量的拼接
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
print(t1.shape)

# %%
# 算术运算
# 计算两个张量之间矩阵乘法的⼏种⽅式。 y1, y2, y3 最后的值是⼀样的 dot
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)
# 计算张量逐元素相乘的⼏种⽅法。 z1, z2, z3 最后的值是⼀样的。
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# %%
# 单元素的张量
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# %%
# In-place操作
# 把计算结果存储到当前操作数中的操作就称为就地操作。含义和pandas中inPlace参
# 数的含义⼀样。pytorch中，这些操作是由带有下划线 _ 后缀的函数表⽰。
print(tensor, "\n")
tensor.add_(5)
print(tensor)


# %%
# 与numpy之间的转换
# CPU 和 NumPy 数组上的张量共享底层内存位置，所以改变⼀个另⼀个也会变。
# 张量到numpy数组
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# %%
# 张量值的变更也反映在关联的NumPy 数组中
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# %%


# %%
# Numpy数组到张量
n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

# %%
import torch
from torchviz import make_dot
# 定义矩阵 A，向量 b 和常数 c
A = torch.randn(10, 10,requires_grad=True)
b = torch.randn(10,requires_grad=True)
c = torch.randn(1,requires_grad=True)
x = torch.randn(10, requires_grad=True)
# 计算 x^T * A + b * x + c
result = torch.matmul(A, x.T) + torch.matmul(b, x) + c
# ⽣成计算图节点
dot = make_dot(result, params={'A': A, 'b': b, 'c': c, 'x': x})
# 绘制计算图
dot.render('expression', format='png', cleanup=True, view=False)



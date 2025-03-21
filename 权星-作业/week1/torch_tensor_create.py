import torch
import numpy as np

# 从列表创建张量
data = torch.tensor([[1, 2], [3, 4]])
print(data)

# 从numpy数组创建张量
np_array = np.array([[1, 2], [3, 4]])
data2 = torch.from_numpy(np_array)
print(np_array)
print(data2)

# 通过已知张量维度，创建全1张量
data3 = torch.ones_like(data2)
print(data3)
print(data3.dtype)

# 通过已知张量维度，创建全0张量
data4 = torch.zeros_like(data2)
print(data4)
print(data4.dtype)

# 通过已知张量维度，创建随机数张量
data5 = torch.rand_like(data2, dtype=torch.float)
print(data5)
print(data5.dtype)


shape = (2, 2)
# 创建全1张量
data6 = torch.ones(shape)
print(data6)

# 创建全0张量
data7 = torch.zeros(shape)
print(data7)

# 创建随机数张量
data8 = torch.rand(shape)
print(data8)

# 基于现有 tensor 构建，但使⽤新值填充
m = torch.ones(5,3, dtype=torch.double)
n = torch.rand_like(m, dtype=torch.float)

# 获取 tensor 的⼤⼩
print(m.size()) # torch.Size([5,3])

# 均匀分布
print(torch.rand(5,3))

# 标准正态分布
print(torch.randn(5,3))

# 离散正态分布
print(torch.normal(mean=.0, std=1.0, size=(5,3)))
                                   
# 线性间隔向量(返回⼀个1维张量，包含在区间start和end上均匀间隔的steps个点)
print(torch.linspace(start=1,end=10,steps=20))

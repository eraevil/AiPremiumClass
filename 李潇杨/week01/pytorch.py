# %%
import torch as tr
import numpy as np

data = [ 1, 2],[3, 4]
x_data = tr.tensor(data)

aa=np.array([[ 1, 2],[3, 4]])
aa

tr.from_numpy(aa)


# %%
x_ones = tr.ones_like(x_data)
x_ones

x_rand = tr.rand_like(x_data, dtype=tr.float)
x_rand

shape = (2,3,)
rand_tensor = tr.rand(shape)
ones_tensor = tr.ones(shape)
zeros_tensor = tr.zeros(shape)
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# %%
# 基于现有tensor构建，但使⽤新值填充
m = tr.ones(5,3, dtype=tr.double)
print(m)
n = tr.rand_like(m, dtype=tr.float)
print(n)
# 获取tensor的⼤⼩
print(m.size()) # torch.Size([5,3])
# 均匀分布
m=tr.rand(5,3)
print(m)
# 标准正态分布
m=tr.randn(5,3)
print(m)
# 离散正态分布
m=tr.normal(mean=.0,std=1.0,size=(5,3))
print(m)
# 线性间隔向量(返回⼀个1维张量，包含在区间start和end上均匀间隔的steps个点)
m=tr.linspace(start=1,end=10,steps=20)
print(m)

# %%
tensor = tr.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# %%
tensor = tr.ones(4, 4)
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[:2, -1])
tensor[:,1] = 0
print(tensor)

# %%
t1 = tr.cat([tensor, tensor, tensor], dim=0)
print(t1)
print(t1.shape)

t1 = tr.cat([tensor, tensor, tensor], dim=1)
print(t1)
print(t1.shape)


# %%
y1 = tensor @ tensor.T
print(y1)
x1=t1@t1.T
print(x1)
y2 = tensor.matmul(tensor.T)
print(y2)
y3 = tr.rand_like(tensor)
tr.matmul(tensor, tensor.T, out=y3)
print(y3)

# %%
agg = tensor.sum()
print(agg, type(agg))
agg_item = agg.item()
print(agg_item, type(agg_item))

# %%
print(tensor, "\n")
tensor.add_(5)
print(tensor)

# %%
t = tr.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")




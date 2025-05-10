import torch
import numpy as np

data = torch.tensor([[1,2], [4,5]], dtype=torch.float)
print(data)

np_array = np.array([[1, 2], [3, 4]])
data1 = torch.from_numpy(np_array)
print(data1)
print(data1.dtype)
print(data1.shape)

np_array1 = np.array([[2,1], [2,3]])
data2 = torch.from_numpy(np_array1)
print(data2)
data3 = torch.rand_like(data2, dtype=torch.float)
print(data3)

shape = (2,3)
rand_tensor = torch.rand(shape)
rand_tensor1 = torch.rand((2,3))
print(rand_tensor)
print(rand_tensor1)
ones_tensor = torch.ones(shape)
print(ones_tensor)
zeros_tensor = torch.zeros(shape)
print(zeros_tensor)

ones_tensor1 = torch.ones((5,3), dtype=torch.float)
tensor2 = torch.rand_like(ones_tensor1)
print(ones_tensor1)
print(tensor2)
tensor3 = torch.rand_like(ones_tensor1, dtype=torch.float)
print(tensor3) 

print(tensor3.shape)
print(tensor3.size())
print(tensor3.dtype)

#均匀分布
rand_tensor = torch.rand((2,3))
rand_tensor1 = torch.rand(2,3)
print(rand_tensor1)
print(rand_tensor)

#标准正态分布
print(torch.randn(2, 3))

#从均值为mean,标准差为1.0的正态分布中抽取数据组成(2,3)的张量
print(torch.normal(mean=.0, std=1.0, size=(2,3)))

#线性间隔[start, end] 共 steps
print(torch.linspace(start=1,end=10,steps=4))

shape = (3, 4)
rand_tensor4 = torch.rand(shape)
print(rand_tensor4)
print(rand_tensor4.shape)
print(rand_tensor4.size())
print(rand_tensor4.dtype)
print(rand_tensor4.amax())
print(rand_tensor4.amin())
print(rand_tensor4.device)

#判断本地环境 torch 是否支持GPU cuda
if torch.cuda.is_available():
    device= torch.device("cuda")
    rand_tensor4 = rand_tensor4.to(device)

print(torch.cuda.is_available())
print(rand_tensor4)
print(rand_tensor4.device)

tensor_rand = torch.rand(3,4)
print(tensor_rand)
np_ary = np.array([(2,3), (7,5)], dtype=float)
ary_tensor = torch.from_numpy(np_ary)
print(ary_tensor)
print(ary_tensor.dtype)
print(ary_tensor[0])
print(tensor_rand[:,1])
print(ary_tensor[:, 1])
print(ary_tensor[1, :])
print(tensor_rand[1, :])
print(tensor_rand[1, 1])
print(tensor_rand[..., 1])
print(tensor_rand[-1, :])
tensor_rand[:, 1]=0
print(tensor_rand)

ones_t = torch.ones((3,3))
# dim = 0,拼接行
ones_ts1 = torch.cat([ones_t, ones_t], dim=0)
print(ones_t)
print(ones_ts1)
print(ones_ts1 * 2)
print(ones_ts1.shape)
# dim = 1, 拼接列
ones_ts2 = torch.cat([ones_t, ones_t], dim=1)
print(ones_ts2)
print(ones_ts2.shape)

# arange 区间 [1, 10)
arange_tensor = torch.arange(1, 10, dtype=torch.float32)
print(arange_tensor)
resp_arange_tensor = arange_tensor.reshape((3, 3))
print(resp_arange_tensor)

rat = resp_arange_tensor
print("rat", rat)
print(rat.T)
#矩阵乘 叉乘
r1 = rat @ rat.T
print(r1)
r2 = rat.matmul(rat.T)
print(r2)

#点乘 对位相乘
r3 = rat * rat
print("点乘 r3", r3)
r4 = rat.mul(rat)
print(r4)

r5 = torch.rand_like(rat)
print("origin r5", r5)
torch.mul(rat, rat, out=r5)
print(r5)

rat_sum = rat.sum()
print(rat_sum)
# item() 返回常数值
rat_sum_item = rat_sum.item()
print(rat_sum_item, type(rat_sum_item))

print("tensor r3", r3)
np_ary = r3.numpy()
print("numpy r3", np_ary)

print(rat, "\n")
#带下划线的危险函数，会改变原来tensor 的值(改内存)
rat.add_(5)
print(rat)

rat.sub_(5)
print(rat)

import torch
from torchviz import make_dot

# requires_grad 表示需要对A求导，需要梯度
A = torch.randn(10, 10, requires_grad=True)
A = torch.randn((10, 10), requires_grad=True)
print(A, A.shape)
b = torch.randn(10, requires_grad=True)
print(b, b.shape)
c = torch.randn(1, requires_grad=True)
print(c, c.shape)
x = torch.randn(10, requires_grad=True)
xt = x.T
print(x)
print(xt.shape)

# A @ x.T +  b @ X.T + c
# [10, 10] * [10, 1] + [1, 10] * [1, 10] + [1] => [10, 1]
bx = torch.matmul(b, x)
print("bx", bx)

axt = torch.matmul(A, x.T)
print("axt", axt)

result1 = axt + bx + c
result2 = torch.matmul(A, x.T) + torch.matmul(b, x) + c
print("result1", result1)
print("result2", result2)
#matmul()用法与规则

# 生成计算图节点
dot = make_dot(result, params={'A':A, 'b':b, 'c':c, 'x':x})
#绘制计算图(有向无环图)
dot.render('expression', format='png', cleanup=True, view=False)


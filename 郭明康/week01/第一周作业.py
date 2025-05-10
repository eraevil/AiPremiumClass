import numpy as np
#创建数组
arr = np.array([1,2,3], float)
#多维数组
arr = np.array([(1,2,3),(4,5,6),(7,8,9)])
#特殊数组
a = np.zeros((1,2), dtype = np.float32)
a1 = np.ones((3,4), dtype = np.float64)
#等差数列
a2 = np.arange(2,6,0.4)
#单位矩阵
a3 = np.eye(5)
#随机数字分布在[0,1),指定长度
a4 = random.random(10)
         
a5 = np.random.normal(0, 1, 10) 
         
a6 = np.array([(1,1), (3,3), (5,5)])
print(a6[:,-1])

a7 = np.array([(1,2), (3,4), (5,6)])
for i,j in a7:
    print(i+j)         

a8 = np.array([(1,1,1), (2,2,2), (3,3,3)])
print("ndim:", a8.ndim)
print("shape:", a8.shape)
print("size", a8.size)
print("dtype", a8.dtype)
#检查
print(3 in a8)

a9 = np.arange(1,12)
print(a9.shape)
a9 = a9.reshape(3,4,1)
print(a9.shape)

print(a9)
a9 = a9.T
print(a9)

a9 = a9.flatten()
print(a9)
b = np.array([1,,1,2])

a10 = np.array([(2,2), (4,4), (6,6)])
a10 = a10[:,:,np.newaxis]  # 给前两个维度之后的维度加一个新维度
a10.shape

a11 = np.ones((2,2))
b2 = np.array([(-1,1),(-1,1)])
print(a)
print(b2)
print(a+b2)
print(a-b2)
a11.sum()
a11.prod()

a12 = np.array([1,2,3])
print("mean:",a12.mean())
print("var:", a12.var())
print("std:", a12.std())

a13 = np.array([1.12, 3.73, 8.88])
print("argmax:", a13.argmax()) 
print("argmin:", a13.argmin())
print("ceil:", np.ceil(a13))  
print("floor:", np.floor(a13)) 
print("rint:", np.rint(a13))  

a14 = np.array([5,19,77,2,16,66,49])
a14.sort()

import torch
tensor = torch.arange(1,20, dtype=torch.float64).reshape(2,2)

#叉乘返回矩阵
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)


#点乘返回数值
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

a15 = np.array([[2,2,2],[5,5,5]])
b3 = np.array([[1,2,3],[7,5,6]])
a15 * b3

import numpy as np

m1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
m2 = np.array([[5, 6], [7, 8]], dtype=np.float32)
# m1 = m1.reshape(1,4)  # 矩阵1最后1个维度 == 矩阵2倒数第2个维度
# m2 = m2.reshape(4,1)

# 使用 np.dot 进行叉乘
result_dot = np.dot(m1, m2)

# 使用 @ 运算符进行叉乘
result_at = m1 @ m2

print("矩阵 1:")
print(m1)
print("矩阵 2:")
print(m2)
print("使用 np.dot 得到的叉乘结果:")
print(result_dot)
print("使用 @ 运算符得到的叉乘结果:")
print(result_at)

np.save('result.npy',result_at)

result_np = np.load('result.npy')

a16 = np.array([1,1,1])
b4 = np.array([9,9,9])
a16 + b4

a17 = np.array([(1,1), (2,2), (3,3), (4,4)])  
b5 = np.array([-1,1])  #自动填充，从（，2）到（1,2）到（4,2）
a17 + b5

import torch
import numpy as np
         
data = torch.tensor([[0,1],[2,3]], dtype=torch.float32)

np_array = np.array([[2,2],[4,3]])
data2 = torch.from_numpy(np_array)

data3 = torch.rand_like(data2, dtype=torch.float32)

#用给定的shape来生成tensor        
shape = (1,2,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

#使用现有tensor作为大小和属性限制新tensor
m = torch.ones(2,2, dtype=torch.double)
n = torch.rand_like(m, dtype=torch.float)

print(m.size()) # torch.Size([5,3])

# 均匀分布
print(torch.rand(2,2))
# 标准正态分布
print(torch.randn(2,2))
# 离散正态分布
print(torch.normal(mean=.0,std=1,size=(3,3)))
# 线性间隔向量(返回一个1维张量，包含在区间start和end上均匀间隔的steps个点)
print(torch.linspace(start=1,end=5,steps=15))

tensor = torch.rand(3,3)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# 检查pytorch是否支持GPU，并且转换
if torch.cuda.is_available():
    device = torch.device("cuda")
    tensor = tensor.to(device)

print(tensor.device)

tensor = torch.ones(3,3)
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])#三个点代表选择未提及的其他所有维度
tensor[:,-1] = 0
print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=1)#在dim维度进行拼接
print(t1 * 2)
print(t1.shape)

tensor = torch.arange(1,17, dtype=torch.float32).reshape(4,4)
#叉乘
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

# print(y1)
# print(y2)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)
# print(y3)


#点乘
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

#转换python类型
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

np_arr = z1.numpy()

#in-place操作
tensor.add_(5)
         
#计算图
from torchviz import make_dot

# 定义矩阵 A，向量 b 和常数 c
A = torch.randn(10, 10,requires_grad=True)  # requires_grad=True 表示我们要对 A 求导
b = torch.randn(10,requires_grad=True)
c = torch.randn(1,requires_grad=True)
x = torch.randn(10, requires_grad=True)

result = torch.matmul(A, x.T) + torch.matmul(b, x) + c

# 生成计算图节点
dot = make_dot(result, params={'A': A, 'b': b, 'c': c, 'x': x})
# 绘制计算图
dot.render('expression', format='png', cleanup=True, view=False)





         

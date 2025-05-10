# 一、numpy练习
import numpy as np

# 1、创建一维数组
b = np.array([1,2,3], dtype=int)
print(b)
print("维度：", b.shape)
print("数据类型：", b.dtype)
print("大小", b.size)

# 2、创建多维数组
c = np.array([(1,2,3),(4,5,6),(7,8,9)])
print(c)

# 3、创建特殊数组。全为1，全为0，等差数列, 单位矩阵，随机数组，正态分布的随机数组

one1 = np.ones([3,3])
print(one1)

zero1 = np.zeros([3,3])
print(zero1)

deng = np.arange(1, 10, 1, dtype=int)
print(deng)

eArr = np.eye(3)
print(eArr)

randomArr = np.random.random(5)
print(randomArr)

randnArr = np.random.normal(0, 0.1, 5)
print(randnArr)


# 4、numpy数组的访问
arr = np.array([(1,2,3),(4,5,6),(7,8,9)])
print(arr[0])
print(arr[1:])
print(arr[:, :1])
print(arr[2,2])

# 4、numpy数组的遍历
arr = np.array([(1,2,3),(4,5,6),(7,8,9)])
for i in arr:
    print(i)

for i,j,k in arr:
    print(i+j+k)

# 5、numpy数组的常用属性
arr = np.array([(1,2,3), (4,5,6), (7,8,9)])
print(arr.ndim)
print(arr.shape)
print(arr.size)
print(arr.dtype)

# 6、NumPy 数组的基本操作
# 检测数值是否在数组中
arr = np.array([(1,2,3), (4,5,6), (7,8,9)])
print(10 in arr)
print((1,2,3) in arr)

# 数组重排列 二维变一维
arr = np.array([(1,2,3), (4,5,6), (7,8,9)])
print(arr.reshape(9))

# 转置
arr = np.array([(1,2,3), (4,5,6), (7,8,9)])
print(arr.T)

# 多维变一维
arr = np.array([(1,2,3), (4,5,6), (7,8,9)])
print(arr.flatten())

# 增加维度
arr = np.array([1,2,3])
print(arr.ndim)
print(arr.shape)
arr = arr[:,np.newaxis]
print(arr)
print(arr.ndim)
print(arr.shape)


# 7、NumPy 数组的数学操作
# 加减乘除
a = np.ones([2,2])
b = np.array([(-1,1),(-1,1)])
print(a+b,"\n", a-b,"\n", a*b,"\n", a/b,"\n")

# 求和求积
a = np.ones([2,2])
print(a.sum())
print(a.prod())

#平均数、方差、标准差、最大值、最小值
a = np.array([1,2,3])
print(a.mean())
print(a.var())
print(a.std())
print(a.max())
print(a.min())

# 最⼤与最⼩值对应的索引值：argmax,argmin:
# 取元素值上限，下限，四舍五⼊：ceil, floor, rint
a = np.array([1.1,2.6,3,4,5.8])
print(a.argmax())
print(a.argmin())
print(np.ceil(a))
print(np.floor(a))
print(np.rint(a))

# 排序
a = np.array([1,6,3,4,5])
a.sort()
print(a)

# 8、NumPy 线性代数
# 矩阵乘法
m1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
m2 = np.array([[5, 6], [7, 8]], dtype=np.float32)
result1 = m1 @ m2
result2 = np.dot(m1, m2)
print(result1)
print(result2)

"""其他函数
eig : 特征值、特征向量
inv : 逆
qr :QR 分解
svd : 奇异值分解
solve : 解线性⽅程 Ax=b
lstsq : 计算Ax=b 的最⼩⼆乘解"""

# #9、⽂件操作
"""⼆进制⽂件： np.save 、 np.load
⽂本⽂件： np.loadtxt 、 np.savetxt"""

# 10、NumPy ⼴播机制
a = np.array([[1,2],[3,4]])# shape(2,2)
b = np.array([5,6])# shape(2)
print(a+b)


# 二、pytorch基础
import torch

# 1、创建张量
data = [1,2,3]
tensor = torch.tensor(data, dtype=float)
np_arr = np.array(data)
tensor2 = torch.from_numpy(np_arr)

ones = torch.ones_like(tensor2)
print(ones)

rand = torch.rand_like(tensor2, dtype=float)
print(rand)

# 2、使⽤随机值或常量值
shape = (3,3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(rand_tensor)
print(ones_tensor)
print(zeros_tensor)

# 3、基于现有tensor构建，但使⽤新值填充
m = torch.ones([3,5], dtype=torch.double)
print(m.dtype)
n = torch.rand_like(m, dtype=torch.float32)
print(m.size())

# 均匀分布
print(torch.rand(3,5))
# 标准正态分布
print(torch.randn(3,5))
# 离散正态分布
print(torch.normal(mean=.0, std=1.0, size=(3,5)))
# 线性间隔向量(返回⼀个1维张量，包含在区间start和end上均匀间隔的steps个点)
print(torch.linspace(start= 1, end=10, steps = 10))

# 4、张量的属性
tensor = torch.tensor([1,2,3], dtype=torch.double)
print("数据类型：", tensor.dtype)
print("形状：", tensor.shape)
print("维度：", tensor.ndim)
print("设备：", tensor.device)

# 5、张量运算

# 张量的索引和切⽚：
tensor = torch.ones(4,4)
print("first row: ", tensor[0])
print("first cloumn: ", tensor[:,0])
print("last cloumn: ", tensor[:,-1])
tensor[:, 1] = 0
print(tensor)

# 张量的拼接
tensor = torch.ones(4,4)
t = torch.cat([tensor, tensor], dim = 1)# dim = 1为列方向
print(t)

t1 = torch.stack([tensor, tensor], dim = 0)
print(t1)

# 算术运算
# 矩阵乘法
tensor = torch.ones(3,3)
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)
print(y1)
print(y2)
print(y3)

# 计算张量逐元素相乘的⼏种⽅法
tensor = torch.ones(3,3)
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor,tensor, out=z3)
print(z1,z2,z3)

# 单元素张量
tensor = torch.ones(3,3)
sum = tensor.sum()
print(sum)
sum_item = sum.item()
print(sum_item, type(sum_item))

# In-place操作
tensor = torch.ones(3,3)
tensor.add_(4)
print(tensor)

# 6、与numpy之间的转换
# 张量转换为numpy
t = torch.ones(3,3)
print(t)
n = t.numpy()
print(n)
t.add_(3)
print(t)
print(n)

# numpy转换为张量
n = np.ones(3)
t = torch.from_numpy(n)
print(n)
print(t)
np.add(n, 1, out = n)
print(n)
print(t)

# 7、计算图
import torch
from torchviz import make_dot
A = torch.randn(10, 10, requires_grad=True)
b = torch.randn(10, requires_grad=True)
c = torch.randn(1, requires_grad=True)
x = torch.randn(10, requires_grad=True)

result = torch.matmul(A, x.T) + torch.matmul(b, x) + c

dot = make_dot(result, params={'A': A, 'b': b, 'c': c, 'x': x})
dot.render('expression1', format='png', cleanup=True, view=False)




import numpy as np
# 创建数组
arr = np.array([1,2,3,4,5], float)
print(arr)
arr1 = np.array([1,2,3,4,5], int)
print(arr1)
arr = np.array([(3,2,1),(33,22,11),(333,222,111)])
arr
# 全0数组 5行6列
a = np.zeros((5,6), dtype = np.int32)
print(a)
# 默认为float类型
b = np.zeros((5,6))
print(b)
# 等差数列 [)
a = np.arange(1, 5, 1)
a
# 左上右下对角为1的4行4列矩阵
a = np.eye(4)
a
# 生成5个[0,1)之间的随机值
# 用于模型运算参数初始化
a = np.random.random(5)
a
# 正态分布随机数组
# 均值为0 标准差为0.2 数量6个
# 用于模型运算参数初始化
a = np.random.normal(0, 0.2, 6)
a
a = np.array([(1,2,12), (3,4,34), (5,6,56)])
a
print("1", a[:,1])
print("2", a[:,-1])
print("3", a[0,:])
print("4", a[-3,:])
a = np.array([(1,2), (3,4), (5,6)])
i, j = a[0]
print(i, j,"\n")
for i, j in a:
    print(i,j)
a = np.array([[(1,2,3), (4,5,6)],[(1,2,3), (4,5,6)]])
# 数组维度 秩
print("ndim:", a.ndim)
# 数组大小 如几行几列
print("shape:", a.shape)
# 元素总个数 等于a.shape中的元素乘积
print("size", a.size)
# 元素类型
print("dtype", a.dtype)
# 基于单一的元素进行检测
print(3 in a)
print((1,2,3) in a)
#t = np.array([(1,2,3),(4,5,6)])
#print(t in a)
a = np.arange(1,20,2)
print(a)
print(a.shape)
a = a.reshape(2,5)
print(a)
print(a.shape)
# 加载图片
# img.shape[1,3, 150, 200]1样本 3通道 高*宽150*200
print(a)
a = a.T
print(a)
a = a.flatten()
print(a)
a = np.array([(1,2), (3,4), (5,6)])
a = a[:, :, np.newaxis]
a.shape
a = np.array([(1,2),(3,4)])
b = np.array([(4,2),(1,3)])
print(a)
print(b)
print(a+b)
print(a-b)
print(a.sum())
print(a.prod())
a = np.array([10.9,5.3,1.2])
# 平均值
print("mean:",a.mean())
# 方差
print("var:", a.var())
# 标准差
print("std:", a.std())
# 最大值
print("max:", a.max())
# 最小值
print("min:", a.min())
# 最大值的索引
print("argmax:", a.argmax())
# 最小值的索引
print("argmin:", a.argmin())
# 向上取整
print("ceil:", np.ceil(a))
# 向下取整
print("floor:", np.floor(a))
# 四舍五入
print("rint:", np.rint(a))
a.sort()
print(a)
import torch
tensor = torch.arange(1,10, dtype=torch.float32).reshape(3, 3)

print(tensor.dtype)
print(tensor)
# 计算两个张量之间矩阵乘法的几种方式。 y1, y2, y3 最后的值是一样的 dot
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)


# 计算张量逐元素相乘的几种方法。 z1, z2, z3 最后的值是一样的。
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
a = np.array([[1,2,3],[3,1,2]])
b = np.array([[200,1,6],[1,2,3]])
a * b
import numpy as np
# 定义两个简单的矩阵
m1 = np.array([[2, 1], [4, 2]] , dtype=np.float32)
m2 = np.array([[5, 6], [7, 8]] , dtype=np.float32)
# 使⽤ np.dot 进⾏矩阵乘法
result_dot = np.dot(m1, m2)
# 使⽤ @ 运算符进⾏矩阵乘法
result_at = m1 @ m2
print("矩阵 1:")
print(m1)
print("矩阵 2:")
print(m2)
print("使⽤ np.dot 得到的矩阵乘法结果:")
print(result_dot)
print("使⽤ @ 运算符得到的矩阵乘法结果:")
print(result_at)
# 创建⼀个全零矩阵，⽤于存储⼿动推演的结果
# 结果矩阵的⾏数等于 matrix1 的⾏数，列数等于 matrix2 的列数
manual_result = np.zeros((m1.shape[0], m2.shape[1]), dtype=np.float32)
# 外层循环：遍历 matrix1 的每⼀⾏
# i 表⽰结果矩阵的⾏索引
for i in range(m1.shape[0]):
 # 中层循环：遍历 matrix2 的每⼀列
 # j 表⽰结果矩阵的列索引
 for j in range(m2.shape[1]):
  # 初始化当前位置的结果为 0
  manual_result[i, j] = 0
  # 内层循环：计算 matrix1 的第 i ⾏与 matrix2 的第 j 列对应元素的乘积之和
  # k 表⽰参与乘法运算的元素索引
  for k in range(m1.shape[1]):
   # 打印当前正在计算的元素
   print(f"{m1[i, k]} * {m2[k, j]} = {m1[i, k] * m2[k, j]}")
   # 将 matrix1 的第 i ⾏第 k 列元素与 matrix2 的第 k ⾏第 j 列元素相乘，并累
   manual_result[i, j] += m1[i, k] * m2[k, j]
  # 打印当前位置计算完成后的结果
  print(f"结果矩阵[{i+1},{j+1}]:{manual_result[i, j]}\n")
print("⼿动推演结果:")
print(manual_result)
#17 20
#34 40
np.save('result.npy', manual_result)
res = np.load('result.npy')
res
a = np.array([(1,3), (2,4), (3,5), (4,6), (5,7)]) # shape(5,2)
b = np.array([1,-1]) # shape(2)-> shape(1,2) -> shape(5,2) [[1,-1],[1,-1],[1,-1],[1,-1],[1,-1]]
a + b

import torch
# 张量
data = torch.tensor([[21,2],[31,4]], dtype=torch.float32)
data
import numpy as np
# 利用数组创建张量
np_array = np.array([[5,2],[1,4]])
data2 = torch.from_numpy(np_array)
data2
# 张量类型
data2.dtype
# 通过已知张量维度，创建新张量
# 通过dtype指定类型
data3 = torch.rand_like(data2, dtype=torch.float)
data3
shape = (4,3,)
# 随机值
rand_tensor = torch.rand(shape)
# 全1
ones_tensor = torch.ones(shape)
# 全0
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
# 基于现有tensor构建，但使用新值填充
m = torch.ones(2,3, dtype=torch.double)
n = torch.rand_like(m, dtype=torch.float)

# 获取tensor的大小
print(m.size()) # torch.Size([2,3])

# 均匀分布
print(torch.rand(2,3))
# 标准正态分布
print(torch.randn(2,3))
# 离散正态分布
print(torch.normal(mean=.0,std=1.0,size=(2,3)))
# 线性间隔向量(返回一个1维张量，包含在区间start和end上均匀间隔的steps个点)
# step：拆分成多少份
print(torch.linspace(start=1,end=10,steps=5))
tensor = torch.rand(2,3)
# 维度
print(f"Shape of tensor: {tensor.shape}")
# 类型，默认float32
print(f"Datatype of tensor: {tensor.dtype}")
# 设备
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
tensor = torch.tensor([(1, 2), (3,4)])
# 第0行的张量
print('First row: ', tensor[0])
# 所有行的第0列
print('First column: ', tensor[:, 0])
# 使用...代替所有维度
print('Last column:', tensor[..., -1])
tensor[:,1] = 0
print(tensor)
# dim 在第几维进行拼接
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1 * 3)
print(t1.shape)
t2 = torch.stack([tensor, tensor, tensor], dim=0)
print(t2)
import torch
tensor = torch.arange(1,10, dtype=torch.float32).reshape(3, 3)
print(tensor)
# 计算两个张量之间矩阵乘法的几种方式。 y1, y2, y3 最后的值是一样的 dot
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

print(y1)
print(y2)

y3 = torch.rand_like(tensor)
# out 填充到y3里
torch.matmul(tensor, tensor.T, out=y3)
# print(y3)


# 计算张量逐元素相乘的几种方法。 z1, z2, z3 最后的值是一样的。
# 按位相乘
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

print(z1)
print(z3)
# agg类型为张量
agg = tensor.sum()
# agg_item类型为python原生数据类型
agg_item = agg.item()
print(agg_item, type(agg_item))
# 转化为数组
np_arr = z1.numpy()
np_arr
print(tensor, "\n")
# ..._() 没有返回值，直接修改自身
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

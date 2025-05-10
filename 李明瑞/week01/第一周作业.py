# %%
#1. numpy 多维数组运算库
#数据类型：bool_  int_  uint_  float_  complex_
#numpy的数据类型实际上是dtype对象的实例，对应唯一的字符
# ndarray  numpy数组  由数组实际数据与描述这些数据的元数据组成
#1.1 numpy数组创建
#用列表转换
import numpy as np
a = [1,2,3]
b = np.array(a)
b

# %%
arr1 = np.array([1,2,3], float)
arr2 = np.array((3,4,5),int)
arr3 = np.array([4,5,6],dtype = float)
print(arr1, arr1.dtype)  #打印时省略了array修饰
print(arr2, arr2.dtype)  #Jupyter Notebook默认会显示最后赋值的变量结果  因此用print
arr3  

# %%
#多维数组创建
a = np.array(([1,2,3],[4,5,6],[7,8,9]))
a

# %%
#特殊数组创建
a = np.zeros((3,4), float)  #全0数组
a

# %%
b = np.ones((3,2))  #全1数组
b

# %%
c = np.arange(100,10,-10)  #等差数组
c

# %%
d = np.eye(3)  #单位矩阵1
d

# %%
e = np.identity(3)  #单位矩阵2
e 

# %%
f = np.random.rand(4)  #随机数数组
f

# %%
g = np.zeros_like([[1,2,3],[4,5,6]])  #0矩阵
g

# %%
h = np.full((2,3),5)  #按形状填充特定值
h

# %%
i = np.diag([3,4,5])  #对角矩阵
i

# %%
mu,sigma = 1,0.2  #正太分布数组
np.random.normal(mu,sigma,5)

# %%
#1.2 np数组访问
a = np.array(([1,2],[3,4]))
print(a[0])
print(a[1:])
print(a[:1])
print(a[:,:1])
print(a[1][1])

# %%
# 数组的遍历
a = np.array([1,2,3])
for i in a:
    print(i)
b = np.array(([1,2],[4,5],[6,7]))
for j,k in b:
    print(j*k)

# %%
#数组的属性
a = np.array([(1,2,3), (4,5,6), (7,8,9)])
print("ndim:", a.ndim)  #维度、秩
print("shape:", a.shape)  #大小
print("size", a.size)  #元素个数
print("dtype", a.dtype)  #元素类型

# %%
# 1.3数组的基本操作
# 检测数值是否在组中
a = np.array([(1,2,3),(4,5,6)])
print(4 in a)
print(7 in a)

# %%
# 数组的重排列
a = np.zeros([2,3])
a
a.reshape(6)  #array([0.,0.,0.,0.,0.,0.])
a.reshape(3,2)

# %%
# 数组的转置
a = np.array(([1,2,3,4],[5,6,7,8]))
a.T  #a.transpose()

# %%
# 多维数组转换一维
a = np.array([(1,2,3),(4,5,6)])
a.flatten()

# %%
#增加维度
a = np.array((1,2,3,4))
print(a.shape)
a = a[:, np.newaxis]
a

# %%
# 1.4 数组的数学操作
a = np.array(((1,2),(2,1)))
b = np.ones((2,2))
print(a)
print(b)
print(a+b)
print(a*b)
print(a.sum())
print(a.prod())  #积

# %%
a = np.array((2,4,6))
print("mean:",a.mean())  #平均值
print("var:", a.var())  #方差
print("std:", a.std())  #标准差
print("max:", a.max())  
print("min:", a.min())

# %%
a = np.array([1.2, 3.8, 4.9])
print("argmax:", a.argmax())  #返回元素最大值位置
print("argmin:", a.argmin())  #返回元素最小值位置
print("ceil:", np.ceil(a))  #向上取整
print("floor:", np.floor(a))  #向下取整
print("rint:", np.rint(a))  #四舍五入

# %%
#排序
a = np.array([16,31,12,28,22,31,48])
a.sort()
a

# %%
#numpy线性代数
a = np.array(((1,2),(3,4)))
b = np.array(((2,2),(3,3)))
result_dot = np.dot(a,b)  #矩阵乘法
result_at = a @ b  #矩阵乘法
print(result_dot)
print(result_at)

# %%
m1 = np.array([[1, 2], [3, 4]] , dtype=np.float32)
m2 = np.array([[5, 6], [7, 8]] , dtype=np.float32)
#结果矩阵的⾏数等于 matrix1 的⾏数，列数等于 matrix2 的列数
manual_result = np.zeros((m1.shape[0], m2.shape[1]), dtype=np.float32) 
for i in range(m1.shape[0]):
    for j in range(m1.shape[0]):
        manual_result[i, j] = 0
    for k in range(m1.shape[1]):
        print(f"{m1[i, k]} * {m2[k, j]} = {m1[i, k] * m2[k, j]}")
        manual_result[i, j] += m1[i, k] * m2[k, j]
    print(f"结果矩阵[{i+1},{j+1}]:{manual_result[i, j]}\n")
print("⼿动推演结果:")
print(manual_result)

# %%
#文件操作
#二进制文件 np.save  np.load  .npy格式
#文本文件 np.loadtxt  np.savetext
np.save('result.npy',manual_result)

# %%
result_np = np.load('result.npy')
result_np

# %%
#numpy广播机制
#不同形状数组数值运算
a = np.array(((1,2,3),(4,5,6)))  #(3,2)
b = np.array((1,2,3))  #(1,3)
a + b

# %%
#2. pytorch基础
#2.1 张量  pytorch中的基本单位  承载数据
#张量的创建 
import torch
data = [[1,2],[3,4]]  #从数据中创建
data1 = torch.tensor(data, dtype=torch.float32)
data1

# %%
np_array = np.array(data)
data2 = torch.from_numpy(np_array)  #从np数组创建
data2

# %%
data2.type  #查看数据类型

# %%
data3 = torch.ones_like(data2, dtype=torch.float)  #通过已知张量维度创建
data3

# %%
#创建指定大小的张量
shape = (4,5)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# %%
# 基于现有tensor构建，使用新值填充
a = torch.ones(3,4, dtype=torch.double)  #双精度
b = torch.rand_like(a, dtype=torch.float)
print(b.size())  #获得tensor大小
b

# %%
# 均匀分布
print(torch.rand(5,3))
# 标准正态分布
print(torch.randn(5,3))
# 离散正态分布
print(torch.normal(mean=.0,std=1.0,size=(5,3)))  #均值 标准差 方差
# 线性间隔向量(返回一个1维张量，包含在区间start和end上均匀间隔的steps个点)
print(torch.linspace(start=1,end=10,steps=10))

# %%
#2.2张量的属性
tensor = torch.rand(3,2)

print(f"Shape of tensor: {tensor.shape}")   #torch.size
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")  #存储设备


# %%
# 检查pytorch是否支持GPU
tensor = torch.rand(3,2)  #默认存储在cpu上
if torch.cuda.is_available():  #检查是否有可用的gpu
    device = torch.device("cuda")  #设置当前计算设备为GPU
    tensor = tensor.to(device)  #将张量移动到指定设备
print(tensor)
print(tensor.device)

# mac上没有GPU，使用M系列芯片
if torch.backends.mps.is_available():
    device = torch.device("mps")
    tensor = tensor.to(device)
print(tensor)
print(tensor.device)


# %%
#2.3张量运算
#索引和切片
tensor = torch.ones(2,3)
print('First row: ', tensor[0])  #第1行
print('First column: ', tensor[:, 0])  #所有行的第1列
print('Last column:', tensor[..., -1])  #所有行的最后一列
tensor[:,1] = 0  #所有行的第一列为0
print(tensor)

# %%
#张量的拼接
#PyTorch 的默认维度顺序是 [批次 (B), 然后是特征 (C), 高度 (H), 宽度 (W)]）拼接起来
tensor = torch.tensor([[1,2],[3,4]])
t1 = torch.cat([tensor, tensor, tensor], dim=1)  #行拼接
t2 = torch.stack([tensor,tensor], dim=1)  #列堆叠
print(t1)
print(t1.shape)
print(t2)
print(t2.shape)

# %%
# torch.stack(tensors, dim=0, out=None)

a = torch.randn(2, 3)
b = torch.randn(2, 3)

print("张量 a：")
print(a)
print("\n张量 b：")
print(b)

# 沿第一个维度堆叠
c = torch.stack([a, b], dim=0)  #[2,2,3] 在第0个维度增加2个样本
print("\n堆叠后的张量 c：")
print(c)

# 沿第二个维度堆叠
d = torch.stack([a, b], dim=1)  #[2,3,2] 即在第2个维度增加了两倍的通道数
print("\n堆叠后的张量 d：")
print(d)


# %%
a = torch.randn(2, 3)
b = torch.randn(2, 3)
print(a)
print(b)

cat_result1 = torch.cat([a, b], dim=0)
print("Concatenated tensor shape:", cat_result1.shape) 

cat_result2 = torch.cat([a, b], dim=1)
print("Concatenated tensor shape:", cat_result2.shape) 

stacked1 = torch.stack([a, b], dim=0)
print(stacked1, "Stacked tensor shape:", stacked1.shape)

stacked2 = torch.stack([a, b], dim=1)
print(stacked2, "Stacked tensor shape:", stacked2.shape)

stacked3 = torch.stack([a, b], dim=2)
print(stacked3, "Stacked tensor shape:", stacked3.shape)

# %%
#算数运算
tensor = torch.arange(1,10, dtype=torch.float32).reshape(3, 3)
print(tensor)
# 计算两个张量之间矩阵乘法的几种方式。 y1, y2, y3 最后的值是一样的 dot
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

print(y1)
print(y2)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)
print(y3)


# 计算张量逐元素相乘的几种方法。 z1, z2, z3 最后的值是一样的。
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

print(z1)
print(z3)

# %%
#单元素张量  值聚合计算
a = tensor.sum()
a_item = a.item()
print(a_item, type(a_item))

# %%
#inplace操作  把计算结果存储到当前操作数中的操作就称为就地操作
print(tensor, "\n")
tensor.add_(5)
# tensor = tensor + 5
# tensor += 5
print(tensor)

# %%
#tensor与numpy数组的转换
np_arr = tensor.numpy()
print(tensor)
np_arr

# %%
a = np.ones(5)
b = torch.from_numpy(a)
print(b)

# %%
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# %%
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# %%
n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

# %%
#计算图
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



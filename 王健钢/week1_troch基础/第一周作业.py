import numpy as np

arr = np.array([1, 2, 3, 5, 20])
arr
a = np.array([1, 2, 3, 4, 5], float)
a
a = np.array([(1,2,3), (7,8,9)])
a
a1 = np.zeros((1,3))
a1
a2 = np.ones((1,3))
a2
a3 = np.arange(1,3,0.2)
a3
a4 = np.eye(3, 3)  # 3x3的单位矩阵
a4
a5 = np.random.random(5)    
a5
a5 = np.random.normal(0, 1, 5)   # 5个标准正态分布的随机数
a5
a6 = np.array([[1, 2, 3], [4, 5, 6]])
print("arr:",  a6)
print("part:",a6[:,1])
a7 = np.array([(1,2,3),(4,5,6),(7,8,9)])
for i,j,z in a7:
    print(i,j)

a = np.array([(1,2,3), (7,8,9)])
print("ndim:", a.ndim)
print("shape:", a.shape)
print("size:", a.size)
print("dtype:", a.dtype)
print(3 in a)
a7 = np.arange(1,8)
print(a7)
print(a7.shape)
a7 = a7.reshape(1,7) 
print(a7)
print(a7.shape) 

print(a)
a= a.T
print(a)
a = a.flatten() 
print("a.flatten():", a)
a8 = np.array([(1, 2, 3), (4, 5, 6)], dtype=np.int32)
print(a8)
a8.shape

a8 = a8[:,:,np.newaxis]
print(a8)
a8.shape
a = np.ones((2, 2), dtype=int)
b = np.array([(-1,1),(-1,1)])
print(a)
print(b)
print(a+b)
print(a-b)
a.sum()
a.prod()
a = np.array([2,4.3,8.6,9.5,10.1])
print("mean",a.mean())     #求平均值
print("max",a.max())       #求最大值
print("min",a.min())       #求最小值
print("std",a.std())       #求标准差
print("var",a.var())       #求方差
print("argmax",a.argmax()) #求最大值的下标
print("argmin",a.argmin()) #求最小值的下标
print("cumsum",a.cumsum()) #求累计和
print("ceil",np.ceil(a))   #向上取整
print("floor",np.floor(a)) #向下取整
print("rint ",np.rint(a))  #四舍五入取整
print("sign",np.sign(a))   #返回各元素的正负号

a.sort() #排序
a
import torch
tensor = torch.arange(1,10, dtype=torch.float32).reshape(3, 3)

print(tensor)
print(tensor.shape)
print(tensor.dtype)

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)

z1 = tensor * tensor
print(z1)
z2 = tensor.mul(tensor)
print(z2)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

a9 = np.array([[1,2,3],[4,5,6]])
b9 = np.array([[4,5,6],[7,8,9]])

a9 * b9
import numpy as np

# 定义两个简单的矩阵
m1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
m2 = np.array([[5, 6], [7, 8]], dtype=np.float32)
# m1 = m1.reshape(2,2)  # 矩阵1最后1个维度 == 矩阵2倒数第2个维度
# m2 = m2.reshape(2,2)

# 使用 np.dot 进行矩阵乘法
result_dot = np.dot(m1, m2)

# 使用 @ 运算符进行矩阵乘法
result_at = m1 @ m2

print("矩阵 1:")
print(m1)
print("矩阵 2:")
print(m2)
print("使用 np.dot 得到的矩阵乘法结果:")
print(result_dot)
print("使用 @ 运算符得到的矩阵乘法结果:")
print(result_at)

# 创建一个全零矩阵，用于存储手动推演的结果
# 结果矩阵的行数等于 matrix1 的行数，列数等于 matrix2 的列数
manual_result = np.zeros((m1.shape[0], m2.shape[1]), dtype=np.float32)
np.save("result.npy", manual_result)

result_up = np.load("result.npy")
result_up
a = np.array([1,2,3])
b = np.array([4,5,6])
a + b


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
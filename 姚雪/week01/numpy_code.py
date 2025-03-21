import numpy as np
# 创建ndarray数组
arr = np.array([1,7,8,2,5], float)
arr
a = np.array([(1,2,3), (4,5,6), (7,8,9)])
a
a1 = np.zeros((2,3), dtype=np.float32)
a1
a2 = np.arange(1,6, 0.3)
a2
a3 = np.eye(3)
a3  # one-hot编码
a4 = np.random.random(5)  # 模型运算参数初始值
a4
a5 = np.random.normal(0, 0.1, 5)  # 模型运算参数初始值
a5
a6 = np.array([(1,2), (3,4), (5,6)])
# print(a6)
print(a6[:,1])
a7 = np.array([(1,2), (3,4), (5,6)])
# i,j = a7[0]
for i,j in a7:
    print(i,j)
a = np.array([(1,2,3), (4,5,6), (7,8,9)])
print("ndim:", a.ndim)
print("shape:", a.shape)
print("size", a.size)
print("dtype", a.dtype)
print(3 in a)
a7 = np.arange(1,10)
print(a7)
print(a7.shape)

a7 = a7.reshape(3,3,1)  # 维度大小乘积 == 元素个数
print(a7)

print(a7.shape)  # 高维矩阵，每个维度都有含义
print(a)
a = a.T
print(a)
a = a.flatten()
print(a)
a8 = np.array([(1,2), (3,4), (5,6)])
a8 = a8[:,:,np.newaxis]  # [3,1,2]
a8.shape
a = np.ones((2,2))
b = np.array([(-1,1),(-1,1)])
print(a)
print(b)
print()
print(a+b)
print()
print(a-b)
a.sum()
a.prod()
a = np.array([5,3,1])
print("mean:",a.mean())
print("var:", a.var())
print("std:", a.std())
a = np.array([1.02, 3.8, 4.9])
print("argmax:", a.argmax()) # 最大值的索引
print("argmin:", a.argmin()) # 最小值的索引

print("ceil:", np.ceil(a))  # 向上取整
print("floor:", np.floor(a)) # 向下取整
print("rint:", np.rint(a))  # 四舍五入
a = np.array([16,31,12,28,22,31,48])
a.sort()  # 排序
a
import torch
tensor = torch.arange(1,10, dtype=torch.float32).reshape(3, 3)

print(tensor.dtype)

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

# 外层循环：遍历 matrix1 的每一行
# i 表示结果矩阵的行索引
for i in range(m1.shape[0]):
    # 中层循环：遍历 matrix2 的每一列
    # j 表示结果矩阵的列索引
    for j in range(m2.shape[1]):
        # 初始化当前位置的结果为 0
        manual_result[i, j] = 0
        # 内层循环：计算 matrix1 的第 i 行与 matrix2 的第 j 列对应元素的乘积之和
        # k 表示参与乘法运算的元素索引
        for k in range(m1.shape[1]):
            # 打印当前正在计算的元素
            print(f"{m1[i, k]} * {m2[k, j]} = {m1[i, k] * m2[k, j]}")
            # 将 matrix1 的第 i 行第 k 列元素与 matrix2 的第 k 行第 j 列元素相乘，并累加到结果矩阵的相应位置
            manual_result[i, j] += m1[i, k] * m2[k, j]
        # 打印当前位置计算完成后的结果
        print(f"结果矩阵[{i+1},{j+1}]:{manual_result[i, j]}\n")

print("手动推演结果:")
print(manual_result)

np.save('result.npy',manual_result)

result_np = np.load('result.npy')
result_np

a = np.array([1,2,3])
b = np.array([4,5,6])
a + b

a = np.array([(1,2), (2,2), (3,3), (4,4)])  # shape(4,2)
b = np.array([-1,1])  # shape(2)-> shape(1,2) -> shape(4,2)  [[-1,1],[-1,1],[-1,1],[-1,1]]
a + b




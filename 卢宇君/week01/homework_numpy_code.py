import numpy as np

ary = np.array([1,2,3,4,5], float)
print("ary", ary)

ary1 = np.array([[1,2,3],[4,5,6]], dtype=np.float32)
print("ary1", ary1)

ary2 = np.array(((2,3,4), (5,6,7)))
print("ary2", ary2)

z1 = np.zeros((3, 3), float)
print("z1", z1)

shape = (2,3)
o1 = np.ones(shape)
print("o1", o1)

# [1, 11)
a1 = np.arange(1, 11, 2)
print("a1", a1)

e1 = np.eye(4)
print("e1", e1)

r1 = np.random.random((3, 3))
print("r1", r1)

#[0, 1) 
r2 = np.random.random(5)
print("r2", r2)

#均值loc，标准差scale，的正态分布中抽取size个数据
n1 = np.random.normal(0, 1, 5)
print("n1", n1)
n2 = np.random.normal(loc=0, scale=1, size=(3, 5))
print("n2", n2)

import torch
tn1 = torch.normal(mean=0, std=2, size=(3,3))
print("tn1", tn1)

a4 = np.array([[1,2,3], [2,3,4], [3,4,5]])
print(a4[:,1]) # 2,3,4
print(a4[0,:]) # 1,2,3

# 一行一行遍历
for i,j,k in a4:
    print(i, j, k)

a5 = np.array([[1,2], [2,3], [3,4]])
for i, j in a5:
    print(i, j)

print("\n")
print(a4.shape)
print(a4.ndim)
print(a4.size)
print(a4.dtype)

print(3 in a4)
print([1,2,3] in a4)
print((2,3,4) in a4)

# [1, 10)
a6 = np.arange(1, 10)
print("a6", a6)
print(a6.shape)
a6 = a6.reshape((3,3))
print("a6", a6)
print(a6.shape)
print("a6.T", a6.T)
print("a6.flatten", a6.flatten())

a7 = np.ones((3, 3))
b7 = np.array([[-1, 1, 1],[1, 1, -1], [1, 1, 1]])
print("a7", a7)
print("b7", b7)
print("\n")
print(a7 + b7)
print(a7 - b7)
print("a7.sum()", a7.sum())
#计算 a7成员的乘积
print("a7.prod()", a7.prod())

a = np.array([1, 3, 2])
print(a.mean())
print(a.sum())
print(a.prod())
print(a.var()) #方差
print(a.std()) #标准差

b = np.array([1.2, 3.8, 4.9])
print(b.argmax())
print(b.argmin())
print(b.max())
print(b.min())
print(np.ceil(b))  # 向上取整
print(np.floor(b)) # 向上取整
print(np.rint(b)) # 四舍五入

a8 = np.array([16, 3, 45, 7, 89, 37, 8])
print(a8)
a8.sort()
print(a8)
import torch

at = torch.arange(1, 10, dtype=torch.float32).reshape(3,3)
print("at", at)

#叉乘
r1 = at @ at.T
r2 = at.matmul(at.T)
print("r1", r1)
print("r2", r2)

#对位相乘
r3 = at * at
r4 = at.mul(at)
print("r3", r3)
print("r4", r4)

lat = torch.rand_like(at)
print("lat", lat)
torch.mul(at, at, out=lat)
print("lat", lat)

ra1 = np.random.random((2,3))
ra2 = np.random.random((2,3))

rad1 = ra1 * ra2
print("rad1", rad1)

m1 = np.arange(1,5, dtype=np.float32).reshape(2,2)
m2 = np.arange(5,9, dtype=np.float32).reshape(2,2)
print("m1", m1)
print("m2", m2)

# 矩阵叉乘 类似 torch.matmul()
rd = np.dot(m1, m2)
print("rd", rd)

rd1 = m1 @ m2
print("rd1", rd1)

mr = np.zeros((m1.shape[0], m2.shape[1]), dtype=np.float32)
print("m1.shape[0]", m1.shape[0])
print("m2.shape[1]", m2.shape[1])
print("mr", mr)

# 实现叉乘 m[i][j] 同 m[i, j]
for i in range(m1.shape[0]):
    for j in range(m2.shape[1]):
        mr[i][j] = 0
        for k in range(m1.shape[1]):
            #m1 行不变，m2列不变
            mr[i][j] += m1[i, k] * m2[k, j]
print("手动叉乘结果", mr)
print("叉乘结果", m1 @ m2)

np.save("result.npy", mr)
mr_np = np.load("result.npy")
print("mr_np", mr_np)

a = np.arange(1, 4, dtype=np.int32)
b = np.arange(4, 7, dtype=np.int32)
print("a", a)
print("b", b)
print("a+b", a + b)

a = np.array([[1,2], [2,3], [3,3]])
b = np.array([1,-1])
print("a", a)
print("b", b)
print("a+b", a+b)
print("a-b", a-b)

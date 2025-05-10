import numpy as np

#创建全0数组
a = np.zeros((2, 3))
print(a)
print(type(a))
print(type(a[0]))
print(type(a[0][0]))

#创建全1数组
b = np.ones((2, 3))
print(b)
print(type(b))
print(type(b[0]))
print(type(b[0][0]))

#创建等差数列数组
c = np.arange(1, 10, 2)
print(c)

c = np.arange(1, 10, 0.5)
print(c)

#创建单位矩阵，或者对角矩阵
d = np.eye(3)
print(d)

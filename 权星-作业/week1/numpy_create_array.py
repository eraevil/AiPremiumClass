import numpy as np

#使用列表创建numpy数组
a = [1, 2, 3]
b = np.array(a)
print(a)
print(type(a))
print(b)
print(type(b))
print(type(b[0]))

b = np.array(a, float)
print(b)
print(type(b))
print(type(b[0]))


#创建二维数组
c = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(c)
print(type(c))
print(type(c[0]))
print(type(c[0][0]))

d = np.array(c)
print(d)
print(type(d))
print(type(d[0]))
print(type(d[0][0]))

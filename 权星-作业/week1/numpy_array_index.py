import numpy as np

# 数组的索引和切片
a = np.array([(1, 2), (3, 4), (5, 6)])
print(a)
print(a[0]) # 第0行
print(a[0][1]) # 第0行第1列
print(a[:, 1]) # 第1列
print(a[1, :]) # 第1行

# 一维数组的遍历
b = np.array([1, 2, 3, 4, 5, 6])
for i in b:
    print(i)

# 二维数组的遍历
c = np.array([(1, 2), (3, 4), (5, 6)])
for i in c:
    print(i)

for i, j in c:
    print(i)
    print(j)

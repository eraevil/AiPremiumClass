import  numpy as np

a = np.ones((2, 3))
b = np.array([[1, 2, 3], [4, 5, 6]])
print("a:\n", a)
print("b:\n", b)
print("a+b:\n", a+b) # 逐元素相加，维度必须相同
print("a-b:\n", a-b) # 逐元素相减，维度必须相同
print("a*b:\n", a*b) # 逐元素相乘，维度必须相同
print("b*a:\n", b*a) # 逐元素相乘，维度必须相同
print("a*b:\n", np.multiply(a, b)) # 逐元素相乘，维度必须相同

# print("a·b:\n", a@b) # 矩阵点乘，a的行数必须等于b的列数
# print("b·a:\n", b@a) # 矩阵点乘，b的行数必须等于a的列数
print("a/b:\n", a/b) # 逐元素相除，维度必须相同

c = np.array([[1, 2], [3, 4]])
d = np.array([[1], [2]])
# 矩阵1最后一个维度 == 矩阵2倒数第二个维度
print("c·d:\n", c@d) # 矩阵点乘，c的行数必须等于d的列数
print("c.shape:", c.shape)
print("d.shape:", d.shape)
# print("d·c:\n", d@c) # 矩阵点乘，d的行数必须等于c的列数
print("c·d:\n", np.dot(c, d)) # 矩阵点乘，c的行数必须等于d的列数

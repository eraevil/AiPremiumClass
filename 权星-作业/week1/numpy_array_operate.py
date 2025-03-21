import numpy as np

# 检测数值是否在数组中
a = np.array([1, 2, 3, 4, 5])
print("array:", a)

print("2 in array:", 2 in a)

b = np.array([[1, 2, 3], [4, 5, 6]])
print("array:", b)

print("2 in array:", 2 in b)
print("2 in array[0]:", 2 in b[0])
print("2 in array[1]:", 2 in b[1])

# print("(1,2) in array[1][1]:", (1,2) in b)
# 报错 operands could not be broadcast together with shapes (2,3) (2,) 
# 元组的维度必须相同
print("(1,2,3) in array:", (1,2,3) in b)



# 数组的重新排列，排列后会返回一个新的数组
c = np.zeros((2, 3, 4))
print("array:\n", c)

print("array.reshape(24):\n", c.reshape(24))
print("array.reshape(24):\n", c.reshape(24).shape)

print("array.reshape(3, 8):\n", c.reshape(3, 8))
print("array.reshape(3, 8):\n", c.reshape(3, 8).shape)

print("array.reshape(2, 2, 6):\n", c.reshape(2, 2, 6))

# print("array.reshape(2, 2, 5):\n", c.reshape(2, 2, 5))
# 报错 cannot reshape array of size 24 into shape (2,2,5)
# 数组的个数转换前后必须相同


# 数组的转置，转置后会返回一个新的数组
d = np.array([[1, 2, 3], [4, 5, 6]])
print("array:\n", d)
print("array.transpose:\n", d.transpose())
print("array.T:\n", d.T)

# 把多维数组转换为一维数组
e = np.array([[1, 2, 3], [4, 5, 6]])
print("array:\n", e)
print("array.flatten:\n", e.flatten())

# 数组增加维度
f = np.array([(1, 2), (3, 4), (5, 6)])
print("array:\n", f)
print("array.shape:\n", f.shape)

g = f[:, np.newaxis, :]
print("array:\n", g)
print("array.shape:\n", g.shape)

h = f[:, :, np.newaxis]
print("array:\n", h)
print("array.shape:\n", h.shape) 

import numpy as np

# 创建一个数组
a = np.array([(1, 2, 3), (4, 5, 6)])

# 数组的维度
print("ndim:", a.ndim)

# 数组的形状
print("shape:", a.shape)

# 数组的元素总数
print("size:", a.size)

# 数组的元素类型
print("dtype:", a.dtype)

# 数组的元素大小（以字节为单位）
print("itemsize:", a.itemsize)

# 数组的内存地址
print("data:", a.data)

import numpy as np
import torch


###numpy.array(object, dtype=None, copy=True, order='K', subok=False, ndmin=0)###
# 用于创建一个 NumPy 数组
# 参数说明
# object: 要转换为数组的对象，可以是列表、元组、嵌套序列等。
# dtype: 可选参数，指定数组的数据类型（如 int, float, str 等）。如果未指定，NumPy 会根据输入数据自动推断数据类型。
# copy: 可选参数，默认为 True。如果为 False，则如果可能的话，返回输入对象的视图，而不是复制数据。
# order: 可选参数，指定多维数组的存储顺序。可以是：
    # 'C': 按行优先（C-style）
    # 'F': 按列优先（Fortran-style）
    # 'K': 保持输入的顺序
# subok: 可选参数，默认为 False。如果为 True，则返回子类数组（如果输入是子类数组）。
# ndmin: 可选参数，指定返回数组的最小维数。如果输入数组的维数小于 ndmin，则会在前面添加维度。
# 返回值
# 返回一个 NumPy 数组。

# 从列表创建一维数组
array_1d = np.array([1, 2, 3])
print("一维数组:", array_1d)

# 从嵌套列表创建二维数组
array_2d = np.array([[1, 2, 3], [4, 5, 6]])
print("二维数组：\n", array_2d)

# 指定数据类型为浮点数
array_float = np.array([1, 2, 3], dtype=float)
print("浮点数数组：", array_float)


###numpy.zeros(shape, dtype=None, order='C')###
# 用于创建一个指定形状的数组，并用零填充该数组
# 参数说明
# shape: 必需参数，指定数组的形状。可以是一个整数（表示一维数组的长度）或一个元组（表示多维数组的形状）。
# dtype: 可选参数，指定数组的数据类型（如 int, float, bool 等）。如果未指定，NumPy 会默认使用 float 类型。
# order: 可选参数，指定多维数组的存储顺序。可以是：
    # 'C': 按行优先（C-style）
    # 'F': 按列优先（Fortran-style）
# 返回值
# 返回一个用零填充的 NumPy 数组。


# 创建一个长度为 5 的一维数组
array_1d = np.zeros(5)
print("一维数组：", array_1d)

# 创建一个 3x4 的二维数组
array_2d = np.zeros((3, 4))
print("二维数组：\n", array_2d)

# 创建一个 2x3 的整数类型数组
array_int = np.zeros((2, 3), dtype=int)
print("整数类型的二维数组：\n", array_int)

# 创建一个 2x3x4 的三维数组
array_3d = np.zeros((2, 3, 4))
print("三维数组形状：", array_3d.shape)
print("三维数组内容：\n", array_3d)


###numpy.arange([start, ]stop[, step, ], dtype=None)
# 用于创建一个包含等间隔值的NumPy数组
# 参数说明
# start: 可选参数，表示序列的起始值，默认为 0。
# stop: 必需参数，表示序列的结束值（不包括该值）。
# step: 可选参数，表示两个值之间的间隔，默认为 1。可以是正数或负数。
# dtype: 可选参数，指定返回数组的数据类型。如果未指定，NumPy 会根据输入数据自动推断数据类型。
# 返回值
# 返回一个包含等间隔值的 NumPy 数组。

# 创建一个从 0 到 9 的一维数组
array_1d = np.arange(10)
print("从 0 到 9 的数组：", array_1d)

# 创建一个从 5 到 15 的一维数组
array_range = np.arange(5, 15)
print("从 5 到 14 的数组：", array_range)

# 创建一个从 0 到 20 的数组，步长为 2
array_step = np.arange(0, 20, 2)
print("从 0 到 18 的数组，步长为 2：", array_step)

# 创建一个从 0 到 5 的数组，指定数据类型为浮点数
array_float = np.arange(0, 5, 0.5, dtype=float)
print("从 0 到 5 的浮点数数组：", array_float)


### numpy.eye(N, M=None, k=0, dtype=<class 'float'>) ###
# 用于创建一个单位矩阵（对角线元素为 1，其余元素为 0）
# 参数说明
#     N: 必需参数，表示矩阵的行数。
#     M: 可选参数，表示矩阵的列数。如果未指定，默认为 N，即生成一个方阵。
#     k: 可选参数，表示对角线的偏移量。默认值为 0，表示主对角线。正值表示上方的对角线，负值表示下方的对角线。
#     dtype: 可选参数，指定返回数组的数据类型。默认情况下为 float。
# 返回值
# 返回一个形状为 (N, M) 的数组，其中对角线上的元素为 1，其余元素为 0。


# 创建一个 3x3 的单位矩阵
identity_matrix = np.eye(3)
print("3x3 单位矩阵：\n", identity_matrix)

# 创建一个 4x2 的矩阵
matrix_4x2 = np.eye(4,2)
print("4x2 矩阵：\n", matrix_4x2)

# 创建一个 5x5 的单位矩阵，偏移 1
offset_matrix = np.eye(5, k=1)
print("偏移 1 的单位矩阵：\n", offset_matrix)


### numpy.random 模块的一个命名空间。该模块提供了多种生成随机数和随机样本的函数 ###
# 用于生成一个包含 5 个随机数的数组，这些随机数是从均匀分布中抽取的，范围在 [0.0, 1.0) 之间（包括 0.0，但不包括 1.0）
random5 = np.random.random(5)
print(random5)

## np.random.normal(loc, scale, size) ##
# 用于生成一个包含5个随机数的数组，这些随机数是从正态分布（高斯分布）中抽取的
# 参数:
# loc: 正态分布的均值。
# scale: 正态分布的标准差。
# size: 指定生成随机数的数量或形状。
# 返回值: 返回一个包含指定数量的随机数的数组，这些随机数服从以 loc 为均值、scale 为标准差的正态分布。

# 生成一个包含 5 个随机数的数组，均值为 0，标准差为 0.1
random_numbers = np.random.normal(0, 0.1, 5)
print("生成的随机数数组：", random_numbers)

# 创建一个4x2的NumPy数组
a2 = np.array([(1, 2), (3, 4), (5, 6), (7, 8)])
# 提取数组中每一行的第二个元素
print(a2[:, 1])

a = np.array([(1,2,3), (4,5,6), (7,8,9)])
# 输出数组的维度
print("ndim:", a.ndim)
# 输出数组的形状
print("shape:", a.shape)
# 输出数组的大小
print("size:", a.size)
# 输出数组的数据类型
print("dtype:", a.dtype)

# 创建一个一维数组，包含从1到9的整数（不包括 10）
a7 = np.arange(1,10)
print(a7)
# 将一维数组的形状调整为三维数组，形状为 (3, 3, 1)。
# 这意味着新数组将有 3 个 "层"，每层包含 3 行 1 列的数据。
a7 = a7.reshape(3,3,1)
print(a7)
# a7.T是用来获取数组 a7 的转置（transpose）
# 转置的定义：
# 对于一个二维数组，转置操作会将第 i 行变为第 i 列。
# 对于三维数组，转置操作会在更高维度上进行，具体的转置方式可以通过 np.transpose() 函数或 .T 属性来实现。
# 示例：
# 假设 a7 是一个三维数组，形状为 (3, 3, 1)，转置操作会根据数组的维度进行调整。对于三维数组，转置的结果通常会改变最后两个维度的顺序。
print(a7.T)

# flatten函数
# 其展平为一维数组
a = a.flatten()
print(a)

# np.newaxis特定位置添加新轴（维度）将其转换为三维数组
a8 = np.array([(1,2), (3,4), (5,6)])
print(a8)
print(a8.shape)
# 在最后一个维度添加一个新轴
a8 = a8[:,:,np.newaxis]
print(a8)
print(a8.shape)

# 创建一个2x2的全1数组
a = np.ones((2,2)),
b = np.array([(-1,1),(-1,1)])
print(a)
print(b)
print()
print(a+b)
print(a-b)

# 计算该数组的均值、方差和标准差
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

# a.sort()  #排序
a = np.array([16,31,12,28,22,31,48])
a.sort()  #排序
print(a)

# 创建一个3x3 的张量，元素为1 到9，数据类型为float32
tensor = torch.arange(1,10, dtype=torch.float32).reshape(3, 3)
print(tensor.dtype)
# 计算两个张量之间矩阵乘法的几种方式。 y1, y2, y3 最后的值是一样的 dot
# 矩阵乘法：使用 @ 运算符
y1 = tensor @ tensor.T
# 矩阵乘法：使用 matmul 方法
y2 = tensor.matmul(tensor.T)
# 创建一个与 tensor 形状相同的随机张量
y3 = torch.rand_like(tensor)
# 矩阵乘法：将结果存储到 y3 中
torch.matmul(tensor, tensor.T, out=y3)
# 计算张量逐元素相乘的几种方法。 z1, z2, z3 最后的值是一样的
# 逐元素乘法：使用 * 运算符
z1 = tensor * tensor
# 逐元素乘法：使用 mul 方法
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)


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
# # 外层循环：遍历 matrix1 的每一行
# # i 表示结果矩阵的行索引
# for i in range(m1.shape[0]):
#     # 中层循环：遍历 matrix2 的每一列
#     # j 表示结果矩阵的列索引
#     for j in range(m2.shape[1]):
#         # 初始化当前位置的结果为 0
#         manual_result[i, j] = 0
#         # 内层循环：计算 matrix1 的第 i 行与 matrix2 的第 j 列对应元素的乘积之和
#         # k 表示参与乘法运算的元素索引
#         for k in range(m1.shape[1]):
#             # 打印当前正在计算的元素
#             print(f"{m1[i, k]} * {m2[k, j]} = {m1[i, k] * m2[k, j]}")
#             # 将 matrix1 的第 i 行第 k 列元素与 matrix2 的第 k 行第 j 列元素相乘，并累加到结果矩阵的相应位置
#             manual_result[i, j] += m1[i, k] * m2[k, j]
#         # 打印当前位置计算完成后的结果
#         print(f"结果矩阵[{i+1},{j+1}]:{manual_result[i, j]}\\n")
# print("手动推演结果:")
# print(manual_result)

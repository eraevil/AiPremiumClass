import numpy as np

# 创建Ndarray数组

# 从列表创建
np_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=int)
print("从列表创建:", np_array)
# ndarray转换为列表
np_list = np_array.tolist()
print("ndarray转换为列表:", np_list)

# 从元组创建
np_array = np.array((1, 2, 3, 4, 5))
print("从元组创建:", np_array)


# 使用内置函数
# 全零数组
np_array = np.zeros((2, 3), dtype=np.float32)
print("全零数组:", np_array)

# 全一数组
np_array = np.ones((2, 3), dtype=np.int8)
print("全一数组:", np_array)

# one-hot编码
np_array = np.eye(3, dtype=np.int8)
print("one-hot编码:", np_array)



# 等间隔数组
np_array = np.arange(1, 10, 2)
print("生成从1到10的序列，步长为2：:", np_array)

np_array = np.linspace(1, 10, 5)
print("生成5个在1到10之间均匀分布的数：:", np_array)



# 一维随机数组
np_array = np.random.rand(2)
print("np.random.rand：0~1之间随机数:", np_array)
np_array = np.random.random(2)
print("np.random.random：0~1之间随机数:", np_array)

# 二维(0,1]随机数组
np_array = np.random.rand(2, 3)
print("np.random.rand：0~1之间随机数组 组成2*3的矩阵:", np_array)
np_array = np.random.random((2, 3))
print("np.random.random：0~1之间随机数 组成2*3的矩阵:", np_array)

# 二维指定范围随机数组
np_array = np.random.randint(1, 10, size=(2, 3))
print("1-10之间随机数 组成2*3的矩阵:", np_array)

# 随机样本
np_array = np.random.randn(2, 3)
print("均值为 0、标准差为 1 的随机样本:", np_array)
np_array = np.random.normal(2, 4, size=(2, 3))
print("均值、标准差指定 的随机样本:", np_array)


# 基本操作

# 数组的形状和维度
np_array = np.array([[[2, 4],[3, 6],[5, 7]], [[1, 3],[5, 7],[6,8]]])
print("数组的形状：", np_array.shape)
print("数组的维度：", np_array.ndim)
print("数组的元素个数：", np_array.size)
print("数组的元素类型：", np_array.dtype)

# 数组的索引和切片
print("数组的第一个元素：", np_array[0, 0, 0])
print("数组的第一行：", np_array[0, 0, :])
print("数组的最后一维元素：", np_array[..., -1])

# 判定元素是否在数组中
print("元素 5 是否在数组中：", 5 in np_array)
print("元素 [2,4] 是否在数组中：", (2,4) in np_array)
# print("元素 [2,4,8] 是否在数组中：", (2,4,8) in np_array)

# 遍历
for i,j,k in np_array:
    print(i,j,k)
    
# 基础运算
np_array1 = np.array([[1, 2], [3, 4]])
np_array2 = np.array([[5, 6], [7, 8]])
print("数组相加：", np_array1 + np_array2)
print("数组相减：", np_array1 - np_array2)
print("数组相乘-元素级乘法-对应位置相乘：", np_array1 * np_array2)
print("数组相乘-矩阵乘法-内积运算1：", np_array1 @ np_array2)
print("数组相乘-矩阵乘法-内积运算2：", np.dot(np_array1, np_array2))
print("数组相除：", np_array1 / np_array2)
print("数组的转置1：", np_array1.T)
print("数组的转置2：", np_array1.transpose())

np_array3 = np.array([1.5, 2.6, 3.4])
print("求和：", np.sum(np_array3))
print("乘积：", np.prod(np_array3))
print("均值：", np.mean(np_array3))
print("标准差：", np.std(np_array3))
print("方差：", np.var(np_array3))
print("最大值：", np.max(np_array3))
print("最小值：", np.min(np_array3))
print("求和：", np.sum(np_array3))
print("最大值下标索引：", np.argmax(np_array3))
print("最小值下标索引：", np.argmin(np_array3))
print("向上取整：", np.ceil(np_array3))
print("向下取整：", np.floor(np_array3))
print("四舍五入1：", np.round(np_array3))
print("四舍五入2：", np.rint(np_array3))
print("取绝对值：", np.abs(np_array3))
print("排序:", np.sort(np_array3))

# 改变数组维度
np_array = np.arange(1,13)
print(np_array)
print(np_array.shape)
# 维度大小乘积 == 元素个数
np_array = np_array.reshape(3,4)
print(np_array)


# 动态增加数组维度
np_array = np.array([[1, 2, 3], [4, 5, 6]])
print("增加维度前：", np_array)
print("增加维度前-维度", np_array.shape)
np_array = np_array[np.newaxis, :]
print("增加维度后1：", np_array)
print("增加维度后1-维度", np_array.shape)
np_array = np_array[:, np.newaxis, :]
print("增加维度后2：", np_array)
print("增加维度后2-维度", np_array.shape)
np_array = np_array[:, :, np.newaxis]
print("增加维度后3：", np_array)
print("增加维度后3-维度", np_array.shape)


# 多维数组转一维数组
np_array = np.array([[1, 2, 3], [4, 5, 6]])
print(np_array.flatten())


# 广播机制-允许不同维度的数组进行运算-自动
# 自动触发广播机制--最后一维的维度要相同
np_array = np.array([2, 3, 4])  # shape(3)
b = 2  # shape(1) -> shape(3) -> [2,2,2]
# 对应位置相乘 or 相加
print(np_array * b)
print(np_array + b)


# 文件读写
np_array = np.array([(1,2), (3,4), (5,6)])
np.savetxt('numpy_array.txt', np_array)
np.save('numpy_array.npy', np_array)

np_array1 = np.loadtxt('numpy_array.txt')
print(np_array1)
np_array2 = np.load('numpy_array.npy')
print(np_array2)
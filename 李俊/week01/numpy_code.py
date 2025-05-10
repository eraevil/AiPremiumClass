import numpy as np

#一维数组创建
arr1 = np.array([4,5,6,7], float)
print(arr1)

#二维数组创建
arr2 = np.array([[5,6,7],[9,3,5]])
print(arr2)
#全零数组（一维，二维）
arr3 = np.zeros(3, dtype=np.float32)
arr4=np.zeros((5,5),np.int64)
print(arr3,arr4)

#步进数组
arr5=np.arange(1,10,0.3)
print(arr5)
#对角为1数组
arr6=np.eye(3)
print(arr6)

#0-1数组(均匀分布)
arr7=np.random.random((3,5))
print(arr7)
#服从正态分布数组(均值1，方差1）
arr8=np.random.normal(1,1,(2,5))
print(arr8)
#数组切片
arr9=np.array([[2,4,5],[3,7,9]])
print(arr9[:],arr9[:,-1],arr9[-1:,1])
#数组遍历
arr10=np.array([[2,4,5],[3,7,9]])
for i,j,q in arr10:
    print(i,j,q)
#数组参数：维度，形状，元素个数，数据类型
arr11=np.array([[4,6,8],[3,5,6]])
print(arr11.ndim)
print(arr11.shape)
print(arr11.size)
print(arr11.dtype)
#元素是否在数组中
print(11 in arr11)
#reshape 调整形状
arr12 = np.arange(1,5,0.1)
print(arr12.shape)
arr13 = arr12.reshape(5,8)
print(arr13)
#数组转置
arr14 = np.random.random((5,8))
print(arr14.shape)
print(arr14.T)
#数组展平
arr15 = arr14.flatten()
print(arr15)
#维度增加
arr16 = np.random.random((3,5))
arr17=arr16[:,:,np.newaxis]
print(arr16)
print(arr17)
print(arr17.shape)
#数组加法
arr18=np.ones((2,2))
arr19=np.array([[1,1],[2,2]])
print(arr18+arr19)
#数组求和,求积
arr20 = arr19.sum()
print(arr20)
arr21=arr19.prod()
print(arr21)
#均值，方差，标准差
print(np.mean(arr19))
print(np.var(arr19))
print(np.std(arr19))
#最大，最小值，向下，向上取整，四舍五入
arr20 = np.array([1.02, 3.8, 4.9])
print("argmax:", arr20.argmax()) # 最大值的索引
print("argmin:", arr20.argmin()) # 最小值的索引

print("ceil:", np.ceil(arr20))  # 向上取整
print("floor:", np.floor(arr20)) # 向下取整
print("rint:", np.rint(arr20))  # 四舍五入
#排序
a = np.array([16,32,12,24,22,52,48])
a.sort()  # 排序
print(a)
a9 = np.array([[3,4,5],[1,1,1]])
b9 = np.array([[4,5,6],[7,8,9]])

print(a9*b9)


# 定义两个简单的矩阵
m1 = np.array([[6, 9], [5, 4]], dtype=np.float32)
m2 = np.array([[1, 3], [3, 8]], dtype=np.float32)
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

#混淆点：矩阵的乘法 np.dot() 满足第一个的列 = 第二个的行相乘  @    逐个元素相乘  *
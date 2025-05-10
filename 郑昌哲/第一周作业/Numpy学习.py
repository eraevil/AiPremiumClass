###{一}、Numpy基础

##一、numpy制作数组
#1.设置一维数组
import numpy as np
a = [1,2,3,4,5]
b = np.array(a)
print(b)
#输出值 [1 2 3 4 5]

#2.设置数组字符格式，默认格式int
a = np.array([1,2,3,4],float)
print(a)
#输出值[1. 2. 3. 4.]

#3.创建多维数组
a = np.array([(1,2),(3,4),(5,6),(7,8)])
print(a)
#输出值
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]

##二、Numpy特殊数组创建
#1.使⽤ dtype = int 指定元素类型：
a = np.zeros((1,2),dtype=int)
print(a)
#输出值  [[0 0]]

#2.创建全1的数组
a = np.ones((2,2),dtype=int)
print(a)
#输出值
# [[1 1]
#  [1 1]]

#3.创建等差数列
a = np.arange(1,7,0.6)
print(a)
#输出值  [1.  1.6 2.2 2.8 3.4 4.  4.6 5.2 5.8 6.4]

#4.创建单位矩阵。单位矩阵：从左上角到右下角都是1，其他为0。任何一个矩阵*单位矩阵=矩阵本身。
a = np.eye(2,dtype=int)
print(a)
# 输出值
# [[1 0]
#  [0 1]]

#5.生成指定长度，在[0,1)之间平均分步的数组
a = np.random.random(5)
print(a)
#输出值 [0.71014421 0.95000735 0.50199042 0.80930863 0.6148871 ]

#6.⽣成指定⻓度，符合正态分布的随机数组，指定其均值 mu 为 0，标准差 sigma 为 0.1：
mu , sigma = 0, 1
a = np.random.normal(mu, sigma, 3)
print(a)
#输出值 [ 0.42849527  1.78254248 -2.39515235]

##三、Numpy数组的创建
#1.和 list 的访问形式基本⼀致，⽀持切⽚操作，我们可以切⽚每⼀个维度，索引每⼀个维度：
a = np.array([(1,2),(3,4),(5,6)])
print(a[1])
#输出值 [3 4]

#2.第一个以后的数组
print(a[1:])
# 输出值
# [[3 4]
#  [5 6]]

#3.取第一列的数组
print(a[:,:1])
# 输出值
# [[1]
#  [3]
#  [5]]

#4.取值
print(a[1,0])
#输出值 3

##四、数组的遍历
#1.一维数组的遍历
a = np.array([1,2])
for i in a:
    print(i)
#输出值
# 1
# 2

#2.多维数组的遍历
a = np.array([(1,2),(3,4),(5,6)])
for i in a:
    print(i)
#输出值
# [1 2]
# [3 4]
# [5 6]

##五、Numpy数组常用属性
# ⽐较常⽤的属性有：
# 1.ndarray.ndim : 数组的维度（数组轴的个数），等于秩
# 2.ndarray.shape : 数组的⼤⼩。为⼀个表⽰数组在每个维度上⼤⼩的整数元组。例如⼆维
# 3.数组中，表⽰数组的’ ⾏数’ 和’ 列数”
# 4.ndarray.size : 数组元素的总个数，等于 shape 属性中元组元素的乘积
# 5.ndarray.dtype : 表⽰数组中元素类型的对象
a = np.array([(1,2,3), (4,5,6), (7,8,9)])
print("ndim:", a.ndim)       #数组维度
print("shape:", a.shape)     #数组大小
print("size", a.size)        #元素总个数
print("dtype", a.dtype)      #数组元素的字符类型
#输出值
# ndim: 2
# shape: (3, 3)
# size 9
# dtype int64

##六、Numpy基本操作
#1.检测数值是否在数组中
a = np.array([(1,2),(3,4),(5,6)])
print(1 in a)
print(7 in a)
# 输出值
# True
# False

#2.数组重排列，例如将⼀个 2 维数组转变为 1 维（元素数⼀定要保持不变）
a = np.zeros([2,3])
print(a)
#输出值（显示2维数组）
# [[0. 0. 0.]
#  [0. 0. 0.]]

#多维数组重排列
print(a.reshape(6))
#输出值
# [0. 0. 0. 0. 0. 0.]

#3.转置 (可以直接.T）
a = np.array([(1,2,3),(4,5,6),(9,8,7)])
print(a)
# 输出值
# [[1 2 3]
#  [4 5 6]
#  [9 8 7]]

#转置T
print(a.transpose())
# 输出值
# [[1 4 9]
#  [2 5 8]
#  [3 6 7]]

#4.把多维数组转换为⼀维数组，注意每个元组的⻓度是相同的
a = np.array([(1,2),(3,4),(5,6)])
print(a.flatten())
#输出值 [1 2 3 4 5 6]

#5.增加维度
a = np.array([1,2,3])
a.reshape
a = a[:,np.newaxis]
print(a)
# 输出值
# [[1]
#  [2]
#  [3]]
print(a.shape)
#输出值 (3, 1)

##七、Numpy数组的数学操作
#1.加减乘除
#星乘表⽰矩阵内各对应位置相乘，点乘表⽰求矩阵内积，⼆维数组称为矩阵积（mastrix product）：
a = np.ones((2,2))
b = np.array([(-1,1),(-1,1)])
print("显示a:\n",a)
print("显示b:\n",b)
print("加法，a+b:\n",a+b)
print("减法，a-b:\n",a-b)
print("乘法，a*b:\n",a*b)
print("除法，a/b:\n",a/b)
#输出值
# 显示a:
#  [[1. 1.]
#  [1. 1.]]
# 显示b:
#  [[-1  1]
#  [-1  1]]
# 加法，a+b:
#  [[0. 2.]
#  [0. 2.]]
# 减法，a-b:
#  [[2. 0.]
#  [2. 0.]]
# 乘法，a*b:
#  [[-1.  1.]
#  [-1.  1.]]
# 除法，a/b:
#  [[-1.  1.]
#  [-1.  1.]]

#2.元素值运算
a = np.array([1,3,1])
print("所有元素的和",a.sum())
print("所有元素的乘积",a.prod())
print("平均数",a.mean())
print("方差",a.std())
print("标准差",a.var())
print("最大值",a.max())
print("最小值",a.min())
#输出值
# 所有元素的和 5
# 所有元素的乘积 3
# 平均数 1.6666666666666667
# 方差 0.9428090415820634
# 标准差 0.888888888888889
# 最大值 3
# 最小值 1

#下标运算
a = np.array([6.2,1.2, 3.8, 4.9])
print("找出最大索引", a.argmax())
print("找出最小索引:", a.argmin())
print("数组元素向上取整:", np.ceil(a))
print("数组元素向上取整:", np.floor(a))
print("数组元素四舍五入取整:", np.rint(a))
print("数组进行排序", np.sort(a))
#输出值
# 找出最大索引 0
# 找出最小索引: 1
# 数组元素向上取整: [7. 2. 4. 5.]
# 数组元素向上取整: [6. 1. 3. 4.]
# 数组元素四舍五入取整: [6. 1. 4. 5.]
# 数组进行排序 [1.2 3.8 4.9 6.2]

##8.Numpy线性代数
#1.矩阵乘法
#（1）两个⼀维的数组，计算的是这两个数组对应下标元素的乘积和(数学上称之为内积)；
#（2）对于⼆维数组，计算的是两个数组的矩阵乘积；
#（3）对于多维数组，它的通⽤计算公式如下，即结果数组中的每个元素都是：数组 a 的最后⼀维上的所有元素与数组 b 的倒数第⼆位上的所有元素的乘积和
# ：dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])

# 定义两个简单的矩阵
m1 = np.array([[1, 2], [7, 8] ], dtype=np.float32)
m2 = np.array([[5, 6], [3, 4] ], dtype=np.float32)
# 使⽤ np.dot 进⾏矩阵乘法
result_dot = np.dot(m1, m2)
# 使⽤ @ 运算符进⾏矩阵乘法
result_at = m1 @ m2
print("矩阵 1:")
print(m1)
print("矩阵 2:")
print(m2)
print("使⽤ np.dot 得到的矩阵乘法结果:")
print(result_dot)
print("使⽤ @ 运算符得到的矩阵乘法结果:")
print(result_at)
# 创建⼀个全零矩阵，⽤于存储⼿动推演的结果
# 结果矩阵的⾏数等于 m1 的⾏数，列数等于 m2 的列数
manual_result = np.zeros((m1.shape[0], m2.shape[1]), dtype=np.float32)
# 外层循环：遍历 matrix1 的每⼀⾏
# i 表⽰结果矩阵的⾏索引
for i in range(m1.shape[0]):
 # 中层循环：遍历 m2 的每⼀列
 # j 表⽰结果矩阵的列索引
 for j in range(m2.shape[1]):
 # 初始化当前位置的结果为 0
    manual_result[i, j] = 0
 # 内层循环：计算 m1 的第 i ⾏与 m2 的第 j 列对应元素的乘积之和
 # k 表⽰参与乘法运算的元素索引
 for k in range(m1.shape[1]):
    # 打印当前正在计算的元素
    print(f"{m1[i, k]} * {m2[k, j]} = {m1[i, k] * m2[k, j]}")
    # 将 m1 的第 i ⾏第 k 列元素与 m2 的第 k ⾏第 j 列元素相乘，并累加
    manual_result[i, j] += m1[i, k] * m2[k, j]
 # 打印当前位置计算完成后的结果
 print(f"结果矩阵[{i+1},{j+1}]:{manual_result[i, j]}\n")
print("⼿动推演结果:")
print(manual_result)

# eig:特征值、特征向量
# inv：逆
# QR：QR分解
# svd:奇异值分解
# solve:解线性方程：Ax = b
# Istsq:计算 Ax = b 的最小二乘解

#9.文件操作
#二进制文件：np.save、np.load
#文本文件：np.loadxt、np.savetxt

#10.Numpy广播机制
a = np.array([1,2,3])
b = np.array([4,5,6])
print("广播机制",a + b)
#输出值 广播机制 [5 7 9]

#当运算中的 2 个数组的形状不同时，numpy 将⾃动触发⼴播机制
a = np.array([(1,2), (2,2), (3,3), (4,4)]) # shape(4,2)
b = np.array([-1,1]) # shape(2)-> shape(1,2) -> shape(4,2) [ -1,1],[-1,1],[-1,1],[-1
print(a + b)
# 输出值
# [[0 3]
#  [1 3]
#  [2 4]
#  [3 5]]

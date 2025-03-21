# %%
# 导入numpy包
import numpy as np

# %%
 # NumPy基础数组创建
a = [1,2,3]
b = np.array(a)
b

# %%
# 创建一个浮点类型的数组
a = np.array([1,2,3],float)
a

# %%
# 创建一个多维数组
a = np.array([(1,2,3),(4,5,6),(7,8,9)])
a

# %%
# 创建一个浮点类型所有值为0的数组
a = np.zeros((2,3),dtype = float)
a

# %%
# 创建一个int类型所有值全为1的数组
a = np.ones((3,3,3),dtype = int)
a

# %%


# %%
# 创建等差数列，从 1 开始，5 结束，2 为差的等差数列，最后⼀项⼀定⼩于 5
a  = np.arange(1,5,2)
a

# %%
# 创建单位矩阵：（从左上⻆到右下⻆的对⻆线（称为主对⻆线）上的元素均为 1）
a = np.eye(3)
a

# %%
# ⽣成个数为4，在 [0,1) 之间平均分布的随机数组
np.random.random(5)

# %%
# ⽣成指定⻓度，符合正态分布的随机数组，指定其均值为 0，标准差为 0.1：
mu,sigma = 0, 0.1
np.random.normal(mu, sigma, 5)


# %%
# NumPy数组的访问
a = np.array([(1,2),(3,4),(5,6)])
a[2]

# %%
# 展示下标为1和1以后的数据
a[1:]

# %%
# 展示所有元素中下标为0的数据
a[:,:1]

# %%
# 查找指定的数组元素
a[1][1]

# %%
# 一维数组的遍历
a = np.array([1,2,3])
for i in a:
    print(i)       

# %%
# 多维数组的遍历
a = np.array([(1,2),(3,4),(5,6)])
for i,j in a:
    print(i,j)

# %%
# NumPy 数组的常⽤属性
# ndarray.ndim : 数组的维度（数组轴的个数），等于秩
# ndarray.shape : 数组的⼤⼩。为⼀个表⽰数组在每个维度上⼤⼩的整数元组。例如⼆维数组中，表⽰数组的’ ⾏数’ 和’ 列数”
# ndarray.size : 数组元素的总个数，等于 shape 属性中元组元素的乘积
# ndarray.dtype : 表⽰数组中元素类型的对象
a = np.array([(1,2,3), (4,5,6), (7,8,9)])
print("ndim:", a.ndim)
print("shape:", a.shape)
print("size", a.size)
print("dtype", a.dtype)



# %%
# 检测数值是否在数组中
a = np.array([(1,3),(5,8)])
print(2 in a)
print(5 in a)


# %%
# 构建多维数组
a = np.zeros([2,3,4],dtype=int)
a

# %%
# 数组的重排列，例如将⼀个 3 维数组转变为 1 维（元素数⼀定要保持不变）
a.reshape(24)

# %%
# 转置
a = np.array([(1,2,3),(4,5,6),(7,8,9)])
a.transpose()

# %%
# 转置
a.T

# %%
# 把多维数组转换为⼀维数组，注意每个元组的⻓度是相同的
a = np.array([(1,2), (3,4), (5,6)])
a.flatten()

# %%
# 增加维度
a = np.array([1,2,3])
a.shape

# %%
a = a[ :, np.newaxis]
a

# %%
a.shape

# %%

a = np.ones((2,2))
b = np.array([(-1,1),(-1,1)])
print(a)
print(b)

# %%
# 加减操作
a + b

# %%
# 求和
a = np.array([1,2,1])
a.sum()

# %%
# 求积
a.prod()

# %%
# 平均数，⽅差，标准差，最⼤值，最⼩值
a = np.array([5,3,1])
print("mean:",a.mean())
print("var:", a.var())
print("std:", a.std())
print("max:", a.max())
print("min:", a.min())

# %%
# 最⼤与最⼩值对应的索引值：argmax,argmin:
# 取元素值上限，下限，四舍五⼊：ceil, floor, rint
a = np.array([1.2, 3.8, 4.9])
print("argmax:", a.argmax())
print("argmin:", a.argmin())
print("ceil:", np.ceil(a))
print("floor:", np.floor(a))
print("rint:", np.rint(a))


# %%
# 排序
a = np.array([16,31,12,28,22,31,48])
a.sort()
a

# %%
import numpy as np
# 定义两个简单的矩阵
m1 = np.array([[1, 2], [3, 4]] , dtype=np.float32)
m2 = np.array([[5, 6], [7, 8]], dtype=np.float32)
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
# 结果矩阵的⾏数等于 matrix1 的⾏数，列数等于 matrix2 的列数
manual_result = np.zeros((m1.shape[0], m2.shape[1]), dtype=np.float32)
# 外层循环：遍历 matrix1 的每⼀⾏
# i 表⽰结果矩阵的⾏索引
for i in range(m1.shape[0]):
 # 中层循环：遍历 matrix2 的每⼀列
 # j 表⽰结果矩阵的列索引
 for j in range(m2.shape[1]):
 # 初始化当前位置的结果为 0
  manual_result[i, j] = 0
 # 内层循环：计算 matrix1 的第 i ⾏与 matrix2 的第 j 列对应元素的乘积之和
 # k 表⽰参与乘法运算的元素索引
 for k in range(m1.shape[1]):
 # 打印当前正在计算的元素
  print(f"{m1[i, k]} * {m2[k, j]} = {m1[i, k] * m2[k, j]}")
 # 将 matrix1 的第 i ⾏第 k 列元素与 matrix2 的第 k ⾏第 j 列元素相乘，并累
 manual_result[i, j] += m1[i, k] * m2[k, j]
 # 打印当前位置计算完成后的结果
print(f"结果矩阵[{i+1},{j+1}]:{manual_result[i, j]}\n")
print("⼿动推演结果:")
print(manual_result)

# %%
# ⼴播 (Broadcast) 是 numpy 对不同形状 (shape) 的数组进⾏数值计算的⽅式，对数
# 组的算术运算通常在相应的元素上进⾏。如果两个数组 a 和 b 形状相同，即满⾜
# a.shape == b.shape，那么 a*b 的结果就是 a 与 b 数组对应位相乘。这要求维数相
# 同，且各维度的⻓度相同。

# %%
a = np.array([1,2,3])
b = np.array([4,5,6])
a + b

# %%
a = np.array([(1,2), (2,2), (3,3), (4,4)]) # shape(4,2)
b = np.array([-1,1]) # shape(2)-> shape(1,2) -> shape(4,2) [ -1,1],[-1,1],[-1,1],[-1
a + b



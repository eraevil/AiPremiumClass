import numpy as np#数组创建方式一：定义数组然后转化为numpy数组
a = [11,12,13]

b = np.array(a)
b
import numpy as np#数组创建方式一：定义数组然后转化为numpy数组
a = [11,12,13]
b = np.array(a)
b

a1 = np.array([[11,12,13], [24,25,26], [37,38,39]])#创建高维数组
a1

a2 = np.zeros((3,5), dtype = int)#创建特殊全0数组，dtype用于指定元素数据类型
a2

a3 = np.ones((5, 3),dtype=int)#全一数组
a3


a4 = np.arange(2,8,1,dtype=int)#创建等差数列，从 2开始，8结束，1 为差的等差数列，最后⼀项⼀定⼩于 8
a4

a5 = np.eye(5,dtype=int)#创建单位矩阵
a5


np.random.random(6)#⽣成指定⻓度，在 [0,1)间平均分布的随机数组：

mu,sigma =0.1,1#⽣成指定⻓度，符合正态分布的随机数组，指定其均值为 0.1，标准差为 1
np.random.normal(mu, sigma, 5)
a = np.array([[1,2,4],[6,5,3],[7,8,9],(77,22,33)])##属性输出
print("dtype:",a.dtype)#数据类型
print("ndim:",a.ndim)#秩
print("shape:",a.shape)#形状
print("size:",a.size)#大小

import numpy as np
#数组的基本操作
a1 = np.array([[1,2,4],[6,6,3],[7,8,9],(77,22,33)])
print(88 in a1)
print(5 in a1)

a3=a1.reshape(12)#括号内个数要与a1的size相同
a3
a1.flatten()#直接调用即可
print(a1)
a1.T#转置
a2=np.array([1,2,3])#np的数组本来是一维的，转换成了二维的
a2.shape
print(a2)
a2=a2[np.newaxis,:]#newaxis在前则是行的维度增加，在后则是列的维度增加
a2.shape

#加减乘除
a3=np.zeros((2,2),dtype=int)
a4=np.array([[5,1],[6,4]],dtype=int)
# a3+a4
# a3-a4
# a3 + a4
a3 * a4#*乘是对应两矩阵对应位置相乘，与点乘要区分
a5=np.ones_like(a3)#加减乘除
a6=np.array([(1,2),(3,4)])
# a5*a6
a5/a6
#求和，求积
a7=np.ones_like(a3)
a8=np.array([(1,2),(3,4)])
# a7.sum()
a8.prod()

a9= np.array([5,8,7])#平均值，方差，不鸟准查
print("mean:",a9.mean())
print("var:", a9.var())
print("std:", a9.std())
print("max:", a9.max())
print("min:", a9.min())

a10 = np.array([1.9, 58, 6.4])
print("argmax:", a10.argmax())
print("argmin:", a10.argmin())
print("ceil:", np.ceil(a10))#向上取整
print("floor:", np.floor(a10))#向下取整
print("rint:", np.rint(a10))#四舍五入
a11=np.array([3,2,45,65,76,1,89,6])#排序
a11.sort()
a11
a12=np.array([(1,2), (3,4), (5,6)])#访问三行两列数组的第一行
# a12[0]
# a12[1:]#访问数组从第二行开始到最后一行，列数不限制
a12[: , :1]#行不限制，列为第一列，:1表示第一个索引
import numpy as np
# 定义两个简单的矩阵
m1 = np.array([[0,1],[2,3]],dtype=np.int32)
m2 =np.array([[4, 6], [5, 8]] , dtype=np.int32)
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
print("使⽤ @ 得到的矩阵乘法结果:")
print(result_at)
# 创建⼀个全零矩阵，⽤于存储⼿动推演的结果
# 结果矩阵的⾏数等于 matrix1 的⾏数，列数等于 matrix2 的列数
test = np.zeros((m1.shape[0], m2.shape[1]), dtype=np.float32)
# 外层循环：遍历 matrix1 的每⼀⾏
# i 表⽰结果矩阵的⾏索引
print(test)
for i in range(m1.shape[0]):
 # 中层循环：遍历 matrix2 的每⼀列
 # j 表⽰结果矩阵的列索引
 for j in range(m2.shape[1]):
 # 初始化当前位置的结果为 0
  test[i, j] = 0
 # 内层循环：计算 matrix1 的第 i ⾏与 matrix2 的第 j 列对应元素的乘积之和
 # k 表⽰参与乘法运算的元素索引
 for k in range(m1.shape[1]):
 # 打印当前正在计算的元素
  print(f"{m1[i, k]} * {m2[k, j]} = {m1[i, k] * m2[k, j]}")
 # 将 matrix1 的第 i ⾏第 k 列元素与 matrix2 的第 k ⾏第 j 列元素相乘，并累
 test[i, j] += m1[i, k] * m2[k, j]
 # 打印当前位置计算完成后的结果
 print(f"结果矩阵[{i+1},{j+1}]:{test[i, j]}\n")
print("⼿动推演结果:")
print(test)
np.save('test.npy',test)#保存
result_np=np.load('test.npy')#文件载入
result_np
a13 = np.array([[1,3],[2,4],[5,6],[8,9]])##广播机制shape（4，2），最后一个维度相同时适用
b14 = np.array([-2,1])#shap(2)-->shape(1,2)-->shape(4,2)四行[-2,1]
a13+b14

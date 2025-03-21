import torch;
import numpy as np;

data = [[1.1, 2.2], [3.3, 4.4]];
torchData = torch.tensor(data); ## 创建张量
torchData

a = np.array([1, 2, 3, 5], float);
torchA = torch.from_numpy(a);
torchA
## torchA.shape

a_ones = torch.ones_like(torchA); ## 参照括号内的张量属性 生成一个元素均为1 的张量  括号内必须为张量 不能为为numpy数组
a_ones

a_rand = torch.rand_like(torchA, dtype=torch.float) 
a_rand

shape = (3, 3, 1)
dimension_tensor1 = torch.rand(shape); ## 按照shape 定义的维度，生成随机数张量
dimension_tensor1

dimension_tensor2 = torch.ones(shape); ## 生成均为1 的张量 
dimension_tensor2

dimension_tensor3 = torch.zeros(shape);
dimension_tensor3

a = torch.ones(3,3, 3, dtype=torch.float);
a
b = torch.rand_like(a, dtype=torch.double);
b
b.size()
torch.rand(5, 3) ##均匀分布 5行3列的张量
torch.randn(4,4) ## 正态分布 4行4列
torch.normal(mean=1,std=1.0,size=(5,3)) ## 离散正态分布  5行三列，均值为1 标准差为1

torch.linspace(start=1,end=20,steps=20) ## 线性间隔张量 起止为1到20 元素数量为20

b.shape ## 获取张量的形状
b.dtype ## 张量的数据类型
b.device ## 存储位置

print(torch.cuda.is_available()); ## 能否移动到GPU上，打印false
print(b)
b[0] # 获取张量b中的第一行数据
b[:, 1]# 获取张量B中的第2列数据
b[..., 1] ## 获取每个数组中的第二列的元素重新生成数组张量

b[:,1] = 0 ## 将张量b中的第二列数据全部赋值为0
b


a = torch.tensor([1.1, 2.2, 3.2]);
b = torch.tensor([4.1,5.1,6.1]);
c = torch.cat([a,b]); ## 拼接张量
c
c = a@b # a b中所有一致下标的元素相乘再进行求和
c
c = a.matmul(b)
c
c = torch.rand_like(a)
torch.matmul(a, a.T, out=c)
c

c = a.sum() # a中所有元素求和
c
c_item = c.item(); #将张量转为python数值
c_item
type(c_item)
print(a, '\n');

a.add_(5);
print(a);


a = torch.ones(5);
a
b = a.numpy();
b
a.add_(2);## a 发生变化时，b同样改变 张量变化  numpy数组同样变化 反之亦然
b

a = np.ones(5);
b = torch.from_numpy(a);
a += 2;
b




import numpy as np
a = [1,3,4]
b = np.array(a)
b
## 生成整数类型的数组

c = np.array([3,4,5], float)
c
## 生成浮点数 类型的数组


d = np.zeros((3,3,3), int)
d
## 生成全为0 的数组


e = np.arange(1, 10, 0.1)
e
## 生成等差数组 从1 -10

f = np.eye(5, int)
f
##   生成对角线均为1 的数组 第一位数字为N 则每个数组中元素为n


np.random.random(10)
## 生成0-1 之间平均分布的随机数组

mu,sigma = 0, 0.2
np.random.normal(mu, sigma, 10)
## 生成均值为0 切标准差为0.2 的数组  10 为数组中的元素数量

a = np.array([1, 2,3,4,5])
a[0]  ## 数组中第0位下标
a[1:] ## 获取数组中第1位下标及之后的数据
a[:] ## 获取所有 []中维度需与 生成的数组维度相同，否则报错
## for i in a: ## 遍历数组中的所有元素
  ##  print(i)
a.ndim ##数组维度
b = np.array([(1,2,3), (1,3,4)])
b.ndim

b.shape ## 数组的大小  (5, )即为 5行一列

b.size ## 数组中所有元素的数量 2*3 = 6

b.dtype ## 数组中元素类型

print(4 in b) ## 判定b中是否包含元素4

c = b.reshape(6) 
print(b)

c+=1 
print(b)## 重新排列 flatten  与 reshape 区别  flatten在生成新的数组做改变时，原数组不变。而reshape改变


b.transpose() ## 转置

a = np.array([1, 2, 3,4])
##a = a[:, :,  np.newaxis]
a = a[:, np.newaxis]
a  ## newaxis升一维 

a = np.array([[1,2], [2,3 ], [3,4]])
b = np.array([5,10])
a + b
a-b
a * b 
a / b
a % b  ## 加减乘除取余

a.mean()  ##均值
a.var()
a.std()
a.max()
a.min()  ## 方差、标准差、最大最小值

a.sort() ## 排序
a

a = np.array(-1)
a = np.abs(a)
a

## 定义两个简单的矩阵
m1 = np.array([[1, 2], [3, 4]] , dtype=np.float32)
m2 = np.array([[5, 6], [7, 8] ], dtype=np.float32)
## dot 与 @ 均为乘法 第一位数字 = 1*5 + 2*7 = 19  2 : 1 * 7 + 2 * 8  34依次类推
result_dot = np.dot(m1, m2)
result_at = m1 @ m2
print("矩阵 1:")
print(m1)
print("矩阵 2:")
print(m2)
print("使⽤ np.dot 得到的矩阵乘法结果:")
print(result_dot)
print("使⽤ @ 运算符得到的矩阵乘法结果:")
print(result_at)
## 创建⼀个全零矩阵，⽤于存储⼿动推演的结果
## 结果矩阵的⾏数等于 matrix1 的⾏数，列数等于 matrix2 的列数
manual_result = np.zeros((m1.shape[0], m2.shape[1]), dtype=np.float32)
## 外层循环：遍历 matrix1 的每⼀⾏
## i 表⽰结果矩阵的⾏索引
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

np.save('result.npy', manual_result) ## 文件名及储存内容  默认存储在当前文件同一文件夹下

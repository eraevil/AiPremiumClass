import numpy as np
import torch
a = np.array([(1,2),(3,4)])
b = torch.from_numpy(a)
print(b)
print(torch.cuda.is_available())
# numpy基础

 # 创建数组
arr = np.array([[1,2,3],[4,5,6]])
print(arr)
a = [1,2,3,4,5]
aa = np.array(a)
print(aa)
b = np.array(a,float)
print(b)

# 创建元素为0的array
zeros = np.zeros((2,2))
print(zeros)

# 创建元素为1的array
one = np.ones((2,2),dtype=np.float32)
print(one)

# 按照顺序和间隔创建array
a = np.arange(0,10,3)
print(a)

# 创建对角线array
a = np.eye(4)
print(a)

# 创建随机数
a = np.random.random((3,4))
print(a)
mu,sigma = 0,1

# 正态分布的随机数生成
a = np.random.normal(mu,sigma,(3,4))
print(a)

# 数组访问与切片
a = np.random.random((3,4))
print(a)
print(a[:,1])
print(a[1,:])
a,b,c,d = a[0]
print(a,b,c,d)

# 数组的重排列
a = np.ones([3,4,5])
print(a)
print(a.reshape(60))

# 数组转置
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a.transpose())
print(a.T)

# 平坦化数组
print(a.flatten())

# 增加维度
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
a = a[:,np.newaxis,:]
print(a)

# numpy的数学操作
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a + b)
print(a.sum(axis=0))
print(a.sum(axis=1))
print(a.sum())
print(a * b)
print(a.dot(b))
print(a.mean())
print(a.var())
print(a.std())
# 找索引
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a.argmax())
print(a.argmin())
# 取整
b = np.array([1.1,2.2,3.3,4.4,5.5,6.6])
print(np.ceil(b))
print(np.floor(b))
print(np.log(b))
print(np.rint(b))
# 排序
a = np.array([12,54,7,56,8,31,43,98,43,8])
print(a.sort())
# 矩阵
# 点积运算
m1 = np.array([[1,2],[3,4],[5,6]])
m2 = np.array([[3,4,5],[5,6,6]])
print(np.dot(m1,m2))

# 广播机制
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = np.array([1,2,3])
print(a+b)

# pytorch基础
data = [[1,2,3],[4,5,6],[7,8,9]]
x_data = torch.tensor(data)
print(x_data)
data_2 = torch.ones_like(x_data)
print(data_2)
data_3 = torch.zeros_like(x_data)
print(data_3)
data_4 = torch.randn_like(data_3,dtype=torch.float)
print(data_4)
data_5 = torch.rand_like(data_3,dtype=torch.float)
print(data_5)
print(torch.normal(0,2,(3,4)))
# 张量属性
print(data_5.shape)
print(data_5.size())
print(data_5.dtype)
print(data_5.device)
device = torch.device('cuda')
data_5 = data_5.to(device)
print(data_5.device)
# 张量的索引和切片
tensor = torch.ones((3,4))
print(tensor[:,1])
# ...跳过前面所有维度
print(tensor[...,-1])
# 张量的拼接
t1 = torch.cat((tensor,tensor),dim=-1)
print(t1)
# 张量的算术运算
tensor = torch.arange(1,10,dtype=torch.float).reshape(3,3)
print(tensor)
a = tensor @ tensor.T
print(a)
b = tensor.matmul(tensor.T)
print(b)
torch.matmul(a,a.T,out = a)
print(a)
z = a*a
print(z)
z2 = a.mul(a)
print(z2)
# 单元素张量 张量类型转换
agg = a.sum()
agg_item = agg.item()
print(type(agg))
print(type(agg_item))
np_arr = a.numpy()
print(type(np_arr))
# 替代操作 就地修改张量的值
print(a)
a.add_(7)
print(a)
# 张量导出到numpy时，如果张量改变numpy也会受到影响
n = a.numpy()
print(n)
a.add_(1)
print(n)
a.add_(2)
print(a)
import numpy as np
#arr = np.array([1,2,3,4],np.int64)
#arr = np.array([1,2,3,4],float)
#arr = np.array([(1,2,3),(4,5,6),(7,8,9)]) # 二维矩阵
#arr = np.array([(1,2,3),(2,2,10)])#列表包含两个元组，每个元组3个数。表示2行3列矩阵

#arr = np.zeros((2,3),dtype=np.float32) #这里是2行3列注意
#arr = np.zeros([(1,2),(2,3)],dtype=np.int64) # 这种写法错误

#arr = np.ones((3,4),dtype=np.float64)

#arr = np.arange(1,5,1) # 是[1,5)区别
#arr = np.arange(9,-2,-1) #输出倒叙
#arr = np.arange(1,5,-1) #直接返回空

#arr = np.eye(2) #创建单位矩阵 结果包含小数点，是因为默认detype是float
#arr = np.eye(3,dtype=np.int16)

#arr = np.random.random(5) # 默认[0,1)平均分布5个随机数
#arr = np.random.random((2,10)) # 生成2行10列个[0,1)随机数
#arr = np.random.uniform(2,10,size = (2,10)) # 生成[2,10) 2行10列个随机数

#arr = np.random.normal(0,0.1,5) #正态分布，均值0标准差0.1
#arr = np.random.normal(0,0.2,10)

#arr = np.array([(1,2),(3,4),(5,6)])
#arr = arr[:,1] # 取全部行，取序号为1的列 切片后类型是一维numpy数组
#arr = arr[1,:]

# arr = np.array([(1,2,2),(3,4,0),(5,6,-1)])
# for i,j,z in arr: # python中可以这么遍历数组
#     print(i,j,z)
#     print(i)

# arr = np.array([(1,2,2),(2,3,1)])
# print(arr.ndim) #几行 维度  轴数
# print(arr.shape)
# print(arr.size)
# print(arr.dtype)

# arr = np.array([(1,2,2),(2,3,1)])
# print(3 not in arr)
# print(-1 not in arr)

# arr = np.arange(1,10)
#arr = arr.reshape(3,3,1) # 转化为3*3*1维
# arr = arr.reshape(3,3)
# arr = arr.T
# arr = arr .flatten()
# arr = np.array([1,2,3,4,5,6,7])
# arr = arr[:,np.newaxis] #增加新的维度，在指定位置插入新轴，这里是列插入新维，变成7行1列
# arr = np.array([(1,2), (3,4), (5,6)])
# arr = arr[:,np.newaxis,:]  
# print(arr.shape) # [3,1,2] 

# arr = np.ones((2,2),dtype=int)
# arr2 = np.array([(-1,1),(1,-1)])
# arr = arr-arr2

# arr = np.array([(1,2,2),(3,4,0),(5,6,-1)])
# print(arr.sum()) #求和,记得加（来调用函数）
# print(arr.prod())#求积,记得加（来调用函数）

# arr = np.array([(1,2,2),(2,3,1)])
# print(arr.ndim) #几行 维度  轴数
# print(arr.mean()) #平均值
# print(arr.var()) #方差
# print(arr.std()) #标准差

# arr = np.array([1.02,3.8,-4.9])
# print(arr.argmax()) # 最大值索引 
# print(arr.argmin()) # 最小值索引
# print(np.ceil(arr)) # 向上取整
# print(np.floor(arr)) # 向下取整
# print(np.rint(arr)) # 四舍五入
# #arr = arr.sort() arr.sort会直接排序，但不返回任何值，因此这样arr被赋值为none
# arr.sort()
# print(arr)

# 这两种方式是相同的，都是创建2*3
#arr = np.array([[1,2,3],[2,3,4]]) # 嵌套列表，外层列表每个元素是一个内层列表，表示一行
#arr2 = np.array([(1,2,3),(2,3,4)]) # 嵌套元组，外层列表每个元素是一个元组，表示一行
# #arr2 = np.array([[4,5,6],[7,8,9]])
# print(arr.shape)
# print(arr2.shape)
# #print(arr*arr2)
#print(np.array_equal(arr,arr2)) # 输出true

# m1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
# m2 = np.array([[5, 6], [7, 8]], dtype=np.float32)
# # 使用 np.dot 进行矩阵乘法
# result_dot = np.dot(m1, m2)
# print(result_dot)
# # 使用 @ 运算符进行矩阵乘法
# result_at = m1 @ m2
# print(result_at)

# m1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
# m2 = np.array([[5, 6], [7, 8]], dtype=np.float32)
# # 创建全零矩阵存放乘法结果，行为m1的行，列为m2的列
# manual_result = np.zeros((m1.shape[0],m2.shape[1]),dtype=np.float32) 
# manual_result = m1 @ m2
# print(manual_result)
# np.save('result.npy',manual_result)
# result_np = np.load('result.npy')
# print(result_np)

# arr1= np.array([(1,2),(2,2),(3,3)]) #3*2
# arr2 = np.array([-1,0]) #广播机制 1*2->3*2，复制下去 (-1,0)(-1,0)(-1,0)
# print(arr2+arr1) 

import torch

# tensors = torch.tensor([(1,2),(3,4)])
# print(tensors)
# tensor = torch.arange(1,10,dtype=torch.float32).reshape(3,3)
# print(tensor.dtype)

# y1 = tensor @ tensor.T # 矩阵乘 m*n * n*p = m*p
# y2 = tensor.matmul(tensor.T) # matmul和@等价,在任何维度都是
# y3 = torch.rand_like(tensor) # 在[0,1)生成与tensor形状相同的随机张量
# y4 = torch.empty(3,3) #分配一个张量，注意形状要与矩阵乘法结果一致 这里用y3就行
# torch.matmul(tensor,tensor.T,out=y4) # y4必须是一个已分配好的张量
# print(y1,y2,y3,y4)

# tensor = torch.arange(1,10,dtype=torch.float32).reshape(3,3)
# z1 = tensor*tensor # 元素对应相乘
# z2 = tensor.mul(tensor) # 元素对应相乘
# z3 = torch.rand_like(tensor)
# torch.mul(tensor,tensor,out = z3)
# print(z1,z2,z3)

# np_array = np.array([(1,2),(3,4)])
# data = torch.from_numpy(np_array) #numpy转pytorch ,转化后两者共享内存
# np_array[0,0] = 4 # 验证共享内存
# print(data)

# shape = (2,3,) #元组定义允许在最后一个元素后加一个逗号
# rand_tensor = torch.rand(shape)
# ones_tensor = torch.ones(shape)
# zeros_tensor = torch.zeros(shape)
# print(f"Random Tensor: \n {rand_tensor} \n")
# print(f"Ones Tensor: \n {ones_tensor} \n")
# print(f"Zeros Tensor: \n {zeros_tensor}")

# m = torch.ones(5,3, dtype=torch.double)
# n = torch.rand_like(m, dtype=torch.float)
# print(n.size()) # torch.Size([5,3])
# print(n.shape) # 也是[5,3]
# # 均匀分布
# print(torch.rand(n.size()))
# # 标准正态分布
# print(torch.randn(n.shape))
# # 离散正态分布
# print(torch.normal(mean=0.0,std=1.0,size=(n.shape)))
# # 线性间隔向量(返回一个1维张量，包含在区间start和end上均匀间隔的steps个点)
# print(torch.linspace(start=1,end=10,steps=21))

# tensor = torch.rand(3,4)
# print(tensor.device)
# if torch.cuda.is_available(): # 检测当前环境是否有可用GPU
#     device = torch.device("cuda") # 使用GPU
#     tensor = tensor.to(device) # 张量移动到GPU
# print(tensor)
# print(tensor.device) 

# tensor = torch.ones(4,4)
# print(tensor[0]) # 第一行 
# print(tensor[0,:]) # 第一行
# print(tensor[...,-1]) #最后一列， ... 表示选择所有前面的维度，相当于：

# tensor = torch.ones(4,4)
# t = torch.cat([tensor,tensor,tensor],dim=1) # 拼接张量，在1维度（列）上拼接
# print(t*3) #对张量t每个元素乘以3

# tensor = torch.arange(1,10, dtype=torch.float32).reshape(3, 3)
# sums = tensor.sum() # 对所有元素求和，变为标量张量
# items = sums.item() # 将标量张量转化为float类型
# tensor+=5
# print(sums,items)
# print(tensor)


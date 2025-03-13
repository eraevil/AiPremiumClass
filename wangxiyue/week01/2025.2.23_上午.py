import numpy as np

# a = [1,2,3]
# b = np.array([1,2,3,4,5],float)
# print( b)


# arr = np.array([(1,2,3),(4,5,6),(7,8,9)])
# print(arr)

# a1 = np.zeros((2,3),dtype=np.float32)
# print(a1)

# a2 = np.ones(shape=(3,3),dtype=np.float32) #初始化 1
# print(a2)

# a3 = np.arange(1,5,0.5) # 递增数组
# print(a3)

# a4 = np.eye(5) # 对角线
# print(a4)

# a4 = np.random.random(5)
# print(a4)

# nu,sigma = 0,0.1
# a5 = np.random.normal(nu,sigma,size=5) # random.normal 正态分布
# print(a5)

# a6 = np.array([(1,2),(3,4),(5,6)])
# # print(a6[1:])
# print(a6[0,:1])

# a7 = np.array([(1,2),(3,4),(5,6)])
# print(a7.ndim) # 数据维度

# for i,j in enumerate(a7):

# a7 = np.arange(1,7)
# print(a7.shape)
# print(a7.reshape(1,2,3))# 维度乘积 = 元素个数


a8 = np.array([(1,2,3),(4,5,6),(7,8,9)])
print(a8)
# print(a8.transpose())# 行列转换
# print(a8.T) # 行列转换 == transpose

# print(a8.flatten()) # 将数组 拉成 一维数组
# # print(a8.T.flatten()) # eg

###?????
# an = a8[:,:,np.newaxis] # [3,3,1] 在原来的数组上增加1个维度
# print(an)
# print(an.shape)

# print("sum = " , a8[:1].sum())
# print(a8.prod())

# print("sum",a8.sum())
# print("平均",a8.mean())
# print("方差",a8.var())
# print("标准差",a8.std())
#
# print("最大值索引" , a8.argmax())
# print("最小值索引",a8.argmin())
# print("向上取整 ， 去掉小数位",np.ceil(a8))
# print("向下取整",np.floor(a8))
# print("四射五入",np.rint(a8))
#
#
# a8.sort()
# print("after sort" , a8)

# 矩阵内积  =  第一行 x 第一列
#矩阵内积 运算 = 两个相同维度的矩阵 第一行成第一例

a = np.array([(1,2),(2,2),(3,3),(4,4)])
b = np.array([-1,1])
## np广播机制 brodcast
## a+b  ： b-> shape(2)-> shape(1,2) -> shape(4,2) [[-1,1],[-1,1],[-1,1],[-1,1]]
print(a+b)

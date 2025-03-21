import numpy as np

#创建随机数数组
a = np.random.rand(5)   #创建一个包含5个随机数的一维数组
print(a)

b = np.random.rand(2, 3)  #创建一个包含2行3列随机数的二维数组
print(b)

#创建正态分别的随机数数组
c = np.random.normal(0, 0.1, 5) #创建一个均值为0，标准差为0.1，包含5个正态分布随机数的一维数组
print(c)

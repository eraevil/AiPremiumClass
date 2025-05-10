import numpy as np
def np_data_attrib(a):
    """打印np矩阵的常见属性"""
    print(a)
    print("ndim:", a.ndim)
    print("shape:", a.shape)
    print("size:", a.size)
    print("dtype", a.dtype)
def np_all_caculate(a):
    """打印np矩阵的常规计算结果"""
    print(a)
    print("mean:",a.mean())
    print("sum:",a.sum())
    print("prod:",a.prod())
    print("var:", a.var())
    print("std:", a.std())
    print("max:", a.max())
    print("min:", a.min())
# 数组创建
a = [1,2,4]
b = np.array(a)
c = np.array([(1,2,3),(2,3,4)], float)
print(c)
a7 =  np.arange(1,10)
print(a7)
a7= a7.reshape(3,3,1)
print(a7)
a8 =a7.transpose()
print(a8)
print(a8.sum())
print(a7+a8)
a2 = [[1,2,3],[3,4,5],[4,5,6]]
a2 = np.array(a2)
b2 = [[1,2,3],[3,4,5],[4,5,6]]
for i in b:
    print(i)
a = np.array([(1, 2,4), (3, 4,5), (5, 6,7)])
for i,j,k in a:
    print(i*j*k)
a = np.array([(1, 2,4), (3, 4,5), (5, 6,7)]) # 二维
a = np.array([((1,2),(2,3)), ((1,3),(2,3)), ((3,2),(5,3))]) # 三维
np_data_attrib(a)
a = np.transpose(a)
np_data_attrib(a)
# 多维转一维
b = a.flatten()
np_data_attrib(b)
c = a[np.newaxis]

np_data_attrib(c)
# 矩阵计算
np_all_caculate(c)

# 矩阵乘积
b2 = np.array(b2)
c2 = a2 * b2
print(c2)
# 矩阵点乘
d2 = np.dot(a2,b2)
print(d2)
result1 = a2 @ b2
print(result1)
# 文件操作
np.save("result.npy", result1)
result2 = np.load("result.npy")
print(result2)
a = np.arange(1,5,0.5)
print(a)
a = np.eye(4)
print(a)
a = np.random.random(5)
print(a)
a = np.random.normal(1,0.1,5)
print(a)

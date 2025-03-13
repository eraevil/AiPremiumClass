import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])

print(a.sum()) # 所有元素的和
print(a.sum(axis=0)) # 每列元素的和
print(a.sum(axis=1)) # 每行元素的和

print(a.prod()) # 所有元素的乘积
print(a.prod(axis=0)) # 每列元素的乘积
print(a.prod(axis=1)) # 每行元素的乘积

print(a.mean()) # 所有元素的平均值
print(a.mean(axis=0)) # 每列元素的平均值
print(a.mean(axis=1)) # 每行元素的平均值

print(a.var()) # 所有元素的方差
print(a.var(axis=0)) # 每列元素的方差
print(a.var(axis=1)) # 每行元素的方差

print(a.std()) # 所有元素的标准差
print(a.std(axis=0)) # 每列元素的标准差
print(a.std(axis=1)) # 每行元素的标准差

print(a.min()) # 所有元素的最小值
print(a.min(axis=0)) # 每列元素的最小值
print(a.min(axis=1)) # 每行元素的最小值

print(a.max()) # 所有元素的最大值
print(a.max(axis=0)) # 每列元素的最大值
print(a.max(axis=1)) # 每行元素的最大值

print(a.argmin()) # 所有元素的最小值的索引
print(a.argmin(axis=0)) # 每列元素的最小值的索引
print(a.argmin(axis=1)) # 每行元素的最小值的索引

print(a.argmax()) # 所有元素的最大值的索引
print(a.argmax(axis=0)) # 每列元素的最大值的索引
print(a.argmax(axis=1)) # 每行元素的最大值的索引

b = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
print(b)
print(np.ceil(b)) # 向上取整
print(np.floor(b)) # 向下取整
print(np.round(b)) # 四舍五入
print(np.round(b, 1)) # 四舍五入
print(np.rint(b)) # 四舍五入

c = np.array([16,31,12,28,22,31,48])
print(c)
c.sort() # 排序
print(c)



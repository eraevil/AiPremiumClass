import numpy as np

a = [1, 2, 3]
b = np.array(a, dtype=np.float64)
# print(b)

c = np.array([[1, 2, 3], [4, 5, 6]])
# print(c)

zeros1 = np.zeros([2, 3], dtype=float)
print(zeros1)
zeros2 = np.zeros((2, 3), dtype=float)
print(zeros2)

ones = np.ones([2, 3], dtype=float)
print(ones)

#  等差数列
cc = np.arange(1, 6, 0.2)
print(cc)

# 单位矩阵
ee = np.eye(4)
print(ee)

# 随机数
ran = np.random.random(5)
print(ran)

# 正太分布
nor = np.random.normal(0, 0.1, 6)
print(nor)

a1 = np.array([[1, 2], [3, 4], [5, 6]])
# print(a[0])
print(a1[:, 1])
print(a1[:, :1])
# print(a1[1,1])
# print(a1[1:])

i, j = a1[0]
print(i, j)

bb = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
for i1, j1, k1 in bb:
    print(i1, j1, k1)

xx = np.arange(1, 10)
print(xx)
aaa = xx.reshape(3, 3)
print(aaa)
print("-----------------------------")
yy = np.array([[3, 4], [5, 6], [7, 8]])
print(yy.shape)

zz = yy[:, np.newaxis, :]
zzz = yy[:, :, np.newaxis]
print(zz.shape)
print(zzz.shape)

print("-------------------")
rr = np.arange(1, 10)
re = rr.reshape(3, 3)
print(re)
print(re.ndim)
print(re.shape)
print(re.size)
print(re.dtype)

print("-------------------")

if 1 in re:
    print(1 in re)

if (1, 2, 3) in re:
    print((1, 2, 3) in re)

#  ValueError: operands could not be broadcast together with shapes (3,3) (2,)
# if (1,2) not in re:
#     print((1,2) not in re)

print("------------------")

# 转置
print(re)
te = re.transpose()
print(te)
print("an other type ", re.T)

print("------------------")
a11 = np.ones((2, 2))
a21 = np.array([[-1, -1], [1, 1]])
print(a11 + a21)
print(a11 * a21)
print(a21.sum())
# 求积
print(a21.prod())
# 求平均值
print(a11.mean())
# 求方差
print(a11.var())
# 求标准差
print(a11.std())
print(a11.max())
print(a11.min())

a = np.array([1.2, 3.8, 4.9])
# 最大值的位置
print(a.argmax())
# 最小值的位置
print(a.argmin())

# 向上取整
print(np.ceil(a11))
# 向下取整
print(np.floor(a11))
# 四舍五入
print(np.rint(a11))

a = np.array([16,31,12,28,22,31,48])
print(a.sort())


ar1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
ar2 = np.ones((3,3))
print(ar1)
print(ar2)
a_res = np.dot(ar1, ar2)
print(a_res)


data = np.random.random(5)

tt = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(tt)
tt1 = tt.reshape(3, 3, 1)
print(tt1)

aaa_bb = np.arange(1, 10, dtype=float).reshape(3, 3)
aaa_cc = np.arange(1, 7, dtype=float).reshape(2, 3)
print(aaa_bb)
print(aaa_cc)

y1 = aaa_bb @ aaa_cc.T
y2 = np.matmul(aaa_bb, aaa_cc.T)
print("---------")
print(y1)
print(y2)



aaa_bb = np.arange(1, 10, dtype=float).reshape(3, 3)
aaa_cc = np.arange(1, 7, dtype=float).reshape(3, 2)
print("xxxxxxxx")
result = np.matmul(aaa_bb, aaa_cc)
print(result)
manual_result = np.zeros((aaa_bb.shape[0], aaa_cc.shape[1]), dtype=float)
for i in range(aaa_bb.shape[0]):
    for j in range(aaa_cc.shape[1]):
        manual_result[i, j] = 0
        for k in range(aaa_bb.shape[1]):
            print(f"{aaa_bb[i, k]} * {aaa_cc[k, j]} = {aaa_bb[i, k] * aaa_cc[k, j]}")
            print(f"{aaa_bb[i, k]} * {aaa_cc[k, j]} = {aaa_bb[i, k] * aaa_cc[k, j]}")
            manual_result[i, j] += aaa_bb[i, k] * aaa_cc[k, j]
        print(f"结果矩阵[{i+1},{j+1}]:{manual_result[i, j]}\n")

np.save("result.npy", result)
data = np.load("result.npy", allow_pickle=True)
print(data)

b1 = np.array([[1,2],[2,2],[3,3],[4,4]])
b2 = np.array([[1,1]])
print(b1 + b2)
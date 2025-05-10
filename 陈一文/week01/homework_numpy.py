import numpy as np
a = [4, 5, 65]
b = np.array(a)
print(b)

a = np.array([2, 4, 6], float)
print(a)

a = np.array([(1, 5, 7), (2, 4, 6), (3, 6, 9)])
print(a)

a = np.zeros((5, 8), dtype = float)
print(a)

a = np.ones((2, 2))
print(a)

a = np.arange(1, 5, 0.2)
print(a)

a = np.eye(6)
print(a)

np.random.random(15)

mu, sigma = 0, 0.5
np.random.normal(mu, sigma, 10)

a = np.array([(3, 6), (2, 4), (1, 5)])
print(a[0], '\n')
print(a[1:], '\n')
print(a[: , :1], '\n')
print(a[0][0], '\n')

for i, j in a:
    print(i * j)

a = np.array([(1, 4, 7), (2, 5, 8), (3, 6, 9)])
print('ndim: ', a.ndim)
print('shape: ', a.shape)
print('size: ', a.size)
print('dtype: ', a.dtype)

a = np.array([(2, 5), (6, 8)])
print(3 in a)
print(2 in a)

a = np.zeros([5, 3, 1])
print(a)
a = a.reshape(5, 3)
print(a)

a = np.array([(1, 4, 7), (2, 5, 8), (3, 6, 9)])
print(a.transpose(), '\n')
print(a.T)

a.T.flatten()

a = np.array([1, 2, 3])
print(a.shape)
a = a[:, np.newaxis]
print(a, a.shape)

a = np.ones((2,2))
b = np.array([(-1, 1), (1, -1)])
print(a)
print(b)

print(a + b, '\n')
print(a - b, '\n')
print(a * b, '\n')
print(a / b, '\n')

a = np.array([1, 5, 9])
print(a.sum(), '\n')
print(a.prod(), '\n')
print(a.mean(), '\n')
print(a.var(), '\n')
print(a.std(), '\n')
print(a.max(), '\n')
print(a.min(), '\n')

a = np.array([2.5, 6.7, 9.9])
print("argmax:", a.argmax())
print("argmin:", a.argmin())
print("ceil:", np.ceil(a))
print("floor:", np.floor(a))
print("rint:", np.rint(a))

import numpy as np
 # 定义两个简单的矩阵
m1 = np.array([[15, 247], [13, 4]], dtype=np.float32)
m2 = np.array([[35, 62], [27, 85]], dtype=np.float32)
              
 # 使⽤np.dot 进⾏矩阵乘法
result_dot = np.dot(m1, m2)

 # 使⽤@运算符进⾏矩阵乘法
result_at = m1 @ m2

print("矩阵1:")
print(m1)
print("矩阵2:")
print(m2)                           
print("使⽤np.dot 得到的矩阵乘法结果:")
print(result_dot)
print("使用@运算符得到的矩阵乘法结果")
print(result_at)

 # 创建⼀个全零矩阵，⽤于存储⼿动推演的结果
# 结果矩阵的⾏数等于matrix1 的⾏数，列数等于matrix2 的列数
manual_result = np.zeros((m1.shape[0], m2.shape[1]), dtype=np.float32)
 # 外层循环：遍matrix1 的每⼀⾏
# i 表⽰结果矩阵的⾏索引
for i in range(m1.shape[0]):
    # 中层循环：遍历matrix2 的每⼀列
    # j 表⽰结果矩阵的列索引
    for j in range(m2.shape[1]):
        # 初始化当前位置的结果为0
        manual_result[i, j] = 0
        # 内层循环：计算matrix1 的第i⾏与matrix2 的第j列对应元素的乘积之和
        # k 表⽰参与乘法运算的元素索引
        for k in range(m1.shape[1]):
            # 打印当前正在计算的元素
            print(f"{m1[i, k]} * {m2[k, j]} = {m1[i, k] * m2[k, j]}")
            # 将matrix1 的第i⾏第k 列元素与matrix2 的第k ⾏第j 列元素相乘，并累
            manual_result[i, j] += m1[i, k] * m2[k, j]
        # 打印当前位置计算完成后的结果
        print(f"结果矩阵[{i+1},{j+1}]:{manual_result[i, j]}\n")
print("⼿动推演结果:")
print(manual_result)

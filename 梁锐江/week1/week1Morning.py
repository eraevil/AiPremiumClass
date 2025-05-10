import numpy as np


def test_func():
    return np.array([1, 2, 3, 4, 5], float)


def test_func2():
    return np.array([(1, 2, 3), (2, 3, 4)])


def test_func3():
    return np.zeros((3, 3), float)


def test_func4():
    return np.ones((3, 3), float)


def test_func5():
    # a = np.array([])
    a = np.array([(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)])
    print(a)
    print("ndim", a.ndim)
    print("shape", a.shape)
    print("size", a.size)
    print("dtype", a.dtype)


def test_func6():
    a = np.array([[[[1], [2]]]])
    print(a.flatten())


def test_func7():
    a = np.eye(4)
    print(a)


def test_func8():
    a = np.random.uniform(low=1, high=10, size=5)
    print(a)


def test_func9():
    mu, sigma = 2, 1
    print(np.random.normal(mu, sigma, 5))


def test_func10():
    a = np.array([(1, 2), (3, 4), (5, 6)])
    print(a[0])
    print(a[1:])
    print(a[:, :1])
    print(a[1, 1])


def test_func11():
    a = np.zeros([2, 3, 4, 5])
    print(a.ndim)
    print(a)


def test_func12():
    a = np.zeros([2, 3, 4, 5])
    b = a.reshape(120)
    print(b)


def test_func13():
    a = np.array([(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)])
    print(a.T)


def test_func14():
    a = np.array([(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)])
    print(a.flatten())


def test_func15():
    a = np.array([(1, 2, 3), (4, 5, 6), (7, 8, 9)])
    print(a.shape)
    a = a[np.newaxis, :, :]
    print(a)
    print(a.shape)


def test_func16():
    a = np.eye(4)
    print(a.sum())
    print(a.prod())


def test_func17():
    a = np.array([(1, 2, 3), (4, 5, 6), (7, 8, 9)])
    print(a.mean())
    print(a.var())
    print(a.std())


def test_func18():
    a = np.array([1.2, 3.8, 4.9])
    print(a.argmax())
    print(a.argmin())
    print(np.ceil(a))
    print(np.floor(a))
    print(np.rint(a))


def test_func19():
    a = np.arange(1, 10)
    print(a)


def test_func20():
    """
    广播机制：
    能否广播必须从数组的最高维向最低维看去
    依次对比两个要进行运算的数组的axis的数据宽度是否相等，
    如果在某一个axis下，一个数据宽度为1，另一个数据宽度不为1，那么numpy就可以进行广播；
    但是一旦出现了在某个axis下两个数据宽度不相等，并且两者全不为1的状况，就无法广播。
    (1,70) + (1,20) X
    (3,4) + (4) -> (3,4) + (1,4) -> (3,4) + (3,4)
    """
    # 定义两个简单的矩阵
    m1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    m2 = np.array([[5, 6], [7, 8]], dtype=np.float32)
    # 使⽤ np.dot 进⾏矩阵乘法
    result_dot = np.dot(m1, m2)
    # 使⽤ @ 运算符进⾏矩阵乘法
    result_at = m1 @ m2
    # 创建⼀个全零矩阵，⽤于存储⼿动推演的结果
    # 结果矩阵的⾏数等于 matrix1 的⾏数，列数等于 matrix2 的列数
    manual_result = np.zeros((m1.shape[0], m2.shape[1]), dtype=np.float32)
    # 外层循环：遍历 matrix1 的每⼀⾏
    # i 表⽰结果矩阵的⾏索引
    for i in range(m1.shape[0]):
        # 中层循环：遍历 matrix2 的每⼀列
        # j 表⽰结果矩阵的列索引
        for j in range(m2.shape[1]):
            # 初始化当前位置的结果为 0
            manual_result[i, j] = 0
            print(f"开始计算结果矩阵[{i},{j}]")
            # 内层循环：计算 matrix1 的第 i ⾏与 matrix2 的第 j 列对应元素的乘积之和
            # k 表⽰参与乘法运算的元素索引
            for k in range(m1.shape[1]):
                # 打印当前正在计算的元素
                print(f"{m1[i, k]} * {m2[k, j]} = {m1[i, k] * m2[k, j]}")
                # 将 matrix1 的第 i ⾏第 k 列元素与 matrix2 的第 k ⾏第 j 列元素相乘，并累
                manual_result[i, j] += m1[i, k] * m2[k, j]
            # 打印当前位置计算完成后的结果
            print(f"结果矩阵[{i + 1},{j + 1}]:{manual_result[i, j]}\n")
    print("⼿动推演结果:")
    print(manual_result)


def test_func21():
    a = np.array([[(1, 2), (3, 4), (5, 6)]])
    b = np.array([(4, 5, 6)])
    print(a + b)


def test_func22():
    a = np.array([[(1, 2), (2, 4), (3, 5)], [(2, 4), (3, 4), (5, 6)]])
    print(a.shape)
    b = np.array([(-1, -1), (-2, -2), (-3, -3)])
    print(b.shape)
    c = a + b
    print(c)
    print(c.shape)


if __name__ == "__main__":
    test_func22()

# 基础学习
## 1. Numpy
### 1.1 简介
    numpy科学计算基础库：提供了处理高维数组各种工具和函数。
### 1.2 安装
    参考：https://note.youdao.com/ynoteshare/index.html?id=37dcaa3e067716ce9ee2515265942416&type=note&_time=1740468223248
### 1.3 重点关注
    Numpy多维数组对象Ndarray, 可以通过下标访问数组中的元素(下标从0开始), 数组中的元素类型必须相同。
### 1.4 基础操作
    ```python
        import numpy as np

        # 创建Ndarray数组

        # 从列表创建
        np_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=int)
        print("从列表创建:", np_array)
        # ndarray转换为列表
        np_list = np_array.tolist()
        print("ndarray转换为列表:", np_list)

        # 从元组创建
        np_array = np.array((1, 2, 3, 4, 5))
        print("从元组创建:", np_array)


        # 使用内置函数
        # 全零数组
        np_array = np.zeros((2, 3), dtype=np.float32)
        print("全零数组:", np_array)

        # 全一数组
        np_array = np.ones((2, 3), dtype=np.int8)
        print("全一数组:", np_array)

        # one-hot编码
        np_array = np.eye(3, dtype=np.int8)
        print("one-hot编码:", np_array)



        # 等间隔数组
        np_array = np.arange(1, 10, 2)
        print("生成从1到10的序列，步长为2：:", np_array)

        np_array = np.linspace(1, 10, 5)
        print("生成5个在1到10之间均匀分布的数：:", np_array)



        # 一维随机数组
        np_array = np.random.rand(2)
        print("np.random.rand：0~1之间随机数:", np_array)
        np_array = np.random.random(2)
        print("np.random.random：0~1之间随机数:", np_array)

        # 二维(0,1]随机数组
        np_array = np.random.rand(2, 3)
        print("np.random.rand：0~1之间随机数组 组成2*3的矩阵:", np_array)
        np_array = np.random.random((2, 3))
        print("np.random.random：0~1之间随机数 组成2*3的矩阵:", np_array)

        # 二维指定范围随机数组
        np_array = np.random.randint(1, 10, size=(2, 3))
        print("1-10之间随机数 组成2*3的矩阵:", np_array)

        # 随机样本
        np_array = np.random.randn(2, 3)
        print("均值为 0、标准差为 1 的随机样本:", np_array)
        np_array = np.random.normal(2, 4, size=(2, 3))
        print("均值、标准差指定 的随机样本:", np_array)


        # 基本操作

        # 数组的形状和维度
        np_array = np.array([[[2, 4],[3, 6],[5, 7]], [[1, 3],[5, 7],[6,8]]])
        print("数组的形状：", np_array.shape)
        print("数组的维度：", np_array.ndim)
        print("数组的元素个数：", np_array.size)
        print("数组的元素类型：", np_array.dtype)

        # 数组的索引和切片
        print("数组的第一个元素：", np_array[0, 0, 0])
        print("数组的第一行：", np_array[0, 0, :])
        print("数组的最后一维元素：", np_array[..., -1])

        # 判定元素是否在数组中
        print("元素 5 是否在数组中：", 5 in np_array)
        print("元素 [2,4] 是否在数组中：", (2,4) in np_array)
        # print("元素 [2,4,8] 是否在数组中：", (2,4,8) in np_array)

        # 遍历
        for i,j,k in np_array:
            print(i,j,k)
            
        # 基础运算
        np_array1 = np.array([[1, 2], [3, 4]])
        np_array2 = np.array([[5, 6], [7, 8]])
        print("数组相加：", np_array1 + np_array2)
        print("数组相减：", np_array1 - np_array2)
        print("数组相乘-元素级乘法-对应位置相乘：", np_array1 * np_array2)
        print("数组相乘-矩阵乘法-内积运算1：", np_array1 @ np_array2)
        print("数组相乘-矩阵乘法-内积运算2：", np.dot(np_array1, np_array2))
        print("数组相除：", np_array1 / np_array2)
        print("数组的转置1：", np_array1.T)
        print("数组的转置2：", np_array1.transpose())

        np_array3 = np.array([1.5, 2.6, 3.4])
        print("求和：", np.sum(np_array3))
        print("乘积：", np.prod(np_array3))
        print("均值：", np.mean(np_array3))
        print("标准差：", np.std(np_array3))
        print("方差：", np.var(np_array3))
        print("最大值：", np.max(np_array3))
        print("最小值：", np.min(np_array3))
        print("求和：", np.sum(np_array3))
        print("最大值下标索引：", np.argmax(np_array3))
        print("最小值下标索引：", np.argmin(np_array3))
        print("向上取整：", np.ceil(np_array3))
        print("向下取整：", np.floor(np_array3))
        print("四舍五入1：", np.round(np_array3))
        print("四舍五入2：", np.rint(np_array3))
        print("取绝对值：", np.abs(np_array3))
        print("排序:", np.sort(np_array3))


        # 改变数组维度
        np_array = np.arange(1,13)
        print(np_array)
        print(np_array.shape)
        # 维度大小乘积 == 元素个数
        np_array = np_array.reshape(3,4)
        print(np_array)


        # 动态增加数组维度
        np_array = np.array([[1, 2, 3], [4, 5, 6]])
        print("增加维度前：", np_array)
        print("增加维度前-维度", np_array.shape)
        np_array = np_array[np.newaxis, :]
        print("增加维度后1：", np_array)
        print("增加维度后1-维度", np_array.shape)
        np_array = np_array[:, np.newaxis, :]
        print("增加维度后2：", np_array)
        print("增加维度后2-维度", np_array.shape)
        np_array = np_array[:, :, np.newaxis]
        print("增加维度后3：", np_array)
        print("增加维度后3-维度", np_array.shape)


        # 多维数组转一维数组
        np_array = np.array([[1, 2, 3], [4, 5, 6]])
        print(np_array.flatten())


        # 广播机制-允许不同维度的数组进行运算-自动
        # 自动触发广播机制--最后一维的维度要相同
        np_array = np.array([2, 3, 4])  # shape(3)
        b = 2  # shape(1) -> shape(3) -> [2,2,2]
        # 对应位置相乘 or 相加
        print(np_array * b)
        print(np_array + b)

        # 文件读写
        np_array = np.array([(1,2), (3,4), (5,6)])
        np.savetxt('numpy_array.txt', np_array)
        np.save('numpy_array.npy', np_array)

        np_array1 = np.loadtxt('numpy_array.txt')
        print(np_array1)
        np_array2 = np.load('numpy_array.npy')
        print(np_array2)



    ```
## 2. PyTorch
### 2.1 简介
    PyTorch是一个基于Python的科学计算包，主要用于深度学习任务。
### 2.2 安装
    参考：https://note.youdao.com/ynoteshare/index.html?id=37dcaa3e067716ce9ee2515265942416&type=note&_time=1740468223248
### 2.3 重点关注
    PyTorch的核心是张量（Tensor），它是一个多维数组，类似于Numpy的ndarray。
### 2.4 基础操作
    ```python
        # 导入PyTorch库
        # 导入PyTorch库
        import torch

        # 初始化张量
        data = torch.tensor([[2, 3],[4,5]], dtype=torch.float32)
        print(data)
        print(data.dtype)

        # 从已知张量维度创建新张量
        data = torch.tensor([[1,2],[3,4]])
        # 维度指定同data
        # 全一
        data1 = torch.ones_like(data)
        print(data1)
        # 全0
        data3 = torch.zeros_like(data)
        print(data3)
        # 0~1 随机数-小数
        data2 = torch.rand_like(data, dtype=torch.float)
        print(data2)
        # 均值为 0，标准差为 1 - 正态分布
        data4 = torch.randn_like(data, dtype=torch.float)
        print(data4)
        # 1~10 随机数-整数
        data5 = torch.randint_like(data, 1, 10)
        print(data5)
        # 初始化全3的张量
        data6 = torch.full_like(data, 3)
        print(data6)
        # 内容不可预测的随机张量
        data7 = torch.empty_like(data)
        print(data7)

        print("========================================")

        # 维度自定的张量
        shape = (2,3)
        # 均匀分布
        data1 = torch.rand(shape)
        print(data1)
        # 正态分布
        data2 = torch.randn(shape)
        print(data2)
        # 离散正态分布
        data3 = torch.normal(mean=.0, std=1.0, size=shape)
        print(data3)
        # 线性间隔向量-在区间start和end上均匀间隔的steps个点
        data4 = torch.linspace(1, 10, steps=5)
        print(data4)
        # 全一
        data4 = torch.ones(shape)
        print(data4)

        print("========================================")

        # 张量属性
        tensor = torch.tensor([[1,2],[3,4]])
        print(f"Shape of tensor: {tensor.shape}")
        print(f"Datatype of tensor: {tensor.dtype}")
        print(f"Device tensor is stored on: {tensor.device}")

        print("========================================")

        # window-识别张量是否存储在GPU上
        tensor = torch.rand(4,4)
        if torch.cuda.is_available():
            device = torch.device("cuda")
            tensor = tensor.to(device)
        else:
            print("No GPU available")
            device = tensor.device
            print("use ", device)

        # mac-识别张量是否存储在GPU上
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            tensor = tensor.to(device)
        else:
            print("No mps available")
            device = tensor.device
            print("use ",device)
            
        print("=======================================")

        # 索引和切片
        tensor = torch.tensor([[1,2],[3,4]])
        print("第二行元素：", tensor[1])
        print("第一列元素：", tensor[:,0])
        print("第二列元素：", tensor[:,1])
        print("最后一列元素：", tensor[...,-1])
        print("对角线元素：", tensor.diag())

        print("=======================================")

        # 张量的拼接
        data1 = torch.tensor([[1,2,3],[4,5,6]])
        data2 = torch.tensor([[7,8,9],[10,11,12]])
        # cat：续接，dim取值[-2, 1], -2=0, -1=1
        # dim=0 表示按行拼接-A/B摞在一起
        data3 = torch.cat([data1, data2], dim=0)
        print(data3)
        # dim=1 表示按列拼接-A/B横着放
        data4 = torch.cat([data1, data2], dim=1)
        print(data4)

        # stack：叠加 
        # dim取值 0 到输入张量的维数
        # 0：左右拼接加维度
        data5 = torch.stack((data1, data2), dim=0)
        print(data5)
        print(data5.shape)
        # 1：每个行进行组合
        data6 = torch.stack((data1, data2), dim=1)
        print(data6)
        print(data6.shape)
        # 2：对相应行中每个列元素进行组合
        data7 = torch.stack((data1, data2), dim=2)
        print(data7)
        print(data7.shape)

        print("=======================================")


        # 算数运算
        tensor = torch.arange(1, 10, dtype=torch.float32).reshape(3,3)
        print(tensor)

        # 加法
        data1 = tensor + 1
        print(data1)   
        # 减法 
        data2 = tensor - 1
        print(data2)
        # 乘法
        data3 = tensor * 2
        print(data3)
        # 逐元素相乘
        data3 = tensor * tensor.T
        print(data3)
        y3 = torch.rand_like(tensor)
        torch.mul(tensor, tensor.T, out=y3)
        print("逐元素相乘", y3)
        # 矩阵乘法-内积
        data3 = tensor @ tensor.T
        print(data3)
        data3 = tensor.matmul(tensor.T)
        print(data3)
        y3 = torch.rand_like(tensor)
        torch.matmul(tensor, tensor.T, out=y3)
        print(y3)
        # 除法
        data4 = tensor / 2
        print(data4)

        print("=======================================")

        # 单元素张量转原生基本类型
        agg = tensor.sum()
        print(agg, type(agg))
        # 原生基本类型
        agg_item = agg.item()
        print(agg_item, type(agg_item))

        print("=======================================")

        # 张量转numpy
        # np_arr = tensor.numpy()
        # print(np_arr, type(np_arr))

        print("=======================================")

        # 危险函数（计算图中慎用）
        # In-place操作(就地操作)-值保存在原变量中
        print(tensor, "\n")
        print(tensor.add_(5), "\n")
        print(tensor)


        print("=======================================")

        import torch
        from torchviz import make_dot

        # ax + b
        a = torch.randn(10, requires_grad=True)
        b = torch.randn(10, requires_grad=True)
        x = torch.randn(10, requires_grad=True)

        y = a * x + b

        dot = make_dot(y, params={'a': a, 'b': b, 'x': x})
        dot.render('onex', format='png', cleanup=True, view=False)

        print("=======================================")

        # 定义矩阵 A，向量 b 和常数 c
        A = torch.randn(10, 10,requires_grad=True)  # requires_grad=True 表示我们要对 A 求导
        b = torch.randn(10,requires_grad=True)
        c = torch.randn(1,requires_grad=True)
        x = torch.randn(10, requires_grad=True)


        # 计算 x^T * A + b * x + c
        result = torch.matmul(A, x.T) + torch.matmul(b, x) + c

        # 生成计算图节点
        dot = make_dot(result, params={'A': A, 'b': b, 'c': c, 'x': x})
        # 绘制计算图
        dot.render('expression', format='png', cleanup=True, view=False)





    ```



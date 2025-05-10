import torch
import numpy as np
from torchviz import make_dot

data = [[1, 2], [3, 4]]


def func1():
    """
    数据表示：
        PyTorch张量（Tensor）：是多维数组的数据结构，支持CPU和GPU上的计算。它是专门为深度学习设计的，具有自动微分的能力，能够高效地执行大规模数值运算。
        普通数组（如Python列表）：是一种动态大小的数组，可以存储任意类型的对象。虽然灵活性高，但在数值计算方面效率较低，不适合进行大规模或高性能的数学运算。
        NumPy中的ndarray：是一个同构的多维数组对象，所有元素必须是相同类型。它被设计用于高效的数值计算，并提供了大量的函数库来进行复杂的数学运算。
    功能特性
        PyTorch Tensor
        支持自动微分，这对于训练神经网络模型至关重要。
        可以轻松地在CPU和GPU之间切换，利用硬件加速提高性能。
        提供了丰富的API，便于构建和训练深度学习模型。
    NumPy ndarray
        不支持自动微分，主要用于科学计算和数据分析。
        高效的内存管理和优化过的计算能力使其成为处理大量数值数据的理想选择。
        广泛应用于各种需要数值计算的领域，比如机器学习、图像处理等。
    普通数组（如Python列表）
        主要用于通用编程任务，不特别针对数值计算优化。
        因其灵活性，适用于需要动态大小和多样化的数据类型的场景。
    互操作性
        PyTorch与NumPy之间的转换非常方便：
        可以通过.numpy()方法将一个PyTorch张量转换为NumPy的ndarray。
        使用torch.from_numpy()可以从NumPy的ndarray创建一个PyTorch张量。
        这种紧密的集成使得可以在两个库之间无缝地转移数据，同时利用各自的优势。
        PyTorch张量与普通数组的关系：
        PyTorch张量不能直接由普通数组创建，但可以通过先将普通数组转换为NumPy的ndarray，再从ndarray创建张量的方式间接实现。
    总结
        PyTorch Tensor 和 NumPy ndarray 都是为了高效处理数值数据而设计的，但前者更侧重于深度学习应用，特别是其对自动微分的支持，使得它非常适合构建和训练神经网络模型。
        普通数组（如Python列表） 则更加灵活，适合一般用途的编程任务，但对于大规模数值计算来说，效率不如前两者。
        PyTorch Tensor 和 NumPy ndarray 在很多情况下可以直接互相转换，这使得开发者可以根据具体需求选择最合适的工具，或者结合两者的优势来解决问题。
    """

    x_data = torch.tensor(data)
    print(x_data)
    return x_data


def fun2():
    """
    [[1, 2], [3, 4]]
    [[1 2]
     [3 4]]
    tensor([[1, 2],
            [3, 4]], dtype=torch.int32)
    """
    print(data)
    np_array = np.array(data)
    print(np_array)
    x_np = torch.from_numpy(np_array)
    print(x_np)
    return x_np


def fun3(x_data):
    x_ones = torch.ones_like(x_data)
    print(f"Ones Tensor:\n {x_ones} \n")


def fun4(x_data):
    x_rand = torch.rand_like(x_data, dtype=torch.float)
    print(f"Random Tensor:\n {x_rand} \n")


def fun5():
    shape = (3, 4,)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)
    print(f"Random Tensor:\n {rand_tensor} \n")
    print(f"Ones Tensor:\n {ones_tensor} \n")
    print(f"Zeros Tensor:\n {zeros_tensor} \n")


def fun6():
    m = torch.ones(5, 3, dtype=torch.double)
    n = torch.rand_like(m, dtype=torch.float)
    print(m)
    print(n)
    print(m.size())


def fun7():
    # 均匀分布
    # 只会输出0,1之间的小数
    print(torch.rand(5, 3))
    # 如果你需要不同范围的随机数 [a,b)
    a = 3
    b = 5
    print(torch.rand(3, 4) * (b - a) + a)


def fun8():
    # 标准正态分布
    print(torch.randn(5, 3))
    # 离散正态分布
    # mean 均值  std 标准差
    print(torch.normal(mean=2.0, std=3.0, size=(5, 3)))
    # 线性间隔向量(返回⼀个1维张量，包含在区间start和end上均匀间隔的steps个点)
    print(torch.linspace(start=5, end=7, steps=7))


def fun9():
    tensor = torch.rand(3, 4)
    # print(f"Shape of tensor: {tensor.shape}")
    # print(f"Datatype of tensor: {tensor.dtype}")
    # print(f"Device tensor is stored on: {tensor.device}")
    return tensor


def fun10(tensor):
    print(tensor)
    if torch.cuda.is_available():
        print("1")
        device = torch.device("cuda")
        tensor = tensor.to(device)
        print(tensor)
        print(tensor.device)


def func11():
    tensor = torch.eye(4, 4)
    print('First row:', tensor[0])
    print('Second column:', tensor[:, 1])
    print('Last column:', tensor[..., -1])
    tensor[:, 1] = 0
    print(tensor)


def fun12():
    a = torch.ones(3, 3)
    b = torch.ones(3, 3)
    print(a)
    print(b)
    t1 = torch.cat([a, b], dim=1)
    print(t1)


def fun13():
    """
        矩阵乘法 (@, matmul)：
        用于计算两个张量的矩阵乘积。
        结果是一个新的张量，其形状取决于输入张量的形状。
        通常用于线性代数中的矩阵运算。
        逐元素乘法 (*, mul)：
        用于计算两个张量的逐元素乘积。
        结果是一个新的张量，其形状与输入张量相同。
        通常用于广播操作和逐元素运算。

        相同点
        数据结构：两者都作用于矩阵（或者更一般地说，张量），即都是针对二维数组（或多维数组）进行操作。
        计算结果类型：两种运算的结果都是一个新的矩阵或张量，其元素由特定规则计算得出。
        不同点
        定义与计算方式：
        矩阵乘法：遵循线性代数中的矩阵乘法规则，其中结果矩阵的每个元素是第一个矩阵的一行与第二个矩阵的一列对应相乘后的和.
        对于两个矩阵 A和B,如果A是m x n矩阵,B是n x p矩阵,则结果矩阵是 m x p 的矩阵,矩阵乘法不是交换的, 即通常AB 不等于BA
        逐元素乘法：也称为哈达玛积（Hadamard Product），指的是两个矩阵中对应位置的元素相乘。两个参与运算的矩阵必须具有相同的尺寸（或其中之一可以广播到另一个的尺寸）。
        逐元素乘法是交换的，意味着对相同尺寸的矩阵A和B,有A X B = B x A

        维度要求：
        矩阵乘法：第一个矩阵的列数必须等于第二个矩阵的行数。
        逐元素乘法：两个矩阵的维度必须完全相同，或者其中一个可以被广播以匹配另一个矩阵的形状。
    """
    tensor = torch.ones(3, 3)
    print(tensor)
    y1 = tensor @ tensor.T
    y2 = tensor @ tensor.T
    y3 = torch.rand_like(tensor)
    a = torch.matmul(tensor, tensor.T, out=y3)
    print(y1)
    print(y2)
    print(y3)
    print(a)


def func14():
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
    agg = tensor.sum()
    print(type(agg))
    agg_item = agg.item()
    print(agg_item, type(agg_item))


def func15():
    """
    In-place操作
    把计算结果存储到当前操作数中的操作就称为就地操作。含义和pandas中inPlace参
    数的含义⼀样。pytorch中，这些操作是由带有下划线_后缀的函数表示
    In-place操作虽然节省了⼀部分内存，但在计算导数时可能会出现
    问题，因为它会⽴即丢失历史记录。因此，不⿎励使⽤它们。
    """
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(tensor, "\n")
    tensor.add_(5)
    print(tensor)


def func16():
    t = torch.ones(5)
    print(f"t: {t}")
    n = t.numpy()
    print(f"n: {n}")
    np.add(n, 1, out=n)
    print(f"n: {n}")
    """
    t: tensor([1., 1., 1., 1., 1.])
    n: [1. 1. 1. 1. 1.]
    n: [2. 2. 2. 2. 2.]
    """


def func17():
    """
    NumPy 数组的变化也反映在张量中。
    """
    n = np.ones(5)
    t = torch.from_numpy(n)
    np.add(n, 1, out=n)
    print(f"t: {t}")
    print(f"n: {n}")


def func18():
    # 定义矩阵 A，向量 b 和常数 c
    A = torch.randn(10, 10, requires_grad=True)
    b = torch.randn(10, requires_grad=True)
    c = torch.randn(1, requires_grad=True)
    x = torch.randn(10, requires_grad=True)

    # 计算 x^T * A + b * x + c
    result = torch.matmul(A, x.T) + torch.matmul(b, x) + c
    # ⽣成计算图节点
    dot = make_dot(result, params={'A': A, 'b': b, 'c': c, 'x': x})
    # 绘制计算图
    dot.render('expression', format='png', cleanup=True, view=False)


def fun19():
    # 定义矩阵 A，向量 b 和常数 c
    A = torch.randn(10, 10, requires_grad=True)
    b = torch.randn(10, requires_grad=True)
    c = torch.randn(1, requires_grad=True)
    x = torch.randn(10, requires_grad=True)  # 确保 x 是一个二维张量，形状为 (10,)

    # 计算 x^T * A + b * x + c
    # 将 x 转换为形状 (10, 1)，以进行矩阵乘法
    x_column = x.unsqueeze(1)  # 转换为形状 (10, 1)

    # 进行矩阵乘法，结果将是形状 (10, 1)
    Ax_result = torch.matmul(A, x_column)

    # b * x 部分可以直接使用点积，因为 b 和 x 都是一维向量
    bx_result = torch.matmul(b.unsqueeze(0), x_column).squeeze()  # 结果是一维向量

    # 注意c的维度，这里假设c是一个标量，因此我们直接广播它
    result = Ax_result.squeeze() + bx_result + c.squeeze()

    # 生成计算图节点
    dot = make_dot(result, params={'A': A, 'b': b, 'c': c, 'x': x})
    # 绘制计算图
    dot.render('expression', format='png', cleanup=True, view=False)

def func20():
    # 定义矩阵 A，向量 b 和常数 c
    A = torch.randn(10, 10, requires_grad=True)
    b = torch.randn(10, requires_grad=True)
    c = torch.randn(1, requires_grad=True)
    x = torch.randn(10, requires_grad=True)

    # 计算 x^T * A + b * x + c
    result = torch.matmul(A, x.T) + torch.matmul(b, x) + c
    # ⽣成计算图节点
    dot = make_dot(result, params={'A': A, 'b': b, 'c': c, 'x': x})
    # 绘制计算图
    dot.render('expression', format='png', cleanup=True, view=False)


if __name__ == '__main__':
    # fun4(func1())
    # fun10(fun9())
    # print(torch.version.cuda)
    func18()

###{二}、Pytorch基础
#张量Tensor是pytorch的基本单位，
import numpy as np
import torch

#张量可以多种方式初始化
#（1）直接从数据
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
print(x_data)
#输出值
# tensor([[1, 2],
#         [3, 4]])

#（2）从Numpy数组
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)
#输出值
# tensor([[1, 2],
#         [3, 4]])

#（3）从另一个张量
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")
x_rand = torch.rand_like(x_data, dtype=torch.float) # 覆盖 x_data的数据类型
print(f"Random Tensor: \n {x_rand} \n")
#输出值
# Ones Tensor:
#  tensor([[1, 1],
#         [1, 1]])
# Random Tensor:
#  tensor([[0.2686, 0.9931],
#         [0.5184, 0.2622]])

#使⽤随机值或常量值：
#shape 是张量维度的元组。在下⾯的函数中，它决定了输出张量的维度。
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
#输出值
# Random Tensor:
#  tensor([[0.3549, 0.8814, 0.8400],
#         [0.7853, 0.3545, 0.3591]])
# Ones Tensor:
#  tensor([[1., 1., 1.],
#         [1., 1., 1.]])
# Zeros Tensor:
#  tensor([[0., 0., 0.],
#         [0., 0., 0.]])

# 其它⼀些创建⽅法
# 基于现有tensor构建，但使⽤新值填充
m = torch.ones(5,3, dtype=torch.double)
n = torch.rand_like(m, dtype=torch.float)
print("m",m)
print("n",n)
#输出值
# m tensor([[1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.]], dtype=torch.float64)
# n tensor([[0.0998, 0.8726, 0.7550],
#         [0.9651, 0.6263, 0.9851],
#         [0.3601, 0.5613, 0.1972],
#         [0.2909, 0.4504, 0.9120],
#         [0.3802, 0.9488, 0.8895]])

#获取tensor的⼤⼩
print(m.size()) # torch.Size([5,3])
#输出值 torch.Size([5, 3])

# 均匀分布
a = torch.rand(5,3)
print("a",a)
# 输出值
# tensor([[0.5049, 0.4567, 0.9965],
#         [0.5732, 0.2231, 0.4051],
#         [0.5365, 0.4532, 0.2709],
#         [0.6969, 0.5965, 0.5304],
#         [0.2952, 0.3445, 0.7441]])

# 标准正态分布
a = torch.randn(5,3)
print("a",a)
#输出值
# a tensor([[ 0.6188, -1.3275, -0.8087],
#         [ 0.4096, -1.5233, -0.3719],
#         [-1.0376,  0.8413, -1.6085],
#         [ 1.9380, -1.8818, -0.4801],
#         [ 1.4482,  0.8509, -1.5715]])

# 离散正态分布
a = torch.normal(mean=.0,std=1.0,size=(5,3))
print("a",a)
#输出值
# a tensor([[-1.4560,  0.8209, -0.8476],
#         [-0.8810,  0.6552, -0.1683],
#         [-0.7489, -0.5633,  1.1053],
#         [ 0.5742, -0.3126,  1.6188],
#         [ 0.6806, -0.7273,  1.0687]])

# 线性间隔向量(返回⼀个1维张量，包含在区间start和end上均匀间隔的steps个点)
a = torch.linspace(start=1,end=10,steps=20)
print("a",a)
#输出值
# a tensor([ 1.0000,  1.4737,  1.9474,  2.4211,  2.8947,  3.3684,  3.8421,  4.3158,
#          4.7895,  5.2632,  5.7368,  6.2105,  6.6842,  7.1579,  7.6316,  8.1053,
#          8.5789,  9.0526,  9.5263, 10.0000])

#
#张量的属性
#张量的属性描述了张量的形状、数据类型和存储它们的设备。以对象的⻆度来判断，
#张量可以看做是具有特征和⽅法的对象。
tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
#输出值
# Shape of tensor: torch.Size([3, 4])
# Datatype of tensor: torch.float32
# Device tensor is stored on: cpu

#张量运算
# 设置张量在GPU上运算
# if torch.cuda.is_available():
#  tensor = tensor.to('cuda')

#张量的索引和切⽚：
tensor = torch.ones(4, 4)
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])
tensor[:, 1] = 0
print(tensor)
#输出值
# First row:  tensor([1., 1., 1., 1.])
# First column:  tensor([1., 1., 1., 1.])
# Last column: tensor([1., 1., 1., 1.])
# tensor([[1., 0., 1., 1.],
#         [1., 0., 1., 1.],
#         [1., 0., 1., 1.],
#         [1., 0., 1., 1.]])

#张量的拼接
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print("t1",t1)
#输出值
# t1 tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
#         [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
#         [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
#         [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])

#算术运算
# 计算两个张量之间矩阵乘法的⼏种⽅式。 y1, y2, y3 最后的值是⼀样的 dot
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)
x1 = torch.matmul(tensor, tensor.T, out=y3)
print("y3",y3)
print("x1",x1)
#输出值
# y3 tensor([[3., 3., 3., 3.],
#         [3., 3., 3., 3.],
#         [3., 3., 3., 3.],
#         [3., 3., 3., 3.]])
# x1 tensor([[3., 3., 3., 3.],
#         [3., 3., 3., 3.],
#         [3., 3., 3., 3.],
#         [3., 3., 3., 3.]])

# 计算张量逐元素相乘的⼏种⽅法。 z1, z2, z3 最后的值是⼀样的。
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
x2 = torch.mul(tensor, tensor, out=z3)
print("z3",z3)
print("x2",x2)
#输出值
# z3 tensor([[1., 0., 1., 1.],
#         [1., 0., 1., 1.],
#         [1., 0., 1., 1.],
#         [1., 0., 1., 1.]])
# x2 tensor([[1., 0., 1., 1.],
#         [1., 0., 1., 1.],
#         [1., 0., 1., 1.],
#         [1., 0., 1., 1.]])

#单元素张量
#如果⼀个单元素张量，例如将张量的值聚合计算，可以使⽤ item() ⽅法将其转换为Python 数值
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))
#输出值 12.0 <class 'float'>

#In-place操作
#把计算结果存储到当前操作数中的操作就称为就地操作。含义和pandas中inPlace参数的含义⼀样。pytorch中，这些操作是由带有下划线 _ 后缀的函数表⽰。
#例如：x.copy_(y) ,  x.t_() , 将改变 x ⾃⾝的值。
print(tensor, "\n")
tensor.add_(5)
print(tensor)
#输出值
# tensor([[1., 0., 1., 1.],
#         [1., 0., 1., 1.],
#         [1., 0., 1., 1.],
#         [1., 0., 1., 1.]])
#
# tensor([[6., 5., 6., 6.],
#         [6., 5., 6., 6.],
#         [6., 5., 6., 6.],
#         [6., 5., 6., 6.]])

#与numpy之间的转换
#CPU 和 NumPy 数组上的张量共享底层内存位置，所以改变⼀个另⼀个也会变。
#张量到numpy数组
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
#输出值
# t: tensor([1., 1., 1., 1., 1.])
# n: [1. 1. 1. 1. 1.]

#张量值的变更也反映在关联的NumPy 数组中
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
#输出值
# t: tensor([2., 2., 2., 2., 2.])
# n: [2. 2. 2. 2. 2.]

#Numpy数组到张量
n = np.ones(5)
t = torch.from_numpy(n)
print("t",t)
#输出值
# t tensor([1., 1., 1., 1., 1.], dtype=torch.float64)

#NumPy 数组的变化也反映在张量中。
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
#输出值
# t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
# n: [2. 2. 2. 2. 2.]

#计算图
#pytorch计算图可视化
#通过torchviz可以实现
# import torch
import graphviz
import torchviz
from torchviz import make_dot
# 定义矩阵 A，向量 b 和常数 c
A = torch.randn(10, 10,requires_grad=True)
b = torch.randn(10,requires_grad=True)
c = torch.randn(1,requires_grad=True)
x = torch.randn(10, requires_grad=True)
print("A",A)
print("b",b)
print("c",c)
print("x",x)
#输出值
# A tensor([[ 0.9331,  0.6759, -0.0877, -0.0534, -1.2935,  0.1262, -2.1144,  1.1065,
#           0.7249,  0.9951],
#         [ 2.7631, -0.2451,  0.7087,  1.6774, -1.1675,  0.0693,  0.7591,  1.3173,
#           2.6567,  1.2660],
#         [ 0.5251,  0.2429, -0.7191,  0.5479, -0.1507, -0.9737, -0.4647,  0.0316,
#           0.5199, -0.6124],
#         [ 0.8462,  0.1660, -0.0279, -0.5222,  0.9925, -2.8333, -1.8784,  1.1686,
#           0.0824,  1.0979],
#         [ 1.0962,  0.3176, -0.2604, -1.4845, -0.5045,  1.2529, -0.0152,  0.6926,
#           0.2340,  0.6921],
#         [-0.5592,  0.2817,  0.8876,  1.0750,  0.3113,  0.3966,  0.1435, -1.2585,
#           0.0179,  1.3115],
#         [-0.0566,  0.4537,  1.6564, -0.5044,  0.6762, -0.3272, -1.4363,  0.8576,
#          -0.1096,  1.4060],
#         [-0.7549,  0.3584, -0.0052, -0.9454, -1.3645,  0.2563,  0.8793,  0.1399,
#           0.2976,  0.8728],
#         [ 2.6459,  1.1079, -0.6104,  0.2920, -0.4279, -1.0213, -0.1578,  1.9906,
#          -1.3653,  0.5285],
#         [ 0.1055,  0.4245,  0.4856, -1.4940, -1.2829, -0.3250, -0.8547, -0.9518,
#           2.0701,  0.6376]], requires_grad=True)
# b tensor([ 0.9966,  1.2854,  1.0467,  0.8371, -0.7971,  0.3061, -0.2665,  0.3681,
#          0.1138, -0.5625], requires_grad=True)
# c tensor([-0.2681], requires_grad=True)
# x tensor([-0.6175,  1.7993,  0.0156, -0.2289, -0.2530,  0.2682,  0.8355, -0.5337,
#          0.3260, -1.1437], requires_grad=True)

# 计算 x^T * A + b * x + c
# result = torch.matmul(A, x.T) + torch.matmul(b, x) + c
# print("result",result)
#输出值
# result tensor([ 1.4043, -0.6950,  2.7171, -4.9846, -3.6220, -4.2557, -2.8987, -0.9236,
#          2.2134,  1.2241], grad_fn=<AddBackward0>)

# ⽣成计算图节点
# dot = make_dot(result, params={'A': A, 'b': b, 'c': c, 'x': x})
# 绘制计算图
# dot.render('expression', format='png', cleanup=True, view=False)
#生成图片： expression.png

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
print("sigmoid(2)")
print(sigmoid(2))
#输出值
# sigmoid(2)
# 0.8807970779778823

#损失函数
def loss_function(y, y_hat):
    e = 1e-8 # 防⽌y_hat计算值为0，添加的极⼩值epsilon
    return - y * np.log(y_hat + e) - (1 - y) * np.log(1 - y_hat + e)

print("Loss function(2,0.1)")
print(loss_function(2,0.1))
#输出值
# Loss function(2,0.1)
# 4.499809481441386

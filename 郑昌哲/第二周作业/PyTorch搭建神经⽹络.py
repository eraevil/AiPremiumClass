#PyTorch搭建神经⽹络

#1.数据预处理
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
#2.
# 下载的数据集⽂件会保存到当前⽤⼾⼯作⽬录的data⼦⽬录中。
# 如果不想下载后就找不到了，建议修改root参数的值。
# 例如"D:\\datasets\\fashionMNIST\\"⼀类的绝对路径
training_data = datasets.FashionMNIST(
 root="data",
 train=True,
 download=True,
 transform=ToTensor(),
)
# 测试集也需要下载，代码和上⾯⼀样。但参数train=False代表不是训练集(逻辑取反
test_data = datasets.FashionMNIST(
 root="data",
 train=False,
 download=True,
 transform=ToTensor(),
)

#下⼀步就是对已加载数据集的封装，把Dataset 作为参数传递给 DataLoader。这样，就在我们的数据集上包装了⼀个迭代器(iterator)，这个迭代器还⽀持⾃动批处理、
# 采样、打乱顺序和多进程数据加载等这些强⼤的功能。这⾥我们定义了模型训练期间，每个批次的数据样本量⼤⼩为64，即数据加载器在迭代中，每次返回⼀批 64 个数据
# 特征和标签。

batch_size = 64
# 创建数据加载器
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
# 测试数据加载器输出
for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

#输出值
# Shape of X [N, C, H, W]:  torch.Size([64, 1, 28, 28])
# Shape of y:  torch.Size([64]) torch.int64

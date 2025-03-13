import numpy as np
# 创建ndarray数组
arr1 = np.array([1,2,3,4,5], float)
arr2 = np.zeros((2,3), dtype=np.float32)
arr3 = np.arange(1,6, 0.3)
arr4 = np.eye(3)
arr5 = np.random.random(5)
arr6 = np.random.normal(0, 0.1, 5)  
arr7 = np.array([(1,2), (3,4), (5,6)])
for i,j in arr7:
    print(i,j)
arr8 = np.array([(1,2), (3,4), (5,6)])
arr8 = arr8[:,:,np.newaxis] 
print(arr8.shape)

a = np.array([(1,2,3), (4,5,6), (7,8,9)])
print("ndim:", a.ndim)
print("shape:", a.shape)
print("size", a.size)
print("dtype", a.dtype)

a2 = np.arange(1,10)
print(a2)
print(a2.shape)

a2 = a2.reshape(3,3,1)  
print(a2)
print(a2.shape)  

a3 = np.array([5,3,2,1.3,8.9,7.71])
print("mean:",a3.mean())
print("var:", a3.var())
print("std:", a3.std())
print("argmax:", a3.argmax()) # 最大值索引
print("argmin:", a3.argmin()) # 最小值索引
print("ceil:", np.ceil(a3))  # 向上取整
print("floor:", np.floor(a3)) # 向下取整
print("rint:", np.rint(a3))  # 四舍五入

a4 = np.array([16,31,12,28,22,31,48])
a4.sort()  # 排序

import torch

data = torch.tensor([[1,2],[3,4]], dtype=torch.float32)
print(data)

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# 检查pytorch是否支持GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    tensor = tensor.to(device)

print(tensor)
print(tensor.device)

# mac上没有GPU，使用M系列芯片
if torch.backends.mps.is_available():
    device = torch.device("mps")
    tensor = tensor.to(device)

print(tensor)
print(tensor.device)

tensor = torch.ones(3, 3)
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])
tensor[:,1] = 0
print(tensor)

import torch
from torchviz import make_dot

a = torch.randn(10, 10,requires_grad=True)  
b = torch.randn(10,requires_grad=True)
c = torch.randn(1,requires_grad=True)
x = torch.randn(10, requires_grad=True)


result = torch.matmul(A, x.T) + torch.matmul(b, x) + c

dot = make_dot(result, params={'A': A, 'b': b, 'c': c, 'x': x})

dot.render('expression', format='png', cleanup=True, view=False)

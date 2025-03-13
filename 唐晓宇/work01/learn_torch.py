import numpy as np
import torch
def torch_data_attrib(a):
    print(f"data of tensor: \n{a}\n")
    print("shape of tensor:", a.shape)
    print("Datatype of tensor:", a.dtype)
    print("Device of tensor:", a.device)

# 数组切张量
data = [[1,3],[3,6]]
x_data = torch.tensor(data)
print(data)

# 创建一个 2x3 的全 0 张量
a = torch.zeros(2, 3)
print(a)

# 创建一个 2x3 的全 1 张量
b = torch.ones(2, 3)
print(b)

# 创建一个 2x3 的随机数张量
c = torch.randn(2, 3)
print(c)

# 从 NumPy 数组创建张量
import numpy as np
numpy_array = np.array([[1, 2], [3, 4]])
tensor_from_numpy = torch.from_numpy(numpy_array)
print(tensor_from_numpy)
# 从另一个张量
x_ones = torch.ones_like(x_data)
print(x_ones)

x_rand = torch.rand_like(x_data,dtype=torch.float)
print(x_rand)
shape  = [2,1,3]
rand_tesor = torch.rand(shape)
print(f"Random Tensor:\n {rand_tesor}\n")
ones_tensor = torch.ones(shape)
print(f"Ones_Tensor:\n {ones_tensor}\n")
zeros_tensor = torch.zeros(shape)
print(f"Zeros Tensor:\n {zeros_tensor}\n")
torch_data_attrib(x_rand)
tensor = torch.rand(3,4)
if torch.cuda.is_available():
    tesor = tensor.to("cuda")
    torch_data_attrib(tesor)
# # 在指定设备（CPU/GPU）上创建张量
device = torch.device("cuda")
d = torch.randn(2, 3, device=device)
torch_data_attrib(d)

tensor = torch.randn(4,4)
torch_data_attrib(tensor)

print("First column:", tensor[:,0])
print("多个 columns:", tensor[:,1:])
print("last columns:", tensor[...,-1])
print("last columns:", tensor[:,-1])

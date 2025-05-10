import torch

# 创建一个随机张量
tensor = torch.rand(2, 2)
print(tensor)                  

# 输出：
print("Shape of tensor:", tensor.shape)
print("Data type of tensor:", tensor.dtype)
print("Device tensor is stored on:", tensor.device)
print("Number of elements in tensor:", tensor.numel())
print("Minimum value in tensor:", tensor.min())
print("Maximum value in tensor:", tensor.max())
print("Mean value in tensor:", tensor.mean())
print("Standard deviation of tensor:", tensor.std())
print("Sum of all elements in tensor:", tensor.sum())
print("Transpose of tensor:", tensor.t())
print("Transpose of tensor:", tensor.transpose(0, 1))

import torch

tensor = torch.arange(1, 10, dtype=torch.float32)
print(tensor)

tensor = torch.arange(1, 10, dtype=torch.float32).reshape(3, 3)
print(tensor)
print(tensor.T)

# 计算两个张量之间逐元素相乘的几种方式
# 方式1
z1 = tensor * tensor
print(z1)

# 方式2
z2 = tensor.mul(tensor)
print(z2)

# 方式3
z3 = torch.mul(tensor, tensor)
print(z3)

# 方式4
z4 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z4)
print(z4)

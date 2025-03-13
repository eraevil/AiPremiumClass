import torch
import numpy as np

tensor = torch.ones(5)
print(f"tensor: {tensor}")

array = tensor.numpy()
print(f"array: {array}")

# 张量值的改变会影响到numpy数组的值
tensor.add_(1)

print(f"tensor: {tensor}")
print(f"array: {array}")


import torch

tensor = torch.arange(1, 10, dtype=torch.float32)
print(tensor)

tensor.add_(2)
print(tensor)

# in-place操作虽然节省了一部分内存，但在计算导数时可能会出现问题，
# 因为它会立即丢失历史记录。
# 因此，不鼓励使用in-place操作。

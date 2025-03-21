import torch
import time
from datetime import datetime

t1 = time.perf_counter()
dt1 = datetime.fromtimestamp(time.time())
print(dt1.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " Start to run torch_tensor_math.py")

tensor = torch.arange(1, 10, dtype=torch.float32)
print(tensor)

tensor = torch.arange(1, 10, dtype=torch.float32).reshape(3, 3)
print(tensor)
print(tensor.T)

# 计算两个张量之间矩阵乘法的几种方式
# 方式1
y1 = tensor @ tensor.T
print(y1)

# 方式2
y2 = tensor.matmul(tensor.T)
print(y2)

# 方式3
y3 = torch.mm(tensor, tensor.T)
print(y3)

# 方式4
y4 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y4)
print(y4)

t2 = time.perf_counter()
dt2 = datetime.fromtimestamp(time.time())
print(dt2.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " End to run torch_tensor_math.py")
print("Time used: " + str(t2 - t1) + "s")

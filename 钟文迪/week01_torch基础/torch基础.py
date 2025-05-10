import torch
from torchviz import make_dot
import numpy as np

data = [[12, 22], [33, 47]]
x_data = torch.tensor(data)
print(x_data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")

shape = (5, 7, )
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")

m = torch.ones(2, 2, dtype=torch.double)
print(m)
print(m.size())
print(torch.rand(2, 2))
print(torch.randn(2, 2))
print(torch.normal(mean = .0, std = 1.0, size=(4, 4)))
print(torch.linspace(start = 2, end = 10, steps = 10))

tensor = torch.rand(3, 4)
print(tensor)
print(tensor.shape)
print(tensor.dtype)
print(tensor.device)
print(tensor.is_cuda)

print(torch.cuda.is_available())

a = np.array([[1, 2, 3], [4, 5, 6]])
data = torch.from_numpy(a)
print(data)
print(data[0])
print(data[:, 0])
print(data[..., -1])
data[:, 1] = 0
print(data)

t1 = torch.cat([data, data, data], dim=1)
print(t1)

data = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
y1 = data @ data.T
print(y1)
y2 = data.matmul(data.T)
print(y2)
y3 = torch.rand_like(data)
print(y3)
torch.matmul(data, data.T, out=y3)
print(y3)

z1 = data * data
print(z1)
z2 = data.mul(data)
print(z2)
z3 = torch.rand_like(data)
print(z3)
torch.mul(data, data, out=z3)
print(z3)

agg = data.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

data.add_(2)
print(data)

t = torch.ones(9)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")


A = torch.randn(10, 10,requires_grad=True)
b = torch.randn(10,requires_grad=True)
c = torch.randn(1,requires_grad=True)
x = torch.randn(10, requires_grad=True)
result = torch.matmul(A, x.T) + torch.matmul(b, x) + c
dot = make_dot(result, params={'A': A, 'b': b, 'c': c, 'x': x})

dot.render('expression', format='png', cleanup=True, view=False)

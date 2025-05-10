import numpy as np

#basic array creation
a = [3,4,5]
b = np.array(a)
c = np.array([6,7,8], float)
d = np.array([[1,2,3],[4,5,6],[7,8,9]])

#special array creation
a = np.zeros((3,2), dtype = int)
a = np.ones((4, 4), dtype = int)
a = np.arange(8, 16, 0.2)
a = np.eye(4)
a = np.random.random(4)
a = np.random.normal(1, 0.2, 6)

#access numparry
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a[0, 1])

#traverse numparry
a = np.array([1,2,3,4])
for i in a:
    print(i)

a = np.array([(7,8),(5,6) , (3,4) , (1,2)])
for i,j in a:
    print(i*j)


#useful properties of numpyarray
a = np.array([[(1,2,3) , (4,5,6)] , [(7,8,9),(10,11,12)]])
print("ndim:", a.ndim)
print("shape:", a.shape)
print("size", a.size)
print("dtype", a.dtype)

#basic operations of numpyarra
a = np.array([(1,2) , (3,4)])
print(3 in a)
print(5 in a)

a = np.zeros((2,2))
a.reshape(4)

a = np.array([(1,2,3) , (4,5,6) , (7,8,9)])
a.T
a.flatten()

a = np.array([[1,2,3],[4,5,6]])
a.shape
a = a[ :, np.newaxis]
a.shape

#methematic operations
a = np.ones((3,3))
b = np.array([(- 1,1,1) , (1,1,2), (2,2,3)])
print(a+b)
print(a-b)
print(a*b)
print(a/b)

a = np.array([1,2,1])
a.sum()
a.prod()

a = np.array([4,5,6])
print("mean:",a.mean())
print("var:", a.var())
print("std:", a.std())
print("max:", a.max())
print("min:", a.min())

a = np.array([1.6, 4.8, 6.9])
print("argmax:", a.argmax())
print("argmin:", a.argmin())
print("ceil:", np.ceil(a))
print("floor:", np.floor(a))
print("rint:", np.rint(a))

a = np.array([16,31,12,28,22,31,48])
a.sort()
a


m1 = np.array([[1, 2] , [3, 4]] , dtype=np.float32)
m2 = np.array([[5, 6] , [7, 8]] , dtype=np.float32)
print(m1.dot(m2))
print(m1@m2)

#numpy broadcast mechanism
a = np.array([(1,2) , (2,2) , (3,3) , (4,4)])
b = np.array([-1,1])
a+b

#pytorch basic
import torch
data = [ [1, 2] , [3, 4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)
x_ones = torch.zeros_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")

shape = (3,4)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

m = torch.ones(5,3, dtype=torch.double)
n = torch.rand_like(m, dtype=torch.float)
print(m.size())

print(torch.rand(5,3))
print(torch.randn(5,3))
print(torch.normal(mean=.0,std=1.0,size=(5,3)))
print(torch.linspace(start=2,end=6,steps=20))

tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

if torch.cuda.is_available(): 
    tensor = tensor.to('cuda')
    print("Tensor is on GPU")
else:
    print("Tensor is on CPU")

tensor = torch.ones(2, 2)
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., - 1])
tensor[:,1] = 0
print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)

z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

print(tensor, "\n")
tensor.add_( 5)
print(tensor)

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

n = np.ones(2)
t = torch.from_numpy(n)
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

#graphs computation
import torch
from torchviz import make_dot

# 定义矩阵 A，向量 b 和常数 c
A = torch.randn(4, 4,requires_grad=True)
b = torch.randn(4,requires_grad=True)
c = torch.randn(1,requires_grad=True)
x = torch.randn(4, requires_grad=True)

# 计算 x^T * A + b * x + c
result = torch.matmul(A, x.T) 
torch.matmul(b, x) + c

# 生成计算图节点
dot = make_dot(result, params={'A': A, 'b': b, 'c': c, 'x': x})
# 绘制计算图
dot.render('expression', format='png', cleanup=True, view=False)

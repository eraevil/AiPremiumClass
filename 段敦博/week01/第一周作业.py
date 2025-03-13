#第一部分numpy_code

import torch

torch.__version__
import numpy as np

np.array([1,2])
import numpy as np
arr = np.array([1,2,3,4,5],float)
arr
a=np.array([(1,2,3),(4,5,6,),(7,8,9)])
a
a1=np.zeros((2,3),dtype=np.float32)
a1
a2=np.arange(1,6,0.2)
a2
a3=np.eye(3)
a3
a4=np.random.random(5)
a4
a5=np.random.normal(0,0.1,5)
a5
a6=np.array([(1,2),(3,4),(5,6)])
print(a6[:,1 ])
a7=np.array([(1,2),(3,4),(5,6)])
#i,j=a7[0]
for i,j in a7:
    print(i,j)
  a = np.array([(1,2,3), (4,5,6), (7,8,9)])
print("ndim:", a.ndim)
print("shape:", a.shape)
print("size:", a.size)
print("dtype:", a.dtype)
a7 = np.array([(1,2), (3,4), (5,6)])
# i,j = a7[0]
for i,j in a7:
    print(i,j)
  print(3 in a)
a7=np.arange(1,10)
print(a7)
print(a7.shape)

a7=a7.reshape(3,3,1)
print(a7)
print(a7.shape)
print(a)

a=a.T
print(a)
a=a.flatten()
print(a)
a8=np.array([(1,2),(3,4),(5,6)])
a8=a8[:,:,np.newaxis]
a8.shape
a = np.ones((2,2))
b = np.array([(-1,1),(-1,1)])
print(a)

print(b)

print(a+b)

print(1-b)
a.sum()
a.prod()
a = np.array([1.02, 3.8, 4.9])
print("argmax:",a.argmax())
print("argmin:",a.argmin())

print("ceil:", np.ceil(a)) 
print("floor:", np.floor(a)) 
print("rint:", np.rint(a)) 
a = np.array([16,31,12,28,22,31,48])
a.sort()
print(a)
import torch
tensor=torch.arange(1,10,dtype=torch.float32).reshape(3,3)
print(tensor.dtype)

y1=tensor@tensor.T
y2=tensor.matmul(tensor.T)

y3=torch.rand_like(tensor)
y=torch.matmul(tensor,tensor.T,out=y3)

print(y1)
print(y2)
print(y3)

z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

print(z1)
print(z2)
print(z3)
a9 = np.array([[1,2,3],[4,5,6]])
b9 = np.array([[4,5,6],[7,8,9]])

a9 * b9
import numpy as np
m1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
m2 = np.array([[5, 6], [7, 8]], dtype=np.float32)
print(m1)
print(m2)
result_dot = np.dot(m1, m2)
print(result_dot)
result_at = m1 @ m2
print(result_at)
manual_result = np.zeros((m1.shape[0], m2.shape[1]), dtype=np.float32)
for i in range(m1.shape[0]):
    for j in range(m1.shape[0]):
        manual_result[i, j] = 0
        for k in range(m1.shape[1]):
            print(f"{m1[i, k]} * {m2[k, j]} = {m1[i, k] * m2[k, j]}")
            manual_result[i, j] += m1[i, k] * m2[k, j]
        print(f"结果矩阵[{i+1},{j+1}]:{manual_result[i, j]}\n")
print(manual_result)
np.save('result.npy',manual_result)
result_np = np.load('result.npy')
result_np
a = np.array([1,2,3])
b = np.array([4,5,6])
a+b
a = np.array([(1,2), (2,2), (3,3), (4,4)])  
b = np.array([-1,1])
a+b

#第二部分pytorch_code
import torch
data = torch.tensor([[1,2],[3,4]], dtype=torch.float32)
data
import numpy as np
np_array = np.array([[1,2],[3,4]])
data2=torch.from_numpy(np_array)
data2
data2.dtype
data3=torch.rand_like(data2,dtype=torch.float)
data3

shape = (2,3,)
rand_tensor=torch.rand(shape)
ones_tensor=torch.ones(shape)
zeros_tensor=torch.zeros(shape)
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

m=torch.ones(5,3,dtype=torch.double)
n=torch.rand_like(m,dtype=torch.float)
print(m.size())
print(torch.rand(5,3))
print(torch.randn(5,3))
print(torch.normal(mean=.0,std=1.0,size=(5,3)))
print(torch.linspace(start=1,end=10,steps=21))

tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    tensor = tensor.to(device)
print(tensor)
print(tensor.device)

tensor = torch.ones(4, 4)
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])
tensor[:,1] = 0
print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=1)
t1
print(t1*3)
print(t1.shape)

import torch
tensor=torch.arange(1,10,dtype=torch.float).reshape(3,3)
print(tensor)

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)
print(y1)
print(y2)
print(y3)
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(z1)
print(z2)
print(z3)

agg = tensor.sum()
print(agg)
agg_item=agg.item()
print(agg_item,type(agg_item))

np_arr = z1.numpy()
np_arr

print(tensor, "\n")
tensor.add_(5)

tensor
import torch
from torchviz import make_dot
A = torch.randn(10, 10,requires_grad=True) 
b = torch.randn(10,requires_grad=True)
c = torch.randn(1,requires_grad=True)
x = torch.randn(10, requires_grad=True)
result = torch.matmul(A, x.T) + torch.matmul(b, x) + c
dot = make_dot(result, params={'A': A, 'b': b, 'c': c, 'x': x})
dot.render('expression', format='png', cleanup=True, view=False)

/tmp/ipykernel_111901/604160287.py:12: UserWarning: The use of `x.T` on tensors of dimension other than 2 to reverse their shape is deprecated and it will throw an error in a future release. Consider `x.mT` to transpose batches of matrices or `x.permute(*torch.arange(x.ndim - 1, -1, -1))` to reverse the dimensions of a tensor. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3683.)
  result = torch.matmul(A, x.T) + torch.matmul(b, x) + c
'expression.png'

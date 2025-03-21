import torch
data = torch.tensor([[1,2],(2,4)])
data

tensor = torch.arange(1,10,dtype=torch.float32).reshape(3,3)
print(tensor)

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)
torch.matmul(tensor,tensor.T,out=y3)
print(y1)
print(y2)
print(y3)

data2 = data.numpy()
print(data2)
data3 = torch.from_numpy(data2)
print(data3)

if torch.backends.mps.is_available():
   device = torch.device("mps")
   data3 = data3.to(device)
print(data3)
print(data3.device)

print(tensor)
tensor = torch.cat([tensor,tensor],dim=1)
print(tensor)

#if torch.cuba.is_available():
 #  device = torch.device("cuda")
  # data = data.to(device)
print(data)
print(data.device)

import torch 
from torchviz import make_dot

A = torch.randn(10,10, requires_grad=True)
B = torch.randn(10, requires_grad=True)  
C = torch.randn(1, requires_grad=True)
X = torch.randn(10, requires_grad=True)  

result = torch.matmul(A,X.T) + torch.matmul(B, X )+ C
dot = make_dot(result, params={"A":A, "B":B, "C":C, "X":X})

dot.render('expression', format='png', cleanup=True, view=False)
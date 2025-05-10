import torch
import pandas
import matplotlib
import numpy as np
from torchviz import make_dot

data = torch.tensor([[1, 2],[3,5]])
print(data)

ab = torch.ones(4, 4)
print(ab)
print(ab.sum().shape)

ab = torch.cat([ab, ab], dim=0)
print(ab)


aa = np.array([[1,3],[2,5],[7,8]])
a1 = torch.from_numpy(aa)
print(a1)

b1 = torch.ones_like(a1)
print(b1)

b11 = torch.rand_like(a1,dtype=torch.float32)
print(b11)

shape=(2,3)
rand_tensor = torch.rand(shape)
one_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(rand_tensor)
print(one_tensor)
print(zeros_tensor)

# 均匀分布
print(torch.rand(5,3))
# 标准正态分布
print(torch.randn(5,3))
# 离散正态分布
print(torch.normal(mean=0.1,std=0.5,size=(5,3)))
print(torch.linspace(1,10,10))

print(rand_tensor.shape)
print(rand_tensor.dtype)
print(rand_tensor.device)

if torch.cuda.is_available():
    print(torch.cuda.is_available())
    torch.cuda.empty_cache()

print(rand_tensor)
print(rand_tensor[0])
print(rand_tensor[0,1])
print(rand_tensor[:,1])
print(rand_tensor[...,-1])

cat_tensor = torch.cat([rand_tensor,one_tensor],dim=0)
print(cat_tensor)
cat_tensor = torch.cat([rand_tensor,zeros_tensor],dim=1)
print(cat_tensor)

m_tensor = rand_tensor @ one_tensor.T
print(m_tensor)
m_tensor = torch.matmul(rand_tensor,one_tensor.transpose(0,1))
print(m_tensor)

res_tensor = rand_tensor * one_tensor
print(res_tensor)
res_tensor = torch.mul(rand_tensor,one_tensor)
print(res_tensor)
sum_tensor = torch.sum(rand_tensor)
print(sum_tensor)
print(sum_tensor.item())

t = torch.ones(5)
print(t.numpy())
t.add_(1)
print(t)

a = torch.randn(10,10,requires_grad=True)
b = torch.randn(10,1,requires_grad=True)
c = torch.randn(1,1,requires_grad=True)
x = torch.randn(1,10, requires_grad=True)
# print(a)
# print(b)
# print(c)
# print(x)
# 计算 x^T * A + b * x + c
print(a)
print(x.T)
result1 = torch.matmul(a,x.T)
print(result1)
print(b)
result2 = torch.matmul(b,x)
print(result1+result2+c)

result = torch.matmul(a,x.T) + torch.matmul(b,x) + c
print(result)
dot = make_dot(result,params={'a':a,'b':b,'c':c,'x':x})
dot.render('expression',format='png',cleanup=True,view=False)





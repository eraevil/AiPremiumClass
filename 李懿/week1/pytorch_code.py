from torchviz import make_dot
import numpy as np
import torch

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

print(x_data)


'''
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones =  torch.ones_like(np_array.shape)
print("Ones Tensor: ", x_ones)

x_rand = torch.rand_like(x_data, dtype = torch.float)
print('Random Tensor: ', x_rand)


shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(rand_tensor, ones_tensor, zeros_tensor)
'''

A = torch.randn(10, 10, requires_grad=True)
b = torch.randn(10, requires_grad=True)
c = torch.randn(1, requires_grad=True)
x = torch.randn(10, requires_grad=True)

result = torch.matmul(A, x.T) + torch.matmul(b, x) + c
dot =make_dot(result, params={'A':A, 'b':b, 'c':c, 'x':x})

dot.render('expression', format='png', cleanup=True, view=False)




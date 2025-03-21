import torch

tensor = torch.ones(4, 4)
print('Tensor: ', tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print('torch.cat(Tensor, Tensor, Tensor, dim=1):\n', t1)
print('cat dim=1: ', t1.shape)


t2 = torch.cat([tensor, tensor, tensor], dim=0)
print('torch.cat(Tensor, Tensor, Tensor, dim=0):\n', t2)
print('cat dim=0: ', t2.shape)


t3 = torch.stack([tensor, tensor, tensor], dim=1)
print('torch.stack(Tensor, Tensor, Tensor, dim=1):\n', t3)
print('stack dim=1: ', t3.shape)


t4 = torch.stack([tensor, tensor, tensor], dim=0)   
print('torch.stack(Tensor, Tensor, Tensor, dim=0):\n', t4)
print('stack dim=0: ', t4.shape)

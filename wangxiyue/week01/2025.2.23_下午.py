#矩阵内积 运算 = 两个相同维度的矩阵 第一行成第一例
import numpy as np
import torch

# 张量 tensor
# data =  torch.tensor([[1,2],[3,4]])
# print(data)
#
#
# np_array = np.array([[1,2,3],[4,5,6]])
# data2 = torch.from_numpy(np_array)
# print(data2)
# #
# data3 = torch.ones_like(data2)
# print(data3)

m = torch.ones(5,3,dtype=torch.double)
n = torch.ones(5,3,dtype=torch.double)

tensor=torch.rand(3,4)
tensor=torch.randn(3,4)
print(tensor)

# if(torch.cuda.is_available()):
#     device = torch.device('cuda')
#     tensor.to(device)
#     print(device)

tensor = torch.ones(4,4)
print("First row " , tensor[0])
print("First column " , tensor[:,0])
print("Last column " , tensor[...,-1])
tensor[:,1] = 0
print(tensor)

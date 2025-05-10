import torch
from torchviz import make_dot

# 定义矩阵 A，向量 b 和常数 c
A = torch.randn(10, 10, requires_grad=True)  # 矩阵 A (10x10)
b = torch.randn(10, requires_grad=True)      # 向量 b (10)
c = torch.randn(1, requires_grad=True)       # 常数 c
x = torch.randn(10, requires_grad=True)      # 向量 x (10)

# 计算 x^T * A + b * x + c
# torch.matmul(A, x.T) 和 torch.matmul(b, x) 计算需要注意维度
result = torch.matmul(A, x) + torch.matmul(b, x) + c

# ⽣成计算图节点
dot = make_dot(result, params={'A': A, 'b': b, 'c': c, 'x': x})

# 绘制计算图
dot.render('expression', format='png', cleanup=True, view=False)

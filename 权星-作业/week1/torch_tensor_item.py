import torch

tensor = torch.arange(1, 10, dtype=torch.float32)
print(tensor)

# 单元数张量
agg = tensor.sum()
print(agg)

# 返回单元值
agg_item = agg.item() 
print(agg_item)

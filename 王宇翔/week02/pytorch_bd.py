"""
1. 数据准备，参数初始化
2. 前向计算
3. 计算损失
4. 计算梯度
5. 更新参数
"""
import torch
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification

# 超参数：学习率
learn_rate = 1e-3
# 1.1 数据准备
# X, y = make_classification(n_features=10)
X,y = load_iris(return_X_y=True)
# 创建张量
tensor_x = torch.tensor(X[:100], dtype=torch.float)
tensor_y = torch.tensor(y[:100], dtype=torch.float)
# 1.2 创建参数并初始化
# x [100,10] * w[1,10]

w = torch.randn(1, 4, requires_grad=True)  # 初始化参数w
b = torch.randn(1, requires_grad=True)  # 初始化参数b
for i in range(5000):
    # 2. 前向运算
    r = torch.nn.functional.linear(tensor_x, w, b)
    r = torch.sigmoid(r)
    # 3. 计算损失
    loss = torch.nn.functional.binary_cross_entropy(r.squeeze(1), tensor_y, reduction='mean')
    # 4.计算梯度
    loss.backward()
    # 5.参数更新
    with torch.autograd.no_grad():  # 关闭梯度计算跟踪
        w -= learn_rate * w.grad  # 更新权重梯度
        w.grad.zero_()  # 清空本次计算的梯度（因为梯度是累加计算，不清空就累
        b -= learn_rate * b.grad  # 更新偏置项梯度
        b.grad.zero_()  # 清空本次计算的梯度

    # item()张量转换python基本类型
    print(f'train loss:{loss.item():.4f}')
# 保存模型参数
torch.save({
    'weights': w,
    'bias': b,
    'input_shape': (1, 4)  # 保存输入维度信息
}, "model_params.pt")
print(f"模型参数已保存至 model_params.pt")
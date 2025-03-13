import numpy as np
import torch
from torchviz import make_dot

def main():
    arr = np.array([(1,2,3), (4,5,6), (7,8,9)])
    print(arr)
    arr = np.array([(1,2,3), (4,5,6), (7,8,9)], dtype=float)
    print(arr)
    arr = np.array([(1,2,3), (4,5,6), (7,8,9)], dtype=np.float64)
    print(arr)
    try:
        arr = np.array([(1,2,3), (4,5,6), (7,8,9)], dtype=np.float128)
    except Exception as e:
        print(f"error: {e}")

    arr = np.zeros((3,3))
    print(arr)
    arr = np.zeros((3,3), dtype=np.int16)
    print(arr)
    arr = np.arange(-1, 6, 0.9)
    print(arr)
    arr = np.arange(-1, 6, 10)
    print(arr)
    arr = np.eye(3)
    print(arr)

    try:
        arr = np.random.random(1.1)
        print(arr)
        arr = np.random.random(-1)
        print(arr)
    except Exception as e:
        print(f"error: {e}")
    arr = np.random.random(0)
    print(arr)
    arr = np.random.random(3)
    print(arr)

    arr = np.random.normal(0, 0.1, 5)
    print(arr)
    try:
        arr = np.random.normal(0, -10, 5)
        print(arr)
    except Exception as e:
        print(f"error: {e}")

    arr = np.array([(1,2,3), (4,5,6), (7,8,9)])
    print(arr.ndim)
    print(arr.shape)
    print(arr.size)
    print(arr.dtype)
    print(arr.T)
    print(arr.flatten())
    print(arr.sum())
    print(arr.prod())
    print(arr.mean())
    print(arr.argmax())
    print(np.rint(arr))
    print(arr * arr)
    print(arr @ arr)



    tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float64)
    print(tensor)
    tensor = torch.rand_like(tensor, dtype=torch.float)
    print(tensor)
    tensor = torch.zeros_like(tensor, dtype=torch.float)
    print(tensor)
    print(torch.rand(5,3))
    print(torch.randn(5,3))
    print(torch.linspace(start=1,end=10,steps=5))
    print(torch.linspace(start=1,end=10,steps=0))
    try:
        print(torch.linspace(start=1,end=10,steps=-1))
    except Exception as e:
        print(f"error: {e}")

def fun():
    # 定义矩阵 A，向量 b 和常数 c
    A = torch.randn(10, 10,requires_grad=True)  # requires_grad=True 表示我们要对 A 求导
    b = torch.randn(10,requires_grad=True)
    c = torch.randn(1,requires_grad=True)
    x = torch.randn(10, requires_grad=True)

    # 计算 x^T * A + b * x + c
    result = torch.matmul(A, x.T) + torch.matmul(b, x) + c

    # 生成计算图节点
    dot = make_dot(result, params={'A': A, 'b': b, 'c': c, 'x': x})
    # 绘制计算图
    dot.render('expression', format='png', cleanup=True, view=False)
    
if __name__ == "__main__":
    # main()
    fun()
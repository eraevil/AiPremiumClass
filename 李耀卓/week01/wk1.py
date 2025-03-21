import numpy as np
import torch
from torchviz import make_dot

# 创建ndarray数组
arr = np.array([1,2,3,4,5,6,7,8], float)
print(arr)

# 创建多维数组
a = np.array([(1,2,3,4), (4,5,6,7), (7,8,9,10)])
print("a=",a)

# 创建特殊数组
a1 = np.zeros((6,6), dtype=np.float32)
print("a1=",a1)

a2 = np.ones((3,4), dtype=np.int32)
print("a2=",a2)

# 创建等差数列
a3 = np.arange(1,10,2) # 起始值为1，终止值为10，步长为2
print("a3=",a3)

# 创建单位矩阵（矩阵的乘法中，有⼀种矩阵起着特殊的作⽤，如同数的乘法中的 1，这
# 种矩阵被称为单位矩阵。它是个⽅阵，从左上⻆到右下⻆的对⻆线（称为主对⻆线）
# 上的元素均为 1。）
a4 = np.eye(5)
print("a4=",a4)

# 创建随机数组
a5 = np.random.random((2,4))
print("a5=",a5)

# ⽣成指定⻓度，符合正态分布的随机数组
a6 = np.random.normal(3,4.1,5)
print("a6=",a6)

# numpy数组的索引和切片 
a7 = np.array([(1,2,3), (4,5,6), (7,8,9)])  
print("a7=",a7[0,1])
print("a7=",a7[0:2,1])
print("a7=",a7[:,1])
print("a7=",a7[1:3, :])
print("a7=",a7[-1])

# 遍历
for row in a7:
    print("遍历a7 row:\n",row)
a8 = np.array([(1,2,3), (4,5,6), (7,8,9)])
print("a8=\n",a8)  
for i,j,k in a8:
    print("i*j*k=\n",i*j*k)

# numpy数组常用属性
a9 = np.array([(1,2,3,4,5,6), (4,5,6,7,8,9), (7,8,9,10,11,12)], float)  
print("a9=",a9)
print("a9.size=",a9.size)
print("a9.ndim=",a9.ndim)
print("a9.shape=",a9.shape)
print("a9.dtype=",a9.dtype)

# numpy数组基本操作
a10 = np.array([(1,2,3), (4,5,6), (7,8,9)])
print(3 in a10)
print(10 in a10)
print(10 not in a10)
print(3 not in a10)

# 重排列
a11 = np.random.random((3,3,4))
print("a11=",a11)   
print("a11.reshape(18,2)=",a11.reshape(18,2))

# 转置
a12 = a11
print("a12=",a12)
print("a12.transpose()=",a12.transpose())
print("a12.T=",a12.T)

# 多维转一维
a13 = a12.flatten()
print("a13=\n",a13)

# 增加维度
a14 = np.array([(1,2,3),(4,5,6)])
print("a14=\n",a14)
print("a14.shape=\n",a14.shape)
a14 = a14[:, np.newaxis]
print("new a14=\n",a14)
print("new a14.shape=\n",a14.shape)

# 数学操作
a15 = np.array([(1,2,3), (4,5,6), (7,8,9)])
a16 = a15
print("a15=\n",a15) 
print("a16=\n",a16) 
print("a15+a16=\n",a15+a16)
print("a15-a16=\n",a15-a16)
print("a15*a16=\n",a15*a16)
print("a15/a16=\n",a15/a16)
print("a15.sum()=",a15.sum())
print("a15.prod()=",a15.prod())
print("a15.max()=",a15.max())
print("a15.min()=",a15.min())
print("a15.mean()=",a15.mean())
print("a15.var()=",a15.var())
print("a15.std()=",a15.std())
print("a15.argmax()=",a15.argmax())
print("a15.argmin()=",a15.argmin())

# 线性代数
a17 = np.array([(1,3,5), (2,4,6), (7,8,9)], float) 
a18 = np.array([(1,1,1), (2,2,2), (1,1,1)], float)
print("a17=\n",a17)
print("a18=\n",a18)
print("np.dot(a17, a18)=",np.dot(a17, a18))
print("a17.dot(a18)=",a17.dot(a18))
print("a17 @ a18=",a17 @ a18)

manual_res = np.zeros((a17.shape[0],a18.shape[1]))
print("first manual_res=\n",manual_res)
for i in range(a17.shape[0]):
    for j in range(a18.shape[1]):
        manual_res[i,j] = 0
        for k in range(a17.shape[1]):
            manual_res[i,j] += a17[i,k] * a18[k,j]
        print(f"结果矩阵[{i+1},{j+1}]:{manual_res[i, j]}\n")
print("manual_res=\n",manual_res)

# 广播机制
a = np.array([(1,2,3), (4,5,6), (7,8,9)])
b = np.array([1,2,3])
print("a=\n",a)
print("b=\n",b)
print("a+b=\n",a+b)

# pytorch基础
data = [[1,2], [3,4]]
t_data = torch.tensor(data)
print("t_data=\n",t_data)

np_arr = np.array(data)
t_np = torch.from_numpy(np_arr)
print("t_np=\n",t_np)

ones = torch.ones_like(t_data)
print("ones=\n",ones)
rand_tensor = torch.rand_like(t_data, dtype=torch.float)
print("rand_tensor=\n",rand_tensor)

# 使用随机或者常量创建张量
shape = (3,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print("rand_tensor=\n",rand_tensor) 
print("ones_tensor=\n",ones_tensor) 
print("zeros_tensor=\n",zeros_tensor)
tensor = torch.rand(3,4)
print("tensor=\n",tensor)

# 张量属性
tensor = torch.rand(5,4)
print("tensor=\n",tensor)
print("tensor.shape=",tensor.shape)
print("tensor.dtype=",tensor.dtype)
print("tensor.device=",tensor.device)

# 张量索引和切片
tensor = torch.rand(4,4)
print("tensor=\n",tensor)
tensor[:,1] = 0
print("tensor=\n",tensor)
print("tensor[0,0]=",tensor[0,0])
print("tensor[0,0].item()=",tensor[0,0].item())
print("tensor[0:2,0:2]=",tensor[0:2,0:2])
print("first row:",tensor[0])
print("first column:",tensor[:,0])
print("last column:",tensor[..., -1])
print("tensor[:,-1]=",tensor[:,-1])

# 张量拼接
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print("t1=\n",t1)
t2 = torch.stack([tensor, tensor, tensor], dim=1)
print("t2=\n",t2)

# 张量运算
# 矩阵乘法
y1 = tensor @ tensor.T
print("y1=\n",y1)   
y2 = tensor.matmul(tensor.T)
print("y2=\n",y2)
y3 = torch.rand_like(tensor)
print("y3=\n",y3)

# 元素相乘
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print("z1=\n",z1)
print("z2=\n",z2)
print("z3=\n",z3)

# 单元素张量
agg = tensor.sum()
agg_item = agg.item()
print("agg_item=",agg_item, "type(agg_item)=",type(agg_item))

# 定义矩阵 A，向量 b 和常数 c
A = torch.randn(18, 8,requires_grad=True)
b = torch.randn(8,requires_grad=True)
c = torch.randn(18,requires_grad=True)
x = torch.randn(8, requires_grad=True)
# 计算 x^T * A + A * b + b * x + 3*c
result = torch.matmul(A, x.T) + torch.matmul(A, b) + torch.matmul(b, x) + 3*c
# ⽣成计算图节点
dot = make_dot(result, params={'A': A, 'b': b, 'c': c, 'x': x})
# 绘制计算图
dot.render('expression', format='png', cleanup=True, view=False)

import numpy as np

# 1. numpy 创建数组
a = [1,2,3]
arr = np.array(a,float)
arr

arr2 = np.array([[1,2,3],[4,5,6],[7,8,9]])
arr2

ones = np.zeros((2,2),dtype=float)
ones

a = np.arange(1,55,5)
a

a = np.eye(5)
a

a = np.random.random(5)
a

mu,sigma = 0, 0.1
# mu为均值， sigma为标准差
np.random.normal(mu,sigma,5)

a = np.array([(1,2),(3,4),(5,6)])
a[1][1]

a = np.array([[1,2],[3,4],[5,6]])
for i,j in a:
    print(i*j)

a = np.array([(1,2,3),(4,5,6),(7,8,9)])
print("ndim",a.ndim)
print("shape:",a.shape)
print("size:",a.size)
print("dtype:",a.dtype)

a = np.array([[[1,2,3],[4,5,6],[7,8,9]],[[3,2,1],[7,4,3],[9,6,5]]])
print("ndim",a.ndim)
print("shape:",a.shape)
print("size:",a.size)
print("dtype:",a.dtype)

# 重置数组维度，将三维数组调整为一维数组
a = a.reshape(18)
a

# 再将一维数组升维到3维数组
a = a.reshape(2,3,3)
a

# 将数组转置,这结果我真是看不懂啊
a.transpose()


# 多维变一维
a.flatten()

# 增加维度
a=a[:,np.newaxis]
a

a = np.array([1,2,3,4])
a = a[:,np.newaxis]
a


# 矩阵乘积
a = np.ones((2,2),int)
b = np.array([(-1,1),(-1,1)])
print(a)
print(b)


print(a*b)
print(a+b)
print(a/b)

print(a.sum())
print(a.prod())

# 平均数，方差，标准差，最值
print(a.mean())
print(a.std())
print(a.var())
print(a.min())
print(a.max())


# 最大值，最小值的索引值：argmax(),argmin()
c = np.array([[(1,2,3),(7,9,-1)],[(6,4,5),(7,7,9)]])
print(c.argmax())
print(c.argmin())


# 取元素上线、下限、四舍五入
d = np.array([1.2, 3.2, 5.3, 6.2, 7.4,8.5,-0.1])
print(np.ceil(d))
print(np.floor(d))
print(np.rint(d))
#通过结果来看，是对全部元素进行操作。这个月np.newaxis功能相同


# 排序
m = np.sort(d,False)
n = d.sort()
print(m)
print(n)

axis1 = np.random.random(5)
axis2 = np.random.random(5)
axis3 = np.random.random(5)
axis4 = np.random.random(5)
axis5 = np.random.random(5)
axis6 = np.random.random(5)


matrix1 = np.array([(axis1,axis2),(axis3,axis4),(axis5,axis6)])
matrix2 = np.array([(axis1,axis2),(axis3,axis4),(axis5,axis6)])
matrix1

## 下面的方法跑不通

resutl = np.dot(matrix1,matrix2)
result2 = matrix1@matrix2
## 看来这两个方法，只适合二维矩阵

matrix3 =np.array([axis1,axis2],dtype=float)

axis7 = np.random.random(2)
axis8 = np.random.random(2)
axis9 = np.random.random(2)
axis10 = np.random.random(2)
axis11 = np.random.random(2)


matrix4 = np.array([axis7,axis8,axis9,axis10,axis11],dtype=float)
result_at = matrix3@matrix4
print(result_at)
result_dot = np.dot(matrix4.transpose(),matrix3.transpose())
print(result_dot)

# 手动推演计算结果
# (1)先看看都是啥
print(matrix3)
print(matrix4)

# (2）开始双遍历
# 根据矩阵相乘公式，m行z列矩阵@z行n列举证，结果为m行n列矩阵
res = np.ones((matrix3.shape[0],matrix4.shape[1]),dtype=float)
 
for m in np.arange(matrix3.shape[0]) :
    for n  in np.arange(matrix4.shape[1]) :
        # 方案一：用系统自带代数积来计算，已证明可行
       # res[m,n] = (matrix3[m,:]*matrix4[:,n]).sum()
       # 方案二：手动注意计算
        temp = float(0)
        for i in np.arange(matrix3.shape[1]) :
           temp = temp + matrix3[m,i]*matrix4[i,n] 
        res[m,n] = temp   

print(res)

# 看看真实结果是啥
res2 = matrix3@matrix4
print(res2)   

# numpy广播机制
arr1 = np.array([(1,2),(3,4),(5,6),(7,9)])
arr2 = np.array([(2,3),(-1,-1)])
res = arr1+arr2
print(res)

# numpy广播机制2：如果行和列都对不上呢--
# ValueError: operands could not be broadcast together with shapes (4,2) (3,3) 
# arr1 = np.array([(1,2),(3,4),(5,6),(7,9)])
# arr2 = np.array([(2,3,4),(-1,-1,-1),(-2,-2,-2)])
# res = arr1+arr2
# 如果行对不上，但是列能对上呢？
arr1 = np.array([(1,2,3),(3,4),(5,6),(7,9)])
arr2 = np.array([(2,3,4),(-1,-1,-1),(-2,-2,-2),(0,0,0)])
res = arr1+arr2
print(res)
# 擦咧，也不行，试验证明，只有列数对上，但是行对不上的情况下，才能广播成功



import numpy as np
import torch

# 创建张量，用普通list和ndarry
x_data = torch.tensor(date1)
x_data2 = torch.from_numpy(data2)
print(x_data)
print(x_data2)

# 由张量创建新张量
# just create one same tesor
x_ones = torch.ones_like(x_data)
print(x_ones)
type(x_ones)
# create one new tesor and cover his attritues
x_new = torch.rand_like(x_data,dtype=torch.float)
print(x_data)
print(x_new)
# why the value is changed??? oh....I see. it is rand_like. means the value is randomed


# create tensor object by system ways
shape = [2,3]
x_tensor1 = torch.rand(shape)
x_tensor2 = torch.ones(shape)
x_tensor3 = torch.zeros(shape)

print(x_tensor1)
print(x_tensor2)
print(x_tensor3)


# create tensor by shape
y_tensor1 = torch.ones(5,3,dtype=torch.float)
print(y_tensor1)


y_tensor2 = torch.rand_like(y_tensor1,dtype=torch.double)


print(y_tensor1.shape)
print(y_tensor1.size())
# so, it is no differents between shape and size. one is attribute, while another one is callable

torch.rand(5,3)

torch.randn(5,3)

torch.normal(mean=0,std=1.0,size=(5,3))

torch.linspace(start=1,end=10,steps=20) 
# there is one mistake happened on code above. just change "step" into "steps"

ten = torch.rand(3,4)

print(ten.shape)
print(ten.dtype)
print(ten.device)

# ?? what is ten.device. "CPU"???


if torch.cuda.is_available:
ten = ten.to('cuda')
# cannot work well on my laptop


tensor = torch.rand(4,3)
print(tensor)
# index and chip
print(tensor[...,-1])
print(tensor[:,-1])
# so , it is same between "..." and ":"

torch.cat([tensor,tensor,tensor],dim=0)
# so , cat is continue with run 
# if dim=0, so si continue with column. else is with row.

torch.stack([tensor,tensor,tensor],dim=1)
# add axis


# calculation on torch
y = tensor@tensor.T
print(y)
print(tensor)

tensor.matmul(tensor.T)
# oh... matmull means sbustact


ye = torch.rand_like(tensor)
ye


torch.matmul(tensor,tensor.T,out=ye)
ye
print(tensor.shape)
print(tensor.T.shape)

tensor.sum()

tensor.sum().item()

print(tensor)

print(tensor)
print(tensor.t_())
print(tensor)
# has changed....

t = torch.rand_like(tensor,dtype=torch.double)
print(t)
print(t.numpy())
t.add_(1)
print(t)


# operate numpy , torch will changed also
n = np.ones(5)
t = torch.from_numpy(n)
n[3]=4
print(t)

import torch
from torchviz import make_dot

A = torch.randn(10,10,requires_grad=True)
B = torch.randn(10,requires_grad=True)
C = torch.randn(1,requires_grad=True)
x = torch.randn(10,requires_grad=True)

result=torch.matmul(A,x.T) + torch.matmul(B,x)+C
dot = make_dot(result,params={'A':A,'b':B,'c':C,'x':x})
dot.render('expression',format='png',cleanup=True,view=False)

import numpy as np
import torch
a=np.array([[1,2,3],[4,5,6],[7,8,9]],'double') #二维矩阵用两个'[]'
b=torch.tensor(a,dtype=float)
print(a)
print(b) #'dtype'对'print'来说非法
c=np.zeros((3,2),dtype=float)#用不用'dtype'一样的效果
d=np.ones([7,8])#创建多维数组一定要有'()'或者'[]'
e=np.arange(1,9,0.8)#有1无9(有头有尾该怎么敲？)
f=np.eye(9)#E数组
g=np.random.random(6)
h=np.random.normal(5,0.1,9)
print(a[1:])
print(a[:,1])#[:,1]代表第一列，结果输出一个一维数组
print(a[:,:1])#':'代表所有，[:,:1]代表第二列之前的所有列
for i in a:
    print(i)
for i,j,k in a:
    print(i*j-k) 
print(np.ndim(a))
print(np.size(a))
print(a.dtype)
print(np.dtype(a))#有什么不同吗,这个编译不通过

print(6 in a)
np.reshape(a,9)
a.reshape(9)
a.T
np.transpose(a)
a.flatten()
a
print(a.shape)
i=a[:,np.newaxis,:]
i.shape


j=np.array([[1,2,3],[4,5,6],[7,8,9]])
k=np.random.normal(7,3,[3,3])
j+k
j*k
a.sum()
a.mean()
a.var()
a.std()
np.ceil(a)
np.argmax(a)
np.rint(a)

print(a.sort())#sort是就地排序法，直接修改原数组
a@a.T
np.dot(a,a.T)
p=np.array([[1,3],[4,5],[5,6]])#二维数组要用两个括号
q=np.array([9,1])
p+q
#pytorch
a=[[1,2],[3,4],[5,6]]
a1=torch.tensor(a,dtype=float)
a1
a2=np.array(a)
a3=torch.tensor(a2)
a3=torch.from_numpy(a2)
a4=torch.zeros_like(a1)
a4
a5=torch.clone(a1)
a5
a6=np.ones_like(b)
a6
a7=torch.rand_like(a1)
a7
a8=torch.rand(4,4)
a8
a9=torch.ones(3,3)
a9
b9=np.ones(3,3)#这种形式torch可以np不行
b9

shape=(3,3,)
torch.randn(shape)
torch.normal(mean=0,std=1.,size=(4,))
torch.linspace(start=1,end=8,steps=8)
torch.arange(1,3,)

tensor=torch.rand(9,9)
k=torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
print(k[...,-1])
print(k[-1,-1])
print(k[-1,...])
torch.cat((k,k,k),dim=0)
print(torch.matmul(k,k.T))
print(k@k.T)
k1=torch.tensor([1])
torch.matmul(k,k.T,out=k1)
k1
k3=k.sum()
k3_item=k3.item()
print(k3_item)

print(k3_item)
j=torch.clone(k)
l=k.numpy()
k.add_(6)
print(l)

from torchviz import make_dot
a=torch.rand([5,5])
b=torch.normal(mean=1,std=1.0,size=[5,5])
result=a+b
dot=make_dot(result,params={'a':a,'b':b}) 
dot.render('jisuan',format='png',cleanup=True,view=False)

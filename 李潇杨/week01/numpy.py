# %%
import torch

tensor=torch.ones(4,4)
print(tensor)


# %%
torch.cuda.is_available()

# %%
import torch
# 当前设备
current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("当前设备: ", current_device)


# %%
import numpy as np

a=np.array([1,2,3,4,5,6])
a.shape
b=np.expand_dims(a,axis=1)


# %%
import numpy as np
a=[1,2,3]
b=np.array(a)
b

b=np.array([1,2,3],float)
b


a=np.array([(1,2,3),(4,5,6),(7,8,9)])
a
a.shape

a=np.zeros((2,3),float)
a

a=np.ones((3,4),float)
a

# 第一，二个参数表示起始位和结束位，第三个参数表示步长,取值范围是左开右闭
a=np.arange(1,5,0.5)
a

# 第三个参数是指矩阵的第一行单位值索引位置
a=np.eye(5,5,0,int,'C')
a

c=np.random
d=c.random(9)
d
# 1：平均值 2：标准差
d=c.normal(0,0.2,6)
d=c.normal(0,0.1,5)
d

v=0.23278609+0.07432719+0.05550536+0.01558179-0.04058723
v=v/5
v

e=np.array([(1,2,3),(3,4,5),(5,6,7)])
e[0]
e[0:]
e[:2]
e[1:]
e[:]
e[:,1:]
e[2,2]

x=np.array([1,2,3])

for i in x :
    print (i)



# %%
a=np.array([(1,2,4),(3,6,9)])

for i,j,k in a:
    print(i)
    print(j)
    print(k)
    print(i*j*k)


a = np.array([(1,2,3), (4,5,6), (7,8,9)])
print("ndim:", a.ndim)
print("shape:", a.shape)
print("size", a.size)
print("dtype", a.dtype)

# %%
a = np.array([(1,7), (3,4)])
print(7 in a)
print(9 in a)

# %%
a = np.zeros([2,3,4])
a

a=a.reshape(3,8)
a

a=a.reshape(24)
a



a = np.array([(1,7), (3,4)])
b=a.transpose()
b


a = np.array([(1,2), (3,4), (5,6)])
a.flatten()

a = np.array([1,2,3])
a.shape

c=a[:,np.newaxis]


c.shape




# %%
a = np.ones((2,2))
b = np.array([(-1,1),(-1,1)])
print(a)
print(b)

a+b
a-b
a*b
c=a/b
c=np.array([7,8,9])
np.sum(c)

c.sum()  #求和
c.prod() #求积





# %%
a = np.array([5,3,1])
print("mean:",a.mean())  #平均数
print("var:", a.var())   #方差
print("std:", a.std())   #标准差

print("max:", a.max())
print("min:", a.min())

# %%
a = np.array([1.2, 3.8, 4.9])
print("argmax:", a.argmax())  # 最大值对应索引
print("argmin:", a.argmin())  # 最小值对应索引
print("ceil:", np.ceil(a))   #向上取整
print("floor:", np.floor(a)) #向下取整
print("rint:", np.rint(a))  #四舍五入

# %%
a = np.array([16,31,12,28,22,31,48])
a.sort()   #排序
a

# %%
import numpy as np
# 定义两个简单的矩阵
m1 = np.array([[1, 2], [3, 4]] , dtype=np.float32)
m2 = np.array([[5, 6], [7, 8]] , dtype=np.float32)
# 使⽤ np.dot 进⾏矩阵乘法
result_dot = np.dot(m1, m2)
result_dot

# %%
# 首先存储数组数据,生成.npy文件
import numpy as np
 
#这里利用相对路径来存储
fileName = './text.npy'
# 生成数组
a = np.arange(24).reshape(2,3,4)
print(a)
#保存到文件中
np.save(fileName,a)
a = np.load(fileName)
a



# %%
a = np.array([1,2,3])
b = np.array([4,5,6])
a + b

a = np.array([(1,2), (2,2), (3,3), (4,4)]) # shape(4,2)
b = np.array([-1,1]) # shape(2)-> shape(1,2) -> shape(4,2) [ -1,1],[-1,1],[-1,1],[-1,1]
a + b



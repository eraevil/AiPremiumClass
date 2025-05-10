# numpy基础
import numpy as  np
arr = [1,2,3,4] #普通数组
#创建ndarrey
arr1 = np.array(arr,float) #把list数组转为np.array
print(arr)
a = np.array([(1,2,3),(4,5,6),(7,8,9)])
print(a)

#创建初始化数组
a1 = np.zeros((2,3),dtype=float)
print(a1)
a2 = np.ones((3,3),dtype= np.float16)
print(a2)

#创建等差数列
a3 = np.arange(1,5,0.5)#从1开始到5结束 步长0.5   1,5包左不包右
print(a3)

#创建one-hot编码矩阵
a4 = np.eye(3)
print(a4)

#生成[0,1)之间x个数量的随机数 可用于模型参数随机初始化
a5 = np.random.random(5)
print(a5,type(a5))

#生成x个正态分布的随机数 mu均值 sigma标准差
mu,sigma = 0,0.1
a6 = np.random.normal(mu,sigma,5)
print(a6)

#ndarray高维切片  普通array要用循环取  numpy底层用C编写 贴近系统底层更快
a7 = np.array([[1,2],[3,4],[5,6]]) # 第0维和第1维
print(a7)
print(a7[:,1]) #  ： 所有行 第一列 注意是以0开始维度

#
a8 = np.array([[1,2],[3,4],[5,6]])
#i,j = a8[0] # 取第0行的两个数给ij  更高维同理
for i,j in a8:
    print(i,j)

#np.array 的一些属性
print(a8.ndim) #维度
print(a8.shape) #形状
print(a8.size) #元素个数 等于shape各值相乘
print(a8.dtype) #数据类型
print(3 in a8) # 检测元素是否存在于数组中
print([1,4] in a8) #高维只可以按维数检测
print((1,3) in a8)
print(a8.reshape(2,3,1)) #改变数组维度 维度大小乘积 == 元素个数 高维矩阵 每维都有含义
#维度转置transpose
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
a = a.transpose()
print(a)
a = a.T
print(a)
#多维数组 变为一维数组
a = a.flatten()#拉平
print(a)    
#增加一个维度 有时为了方便运算
a = a[:,np.newaxis]
print(a,a.shape)    

#矩阵的运算 要符合矩阵运算规律
a = np.ones((2,2))
b = np.array([[-2,2],[2,-2]])
print(a + b)
#高维数组所有元素求和
print(a.sum())
#高维数组所有元素乘积
print(a.prod())
a.mean() #均值
a.var() #方差
a.std() #标准差 
a.max()
a.min()

#总结 ：数组、矩阵 都是数据的组织结构形式  就像 年级-班级-小组-行-列  
#       可以想象成一个小区内的通讯地址

#找数组中最值所在位置的索引
a = np.array([1.2,3.8,4.9])
print(a.argmax()) #最大值所在索引 0开始
print(a.argmin()) #最小值所在索引 0开始
#所有元素数据上的操作
print(np.ceil(a)) #所有元素向上取整
print(np.floor(a)) #所有元素向下取整
print(np.rint(a)) #所有元素四舍五入
#数组排序
b = np.array([1,4,9,7,6])
print(b.sort())
#numpy的其他功能和数学符号相同 直接你拿来用即可 如sin


#矩阵运算
a9 = np.array([[1,2],[4,5]])
b9 = np.array([[4,5],[7,8]])
print(a9 * b9) #矩阵各元素对位相乘

res1 = np.dot(a9,b9) #矩阵乘法
res2 = a9 @ b9 #也是矩阵乘法 同上一行
print(res1)
print(res2)

#用numpy进行文件保存
# np.save('result.npy',res1)
# np.load('res.npy')

# numpy的广播机制broadcast 将两个不同维度的矩阵按照某规律自动补全 目的是两个矩阵可以做运算
a = np.array([[1,1],[2,2],[3,3],[4,4]])
b = np.array([-1,1]) #shape(2) ->shape(1,2)->shape(4,2) [[-1,1],[-1,1],[-1,1],[-1,1]]
print(a + b)#broadcast机制发挥作用前提 在两个要广播的高维数组最后一维对齐 因为brocast从最低为开始升维






#pytorch基础
#Pytorch基础
import torch
import numpy as np
from torchviz import make_dot

#Pytorch 的操作很想 Numpy
#生成一个张量 
#方式一
data = torch.tensor([[1,2],[3,4]]) #张量 pytorch的基本数据单位 数据的容器 类比numpy的ndarray
print(data)
#方式二
data2 = np.array([[1,2],[3,4]])
data2 = torch.from_numpy(data2)
print(data2)

#参照已知张量形状 生成一个新的高维张量
data3 = torch.ones_like(data2)
#data3 = torch.zeros_like(data2)
print(data3)

#张量的数据类型
print(data2.dtype) #长整型Long
#参照已知张量形状 创建随机数张量
data4 = torch.rand_like(data2,dtype = float) #张量数据类型默认Long即int64 随机数是小数

#指定形状生成张量 shape 是张量维度的元组
shape = (2,2)
print(torch.rand(shape))
print(torch.ones(shape))
print(torch.zeros(shape))
#获取张量形状
print(data4.size())
print(data4.shape)
#随机分布 [0,1)
print(torch.rand(2,3))
#正态分布
print(torch.randn(2,3))
#自定义均值 标准差 的正态分布
print(torch.normal(mean= 0,std = 1.0,size = (2,3)))
#生成一个等差数列形的一行张量
print(torch.linspace(start = 1,end = 10,steps = 20)) # steps 点数 分成多少份

#张量的属性
tensor  = torch.rand(3,4)
#形状   
print(tensor.shape) #等价于tensor.size()
#dtype
print(tensor.dtype)
#该张量存储所在的设备
print(tensor.device)


#想让张量在GPU运行
#先检查当前pytorch是否有可行的GPU 
print(torch.cuda.is_available())
#如果可以 说明可以让张量在GPU端运行
if torch.cuda.is_available():
    device = torch.device('cuda:1') #cuda:x第几块gpu 不写默认第一块
    tensor = tensor.to(device)
print(tensor)
print(tensor.device)

#张量的切片 用法同numpy
tensor = torch.ones(3,4)
print(tensor[0])
print(tensor[:,0])
print(tensor[...,-1]) #张量维度很高 选以前所有维 最后一维的最后一列的值
tensor[:,1] = 0 #修改所有行的维度是1的列的值为x
print(tensor)
#高位矩阵运算 要满足 第一个矩阵最后一维 == 第二个矩阵倒数第二维

#张量的拼接
t1 = torch.cat([tensor,tensor,tensor],dim = 1) #0是行 1是列 把teensorx用列表拼接起来
print(t1,t1.size())

#张量的矩阵运算
tensor =  torch.arange(1,10,dtype = float).reshape(3,3)
#1矩阵相乘
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)  #y1 == y2
y3 = torch.rand_like(tensor)
torch.matmul(tensor,tensor.T,out = y3) #把tensor@tensor.T的结果赋值给out = y3
print(y1)
print(y2)
print(y3)    # y1 == y2 == y3

#2矩阵个元素对位相乘
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor,tensor,out = z3)

#把张量转为 普通python数值
agg = tensor.sum()
agg_item = agg.item()
print(agg_item,type(agg_item))  #numpy torch 的查看数据类型用dtype 普通python数值用type


#张量转numpy的ndarray数组
print(tensor.numpy())

#in-place操作 没有返回值 直接在原张量内部做出修改 速度更快 在模型中不鼓励用  使用了该操作的特征： _
print(tensor,'\n')
tensor.add_(5) #用了in-place 操作  等价于 tensor += 5 对矩阵中每个元素 + 5
print(tensor) 

#pytorch 1.计算图（前向传播图DAG）(torchviz) 2.反向传播调参

#定义矩阵A 向量b 和常数c
A = torch.randn(10,10,requires_grad = True)
b = torch.randn(10,requires_grad= True)
c = torch.randn(1,requires_grad=True)
x = torch.randn(10,requires_grad=True)

#计算x.T *A + b *x + c
result =  torch.matmul(A,x.T) + torch.matmul(b,x) + c
#生成计算图点
dot = make_dot(result,params = {'A':A,'b':b,'c':c,'x':x})

#绘制计算图
dot.render('expression',format = 'png',cleanup=True,view= False)

import numpy as np 
a= np.eye(3)
a
b = np.random.normal(0,1,(5,5))
# print(b)
# print(b[:2])
i,j = np.where(b>0)
# print(i,j)
# print(b[i,j])
# print(b[b>0])
# print(b[b>0].shape)
print("b[b>0].reshape(1,-1)",b[b>0].reshape(1,-1))
print("b[b>0].reshape(-1,1)",b[b>0].reshape(-1,1))
print("b[b>0].reshape(-1,1).shape",b[b>0].reshape(-1,1).shape)
print("b[b>0].reshape(-1,1).T",b[b>0].reshape(-1,1).T)
print("b[b>0].reshape(-1,1).T.shape",b[b>0].reshape(-1,1).T.shape)
print("b[b>0].reshape(-1,1).T[0]", b[b>0].reshape(-1,1).T[0])
print("b[b>0].reshape(-1,1).T[0].shape", b[b>0].reshape(-1,1).T[0].shape)
print("b[b>0].reshape(-1,1).T[0].T",b[b>0].reshape(-1,1).T[0].T)
print("b[b>0].reshape(-1,1).T[0].T.shape",b[b>0].reshape(-1,1).T[0].T.shape)
print("b[b>0].reshape(-1,1).T[0].T[0]", b[b>0].reshape(-1,1).T[0].T[0])
print("b[b>0].reshape(-1,1).T[0].T[0].shape", b[b>0].reshape(-1,1).T[0].T[0].shape)
print("b[b>0].reshape(-1,1).T[0].T[0].T", b[b>0].reshape(-1,1).T[0].T[0].T)
print("b[b>0].reshape(-1,1).T[0].T[0].T.shape", b[b>0].reshape(-1,1).T[0].T[0].T.shape)

c= np.eye(3)
c = c[:,:,np.newaxis]
print(c)
print(c.shape)
print(np.ceil(c))

a = np.array([(1,2), (3,4), (5,6)])
a[: , 0]

# 定义两个数组
aa = np.array([1, 2, 3])
bb = np.array([[1, 2], [3, 4], [5, 6]])
bb = bb.reshape(2,3)
 
# 使用广播机制进行加法运算
result = aa + bb
 
print(result)

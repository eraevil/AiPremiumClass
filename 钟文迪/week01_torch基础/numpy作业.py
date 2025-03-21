import numpy as np

a = [1, 2, 3, 4, 5]
b = np.array(a)
print(b)

a = np.array([1, 2, 3, 4, 5], float)
print(a)
a = np.array([1, 2, 3, 4, 5], int)
print(a)

a = np.array([[11, 22, 33], [44, 55, 66], [77, 88, 99]], float)
print(a)

a = np.zeros((2, 3), float)
print(a)

a = np.ones((2, 3), float) 
print(a)

a = np.arange(1, 10, 2.5)
print(a)

a = np.eye(3)
print(a)

a = np.random.rand(2, 3)
print(a)

a = np.random.normal(0, 0.5, 5)
print(a)

a = np.array([[100, 222], [356, 488], [666, 590]], float)
print(a[0])
print(a[1][1])
print(a[:, 1])
print(a[1, :])

for i in a:
    print(i)

a = np.array([(1, 2, 3), (2, 4, 5), (3, 4, 5)], float)
print(a.ndim)
print(a.shape)
print(a.size)
print(a.dtype)
print(2 in a)
print(10 in a)

a = np.zeros([2, 4, 4])
print(a)
a.reshape(32)
print(a)

a = np.array([(1, 2, 3), (6, 4, 9), (8, 7, 5)], float)
print(a)
print(a.T)
print(a.transpose())

a = np.array([(1, 2, 3), (6, 4, 9), (8, 7, 5)], float)
print(a.flatten())

a = np.array([(1, 2, 3), (6, 4, 9), (8, 7, 5)], float)
print(a.shape)
print(a)
a = a[:, np.newaxis, :]
print(a.shape)
print(a)

a = np.ones((2, 3))
b = np.array([(3, 3, 1), (-2, 6, 1)])
print(a + b)
print(a - b)
print(a * b)
print(a / b)

a = np.array([4, 6, 10])
print(a.sum())
print(a.prod())

a = np.array([15, 23, 6])
print("mean:", a.mean())
print("var:", a.var())
print("std:", a.std())
print("max:", a.max())
print("min:", a.min())

a = np.array([3.2, 8.4, 6.9])
print("argmax:", a.argmax())
print("argmin:", a.argmin())
print("ceil:", np.ceil(a))
print("floor:", np.floor(a))
print("rint:", np.rint(a))

a = np.array([1, 2, 7, 9, 5, 3, 8, 4])
a.sort()
print(a)

m1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
m2 = np.array([[5, 6], [7, 8]], dtype=np.float32)
print(m1)
print(m2)
print(m1 + m2)
print(m1 - m2)
print(m1 * m2)
print(m1 / m2)
result = np.dot(m1, m2)
print(result)
result = m1 @ m2
print(result)

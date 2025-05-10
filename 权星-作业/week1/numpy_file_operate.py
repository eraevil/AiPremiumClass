import numpy as np

a = np.array([(1, 2, 3), (4, 5, 6)])
print("数组a:\n", a)

np.save('a.npy', a)
a_load = np.load('a.npy')
print("加载的数组a:\n", a_load)

np.savetxt('a.txt', a)
a_loadtxt = np.loadtxt('a.txt')
print("加载的数组a:\n", a_loadtxt)

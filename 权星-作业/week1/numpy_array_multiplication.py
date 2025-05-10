import  numpy as np

a = np.array([[1, 2], [3 ,4]])
b = np.array([[5, 6], [7, 8]])

result_dot = np.dot(a, b)
result_dot1 = np.dot(b, a)

result_matmul = np.matmul(a, b)
result_at = a @ b

print("矩阵a:\n", a)
print("矩阵b:\n", b)
print("矩阵a和b的点积:\n", result_dot)
print("矩阵b和a的点积:\n", result_dot1)

print("矩阵a和b的矩阵积:\n", result_matmul)
print("矩阵a和b的矩阵积:\n", result_at)


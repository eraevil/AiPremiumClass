import numpy as np

m1 = np.array([[1, 2], [3, 4]], dtype = np.float32)
m2 = np.array([[5, 6], [7, 8]], dtype = np.float32)

result_dot = np.dot(m1, m2)
result_at = m1 @ m2

print('result_dot：', result_at)
print('result_at: ', result_at)

manual_result = np.zeros((m1.shape[0], m2.shape[1]), dtype = np.float32)

for j in range(m2.shape[1]):
    for i in range(m1.shape[0]):
        manual_result[i, j] = 0
        for k in range(m1.shape[1]):
            manual_result[i,j] += m1[i, k] * m2[k, j]

print(manual_result)

Python 3.12.7 | packaged by Anaconda, Inc. | (main, Oct  4 2024, 13:17:27) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy as np
>>> # 创建ndarray数组
>>> arr = np/array([1,2,3,4,5])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'array' is not defined. Did you forget to import 'array'?
>>>  arr = np.array([1,2,3,4,5])
  File "<stdin>", line 1
    arr = np.array([1,2,3,4,5])
IndentationError: unexpected indent
>>> arr = np.array([1,2,3,4,5], float)
>>> print arr
  File "<stdin>", line 1
    print arr
    ^^^^^^^^^
SyntaxError: Missing parentheses in call to 'print'. Did you mean print(...)?
>>> arr = np.array([1,2,3,4,5], float)
>>> print(arr)
[1. 2. 3. 4. 5.]
>>> a = np.array([1,2,3,4],[5,6,7,8])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: Field elements must be 2- or 3-tuples, got '5'
>>> a = np.array([1,2,3,4],[5,6,7,8],[9,0,1,2],[3,4,5,6])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: array() takes from 1 to 2 positional arguments but 4 were given
>>> a = np.array([(1,2,3), (4,5,6), (7,8,9)])
>>> print(a)
[[1 2 3]
 [4 5 6]
 [7 8 9]]
>>> a = np.array([[1,2,3,4],[2,3,4,5],[4,5,6,7]])
>>> print(a)
[[1 2 3 4]
 [2 3 4 5]
 [4 5 6 7]]
>>> import numpy as np
>>> a = np.zeros((2,3),dtype = float)
>>> print(a)
[[0. 0. 0.]
 [0. 0. 0.]]
>>> a = np.zeros(4,5),dtype = float)
  File "<stdin>", line 1
    a = np.zeros(4,5),dtype = float)
    ^^^^^^^^^^^^^^^^^
SyntaxError: invalid syntax. Maybe you meant '==' or ':=' instead of '='?
>>> a = np.zeros((4, 5), dtype=float)
>>> print(a)
[[0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]]
>>> a1 = np.zeros((2,3), dtype=np.float64)
>>> print(a1)
[[0. 0. 0.]
 [0. 0. 0.]]
>>> a1 = np.zeros((2,3), dtype=np.float32)
>>> print(a1)
[[0. 0. 0.]
 [0. 0. 0.]]
>>> a = np.ones((3,3))
>>> print(a)
[[1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]]
>>> a2 = np.arange(2,7,0.4)
>>> print(a2)
[2.  2.4 2.8 3.2 3.6 4.  4.4 4.8 5.2 5.6 6.  6.4 6.8]
>>> a3 = np.eye(4)
>>> print(a3)
[[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]]
>>> a4 = np.random.random(8)
>>> print(a4)
[0.83346411 0.3204628  0.73947591 0.67589662 0.62354671 0.89723767
 0.93206372 0.54555902]
>>> a5 = np.random.normal(0,0.1,7)
>>> print(a5)
[ 0.09008309 -0.10237061  0.0360901  -0.03197009  0.12129384 -0.01377393
 -0.1492122 ]
>>> a6 = np.array([(1,2),(3,4),(5,6),(7,8)])
>>> print(a6)
[[1 2]
 [3 4]
 [5 6]
 [7 8]]
>>> print(a6[:,1])
[2 4 6 8]
>>> a7 = np.array([(1,2), (3,4), (5,6),(7,8)])
>>> print(a7[0])
[1 2]
>>> i,j = a7[0]
>>> print(i,j)
1 2
>>> for i,j in a7:
... a7 = np.array([(1,2), (3,4), (5,6),(7,8)])
  File "<stdin>", line 2
    a7 = np.array([(1,2), (3,4), (5,6),(7,8)])
    ^
IndentationError: expected an indented block after 'for' statement on line 1
>>>
>>> a7 = np.array([(1,2), (3,4), (5,6),(7,8)])
>>> i,j = a7[0]
>>> for i,j in a7:
...
  File "<stdin>", line 2

    ^
IndentationError: expected an indented block after 'for' statement on line 1
>>> import numpy as np
>>>
>>> a7 = np.array([(1, 2), (3, 4), (5, 6), (7, 8)])
>>>
>>> # 将 a7 的第一行元素拆解并赋值给 i, j
>>> i, j = a7[0]
>>>
>>> # 循环遍历 a7 数组中的每个元素，i 和 j 分别取每个元组中的两个值
>>> for i, j in a7:
...     print(f"i: {i}, j: {j}")
...
i: 1, j: 2
i: 3, j: 4
i: 5, j: 6
i: 7, j: 8
>>> a = np.array([(1,2,3), (4,5,6), (7,8,9)])print("ndim:", a.ndim)print("shape:", a.shape)print("size", a.size)print("dtype", a.dtype)
  File "<stdin>", line 1
    a = np.array([(1,2,3), (4,5,6), (7,8,9)])print("ndim:", a.ndim)print("shape:", a.shape)print("size", a.size)print("dtype", a.dtype)
                                             ^^^^^
SyntaxError: invalid syntax
>>> a = np.array([(1,2,3), (4,5,6a = np.array([(1,2,3), (4,5,6), (7,8,9)]))])
  File "<stdin>", line 1
    a = np.array([(1,2,3), (4,5,6a = np.array([(1,2,3), (4,5,6), (7,8,9)]))])
                                ^
SyntaxError: invalid decimal literal
>>> a = np.array([(1,2,3), (4,5,6), (7,8,9)])
>>> print("ndim:", a.ndim)
ndim: 2
>>> print("shape:", a.shape)print("size", a.size)print("dtype", a.dtype)
  File "<stdin>", line 1
    print("shape:", a.shape)print("size", a.size)print("dtype", a.dtype)
                            ^^^^^
SyntaxError: invalid syntax
>>> print("shape:", a.shape)
shape: (3, 3)
>>> print("size", a.size)
size 9
>>> print("dtype", a.dtype)
dtype int64
>>> a = np.array([(1,2,3),(2,3,4),(4,5,6),(6,7,8)])
>>> print("ndim:", a.ndim)
ndim: 2
>>> print("shape:", a.shape)
shape: (4, 3)
>>> print("size", a.size)
size 12
>>> print("dtype", a.dtype)
dtype int64
>>> print (9 in a)
False
>>> print((7,8,9) in a)
False
>>> a7 = np.arange(1,7)
>>> print(a7)
[1 2 3 4 5 6]
>>> print(a7.reshape)
<built-in method reshape of numpy.ndarray object at 0x0000025D7A744390>
>>> a7 = a7.reshape(3,2)
>>> print(a7)
[[1 2]
 [3 4]
 [5 6]]
>>> a = ((1,2,3),(2,3,4),(5,6,7))
>>> print (a)
((1, 2, 3), (2, 3, 4), (5, 6, 7))
>>> a = np.array([(1,2,3), (4,5,6), (7,8,9)])
>>> print (a)
[[1 2 3]
 [4 5 6]
 [7 8 9]]
>>> a = a.T
>>> print(a)
[[1 4 7]
 [2 5 8]
 [3 6 9]]
>>> a = flatten()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'flatten' is not defined
>>> a= a.flatten()
>>> print(a)
[1 4 7 2 5 8 3 6 9]
>>> a8 = np.array([(1,2), (3,4), (5,6),(7,8)])
>>> a8 = a8[:,np.newaxis]
>>> print(a8)
[[[1 2]]

 [[3 4]]

 [[5 6]]

 [[7 8]]]
>>> a8 = a8[:,:,np.newaxis]
>>> print(a8)
[[[[1 2]]]


 [[[3 4]]]


 [[[5 6]]]


 [[[7 8]]]]
>>> a8 = a8[:,:,np.newaxis]
>>> a8 = a8.shape
>>> print(a8)
(4, 1, 1, 1, 2)
>>> a = np.ones((2,2,2))
>>> print(a)
[[[1. 1.]
  [1. 1.]]

 [[1. 1.]
  [1. 1.]]]
>>> b = np.array([(-1,1),(-1,1),(-1,-1]))
  File "<stdin>", line 1
    b = np.array([(-1,1),(-1,1),(-1,-1]))
                                      ^
SyntaxError: closing parenthesis ']' does not match opening parenthesis '('
>>> b = np.array([(-1,1),(-1,1),(-1,-1)])
>>> print(b)
[[-1  1]
 [-1  1]
 [-1 -1]]
>>> print (a+b)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: operands could not be broadcast together with shapes (2,2,2) (3,2)
>>> a = np.ones((2,2))
>>> b = np.array([(-1,1),(-1,1)])
>>> print (a+b)
[[0. 2.]
 [0. 2.]]
>>> print(a-b)
[[2. 0.]
 [2. 0.]]
>>> print(a*b)
[[-1.  1.]
 [-1.  1.]]
>>> a = a.sum()
>>> print(a)
4.0
>>> a =a.prod()
>>> print(a)
4.0
>>> a = np.array([5,3,1,8])
>>> print("mean:",a.mean())
mean: 4.25
>>> print("var:", a.var())
var: 6.6875
>>> print("std:", a.std())
std: 2.5860201081971503
>>> a = np.array([3.6,3.9,4.9])
>>> print("argmax:", a.argmax())
argmax: 2
>>> print("argmin:", a.argmin())
argmin: 0
>>> print("ceil:", np.ceil(a))
ceil: [4. 4. 5.]
>>> print("floor:", np.floor(a))
floor: [3. 3. 4.]
>>> print("rint:", np.rint(a))
rint: [4. 4. 5.]
>>> a=np.array([23,45,12,56,34])
>>> a=a.sort()
>>> print(a)
None
>>> sorted_a = np.sort(a)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\Users\stephanie.chen\miniconda3\Lib\site-packages\numpy\_core\fromnumeric.py", line 1122, in sort
    a.sort(axis=axis, kind=kind, order=order, stable=stable)
numpy.exceptions.AxisError: axis -1 is out of bounds for array of dimension 0
>>> a=np.array([23,45,12,56,34])
>>> sorted_a = np.sort(a)
>>> print(a)
[23 45 12 56 34]
>>> a9 = np.array([[1,2,3],[4,5,6]])
>>> b9 = np.array([[1,2,3],[4,5,6]])
>>> print(a9*b9)
[[ 1  4  9]
 [16 25 36]]
>>> m1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
>>> m2 = np.array([[5, 6], [7, 8]], dtype=np.float32)
>>> result_dot = np.dot(m1, m2)
>>> result_at = m1 @ m2
>>> print("矩阵 1:")
矩阵 1:
>>> print(m1)
[[1. 2.]
 [3. 4.]]
>>> print(m2)
[[5. 6.]
 [7. 8.]]
>>> tprint(result_dot)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'tprint' is not defined. Did you mean: 'print'?
>>> print(result_dot)
[[19. 22.]
 [43. 50.]]
>>> print(result_at)
[[19. 22.]
 [43. 50.]]
>>> np.save('result.npy',manual_result)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'manual_result' is not defined
>>> manual_result = np.array([1, 2, 3, 4, 5])
>>> np.save('result.npy', manual_result)
>>> result_np = np.load('result.npy')
>>> print(result_np)
[1 2 3 4 5]
>>> a = np.array([1,2,3])
>>> b = np.array([4,5,6])
>>> print(a+n)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'n' is not defined. Did you mean: 'np'?
>>> print(a+b)
[5 7 9]
>>> a = np.array([(1,2), (2,2), (3,3), (4,4)])
>>> b = np.array([-1,1])
>>> print(a+b)
[[0 3]
 [1 3]
 [2 4]
 [3 5]]

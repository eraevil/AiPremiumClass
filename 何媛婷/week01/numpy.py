{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1. 2. 3. 4. 5. 6.]\n",
      "[1 2 3 4 5 6]\n",
      "[[1 2]\n",
      " [3 4]]\n",
      "[[1 2 3]\n",
      " [4 5 6]\n",
      " [7 8 9]]\n",
      "[[1. 2. 3.]\n",
      " [4. 5. 6.]\n",
      " [7. 8. 9.]]\n"
     ]
    }
   ],
   "source": [
    "#  创建一个新数组\n",
    "array = np.array([1,2,3,4,5,6],float)\n",
    "print(array)\n",
    "array2 = np.array([1,2,3,4,5,6],int)\n",
    "print(array2)\n",
    "\n",
    "# 创建一个 2x2 的数组\n",
    "array2x2 = np.array([(1,2),(3,4)])\n",
    "print(array2x2)\n",
    "# 得出结论：创建数组的时候默认括号内的为一行\n",
    "\n",
    "# 尝试是否能自动补充,程序会报错，如果缺少维度是不会自动补充的\n",
    "# array2x2_1 = np.array([(1,2),(3)])\n",
    "# print(array2x2_1)\n",
    "\n",
    "# 创建一个3x3的数组\n",
    "array3x3 = np.array([(1,2,3),(4,5,6),(7,8,9)])\n",
    "print(array3x3)\n",
    "\n",
    "# 尝试切换数据类型\n",
    "array3x3_1 = np.array([(1,2,3),(4,5,6),(7,8,9)],float)\n",
    "print(array3x3_1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0 0 0]\n",
      " [0 0 0]\n",
      " [0 0 0]]\n",
      "[[1 1 1]\n",
      " [1 1 1]\n",
      " [1 1 1]]\n",
      "[1 2 3 4 5 6 7 8 9]\n",
      "[[1 0 0]\n",
      " [0 1 0]\n",
      " [0 0 1]]\n",
      "[0.92313342 0.9607827  0.39879273 0.49775707 0.82148994]\n",
      "[1.21189912 1.49086291 1.96364027 1.84775521 2.24204272 2.86368737\n",
      " 0.98110598 3.96011222 1.91159954 2.61879131]\n"
     ]
    }
   ],
   "source": [
    "# NumPy 特殊数组创建\n",
    "a = np.zeros((3,3),dtype = int)\n",
    "print(a)\n",
    "a1 = np.ones((3,3),dtype = int)\n",
    "print(a1)\n",
    "# 得出结论，如果直接用ones ，默认的好像是float类型，可以单独指定类型为int\n",
    "\n",
    "# 创建等差数列\n",
    "#  arange(数列首项，数列尾项，差值)\n",
    "a2 = np.arange(1,10,1)\n",
    "print(a2)\n",
    "\n",
    "# 创建单位矩阵,如果不指定数据类型，默认float\n",
    "a3 = np.eye(3,dtype=int)\n",
    "print(a3)\n",
    "\n",
    "# 创建在[0,1) 之间平均分布的随机数组\n",
    "a4 = np.random.random(5)\n",
    "print(a4)\n",
    "\n",
    "# 创建随机数组，指定均值和标准差\n",
    "average = 2\n",
    "standard = 1\n",
    "a5 = np.random.normal(average,standard,10)\n",
    "print(a5)\n",
    "# 问题1： 怎么设置为 int 格式呢"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "打印数组:[[11 12 13 14]\n",
      " [21 22 23 24]\n",
      " [31 32 33 34]]\n",
      "\n",
      "输出第一行:[11 12 13 14]\n",
      "\n",
      "输出第二列:[12 22 32]\n",
      "\n",
      "输出第一行第二列:12\n",
      "\n",
      "输出第一行第二列:12\n"
     ]
    }
   ],
   "source": [
    "# numpy数组的查询\n",
    "array = np.array([(11,12,13,14),(21,22,23,24),(31,32,33,34)])\n",
    "print('\\n打印数组:'+str(array))\n",
    "\n",
    "print('\\n输出第一行:'+str(array[0]))\n",
    "\n",
    "print('\\n输出第二列:'+str(array[:,1]))\n",
    "\n",
    "print('\\n输出第一行第二列:'+str(array[0,1]))\n",
    "\n",
    "print('\\n输出第一行第二列:'+str(array[0][1]))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "11\n",
      "12\n",
      "13\n",
      "14\n",
      "21\n",
      "22\n",
      "23\n",
      "24\n",
      "31\n",
      "32\n",
      "33\n",
      "34\n",
      "11 12 13 14\n",
      "21 22 23 24\n",
      "31 32 33 34\n"
     ]
    }
   ],
   "source": [
    "# 数组遍历\n",
    "array = np.array([(11,12,13,14),(21,22,23,24),(31,32,33,34)])\n",
    "\n",
    "# 打印单个元素\n",
    "for i in array:\n",
    "    for j in i:\n",
    "        print(j)\n",
    "# 打印单个元素\n",
    "for a,b,c,d in array:\n",
    "    print(a,b,c,d)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "数组大小：12\n",
      "数组是否包含：True\n",
      "数组打平：[11 12 13 14 21 22 23 24 31 32 33 34]\n",
      "数组打平：[11 12 13 14 21 22 23 24 31 32 33 34]\n",
      "数组转置：[[11 21 31]\n",
      " [12 22 32]\n",
      " [13 23 33]\n",
      " [14 24 34]]\n",
      "数组转置：[[11 21 31]\n",
      " [12 22 32]\n",
      " [13 23 33]\n",
      " [14 24 34]]\n",
      "数组行列：(3, 4)\n",
      "增加维度后：(3, 1, 4)\n"
     ]
    }
   ],
   "source": [
    "array = np.array([(11,12,13,14),(21,22,23,24),(31,32,33,34)])\n",
    "\n",
    "print('数组大小：'+ str(array.size))\n",
    "print('数组是否包含：'+ str(11 in array))\n",
    "print('数组打平：'+ str(array.reshape(array.size)))\n",
    "print('数组打平：'+ str(array.flatten()))\n",
    "print('数组转置：'+ str(array.transpose()))\n",
    "print('数组转置：'+ str(array.T))\n",
    "print('数组行列：'+ str(array.shape))\n",
    "# 增加维度\n",
    "array = array[:,np.newaxis]\n",
    "print('增加维度后：'+str(array.shape))\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 5 12]\n",
      " [21 32]]\n",
      "[[ 6  8]\n",
      " [10 12]]\n",
      "[[-4 -4]\n",
      " [-4 -4]]\n",
      "[[0.2        0.33333333]\n",
      " [0.42857143 0.5       ]]\n",
      "数组内和： 10\n",
      "数组内乘积： 24\n",
      "数组内最大值： 4\n",
      "数组内最小值： 1\n",
      "数组内平均值： 2.5\n",
      "数组内方差： 1.25\n",
      "数组内标准差： 1.118033988749895\n",
      "数组排序后： [[6 7]\n",
      " [7 8]\n",
      " [1 2]\n",
      " [3 4]]\n"
     ]
    }
   ],
   "source": [
    "# 数学运算\n",
    "arr = np.array([(7,6),(7,8),(1,2),(3,4)])\n",
    "arr1 = np.array([(1,2),(3,4)])\n",
    "arr2 = np.array([(5,6),(7,8)])\n",
    "\n",
    "print(arr1*arr2)\n",
    "print(arr1+arr2)\n",
    "print(arr1-arr2)\n",
    "print(arr1/arr2)\n",
    "print('数组内和：',arr1.sum())\n",
    "print('数组内乘积：',arr1.prod())\n",
    "print('数组内最大值：',arr1.max())\n",
    "print('数组内最小值：',arr1.min())\n",
    "print('数组内平均值：',arr1.mean())\n",
    "print('数组内方差：',arr1.var())\n",
    "print('数组内标准差：',arr1.std())\n",
    "arr.sort()\n",
    "print('数组排序后：',arr)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "矩阵 1: [[1 2]\n",
      " [3 4]]\n",
      "矩阵 2: [[5 6]\n",
      " [7 8]]\n",
      "使⽤ np.dot 得到的矩阵乘法结果:\n",
      "[[19 22]\n",
      " [43 50]]\n",
      "使⽤ @ 运算符得到的矩阵乘法结果:\n",
      "[[19 22]\n",
      " [43 50]]\n",
      "1 * 5 = 5\n",
      "2 * 7 = 14\n",
      "1 * 6 = 6\n",
      "2 * 8 = 16\n",
      "结果矩阵[1,2]:22.0\n",
      "\n",
      "3 * 5 = 15\n",
      "4 * 7 = 28\n",
      "3 * 6 = 18\n",
      "4 * 8 = 32\n",
      "结果矩阵[2,2]:50.0\n",
      "\n",
      "⼿动推演结果:\n",
      "[[19. 22.]\n",
      " [43. 50.]]\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# 定义两个简单的矩阵\n",
    "m1 = np.array([[1, 2], [3, 4]] , dtype=np.int_)\n",
    "m2 = np.array([[5, 6], [7, 8]] , dtype=np.int_)\n",
    "# 使⽤ np.dot 进⾏矩阵乘法\n",
    "result_dot = np.dot(m1, m2)\n",
    "# 使⽤ @ 运算符进⾏矩阵乘法\n",
    "result_at = m1 @ m2\n",
    "print(\"矩阵 1:\",m1)\n",
    "print(\"矩阵 2:\",m2)\n",
    "print(\"使⽤ np.dot 得到的矩阵乘法结果:\")\n",
    "print(result_dot)\n",
    "\n",
    "print(\"使⽤ @ 运算符得到的矩阵乘法结果:\")\n",
    "print(result_at)\n",
    "# 创建⼀个全零矩阵，⽤于存储⼿动推演的结果\n",
    "# 结果矩阵的⾏数等于 matrix1 的⾏数，列数等于 matrix2 的列数\n",
    "manual_result = np.zeros((m1.shape[0], m2.shape[1]), dtype=np.float32)\n",
    "# 外层循环：遍历 matrix1 的每⼀⾏\n",
    "# i 表⽰结果矩阵的⾏索引\n",
    "for i in range(m1.shape[0]):\n",
    "    # 中层循环：遍历 matrix2 的每⼀列\n",
    "    # j 表⽰结果矩阵的列索引\n",
    "    for j in range(m2.shape[1]):\n",
    "        # 初始化当前位置的结果为 0\n",
    "        manual_result[i, j] = 0\n",
    "        # 内层循环：计算 matrix1 的第 i ⾏与 matrix2 的第 j 列对应元素的乘积之和\n",
    "        # k 表⽰参与乘法运算的元素索引\n",
    "        for k in range(m1.shape[1]):\n",
    "            # 打印当前正在计算的元素\n",
    "            print(f\"{m1[i, k]} * {m2[k, j]} = {m1[i, k] * m2[k, j]}\")\n",
    "            # 将 matrix1 的第 i ⾏第 k 列元素与 matrix2 的第 k ⾏第 j 列元素相乘，并累\n",
    "            manual_result[i, j] += m1[i, k] * m2[k, j]\n",
    "    # 打印当前位置计算完成后的结果\n",
    "    print(f\"结果矩阵[{i+1},{j+1}]:{manual_result[i, j]}\\n\")\n",
    "print(\"⼿动推演结果:\")\n",
    "print(manual_result)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "py3124",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

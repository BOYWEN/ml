from pickle import TRUE
from re import X
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
# 原始形式(例2.2  p45)
# 数据准备
x_train = np.array([[3, 3],
              [4, 3],
              [1, 1]])

y_train= np.array([1, 1, -1])

# 可视化
fig = plt.figure()
marker = ["o", "^"]
for i in range(np.shape(y_train)[-1]):
    plt.scatter(x_train[i, 0], x_train[i, 1], color="red", marker=marker[0] if y_train[i]==-1 else marker[1])

plt.show()

# 参数设置
alpha = np.zeros(np.shape(y_train)) # 误分类次数列表
b = 0
yita = 1
# 计算Gram矩阵
Gram_mat = np.dot(x_train, x_train.T)
print(Gram_mat)

iter = 0
# 开始训练
while iter <= np.shape(y_train)[0]-1:
    # 误分条件
    if_upgrade = y_train[iter] * (np.sum(alpha*y_train*Gram_mat[iter,:])+b)
    print("判据：",if_upgrade)
    if if_upgrade <= 0:
        # 更新参数
        alpha[iter] += yita
        b += yita * y_train[iter]
        print(alpha)
        # 更新后索引置零
        iter = 0
        continue
    iter += 1

# 计算 w
w = x_train * y_train[:, np.newaxis]
w = np.dot(alpha, w)
print("w = ", w)
print("b = ", b)

# 可视化分类结果
for i in range(np.shape(y_train)[-1]):
    plt.scatter(x_train[i, 0], x_train[i, 1], color="red", marker=marker[0] if y_train[i]==-1 else marker[1])
x_line = np.array([[0, -b/w[0]],[-b/w[1], 0]])
plt.plot(x_line[0], x_line[1])
plt.show()
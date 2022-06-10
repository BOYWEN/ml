from pickle import TRUE
from re import X
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
# 原始形式(例2.1  p40)
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
# 初始化参数
w = np.zeros(np.shape(x_train)[1])
b = 0
yita = 1

# 训练
iter = 0
while iter <= np.shape(y_train)[0]-1:
    # 对负样本进行参数更新
    # 误分类条件
    if_upgrade = y_train[iter]*(np.dot(w, x_train[iter])+b)
    print("判据：", if_upgrade)
    if if_upgrade <= 0:
        print(iter)
        w = w + yita*y_train[iter]*x_train[iter]
        b = b + yita*y_train[iter]
        print(w)
        print(b)
        # 更新参数后重新循环，索引置零
        iter = 0
        continue
    iter += 1


# 可视化分类结果
for i in range(np.shape(y_train)[-1]):
    plt.scatter(x_train[i, 0], x_train[i, 1], color="red", marker=marker[0] if y_train[i]==-1 else marker[1])
x_line = np.array([[0, -b/w[0]],[-b/w[1], 0]])
plt.plot(x_line[0], x_line[1])
plt.show()
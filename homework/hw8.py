# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:04:38 2021

@author: htchen
"""
# If this script is not run under spyder IDE, comment the following two lines.
# from IPython import get_ipython
# get_ipython().run_line_magic('reset', '-sf')

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

# 读取数据
hw8_csv = pd.read_csv('E:/homework/data/hw8.csv')
hw8_dataset = hw8_csv.to_numpy(dtype=np.float64)

X0 = hw8_dataset[:, 0:2]  # 特征
y = hw8_dataset[:, 2]      # 标签 (1 或 -1)

print(f"数据形状: X={X0.shape}, y={y.shape}")
print(f"类别分布: Class 1={np.sum(y==1)}, Class -1={np.sum(y==-1)}")

# ========== 训练分类器 ==========
# 你可以选择以下任一方法，这里使用 SVM

# 方法1: SVM (Support Vector Machine) - 推荐
classifier = SVC(kernel='rbf', C=10.0, gamma='auto')

# 方法2: 神经网络
# classifier = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

# 方法3: AdaBoost
# classifier = AdaBoostClassifier(n_estimators=100, random_state=42)

# 方法4: 逻辑回归
# classifier = LogisticRegression(max_iter=1000)

# 训练模型
classifier.fit(X0, y)

# 计算训练准确率
train_accuracy = classifier.score(X0, y)
print(f"\n训练准确率: {train_accuracy * 100:.2f}%")

# ========== 创建网格用于可视化决策边界 ==========
# 确定绘图范围
x1_min, x1_max = X0[:, 0].min() - 1, X0[:, 0].max() + 1
x2_min, x2_max = X0[:, 1].min() - 1, X0[:, 1].max() + 1

# 创建网格点
h = 0.1  # 网格步长
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h),
                       np.arange(x2_min, x2_max, h))

# 对网格上的每个点进行预测
X_grid = np.c_[xx1.ravel(), xx2.ravel()]
Z = classifier.predict(X_grid)
Z = Z.reshape(xx1.shape)

# ========== 绘图 ==========
fig = plt.figure(figsize=(10, 8), dpi=150)

# 绘制决策边界的着色区域
# 使用 contourf 填充不同分类区域
plt.contourf(xx1, xx2, Z, levels=[-1.5, 0, 1.5], 
             colors=['lightgreen', 'darkseagreen'], 
             alpha=0.6)

# 绘制决策边界线
plt.contour(xx1, xx2, Z, levels=[0], 
            colors='black', linewidths=2, linestyles='--')

# 绘制原始数据点
plt.scatter(X0[y == 1, 0], X0[y == 1, 1], 
           c='red', s=40, edgecolors='black', linewidth=0.5,
           label='$\omega_1$', zorder=3)
plt.scatter(X0[y == -1, 0], X0[y == -1, 1], 
           c='blue', s=40, edgecolors='black', linewidth=0.5,
           label='$\omega_2$', zorder=3)

plt.xlabel('$x_1$', fontsize=12)
plt.ylabel('$x_2$', fontsize=12)
plt.title(f'Binary Classification with SVM\nAccuracy: {train_accuracy*100:.2f}%', 
          fontsize=14)
plt.xlim([x1_min, x1_max])
plt.ylim([x2_min, x2_max])
plt.axis('equal')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ========== 输出分类器信息 ==========
print(f"\n使用的分类器: {type(classifier).__name__}")
if isinstance(classifier, SVC):
    print(f"SVM 参数: kernel={classifier.kernel}, C={classifier.C}, gamma={classifier.gamma}")
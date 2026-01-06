# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:04:38 2021

@author: htchen
"""

# from IPython import get_ipython
# get_ipython().run_line_magic('reset', '-sf')

import math
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd


# calculate the eigenvalues and eigenvectors of a squared matrix
# the eigenvalues are decreasing ordered
def myeig(A, symmetric=False):
    if symmetric:
        lambdas, V = np.linalg.eigh(A)
    else:
        lambdas, V = np.linalg.eig(A)
    # lambdas, V may contain complex value
    lambdas_real = np.real(lambdas)
    sorted_idx = lambdas_real.argsort()[::-1] 
    return lambdas[sorted_idx], V[:, sorted_idx]


# class 1
mean1 = np.array([0, 5])
sigma1 = np.array([[0.3, 0.2],[0.2, 1]])
N1 = 200
X1 = np.random.multivariate_normal(mean1, sigma1, N1)

# class 2
mean2 = np.array([3, 4])
sigma2 = np.array([[0.3, 0.2],[0.2, 1]])
N2 = 100
X2 = np.random.multivariate_normal(mean2, sigma2, N2)

# m1: mean of class 1
# m2: mean of class 2
m1 = np.mean(X1, axis=0, keepdims=True)  # shape: (1, 2)
m2 = np.mean(X2, axis=0, keepdims=True)  # shape: (1, 2)

# ========== LDA 計算 ==========
# Step 1: 計算類內散布矩陣 Sw (Within-class scatter matrix)
# S1 = Σ(x - m1)(x - m1)^T for all x in class 1
S1 = np.zeros((2, 2))
for i in range(N1):
    diff = (X1[i:i+1, :] - m1).T  # shape: (2, 1)
    S1 += diff @ diff.T

# S2 = Σ(x - m2)(x - m2)^T for all x in class 2
S2 = np.zeros((2, 2))
for i in range(N2):
    diff = (X2[i:i+1, :] - m2).T  # shape: (2, 1)
    S2 += diff @ diff.T

# 類內散布矩陣
Sw = S1 + S2

# Step 2: 計算最佳投影方向 w
# w = Sw^(-1) * (m2 - m1)^T
mean_diff = (m2 - m1).T  # shape: (2, 1)
w = la.inv(Sw) @ mean_diff  # shape: (2, 1)

# 標準化 w（單位向量）
w = w / la.norm(w)

print(f"投影方向 w = {w.flatten()}")
print(f"w 的長度 = {la.norm(w)}")

# Step 3: 計算投影
# 將每個點投影到 w 方向上，得到標量投影值
proj_scalar_1 = X1 @ w  # shape: (N1, 1) - 每個點在 w 上的標量投影
proj_scalar_2 = X2 @ w  # shape: (N2, 1)

# 將標量投影轉換回2D空間中的點
proj_points_1 = proj_scalar_1 @ w.T  # shape: (N1, 2)
proj_points_2 = proj_scalar_2 @ w.T  # shape: (N2, 2)

# ========== 繪圖 ==========
plt.figure(figsize=(8, 6), dpi=150)

# 繪製原始數據點
plt.plot(X1[:, 0], X1[:, 1], 'r.', markersize=6, alpha=0.6, label='Class 1')
plt.plot(X2[:, 0], X2[:, 1], 'g.', markersize=6, alpha=0.6, label='Class 2')

# 繪製投影點（在 w 方向上）
# 使用較粗的線條來顯示投影後的數據分布
plt.plot(proj_points_1[:, 0], proj_points_1[:, 1], 'r-', 
         linewidth=8, alpha=0.7, solid_capstyle='round')
plt.plot(proj_points_2[:, 0], proj_points_2[:, 1], 'g-', 
         linewidth=8, alpha=0.7, solid_capstyle='round')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Linear Discriminant Analysis (LDA)')
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 輸出統計信息
print(f"\n類別 1 均值: {m1.flatten()}")
print(f"類別 2 均值: {m2.flatten()}")
print(f"\n類別 1 投影範圍: [{proj_scalar_1.min():.4f}, {proj_scalar_1.max():.4f}]")
print(f"類別 2 投影範圍: [{proj_scalar_2.min():.4f}, {proj_scalar_2.max():.4f}]")
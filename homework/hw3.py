# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 12:38:53 2024

@author: htchen
"""

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

# SVD: A = U * Sigma * V^T
# V: eigenvector matrix of A^T * A; U: eigenvector matrix of A * A^T 
def mysvd(A):
    lambdas, V = myeig(A.T @ A, symmetric=True)
    lambdas, V = np.real(lambdas), np.real(V)
    # if A is full rank, no lambda value is less than 1e-6 
    # append a small value to stop rank check
    lambdas = np.append(lambdas, 1e-12)
    rank = np.argwhere(lambdas < 1e-6).min()
    lambdas, V = lambdas[0:rank], V[:, 0:rank]
    U = A @ V / np.sqrt(lambdas)
    Sigma = np.diag(np.sqrt(lambdas))
    return U, Sigma, V


pts = 50
x = np.linspace(-2, 2, pts)
y = np.zeros(x.shape)

# square wave
pts2 = pts // 2
y[0:pts2] = -1
y[pts2:] = 1

# sort x
argidx = np.argsort(x)
x = x[argidx]
y = y[argidx]

T0 = np.max(x) - np.min(x)
f0 = 1.0 / T0
omega0 = 2.0 * np.pi * f0

# step1: generate X=[1 cos(omega0 x) cos(omega0 2x) ... cos(omega0 nx) sin(omega0 x) sin(omega0 2x) ... sin(omega0 nx)]
n = 5  # 使用前 5 個頻率成分
X = np.ones((pts, 2*n + 1))  # 初始化矩陣，第一列為常數項 1

# 填入 cos 項: cos(kω₀x), k=1,2,...,n
for k in range(1, n + 1):
    X[:, k] = np.cos(k * omega0 * x)

# 填入 sin 項: sin(kω₀x), k=1,2,...,n
for k in range(1, n + 1):
    X[:, n + k] = np.sin(k * omega0 * x)

# step2: SVD of X => X=USV^T
U, Sigma, V = mysvd(X)

# step3: a = V @ Sigma^-1 @ U^T @ y
Sigma_inv = np.linalg.inv(Sigma)
a = V @ Sigma_inv @ U.T @ y

# 預測值
y_bar = X @ a

# 繪圖
plt.figure(figsize=(10, 6))
plt.plot(x, y_bar, 'g-', linewidth=2, label='predicted values') 
plt.plot(x, y, 'b-', linewidth=2, label='true values')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title(f'Fourier Series Approximation (n={n})')
plt.show()

# 顯示擬合係數
print(f"擬合係數 a (a₀, a₁, ..., a_{n}, b₁, ..., b_{n}):")
print(a)
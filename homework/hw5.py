# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 09:37:05 2021

@author: htchen
"""
# If this script is not run under spyder IDE, comment the following two lines.
#from IPython import get_ipython
#get_ipython().run_line_magic('reset', '-sf')

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

def row_norm_square(X):
    return np.sum(X * X, axis=1)

# gaussian weight array g=[ g_1 g_2 ... g_m ]
# g_i = exp(-0.5 * ||x_i - c||^2 / sigma^2)
def gaussian_weight(X, c, sigma=1.0):
    s = 0.5 / sigma / sigma;
    norm2 = row_norm_square(X - c)
    g = np.exp(-s * norm2)
    return g

# xt: a sample in Xt
# yt: predicted value of f(xt)
# yt = (X.T @ G(xt) @ X)^-1 @ X.T @ G(xt) @ y
def predict(X, y, Xt, sigma=1.0):
    ntest = Xt.shape[0] # number of test samples 
    yt = np.zeros(ntest)
    for xi in range(ntest):
        c = Xt[xi, :]
        g = gaussian_weight(X, c, sigma) # diagonal elements in G
        G = np.diag(g)
        w = la.pinv(X.T @ G @ X) @ X.T @ G @ y
        yt[xi] = c @ w
    return yt

# Xs: m x n matrix; 
# m: pieces of sample
# K: m x m kernel matrix
# K[i,j] = exp(-c(|xt_i|^2 + |xs_j|^2 -2(xt_i)^T @ xs_j)) where c = 0.5 / sigma^2
# 更多實作說明, 參考課程oneonte筆記

def calc_gaussian_kernel(Xt, Xs, sigma=1):
    nt, _ = Xt.shape # pieces of Xt
    ns, _ = Xs.shape # pieces of Xs
    
    norm_square = row_norm_square(Xt)
    F = np.tile(norm_square, (ns, 1)).T
    
    norm_square = row_norm_square(Xs)
    G = np.tile(norm_square, (nt, 1))
    
    E = F + G - 2.0 * Xt @ Xs.T
    s = 0.5 / (sigma * sigma)
    K = np.exp(-s * E)
    return K

# n: degree of polynomial
# generate X=[1 x x^2 x^3 ... x^n]
# m: pieces(rows) of data(X)
# X is a m x (n+1) matrix
def poly_data_matrix(x: np.ndarray, n: int):
    m = x.shape[0]
    X = np.zeros((m, n + 1))
    X[:, 0] = 1.0
    for deg in range(1, n + 1):
        X[:, deg] = X[:, deg - 1] * x
    return X  # 修正：返回 X 而不是 deg

hw5_csv = pd.read_csv('E:/homework/data/hw5.csv')
hw5_dataset = hw5_csv.to_numpy(dtype = np.float64)

hours = hw5_dataset[:, 0]
sulfate = hw5_dataset[:, 1]

# ====== 圖1: 濃度 vs 時間散點圖 ======
plt.figure(figsize=(8, 6))
plt.scatter(hours, sulfate, color='red', s=30, alpha=0.6)
plt.title('concentration vs time')
plt.xlabel('time in hours')
plt.ylabel('sulfate concentration (times $10^{-4}$)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ====== 圖2: 使用局部加權回歸繪製預測曲線（藍色曲線）======
# 準備訓練數據
X_train = poly_data_matrix(hours, 1)  # 使用線性基底 [1, x]

# 生成測試數據（更密集的點以得到平滑曲線）
hours_test = np.linspace(hours.min(), hours.max(), 200)
X_test = poly_data_matrix(hours_test, 1)

# 使用局部加權回歸預測（調整 sigma 以獲得平滑效果）
sulfate_pred = predict(X_train, sulfate, X_test, sigma=20.0)

plt.figure(figsize=(8, 6))
plt.plot(hours_test, sulfate_pred, color='blue', linewidth=2, label='predicted values')
plt.scatter(hours, sulfate, color='red', s=30, alpha=0.6, label='true values')
plt.title('concentration vs time')
plt.xlabel('time in hours')
plt.ylabel('sulfate concentration (times $10^{-4}$)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ====== 圖3: log-log scale 散點圖 ======
plt.figure(figsize=(8, 6))
plt.scatter(hours, sulfate, color='red', s=30, alpha=0.6)
plt.xscale("log")
plt.yscale("log")
plt.title('concentration vs time (log-log scale)')
plt.xlabel('time in hours')
plt.ylabel('sulfate concentration (times $10^{-4}$)')
plt.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.show()

# ====== 圖4: log-log scale 下的線性回歸（藍色直線）======
# 對數轉換
log_hours = np.log(hours)
log_sulfate = np.log(sulfate)

# 線性回歸: log(sulfate) = a + b * log(hours)
X_log = poly_data_matrix(log_hours, 1)  # [1, log(hours)]
a_log = la.pinv(X_log.T @ X_log) @ X_log.T @ log_sulfate

# 生成預測曲線
log_hours_test = np.linspace(log_hours.min(), log_hours.max(), 200)
X_log_test = poly_data_matrix(log_hours_test, 1)
log_sulfate_pred = X_log_test @ a_log

# 轉回原始尺度
hours_test_log = np.exp(log_hours_test)
sulfate_pred_log = np.exp(log_sulfate_pred)

plt.figure(figsize=(8, 6))
plt.plot(hours_test_log, sulfate_pred_log, color='blue', linewidth=2, label='regression line')
plt.scatter(hours, sulfate, color='red', s=30, alpha=0.6, label='data')
plt.xscale("log")
plt.yscale("log")
plt.title('concentration vs time (log-log scale)')
plt.xlabel('time in hours')
plt.ylabel('sulfate concentration (times $10^{-4}$)')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.show()

# 輸出回歸係數
print(f"\n在 log-log scale 下的線性回歸結果:")
print(f"log(sulfate) = {a_log[0]:.4f} + {a_log[1]:.4f} * log(hours)")
print(f"或寫成: sulfate = {np.exp(a_log[0]):.4f} * hours^{a_log[1]:.4f}")
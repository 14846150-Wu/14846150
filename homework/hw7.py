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

def scatter_pts_2d(x, y):
    # set plotting limits
    xmax = np.max(x)
    xmin = np.min(x)
    xgap = (xmax - xmin) * 0.2
    xmin -= xgap
    xmax += xgap

    ymax = np.max(y)
    ymin = np.min(y)
    ygap = (ymax - ymin) * 0.2
    ymin -= ygap
    ymax += ygap 

    return xmin, xmax, ymin, ymax

# 定義成本函數
def cost_function(w, x, y):
    """
    J(w) = sum((y_i - w[0] - w[1] * sin(w[2] * x_i + w[3]))^2)
    """
    m = len(x)
    predictions = w[0] + w[1] * np.sin(w[2] * x + w[3])
    errors = y - predictions
    cost = np.sum(errors ** 2)
    return cost

# 解析法計算梯度
def analytic_gradient(w, x, y):
    """
    計算梯度的解析解
    e_i = y_i - w[0] - w[1] * sin(w[2] * x_i + w[3])
    """
    m = len(x)
    
    # 計算誤差 e_i
    predictions = w[0] + w[1] * np.sin(w[2] * x + w[3])
    errors = y - predictions  # e_i
    
    # 計算各個偏導數
    # ∂J/∂w_1 = -Σ 2e_i
    dJ_dw0 = -np.sum(2 * errors)
    
    # ∂J/∂w_2 = -Σ [2e_i * sin(w_3*x_i + w_4)]
    dJ_dw1 = -np.sum(2 * errors * np.sin(w[2] * x + w[3]))
    
    # ∂J/∂w_3 = -Σ [2e_i * x_i * w_2 * cos(w_3*x_i + w_4)]
    dJ_dw2 = -np.sum(2 * errors * x * w[1] * np.cos(w[2] * x + w[3]))
    
    # ∂J/∂w_4 = -Σ [2e_i * w_2 * cos(w_3*x_i + w_4)]
    dJ_dw3 = -np.sum(2 * errors * w[1] * np.cos(w[2] * x + w[3]))
    
    gradient = np.array([dJ_dw0, dJ_dw1, dJ_dw2, dJ_dw3])
    return gradient

# 數值法計算梯度
def numeric_gradient(w, x, y, epsilon=1e-8):
    """
    使用數值微分計算梯度
    ∂J/∂w_k ≈ [J(w + ε*e_k) - J(w)] / ε
    """
    gradient = np.zeros_like(w)
    
    for i in range(len(w)):
        w_plus = w.copy()
        w_plus[i] += epsilon
        
        cost_plus = cost_function(w_plus, x, y)
        cost_current = cost_function(w, x, y)
        
        gradient[i] = (cost_plus - cost_current) / epsilon
    
    return gradient

# 讀取數據
dataset = pd.read_csv('E:/homework/data/hw7.csv').to_numpy(dtype=np.float64)
x = dataset[:, 0]
y = dataset[:, 1]

# 參數設置
alpha = 0.05  # 學習率
max_iters = 500  # 最大迭代次數

print("=" * 60)
print("方法 1: 解析法 (Analytic Method)")
print("=" * 60)

# ========== 方法 1: 解析法 ==========
w_analytic = np.array([-0.1607108, 2.0808538, 0.3277537, -1.5511576])

for iteration in range(1, max_iters):
    # 計算梯度（解析法）
    gradient = analytic_gradient(w_analytic, x, y)
    
    # 更新參數
    w_analytic = w_analytic - alpha * gradient
    
    # 每100次迭代輸出一次成本
    if iteration % 100 == 0 or iteration == 1:
        cost = cost_function(w_analytic, x, y)
        print(f"Iteration {iteration:3d}: Cost = {cost:.6f}, w = {w_analytic}")

print(f"\n最終參數 (解析法): {w_analytic}")

# 生成預測曲線
xmin, xmax, ymin, ymax = scatter_pts_2d(x, y)
xt = np.linspace(xmin, xmax, 100)
yt1 = w_analytic[0] + w_analytic[1] * np.sin(w_analytic[2] * xt + w_analytic[3])

print("\n" + "=" * 60)
print("方法 2: 數值法 (Numeric Method)")
print("=" * 60)

# ========== 方法 2: 數值法 ==========
w_numeric = np.array([-0.1607108, 2.0808538, 0.3277537, -1.5511576])

for iteration in range(1, max_iters):
    # 計算梯度（數值法）
    gradient = numeric_gradient(w_numeric, x, y, epsilon=1e-8)
    
    # 更新參數
    w_numeric = w_numeric - alpha * gradient
    
    # 每100次迭代輸出一次成本
    if iteration % 100 == 0 or iteration == 1:
        cost = cost_function(w_numeric, x, y)
        print(f"Iteration {iteration:3d}: Cost = {cost:.6f}, w = {w_numeric}")

print(f"\n最終參數 (數值法): {w_numeric}")

# 生成預測曲線
yt2 = w_numeric[0] + w_numeric[1] * np.sin(w_numeric[2] * xt + w_numeric[3])

# ========== 繪圖 ==========
fig = plt.figure(figsize=(10, 6), dpi=150)
plt.scatter(x, y, color='k', edgecolor='w', linewidth=0.9, s=60, zorder=3, label='Data')
plt.plot(xt, yt1, linewidth=4, c='b', zorder=1, label='Analytic method', alpha=0.7)
plt.plot(xt, yt2, linewidth=2, c='r', zorder=2, label='Numeric method')
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.xlabel('$x$', fontsize=12)
plt.ylabel('$y$', fontsize=12)
plt.title('Gradient Descent: Fitting Sine Wave', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 比較兩種方法的結果
print("\n" + "=" * 60)
print("結果比較")
print("=" * 60)
print(f"解析法最終參數: {w_analytic}")
print(f"數值法最終參數: {w_numeric}")
print(f"參數差異: {np.abs(w_analytic - w_numeric)}")
print(f"解析法最終成本: {cost_function(w_analytic, x, y):.6f}")
print(f"數值法最終成本: {cost_function(w_numeric, x, y):.6f}")
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 10:46:50 2021

@author: htchen
"""

# from IPython import get_ipython
# get_ipython().run_line_magic('reset', '-sf')
import math
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd
figdpi = 400

# 讀取數據
hw9_csv = pd.read_csv('E:/homework/data/hw9.csv').to_numpy(dtype=np.float64)
t = hw9_csv[:, 0]  # 時間（秒）
flow_velocity = hw9_csv[:, 1]  # 氣體流速（ml/sec）

# ========== 圖1: 氣體流速 vs 時間 ==========
plt.figure(dpi=figdpi)
plt.plot(t, flow_velocity, 'r', linewidth=1.5)
plt.title('Gas Flow Velocity', fontsize=14)
plt.xlabel('time in seconds', fontsize=12)
plt.ylabel('ml/sec', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ========== 圖2: 積分得到淨流量 ==========
# 使用累積和（cumsum）来積分流速，得到淨流量
# 乘以時間步長 0.01 秒
net_vol = np.cumsum(flow_velocity) * 0.01

plt.figure(dpi=figdpi)
plt.plot(t, net_vol, 'r', linewidth=1.5)
plt.title('Gas Net Flow (with drift)', fontsize=14)
plt.xlabel('time in seconds', fontsize=12)
plt.ylabel('ml', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ========== 計算趨勢線（二次多項式擬合）==========
# 構建設計矩陣 A = [1, t, t²]
A = np.zeros((len(t), 3))
A[:, 0] = 1      # 常數項
A[:, 1] = t      # 一次項
A[:, 2] = t * t  # 二次項

# 最小二乘法求解係數
y = net_vol
a = la.inv(A.T @ A) @ A.T @ y

# 計算趨勢線
trend_curve = a[0] + a[1] * t + a[2] * t * t

print("=" * 60)
print("趨勢線擬合系数（二次多項式）")
print("=" * 60)
print(f"y = {a[0]:.4f} + {a[1]:.4f}*t + {a[2]:.6f}*t²")
print("=" * 60)

# ========== 圖3: 去除趨勢後的淨流量 ==========
# 從淨流量中減去趨勢線，得到去除累積誤差後的結果
detrended_net_vol = net_vol - trend_curve

plt.figure(dpi=figdpi)
plt.plot(t, detrended_net_vol, 'r', linewidth=1.5)
plt.title('Gas Net Flow (detrended)', fontsize=14)
plt.xlabel('time in seconds', fontsize=12)
plt.ylabel('ml', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 上圖：原始淨流量 + 趨勢線
axes[0].plot(t, net_vol, 'r-', linewidth=1.5, label='Original Net Flow', alpha=0.7)
axes[0].plot(t, trend_curve, 'b--', linewidth=2, label='Trend Line (Polynomial Fit)')
axes[0].set_title('Gas Net Flow with Drift', fontsize=14)
axes[0].set_xlabel('time in seconds', fontsize=12)
axes[0].set_ylabel('ml', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# 下圖：去除趨勢後的淨流量
axes[1].plot(t, detrended_net_vol, 'r-', linewidth=1.5)
axes[1].axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
axes[1].set_title('Gas Net Flow (Detrended)', fontsize=14)
axes[1].set_xlabel('time in seconds', fontsize=12)
axes[1].set_ylabel('ml', fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ========== 輸出統計信息 ==========
print("\n統計信息:")
print(f"原始淨流量範圍: [{net_vol.min():.2f}, {net_vol.max():.2f}] ml")
print(f"去趨勢後範圍: [{detrended_net_vol.min():.2f}, {detrended_net_vol.max():.2f}] ml")
print(f"去趨勢後均值: {detrended_net_vol.mean():.4f} ml")
print(f"去趨勢後標準差: {detrended_net_vol.std():.4f} ml")
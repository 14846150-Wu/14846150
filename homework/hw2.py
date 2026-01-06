# -*- coding: utf-8 -*-
# from IPython import get_ipython
# get_ipython().run_line_magic('reset', '-sf')

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import cv2


plt.rcParams['figure.dpi'] = 144 


def myeig(A, symmetric=False):
    if symmetric:
        lambdas, V = np.linalg.eigh(A)
    else:
        lambdas, V = np.linalg.eig(A)
    lambdas_real = np.real(lambdas)
    sorted_idx = lambdas_real.argsort()[::-1] 
    return lambdas[sorted_idx], V[:, sorted_idx]


def mysvd(A):
    lambdas, V = myeig(A.T @ A, symmetric=True)
    lambdas, V = np.real(lambdas), np.real(V)
    lambdas = np.append(lambdas, 1e-12)
    rank = np.argwhere(lambdas < 1e-6).min()
    lambdas, V = lambdas[0:rank], V[:, 0:rank]
    U = A @ V / np.sqrt(lambdas)
    Sigma = np.diag(np.sqrt(lambdas))
    return U, Sigma, V


# ========= 讀圖 =========
img = cv2.imread('E:/homework/data/svd_demo1.jpg', cv2.IMREAD_GRAYSCALE)
A = img.astype(dtype=np.float64)

# ========= SVD =========
U, Sigma, V = mysvd(A)
VT = V.T


# ========= (你作業要寫的) Energy function =========
def compute_energy(X: np.ndarray):
    return np.sum(X**2)


# ========= SNR & Noise Energy =========
img_h, img_w = A.shape
keep_r = 201
rs = np.arange(1, keep_r)

energy_A = compute_energy(A)
energy_N = np.zeros(keep_r)

for r in rs:
    A_bar = U[:, 0:r] @ Sigma[0:r, 0:r] @ VT[0:r, :]
    Noise = A - A_bar
    energy_N[r] = compute_energy(Noise)


# ========= SNR plot =========
ASNR = np.zeros(keep_r)
for r in rs:
    ASNR[r] = 10 * np.log10(energy_A / energy_N[r])

plt.figure()
plt.plot(rs, ASNR[1:])
plt.xlabel("r")
plt.ylabel("ASNR (dB)")
plt.title("ASNR vs r")
plt.grid(True)
plt.show()


# ========= Verification =========
lambda_vals, _ = myeig(A.T @ A, symmetric=True)
lambda_vals = np.real(lambda_vals)

verify_err = np.zeros(keep_r)

for r in rs:
    tail_sum = np.sum(lambda_vals[r:])
    verify_err[r] = np.abs(tail_sum - energy_N[r])

print("Max verification error =", np.max(verify_err[1:]))

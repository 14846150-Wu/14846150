# -*- coding: utf-8 -*-
"""
@author: htchen
"""
import numpy as np
import numpy.linalg as la

def gram_schmidt(S1: np.ndarray):
    """
    Parameters
    ----------
    S1 : np.ndarray
        A m x n matrix with columns that need to be orthogonalized using Gram-Schmidt process.
        It is assumed that vectors in S1 = [v1 v2 ... vn] are linear independent.

    Returns
    -------
    S2 : np.ndarray
        S2 = [e1 e2 ... en] is a m x n orthonormal matrix such that span(S1) = span(S2)
    """
    m, n = S1.shape
    S2 = np.zeros((m, n), dtype=np.float64)

    eps = 1e-12  # avoid division by zero (numerical safety)

    for j in range(n):
        # start from v_j
        u = S1[:, j].astype(np.float64).copy()

        # subtract projections onto previous e_i
        for i in range(j):
            proj = np.dot(S2[:, i], u)          # e_i^T u  (since e_i is unit)
            u = u - proj * S2[:, i]             # u <- u - (e_i^T u) e_i

        # normalize
        norm_u = la.norm(u)
        if norm_u < eps:
            raise ValueError(f"Vectors are linearly dependent or numerically unstable at column {j}.")
        S2[:, j] = u / norm_u

    return S2


S1 = np.array([[ 7,  4,  7, -3, -9],
               [-1, -4, -4,  1, -4],
               [ 8,  0,  5, -6,  0],
               [-4,  1,  1, -1,  4],
               [ 2,  3, -5,  1,  8]], dtype=np.float64)
S2 = gram_schmidt(S1)

np.set_printoptions(precision=2, suppress=True)
print(f'S1 => \n{S1}')
print(f'S2.T @ S2 => \n{S2.T @ S2}')

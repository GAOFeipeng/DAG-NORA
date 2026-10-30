"""Sign-preserving fallback for a residual-certified nonsingular M-matrix.

This is Gaussian elimination without row pivoting, NOT a new matrix identity.
Off-diagonal eliminations and RHS updates preserve nonnegativity of the inverse.
Every pivot must stay strictly positive; failure is raised rather than clipped.
"""
import numpy as np
from numba import njit

@njit(cache=True)
def positive_m_inverse(A,s):
    n=len(A);M=-A.copy();B=np.eye(n)
    for i in range(n):M[i,i]+=s
    for k in range(n):
        pivot=M[k,k]
        if pivot<=0 or not np.isfinite(pivot):raise ValueError('nonpositive M-matrix pivot')
        for i in range(k+1,n):
            factor=M[i,k]/pivot
            if factor>0:raise ValueError('off-diagonal M-matrix sign violated')
            for j in range(k+1,n):M[i,j]-=factor*M[k,j]
            for j in range(n):B[i,j]-=factor*B[k,j]
            M[i,k]=0.
    K=np.zeros((n,n))
    for i in range(n-1,-1,-1):
        for j in range(n):
            v=B[i,j]
            for k in range(i+1,n):v-=M[i,k]*K[k,j]
            K[i,j]=v/M[i,i]
    return K

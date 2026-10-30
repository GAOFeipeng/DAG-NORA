import numpy as np
from numba import njit

@njit(cache=True)
def kkt_vec(u, g, lam, skip):
    out=0.
    for i in range(len(u)):
        if i == skip: continue
        if u[i]>0: v=abs(g[i]+lam)
        elif u[i]<0: v=abs(g[i]-lam)
        else: v=max(0.,abs(g[i])-lam)
        out=max(out,v)
    return out

"""Explicit complete-R reference, used only by audit/regression tests."""
from types import SimpleNamespace
import time
import numpy as np
from .pair import reversal_seed
def rank_pairs_reference(G,W,K,mu,s,lam,threshold):
    start=time.perf_counter();ranked=[]
    for i in range(len(W)):
        for j in range(i+1,len(W)):
            strength=max(abs(W[i,j]),abs(W[j,i]))
            if strength<threshold:continue
            # Ranking uses the data score only; no numerical domain computation
            # is needed for candidates that will never be tried.
            J=np.asarray([i,j],dtype=int);old=W[:,J].copy();b=G[:,J]
            p=SimpleNamespace(G=G,old=old,b=b,J=J,data_gradient=G@old-b)
            dominant=0 if abs(W[i,j])>=abs(W[j,i]) else 1
            V=reversal_seed(p,dominant);dv=V-p.old
            dj=.5*np.sum(dv*(G@dv))+np.sum(dv*p.data_gradient)+lam*(np.abs(V).sum()-np.abs(p.old).sum())
            if not np.isfinite(dj):raise FloatingPointError('nonfinite ranking score')
            ranked.append(dict(i=i,j=j,predicted_J_gain=float(-dj),strength=float(strength)))
    ranked.sort(key=lambda x:(-x['predicted_J_gain'],x['i'],x['j']))
    return ranked,time.perf_counter()-start,dict(inversions=0,positive_fallbacks=0)


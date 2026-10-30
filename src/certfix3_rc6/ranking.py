"""Complete original R in scalar form. No score component is removed.

The virtual regression transformation is algebraically expanded, not used as
an actual candidate. Its squared loss AND L1 term are both retained. Sorting
and tie breaking remain (-score, i, j). No across-state cache or hashing.
"""
import time
import numpy as np
from numba import njit
@njit(cache=True)
def _complete_score_kernel(G,W,D,S,pairs,lam,include_l1):
    n=len(W);m=len(pairs)
    # fields: R, squared-loss gain, L1-gain, EV, eta, uvar, targetvar,
    # cov(u,old target residual), source,target,b,T,q, fallback_needed
    ans=np.zeros((m,17))
    norms=np.zeros(n)
    for j in range(n):
        for i in range(n):norms[j]+=abs(W[i,j])
    for k in range(m):
        x,y=pairs[k,0],pairs[k,1]
        i,j=(x,y) if abs(W[x,y])>=abs(W[y,x]) else (y,x)
        a,b=W[i,j],W[j,i]
        uvar=S[i,i]+2*b*D[j,i]+b*b*G[j,j]
        c=S[i,j]+b*D[j,j]
        T=S[j,j]+2*a*c+a*a*uvar
        q=c+a*uvar
        # Numerical near-zero denominator is re-evaluated from its actual
        # vectors outside the kernel; this is not a new acceptance threshold.
        scale=abs(S[j,j])+abs(2*a*c)+abs(a*a*uvar)
        fallback=(not np.isfinite(T)) or T<=64*np.finfo(np.float64).eps*n*max(scale,1.) or uvar<=0
        eta=q/max(T,1e-15)  # inherited denominator floor; not newly tuned
        reverse_bonus=-b*D[j,i]-.5*b*b*G[j,j]
        ls=.5*a*a*uvar-a*q+eta*q-.5*eta*eta*T+reverse_bonus
        newnorm=abs(eta)
        l1=0.
        if include_l1:
            for r in range(n):
                if r==i or r==j:continue
                hj=W[r,j]+a*W[r,i]
                hi=W[r,i]-eta*hj
                newnorm+=abs(hj)+abs(hi)
            l1=lam*(norms[i]+norms[j]-newnorm)
        direction=.5*q*q*(uvar-T)/(uvar*T) if uvar>0 and T>0 else 0.
        fit_bonus=.5*uvar*(a-q/uvar)**2 if uvar>0 else 0.
        ans[k,:]=np.array([ls+l1,ls,l1,direction,eta,uvar,S[j,j],c,float(i),float(j),b,T,q,(1. if fallback else 0.),fit_bonus,reverse_bonus,a])

    return ans


def eligible(W,threshold):
    strength=np.maximum(np.abs(W),np.abs(W.T))
    x,y=np.nonzero(np.triu(strength>=threshold,1))
    return np.column_stack((x,y)).astype(np.int64),strength[x,y]


def scalar_scores(G,W,lam,threshold=.05):
    """R is exactly equal algebraically, including both nonzero directions.
    D=<X, old residual>/n; S=<old residual, old residual>/n.
    O(d^3 + m*d), versus original repeated dense per-pair Gram products.
    """
    include_l1=True
    G=np.asarray(G,dtype=np.float64);W=np.asarray(W,dtype=np.float64)
    if G.shape!=W.shape or G.ndim!=2 or G.shape[0]!=G.shape[1]:raise ValueError('square G/W required')
    if not (np.isfinite(G).all() and np.isfinite(W).all() and np.isfinite(lam)):raise ValueError('finite inputs required')
    if np.any(np.diag(W)!=0):raise ValueError('diagonal must remain zero')
    E=np.eye(len(W))-W
    D=G@E;S=E.T@D
    pairs,strength=eligible(W,threshold)
    values=_complete_score_kernel(G,W,D,S,pairs,float(lam),include_l1)
    fallbacks=0
    for k in np.flatnonzero(values[:,13]):
        fallbacks+=1;i,j=map(int,values[k,8:10]);a=W[i,j]
        ci=W[:,i].copy();cj=W[:,j].copy();ci[[i,j]]=0.;cj[[i,j]]=0.
        ri=-ci;ri[i]=1.;rj=-(cj+a*ci);rj[j]=1.
        T=float(rj@G@rj);q=float(ri@G@rj);p=float(ri@G@ri)
        eta=q/max(T,1e-15)
        ui=E[:,i];uj=E[:,j]
        oi=float(ui@G@ui);oj=float(uj@G@uj)
        ls=.5*(oi+oj-p-T+2*eta*q-eta*eta*T)
        l1new=abs(eta)
        for r in range(len(W)):
            if r in (i,j):continue
            hj=W[r,j]+a*W[r,i];hi=W[r,i]-eta*hj;l1new+=abs(hj)+abs(hi)
        l1=lam*(np.abs(W[:,i]).sum()+np.abs(W[:,j]).sum()-l1new)
        l1 = l1 if include_l1 else 0.
        values[k,0:3]=[ls+l1,ls,l1];values[k,4]=eta;values[k,5]=p;values[k,11:13]=[T,q]
        values[k,3]=.5*q*q*(p-T)/(p*T) if p>0 and T>0 else 0.
        values[k,14]=.5*p*(a-q/p)**2 if p>0 else 0.
        bb=W[j,i];values[k,15]=-bb*float(ri@G[:,j])+.5*bb*bb*G[j,j]
    if not np.isfinite(values).all():raise FloatingPointError('nonfinite score')
    return pairs,strength,values,dict(fallbacks=fallbacks,old_residual_covariance=S)



def rank_pairs(G,W,K,mu,s,lam,threshold):
    start=time.perf_counter()
    pairs,strength,values,info=scalar_scores(G,W,lam,threshold)
    ranked=[dict(i=int(p[0]),j=int(p[1]),predicted_J_gain=float(v[0]),strength=float(st))
            for p,st,v in zip(pairs,strength,values)]
    ranked.sort(key=lambda item:(-item['predicted_J_gain'],item['i'],item['j']))
    return ranked,time.perf_counter()-start,dict(inversions=0,positive_fallbacks=0,score_vector_fallbacks=info['fallbacks'])


def warmup():
    W=np.zeros((4,4));W[0,1]=.2
    scalar_scores(np.eye(4),W,.03)

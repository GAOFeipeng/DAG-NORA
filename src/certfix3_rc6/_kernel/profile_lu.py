"""Scalar conjugate profiling of the EXACT signed L1 DAGMA column objective.
Classical convex duality + Lasso active set, not a new acyclicity characterization.
No fixed candidate screening: all inactive coordinates are checked at every solve.
"""
import numpy as np
from numba import njit
from .kkt import kkt_vec

@njit(cache=True)
def _active_solve(G,b,diagonal,u0,lam,skip,tol,max_iter):
    n=len(b);u=u0.copy();u[skip]=0.
    active=np.zeros(n,np.bool_);signs=np.zeros(n)
    for i in range(n):
        if u[i]!=0: active[i]=True;signs[i]=np.sign(u[i])
    for iteration in range(max_iter):
        ids=np.where(active)[0];p=len(ids)
        if p:
            A=np.empty((p,p));rhs=np.empty(p)
            for a in range(p):
                i=ids[a];rhs[a]=b[i]-lam*signs[i]
                for z in range(p): A[a,z]=G[i,ids[z]]
                A[a,a]+=diagonal[i]
            sol=np.linalg.solve(A,rhs)
            step=1.;cross=False
            for a in range(p):
                i=ids[a]
                if sol[a]*signs[i]<=0:
                    cross=True
                    if u[i]!=sol[a]:step=min(step,u[i]/(u[i]-sol[a]))
            if cross:
                for a in range(p):
                    i=ids[a];u[i]+=step*(sol[a]-u[i])
                    if abs(u[i])<=1e-12:
                        u[i]=0.;active[i]=False;signs[i]=0.
                continue
            for a in range(p):u[ids[a]]=sol[a]
        residual=b-G@u-diagonal*u
        maximum=tol;chosen=-1
        for i in range(n):
            if i==skip or active[i]:continue
            violation=abs(residual[i])-lam
            if violation>maximum:maximum=violation;chosen=i
        if chosen==-1:
            err=kkt_vec(u,-residual,lam,skip)
            return u,iteration+1,err,err<=max(tol*10,1e-8)
        active[chosen]=True;signs[chosen]=np.sign(residual[chosen])
    return u,max_iter,kkt_vec(u,G@u+diagonal*u-b,lam,skip),False

@njit(cache=True)
def profile_column(G,b,c,u0,mu,lam,skip,tol=1e-6,max_pass=2000):
    # -log(1-q) = max_{t>0} [t*q-t+log(t)+1].
    # Given t, optimize Lasso with Hessian G+(2*t/mu)*diag(c).
    # Consistency root: q(u(t))-1+1/t = 0; strictly decreasing.
    u=u0.copy();u[skip]=0.
    q0=np.dot(c,u*u)
    t=1./(1.-q0);low=1.;high=np.inf;total_active=0
    good=True;err=np.inf
    for outer in range(80):
        diagonal=(2.*t/mu)*c
        u,na,lasso_err,ok=_active_solve(G,b,diagonal,u,lam,skip,min(tol*.01,1e-9),max_pass)
        total_active+=na;good=good and ok
        q=np.dot(c,u*u)
        if q<1.:
            g=G@u-b+(2./mu)*c*u/(1.-q)
            err=kkt_vec(u,g,lam,skip)
            if err<=tol:return u,total_active,outer+1,err,good
        residual=q-1.+1./t
        if residual>0:low=t
        else:high=t
        ids=np.where(u!=0)[0];p=len(ids)
        deriv=-1./(t*t)
        if p:
            A=np.empty((p,p));v=np.empty(p)
            for a in range(p):
                i=ids[a];v[a]=c[i]*u[i]
                for z in range(p):A[a,z]=G[i,ids[z]]
                A[a,a]+=diagonal[i]
            dv=np.linalg.solve(A,v)
            deriv-=(4./mu)*np.dot(v,dv)
        newt=t-residual/deriv
        if not np.isfinite(high):
            newt=min(max(newt,low*(1.+1e-8)),2.*low)
        elif newt<=low or newt>=high:
            newt=.5*(low+high)
        if newt==t:break
        t=newt
    # Do not turn a failed inner solve into claimed exact optimization.
    return u,total_active,outer+1,err,False

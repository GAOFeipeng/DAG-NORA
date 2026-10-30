from __future__ import annotations
import time
import numpy as np
import scipy.linalg as sla
from ._kernel.positive_inverse import positive_m_inverse
from ._kernel.profile import profile_column_status
def woodbury(K, old, new, J):
    """U consists of squared-weight DIFFERENCES, not squared differences."""
    U=new*new-old*old
    row=K[J,:]
    T=np.eye(2)-row@U
    return K+(K@U)@np.linalg.solve(T,row),T


def inverse_witness(W,K,s):
    x=K@np.ones(len(W))
    residual=(s*np.eye(len(W))-W*W)@x-1
    err=float(np.max(np.abs(residual)))
    if not np.all(np.isfinite(K)) or np.min(x)<=0 or err>1e-6:
        raise FloatingPointError(f'invalid residual-checked M-matrix witness: {err}')
    return err


class PairProblem:
    """Fix all columns outside J. The joint problem is NOT assumed convex.

    Remove both columns algebraically. R0 is two rows of the deleted inverse.
    Then B=R0 @ V**2 >=0, and rho(B)<1 is an exact reduced M-domain test.
    The reinverse arm recomputes full inverses for the SAME candidate operations.
    """
    def __init__(self,G,W,K,mu,s,lam,J,backend='woodbury', *, comparator, kernel=profile_column_status):
        self.G=G;self.W=W;self.K=K;self.mu=mu;self.s=s;self.lam=lam
        self.comparator=comparator;self.kernel=kernel
        self.J=np.asarray(J,dtype=int);self.d=len(W);self.backend=backend
        self.old=W[:,self.J].copy();self.b=G[:,self.J]
        self.repairs=0;self.positive_fallbacks=0;self.fallback_backward_error=0.;self.inversions=0;self.row_calls=0;self.objective_calls=0;self.backtracks=0
        self.inner_calls=0;self.inner_caps=0;self.local_caps=0;self.max_inner_res=0.
        self.current_inner_caps=0;self.inner_warnings=0
        self.R0=np.linalg.solve(np.eye(2)+K[self.J,:]@(self.old*self.old),K[self.J,:])
        if np.min(self.R0)<-1e-12:
            V0=W.copy();V0[:,self.J]=0.
            self.R0=sla.inv(s*np.eye(self.d)-V0*V0,check_finite=False)[self.J,:]
            self.inversions+=1;self.repairs+=1
        if np.min(self.R0)<-1e-12:
            # V1 failed on a nonnormal but legal M-matrix after pivoted
            # inversion produced a -2e-12 entry. Preserve signs rather than
            # relaxing domain tolerances or clipping significant negatives.
            V0=W.copy();V0[:,self.J]=0.
            M0=s*np.eye(self.d)-V0*V0
            K0=positive_m_inverse(V0*V0,s)
            self.R0=K0[self.J,:];self.inversions+=1;self.positive_fallbacks+=1
            residual=self.R0@M0-np.eye(self.d)[self.J,:]
            self.fallback_backward_error=float(np.linalg.norm(residual,np.inf)/max(1.,np.linalg.norm(self.R0,np.inf)*np.linalg.norm(M0,np.inf)))
            if np.min(K0)<0 or not np.all(np.isfinite(K0)) or self.fallback_backward_error>1e-12:
                raise FloatingPointError('nonnegative deleted-inverse fallback failed residual check')
        self.roundoff_clamps=int(np.sum(self.R0<0))
        self.R0=np.maximum(self.R0,0.)
        # These four entries are known exactly because the two base columns vanish.
        self.R0[:,self.J]=np.eye(2)/s
        self.Told,self.ldold=self.reduced(self.old)
        self.data_gradient=G@self.old-self.b
        if backend=='reinverse':
            self.ldfullold=np.linalg.slogdet(s*np.eye(self.d)-W*W)[1]

    def reduced(self,V):
        B=self.R0@(V*V)
        # Test BOTH diagonal principal minors and determinant, not det alone.
        t00=1.-B[0,0];t11=1.-B[1,1]
        det=t00*t11-B[0,1]*B[1,0]
        if min(t00,t11,det)<=1e-12 or not np.isfinite(det):
            raise FloatingPointError('outside reduced M-matrix domain')
        return np.eye(2)-B,float(np.log(det))

    def feasible(self,V):
        try:self.reduced(V);return True
        except FloatingPointError:return False

    def rows(self,V):
        self.row_calls+=1
        T,_=self.reduced(V)
        if self.backend=='reinverse':
            WV=self.W.copy();WV[:,self.J]=V
            KV=sla.inv(self.s*np.eye(self.d)-WV*WV,check_finite=False)
            self.inversions+=1;inverse_witness(WV,KV,self.s)
            return KV[self.J,:]
        return np.linalg.solve(T,self.R0)

    def delta(self,V):
        self.objective_calls+=1
        _,ld=self.reduced(V)
        delta=V-self.old
        dj=.5*np.sum(delta*(self.G@delta))+np.sum(delta*self.data_gradient)
        dj+=self.lam*(np.abs(V).sum()-np.abs(self.old).sum())
        dh=self.ldold-ld
        if self.backend=='reinverse':
            WV=self.W.copy();WV[:,self.J]=V
            M=self.s*np.eye(self.d)-WV*WV
            KV=sla.inv(M,check_finite=False);self.inversions+=1
            inverse_witness(WV,KV,self.s)
            sign,ldfull=np.linalg.slogdet(M)
            if sign<=0:raise FloatingPointError('det sign')
            dh=self.ldfullold-ldfull
        return float(self.mu*dj+dh)

    def valid_seed(self,V,anchor=None):
        if anchor is None:anchor=np.zeros_like(V)
        V=V.copy();V[self.J[0],0]=0.;V[self.J[1],1]=0.
        for k in range(60):
            if self.feasible(V):return V,k
            V=anchor+.5*(V-anchor);self.backtracks+=1
        raise FloatingPointError('seed backtracking failed')

    def block_kkt(self,V):
        rows=self.rows(V)
        g=self.G@V-self.b+(2./self.mu)*V*rows.T
        z=V-g;prox=np.sign(z)*np.maximum(np.abs(z)-self.lam,0.)
        r=V-prox;r[self.J[0],0]=0.;r[self.J[1],1]=0.
        return float(np.max(np.abs(r)))

    def local_columns(self,V,order=(0,1),max_sweeps=100,tol=1e-6,inner_tol=1e-8):
        """Multi-start nonconvex block relaxation by exact conditional columns.
        No claim of joint global optimum. Only final objective decrease is accepted.
        """
        V=V.copy();prev=self.delta(V);converged=False;last_kkt=np.inf
        for sweep in range(1,max_sweeps+1):
            self.current_inner_caps=0
            V_previous=V.copy()  # certfix2: arithmetic-comparison inputs only
            for a in order:
                j=int(self.J[a]);rows=self.rows(V);old=V[:,a].copy()
                c=rows[a].copy()/(1.+rows[a]@(old*old));c[j]=0.
                if np.min(c)<-1e-12:raise FloatingPointError('negative local c')
                c=np.maximum(c,0.)
                answer=self.kernel(self.G,self.b[:,a].copy(),c,old,self.mu,self.lam,j,inner_tol,2000)
                new,_,_,res,ok=answer[:5]
                if len(answer)==8:self.inner_warnings+=int(answer[6])
                self.inner_calls+=1;self.inner_caps+=int(not ok);self.max_inner_res=max(self.max_inner_res,float(res))
                self.current_inner_caps+=int(not ok)
                V[:,a]=new
                if not self.feasible(V):
                    for back in range(50):
                        new=.5*(old+new);V[:,a]=new;self.backtracks+=1
                        if self.feasible(V):break
                    else:raise FloatingPointError('local line search failed')
            cur=self.delta(V)
            monotone,_=self.comparator.compare_pair(self,V_previous,V,cur-prev,1e-10,'le',not(cur-prev>1e-10),'local_block_monotonicity')
            if not monotone:raise FloatingPointError('local block relaxation increased objective')
            last_kkt=self.block_kkt(V)
            if last_kkt<=tol:
                converged=True;break
            prev=cur
        self.local_caps+=int(not converged)
        return V,dict(local_sweeps=sweep,local_converged=converged,block_kkt=last_kkt,
                      current_profile_unconverged_columns=self.current_inner_caps,
                      inner_warnings=self.inner_warnings)

    def apply(self,V):
        W=self.W.copy();W[:,self.J]=V
        if self.backend=='reinverse':
            K=sla.inv(self.s*np.eye(self.d)-W*W,check_finite=False);self.inversions+=1
        else:K,_=woodbury(self.K,self.old,V,self.J)
        err=inverse_witness(W,K,self.s)
        return W,K,err


def reversal_seed(problem,src_index=0):
    """Finite two-column regression reparametrization; no ground truth.
    Seed only, NOT a restriction on the final two-column candidate support.
    Works even as an imperfect proposal when the current edge is not covered.
    """
    p=problem;z=src_index;t=1-z;src=int(p.J[z]);dst=int(p.J[t]);a=p.old[src,t]
    ci=p.old[:,z].copy();cj=p.old[:,t].copy()
    ci[p.J]=0.;cj[p.J]=0.
    newj=cj+a*ci
    ri=-ci;ri[src]=1.
    rj=-newj;rj[dst]=1.
    denom=rj@p.G@rj
    eta=(ri@p.G@rj)/max(denom,1e-15)
    newi=ci-eta*newj;newi[dst]=eta;newi[src]=0.;newj[dst]=0.
    V=p.old.copy();V[:,z]=newi;V[:,t]=newj
    return V

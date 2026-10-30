"""Unmodified upstream fit controller + fixed forward column minimizer.

Only one raw final-stage state is retained in normal mode. Its parameters are
captured before upstream output thresholding and before mu is incremented.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
from ._upstream.dagma_linear import DagmaLinear
from . import guards as ns
from .numerics import bounded_inverse
from .column import fit_columns_r2
from ._kernel.profile import profile_column

class ForwardOrder:
    def __init__(self,d,audit=False):
        self.order=np.arange(d);self.audit=audit;self.calls=0;self.seconds=0.
    def select(self,G,W,K,mu,lam,sweep):
        self.calls+=1
        return self.order

class ProductionBridge(DagmaLinear):
    def __init__(self,config,context,*,audit=False,checkpoints=None,comparator=None,kernel=profile_column):
        super().__init__('l2',verbose=False,dtype=np.float64)
        self.config=config;self.context=context;self.audit=audit
        self.comparator=comparator;self.kernel=kernel
        self.checkpoints=Path(checkpoints) if checkpoints else None
        self.attempts=[];self.stage_results=[];self.actual_s=[]
        self.last_raw=None;self.last_params=None;self.last_evaluation=None;self.last_lambda=None;self.last_raw_binding=None
        self.start_rejections=0;self.io_seconds=0.;self.scheduler=None
        self.current_attempt=None
    def minimize(self,W,mu,max_iter,s,lr,tol=1e-6,beta_1=.99,beta_2=.999,pbar=None):
        start=time.perf_counter()
        if not np.isfinite(self.cov).all():
            raise FloatingPointError('nonfinite centered Gram matrix')
        e=dict(stage=len(self.actual_s),attempt=len(self.attempts),mu=float(mu),s=float(s),lr=float(lr),
               max_adam_iterations=int(max_iter),lr_applied_to_column_updates=False,success=None)
        self.attempts.append(e);self.current_attempt=e
        try:
            bounded_inverse(W,s,mu)
        except ns.NumericalUnresolved as exc:
            self.start_rejections+=1
            e.update(success=False,rejection='stage_start_numerically_unresolved',exception=str(exc),seconds=time.perf_counter()-start)
            self.current_attempt=None
            return W,False
        if self.scheduler is None:self.scheduler=ForwardOrder(len(W),self.audit)
        cfg=dict(column=self.config['column'],mu=[float(mu)],s=[float(s)],**{'lambda':float(self.lambda1)})
        V,z,_=fit_columns_r2(self.X,cfg,initial_W=W,prepared_G=self.cov,scheduler=self.scheduler,context=self.context,comparator=self.comparator,kernel=self.kernel)
        if not self.audit:z.pop('trace',None)
        self.stage_results.append(z);self.actual_s.append(float(s))
        self.last_raw=V.copy();self.last_params=(float(mu),float(s))
        self.last_lambda=float(self.lambda1)
        self.last_evaluation=self.context.handoff(self.context.cached[1],self.context.K,mu,s,self.lambda1)
        self.last_raw_binding=(self.last_raw,self.last_params,self.last_lambda,self.last_evaluation[4])
        e.update(success=True,native_status=z['status'],sweeps=z['stages'][0]['sweeps'],
                 residual=z['stages'][0]['residual_normalized'],seconds=time.perf_counter()-start)
        if self.checkpoints:
            t=time.perf_counter();self.checkpoints.mkdir(parents=True,exist_ok=True)
            np.savez(self.checkpoints/f'stage_{len(self.actual_s)-1}.npz',W=V,mu=mu,s=s)
            self.io_seconds+=time.perf_counter()-t
        self.current_attempt=None
        if pbar is not None:pbar.update(max_iter)
        # Upstream fit thresholds its output in place. Keep that output buffer
        # separate from the certified continuous state and its cached inverse.
        return V.copy(order='K'),True
    def raw_final_state(self):
        successes=[e for e in self.attempts if e.get('success') is True]
        if not successes or len(successes)!=len(self.actual_s) or self.last_params!=(successes[-1]['mu'],successes[-1]['s']):
            raise RuntimeError('Final raw W/mu/s binding invalid')
        return self.last_raw.copy(),*self.last_params

assert ProductionBridge.fit is DagmaLinear.fit

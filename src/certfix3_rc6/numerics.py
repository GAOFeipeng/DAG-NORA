"""Periodic, bounded inverse safeguards and low-cost rank maintenance.

Rank arithmetic retains the legacy summation order, but does not propagate a
full elementwise error envelope after every column. Each sweep still refreshes
with the inherited independent inverse/gradient gate and checks relative drift.
This is a numerical-strategy change, not a proof of bitwise equivalence.
"""
from __future__ import annotations
import time
import copy
from contextlib import nullcontext
from dataclasses import dataclass
from collections import Counter
import numpy as np
import scipy.linalg as sla
from numba import njit
from . import guards as ns
from .timing import Ledger, metered

from ._kernel.positive_inverse import positive_m_inverse

# Preserve the old independently tested building blocks for bounded refreshes.



def bounded_inverse(W,s,mu=1.,K=None,force_quad=False,ledger=None):
    """At most pivoted double, sign-preserving double, then one binary128 solve.

    No arbitrary-precision witness construction. The inherited M-witness and
    inverse/gradient bounds are retained at full refreshes, not on each rank step.
    """
    ns.STATS['checks']+=1
    if not np.isfinite(s) or s<=0 or not np.isfinite(mu) or mu<=0:
        raise ns.NumericalUnresolved('invalid_stage_parameter')
    if W.ndim!=2 or W.shape[0]!=W.shape[1] or not np.isfinite(W).all():
        raise ns.NumericalUnresolved('invalid_weight_matrix')
    if np.any(np.diag(W)!=0):raise ValueError('zero diagonal required')
    M=s*np.eye(len(W))-W*W
    if not np.isfinite(M).all():raise ns.NumericalUnresolved('nonfinite_squared_weights')
    if K is None:
        try:K=sla.inv(M,check_finite=False)
        except (ValueError,sla.LinAlgError):K=np.zeros_like(W);force_quad=True
    diag,B=ns.check_existing(W,K,s,mu)
    if not force_quad and diag['domain']['legal'] and diag['accuracy']['reliable']:
        sign,ld=np.linalg.slogdet(M)
        if sign>0 and np.isfinite(ld):
            ns.STATS['fast_pass']+=1
            ns.STATS['max_accepted_gradient_error']=max(ns.STATS['max_accepted_gradient_error'],diag['accuracy']['gradient_scaled_bound'])
            return K,float(ld),B,dict(**diag,backend='binary64_refresh')
    if not force_quad:
        try:
            KP=positive_m_inverse(W*W,s);pd,PB=ns.check_existing(W,KP,s,mu)
            if pd['domain']['legal'] and pd['accuracy']['reliable']:
                sign,ld=np.linalg.slogdet(M)
                if sign>0 and np.isfinite(ld):
                    ns.STATS['positive_fallbacks']+=1
                    ns.STATS['max_accepted_gradient_error']=max(ns.STATS['max_accepted_gradient_error'],pd['accuracy']['gradient_scaled_bound'])
                    return KP,float(ld),PB,dict(**pd,backend='binary64_positive_refresh')
        except (FloatingPointError,ValueError):pass
    try:
        with ledger.measure('high_precision_inverse_fallback') if ledger else nullcontext():
            KK,ld,B,qinfo=ns.quad(W,s)
        dom=ns.domain_certificate(W,KK,s);acc,B=ns.accuracy(W,KK,mu,B=B)
        if not dom['legal'] or not acc['reliable'] or not np.isfinite(ld):
            raise ns.NumericalUnresolved('bounded_inverse_refresh_unresolved_after_binary128')
        ns.STATS['max_accepted_gradient_error']=max(ns.STATS['max_accepted_gradient_error'],acc['gradient_scaled_bound'])
        return KK,ld,B,dict(domain=dom,accuracy=acc,backend='binary128_refresh',quad=qinfo)
    except (FloatingPointError,ValueError,OverflowError) as exc:
        ns.STATS['rejected_unresolved']+=1
        raise ns.NumericalUnresolved('bounded_inverse_refresh_failed: '+str(exc)) from exc


@njit(cache=True)
def rank_practical(K,old,new,j):
    """Packed SAFE kernel: same per-element arithmetic, F-contiguous output."""
    d=len(K);u=np.finfo(np.float64).eps/2
    delta=new*new-old*old;z=np.zeros(d);den=1.;absdot=0.
    for k in range(d):
        den-=K[j,k]*delta[k];absdot+=abs(K[j,k])*abs(delta[k])
    floor=8*(d+8)*u*(1.+absdot)
    if not np.isfinite(den) or den<=floor:return K.copy(),den,False
    if K.flags.f_contiguous:
        for k in range(d):
            for i in range(d):z[i]+=K[i,k]*delta[k]
    else:
        for i in range(d):
            for k in range(d):z[i]+=K[i,k]*delta[k]
    Q=np.empty((d,d)).T;ok=True
    for k in range(d):
        for i in range(d):
            term=z[i]*K[j,k]/den
            Q[i,k]=K[i,k]+term
            if not np.isfinite(Q[i,k]):ok=False
    return Q,den,ok


@dataclass(frozen=True)
class Evaluation:
    """Caller-owned checked state; arrays must not be changed while held."""
    W: np.ndarray
    K: np.ndarray
    B: np.ndarray
    metrics: dict
    params: tuple[float, float, float]


class NumericalContext:
    def __init__(self, *, maintenance='practical', audit=False, recovery_alphas=None, recovery_no_progress_retries=1):
        self.ledger=Ledger()
        from .recovery import DEFAULT_ALPHAS
        self.recovery_alphas=tuple(DEFAULT_ALPHAS if recovery_alphas is None else recovery_alphas)
        self.recovery_no_progress_retries=recovery_no_progress_retries
        self.recovery=None;self.recovery_event_path=None;self.budget_state=None
        self.W=self.K=self.B=self.mu=self.s=None
        self.latest=None;self.maintenance_refreshes=0
        self.maintenance=maintenance;self.audit=audit
        self.counts=Counter();self.seconds=Counter();self.state_version=0
        self.since_refresh=0;self.cached=None;self.probes=None
        self.cached_W_owned=None
        self.last_probe_error=0.;self.maximum_probe_error=0.
    def check_deadline(self):
        state=self.budget_state
        if state is not None and (state['triggered'] or time.monotonic()>=state['deadline']):
            state['triggered']=True
            raise TimeoutError('Prespecified hot-fit deadline reached')
    def bind(self,W,K,B,mu,s):
        self.W=W;self.K=K;self.B=B;self.mu=mu;self.s=s
        self.state_version+=1;self.since_refresh=0
        self.cached=None;self.cached_W_owned=None
        if self.probes is None or self.probes.shape[0]!=len(W):
            self.probes=np.column_stack((np.ones(len(W)),np.where(np.arange(len(W))%2, -1.,1.)))
    @metered('state_evaluation')
    def evaluate(self,G,W,mu,s,lam,need_kkt=True, *, commit=True):
        """Full-matrix state evaluation.

        Two proximal-mapping residuals are produced and they are NOT the same
        finite number even though they share the same zeros:

        * ``kkt``  -- step 1 prox mapping of the Q-normalised objective with
          penalty ``lam``: ``max|W - soft(W - grad_smooth_F/mu, lam)|``.
          This is the legacy diagnostic field; its meaning is unchanged.
        * ``residual_normalized`` (and its explicit alias ``stop_residual``) --
          the step-1 prox mapping of ``mu*J + h`` with penalty ``mu*lam``,
          divided by ``mu``:
          ``max|W - soft(W - grad_smooth_F, mu*lam)| / mu`` with the diagonal
          masked out, where ``grad_smooth_F = mu*G@(W-I) + 2*W*K.T`` and
          ``K = (s*I - W*W)**-1`` (``W*W`` is the elementwise square).

        Since rc6-fix the full-column stopping rule uses ``residual_normalized``
        so that the internal stop and the external independent check share one
        mathematical definition.
        """
        self.check_deadline()
        t=time.perf_counter();self.counts['full_evaluate']+=1
        try:
            ti=time.perf_counter()
            with self.ledger.measure('inverse_refresh'):
                K,ld,B,num=bounded_inverse(W,s,mu,ledger=self.ledger)
            self.seconds['inverse_refresh']+=time.perf_counter()-ti
            I=np.eye(len(W));R=I-W
            quadratic_terms=R*(G@R)
            J=.5*np.sum(quadratic_terms)+lam*np.abs(W).sum()
            h=len(W)*np.log(s)-ld;F=mu*J+h
            # Evidence for exceptional endpoint comparisons, NOT an acceptance
            # tolerance. Reuse the logdet from the already checked inverse.
            # The double estimate includes M's elementwise product sensitivity;
            # the quad backend already supplies a logdet rounding estimate.
            eps_scale=64*np.finfo(float).eps*(len(W)+8)
            if num['backend']=='binary128_refresh':
                barrier_error=num['quad']['logdet_rounding_radius']
            else:
                sensitivity=s*np.abs(np.diag(K)).sum()+np.sum((W*W)*np.abs(K.T))
                barrier_error=eps_scale*(1.+sensitivity+abs(ld))
            barrier_error+=eps_scale*(abs(ld)+abs(len(W)*np.log(s)))
            with self.ledger.measure('residual_calculation'):
                grad=mu*(G@(W-I))+2*W*K.T
                z=W-grad/mu;p=np.sign(z)*np.maximum(np.abs(z)-lam,0.);r=W-p;np.fill_diagonal(r,0.)
                z1=W-grad;p1=np.sign(z1)*np.maximum(np.abs(z1)-mu*lam,0.);r1=W-p1;np.fill_diagonal(r1,0.)
                stop_residual=float(np.max(abs(r1)))/mu
            result=dict(objective_abs_scale=float(mu*(.5*np.abs(quadratic_terms).sum()+lam*np.abs(W).sum())+abs(ld)+abs(len(W)*np.log(s))),F=float(F),J=float(J),h=float(h),kkt=float(np.max(abs(r))) if need_kkt else float('nan'),
                        # objective_abs_scale is a legacy post-product scale;
                        # it is NOT a forward error bound for the Gram product.
                        barrier_error_estimate_F=float(barrier_error),
                        stationarity_metric='prox_residual_mu_normalized',
                        stop_residual=stop_residual,
                        residual_absolute=float(np.max(abs(r1))),residual_normalized=stop_residual,
                        witness_res=num['domain']['old_abs_residual'],domain_lower=num['domain']['lower'],
                        inverse_error_bound=num['accuracy']['inverse_relative_bound'],gradient_error_bound=num['accuracy']['gradient_scaled_bound'],precision_backend=num['backend'])
            state=Evaluation(W,K,B,result,(float(mu),float(s),float(lam)))
            if not commit:
                self.counts['detached_evaluations']+=1
                return state
            self.adopt(state)
            return result,K
        finally:self.seconds['full_evaluate']+=time.perf_counter()-t
    def adopt(self,state: Evaluation):
        """Commit a previously evaluated candidate, without recomputing its inverse."""
        self.bind(state.W,state.K,state.B,*state.params[:2])
        self.cached=(self.state_version,state.metrics,state.K,state.B,*state.params)
        self.cached_W_owned=state.W.copy(order='K')
        self.latest=(self.cached_W_owned,float(state.params[0]),float(state.params[1]))
        return state.metrics,state.K

    def check_cached_weights(self):
        """Full-state boundary check, never run per column or hash a matrix."""
        if self.cached is None or self.cached[0]!=self.state_version:
            raise RuntimeError('stale state version in cached weights')
        if self.cached_W_owned is None or not np.array_equal(self.W,self.cached_W_owned):
            self.state_version+=1;self.cached=None;self.cached_W_owned=None
            self.counts['inplace_cache_invalidations']+=1
            raise ns.NumericalUnresolved('unrecorded_inplace_weight_mutation')

    def snapshot(self) -> Evaluation:
        if self.cached is None or self.cached[0]!=self.state_version:
            raise RuntimeError('cannot export a state after an unevaluated column update')
        self.check_cached_weights()
        _,cur,K,B,mu,s,lam=self.cached
        return Evaluation(self.W,K,B,cur,(mu,s,lam))

    def checkpoint(self) -> Evaluation:
        """A rollback-safe owned copy; snapshot() retains its old handoff API."""
        state=self.snapshot()
        return Evaluation(state.W.copy(order='K'),state.K.copy(order='K'),
                          state.B.copy(order='K'),copy.deepcopy(state.metrics),state.params)

    def reuse(self,W,mu,s,lam,precomputed):
        """Explicit caller-owned handoff; never infer freshness from an address/hash.

        precomputed is produced immediately by evaluate or terminal polish. The
        caller may copy W but may not modify it before this handoff.
        """
        cur,K,B,params,version=precomputed
        if version!=self.state_version or self.cached is None or self.cached[0]!=version:
            raise RuntimeError('stale state version in explicit reuse')
        if params!=(float(mu),float(s),float(lam)):raise ValueError('state parameter mismatch')
        if not np.isfinite(W).all() or not np.isfinite(K).all():raise ns.NumericalUnresolved('invalid cached handoff')
        self.check_cached_weights()
        if not np.array_equal(W,self.cached_W_owned):
            self.counts['handoff_weight_mismatches']+=1
            raise ns.NumericalUnresolved('cached_handoff_weights_changed')
        self.adopt(Evaluation(W,K,B,cur,params))
        self.counts['explicit_evaluate_reuse']+=1
        return cur,K
    def handoff(self,cur,K,mu,s,lam):
        if self.cached is None or self.cached[0]!=self.state_version or K is not self.K:
            raise RuntimeError('stale state version in evaluation handoff')
        self.check_cached_weights()
        return (cur,K,self.B,(float(mu),float(s),float(lam)),self.state_version)
    @metered('inverse_refresh')
    def refresh(self, K, *, positive=False):
        """Rebase the current inverse without changing ownership of K.

        A sign-preserving solve is only requested after the usual coefficient
        validity test fails. Its output must pass the inherited domain and
        inverse/gradient checks; it is never replaced by zero or clipped to pass.
        """
        if K is not self.K:
            raise ns.NumericalUnresolved('untracked_refresh')
        t=time.perf_counter()
        try:
            trial=positive_m_inverse(self.W*self.W,self.s) if positive else None
            fresh,_,B,_=bounded_inverse(self.W,self.s,self.mu,K=trial,ledger=self.ledger)
            K[:]=fresh;self.B=B;self.since_refresh=0
            self.state_version+=1;self.cached=None;self.maintenance_refreshes+=1
            self.counts['positive_coefficient_refresh' if positive else 'coefficient_refresh']+=1
            return K
        finally:
            self.seconds['triggered_refresh']+=time.perf_counter()-t

    @metered('condition_coefficients')
    def coefficients(self,K,old,j,negative_tolerance=1e-12):
        t=time.perf_counter();self.counts['coefficients']+=1
        try:
            if K is not self.K:raise ns.NumericalUnresolved('untracked_inverse_state')
            row=K[j].copy();den=1.+row@(old*old)
            if not np.isfinite(row).all() or not np.isfinite(den) or den<=0:
                self.refresh(K,positive=True)
                row=K[j].copy();den=1.+row@(old*old)
            if not np.isfinite(den) or den<=0:
                raise ns.NumericalUnresolved('column_deletion_denominator')
            c=row/den;c[j]=0.
            if np.min(c)<-negative_tolerance:
                self.refresh(K,positive=True)
                row=K[j].copy();den=1.+row@(old*old)
                if not np.isfinite(den) or den<=0:
                    raise ns.NumericalUnresolved('column_deletion_denominator_after_refresh')
                c=row/den;c[j]=0.
            if not np.isfinite(c).all() or np.min(c)<-1e-12:
                raise ns.NumericalUnresolved('invalid_column_coefficients_after_bounded_refresh')
            return c
        finally:self.seconds['coefficients']+=time.perf_counter()-t
    @metered('inverse_maintenance')
    def rank_one(self,K,old,new,j):
        t=time.perf_counter();self.counts['rank_updates']+=1
        try:
            if K is not self.K:raise ns.NumericalUnresolved('untracked_rank_update')
            if not np.isfinite(new).all() or new[j]!=0:raise ns.NumericalUnresolved('invalid_column_update')
            tr=time.perf_counter();Q,den,ok=rank_practical(K,old,new,int(j))
            self.seconds['rank_kernel']+=time.perf_counter()-tr;self.since_refresh+=1
            need_refresh=not ok;trial=None
            if ok and self.since_refresh%16==0:
                tp=time.perf_counter();self.counts['online_probes']+=1
                A=self.W*self.W;A[:,j]=new*new
                Y=Q@self.probes;AY=A@Y
                R=self.probes-self.s*Y+AY
                scale=1.+np.abs(self.probes)+abs(self.s)*np.abs(Y)+A@np.abs(Y)
                err=float(np.max(np.abs(R)/scale))
                self.last_probe_error=err;self.maximum_probe_error=max(self.maximum_probe_error,err)
                threshold=128*(len(K)+8)*np.finfo(float).eps*max(1,self.since_refresh)
                need_refresh=(not np.isfinite(err) or err>threshold or np.any(Y[:,0]<=0))
                self.seconds['online_probes']+=time.perf_counter()-tp
            if need_refresh:
                tr=time.perf_counter();trial=self.W.copy();trial[:,j]=new
                with self.ledger.measure('inverse_refresh'):
                    Q,_,B,_=bounded_inverse(trial,self.s,self.mu,ledger=self.ledger)
                self.seconds['triggered_refresh']+=time.perf_counter()-tr
                self.maintenance_refreshes+=1;self.counts['anomaly_refreshes']+=1
                self.B=B;self.since_refresh=0
            self.K=Q;self.state_version+=1;self.cached=None
            self.cached_W_owned=None
            return Q,float(den)
        finally:self.seconds['rank_total']+=time.perf_counter()-t

    def commit_column(self,K,old,new,j):
        """Existing rank arithmetic and W write form one versioned transaction."""
        if not np.array_equal(self.W[:,j],old):
            raise ns.NumericalUnresolved('column_commit_old_weights_changed')
        Q,den=self.rank_one(K,old,new,j)
        self.W[:,j]=new
        return Q,den
    def summary(self):
        return dict(maintenance=self.maintenance,counts=dict(self.counts),seconds=dict(self.seconds),
                    maintenance_refreshes=self.maintenance_refreshes,maximum_probe_error=self.maximum_probe_error,
                    full_refresh_interval='each complete sweep plus anomalous online probes',
                    probe_every_column_updates=16,elementwise_bound_propagation=False,
                    state_version=self.state_version)

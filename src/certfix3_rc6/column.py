"""Shared full-column engine: continuation and terminal coordination policies.

The same conditional kernel, inverse context and comparator serve every stage.
The column kernel and R policy are unchanged; C1/C2 use finite-error acceptance.
"""
from __future__ import annotations
import time
import numpy as np
import scipy.linalg as sla
from numba import njit
from ._kernel.profile import profile_column, profile_column_status
from .recovery import controller as recovery_controller, RECOVERABLE
from .guards import NumericalUnresolved
from .finite_policy import state_reliable
from .c2_comparison import ComparisonUnresolved
from .soft_events import HardNumericalFailure, require_certified

@njit(cache=True)
def block_value(G,b,c,u,mu,lam):
    q=np.dot(c,u*u)
    if q>=1:return np.inf
    return .5*u@G@u-b@u+lam*np.abs(u).sum()-np.log1p(-q)/mu


PROFILE_OUTER_BUDGET=80


def classify_inner_failure(active_iterations,outer_iterations,residual,max_inner_pass):
    """Cheap classification of a non-converged single-column solve.

    It is derived only from the values the production kernel already returns, so
    it can be recorded on every run without touching the arithmetic:

    * ``local_residual_not_finite`` -- the outer iterate never produced a finite
      local KKT residual (``q >= 1`` at every outer iteration);
    * ``profile_outer_cap`` -- the profile outer loop consumed its whole budget
      (``outer_iterations >= PROFILE_OUTER_BUDGET``);
    * ``root_stagnation`` -- the outer loop stopped earlier than its budget,
      i.e. the Newton update no longer moved the barrier parameter;
    * ``active_set_budget_evidence`` -- True when the summed active-set
      iterations reached the per-call budget; this is a CLUE, not a proof, and
      the exact branch needs the diagnostic probe.
    """
    finite=bool(np.isfinite(residual))
    if not finite:
        kind='local_residual_not_finite'
    elif int(outer_iterations)>=PROFILE_OUTER_BUDGET:
        kind='profile_outer_cap'
    else:
        kind='root_stagnation'
    return dict(kind=kind,
                active_set_budget_evidence=bool(int(active_iterations)>=int(max_inner_pass)),
                active_iterations=int(active_iterations),
                outer_iterations=int(outer_iterations),
                local_residual=None if not finite else float(residual),
                profile_outer_budget=PROFILE_OUTER_BUDGET)


def column_status_kernel(G,b,c,u0,mu,lam,skip,tol,max_pass):
    """Profile solve with the terminal status split (current vs historical).

    Returns ``(new, passes, nroots, resid, current_converged, had_warning,
    warning_count, worst_warning_residual)``.
    """
    return profile_column_status(G,b,c,u0,mu,lam,skip,tol,max_pass)

def prepare(X):
    with np.errstate(over='raise',invalid='raise'):
        Y=np.asarray(X,dtype=np.float64).copy();Y-=Y.mean(axis=0,keepdims=True)
        G=Y.T@Y/len(Y)
    if not np.isfinite(G).all():
        raise FloatingPointError('nonfinite centered Gram matrix')
    return G

def fit_columns_r2(X, config, maintenance='sm',order_kind='forward',order_seed=1803, *, initial_W=None, prepared_G=None, scheduler=None, context=None, comparator=None, kernel=profile_column):
    start=time.perf_counter(); G=prepare(X) if prepared_G is None else prepared_G; d=len(G); I=np.eye(d); W=np.zeros((d,d)) if initial_W is None else initial_W.copy()
    lam=config['lambda']; cc=config['column']; trace=[]; stages=[]
    inversions=0; updates=0; total_passes=0; roots=0; inner_caps=0; clamp_count=0
    inner_failures=0; inner_failure_worst=None
    inner_warnings=0; had_inner_warning=False; worst_inner_residual=0.; profile_unconverged=0
    solve_time=0.;inv_time=0.;update_time=0.;max_drift=0.; max_increase=0.
    stage_weights=[]; status='completed'; max_inner_res=0.
    recovery=recovery_controller(context,comparator);no_progress_recoveries=0
    for stage,(mu,s) in enumerate(zip(config['mu'],config['s'])):
        warn_before=inner_warnings
        t=time.perf_counter(); prev,K=context.evaluate(G,W,mu,s,lam); inversions+=1; inv_time+=time.perf_counter()-t
        if scheduler.audit: trace.append(dict(stage=stage,sweep=0,seconds=time.perf_counter()-start,mu=mu,s=s,**prev))
        converged=False; stage_pass0=total_passes; rejection_retry_pending=False
        for sweep in range(1,cc['max_sweeps']+1):
            current_profile_unconverged=0
            anchor=context.checkpoint();W_previous=anchor.W;K_previous=anchor.K
            location=dict(phase='column',stage=stage,sweep=sweep)
            recovery.begin(G,anchor,location)
            limit=1e-8*max(abs(prev['F']),1e-15)
            interrupted=False
            order=scheduler.select(G,W,K,mu,lam,sweep)
            try:
                try:
                    for j in order:
                        context.check_deadline()
                        old=W[:,j].copy()
                        if maintenance=='reinverse':
                            t=time.perf_counter();K=context.refresh(K);inversions+=1;inv_time+=time.perf_counter()-t
                        c=context.coefficients(K,old,j,negative_tolerance=1e-14)
                        if c.min() < -1e-12: raise FloatingPointError('negative column coefficients')
                        clamp_count+=int(np.sum(c<0));c=np.maximum(c,0.)
                        t=time.perf_counter()
                        try:
                            with context.ledger.measure('column_solve'):
                                new,passes,nroots,resid,ok,warned,warn_n,warn_worst=column_status_kernel(
                                    G,G[:,j].copy(),c,old,mu,lam,int(j),cc['inner_kkt_tol'],cc['max_inner_pass'])
                        except HardNumericalFailure:raise
                        except RECOVERABLE as exc:
                            recovery.reject_column_exception(G,old,j,exc)
                            continue
                        solve_time+=time.perf_counter()-t; total_passes+=int(passes);roots+=int(nroots)
                        max_inner_res=max(max_inner_res,float(resid))
                        if warned:
                            had_inner_warning=True;inner_warnings+=int(warn_n)
                            worst_inner_residual=max(worst_inner_residual,float(warn_worst))
                        if not ok:
                            current_profile_unconverged+=1
                            profile_unconverged+=1
                            inner_caps+=1;inner_failures+=1
                            record=dict(stage=stage,sweep=sweep,column=int(j),
                                **classify_inner_failure(passes,nroots,resid,cc['max_inner_pass']))
                            worst=inner_failure_worst
                            if worst is None or (record['local_residual'] is None) or (
                                    worst['local_residual'] is not None and record['local_residual']>worst['local_residual']):
                                inner_failure_worst=record
                        old_block=block_value(G,G[:,j],c,old,mu,lam)
                        inc=block_value(G,G[:,j],c,new,mu,lam)-old_block
                        max_increase=max(max_increase,inc)
                        block_limit=1e-8*(1.+abs(old_block))
                        if not recovery.certify_or_reject_column(G,G[:,j],c,old,new,j,mu,lam,
                                old_block,inc,block_limit,'base_conditional_column_monotonicity'):
                            continue
                        t=time.perf_counter()
                        if maintenance=='sm': K,_=context.commit_column(K,old,new,j)
                        else:W[:,j]=new
                        updates+=1
                        update_time+=time.perf_counter()-t
                except HardNumericalFailure:raise
                except (ComparisonUnresolved,TimeoutError):
                    recovery.rollback(anchor,'column_unresolved_or_timeout')
                    raise
                except RECOVERABLE as exc:
                    state,safe_delta,drift,scaled=recovery.restore(G,W,anchor,limit,
                        'base_sweep_monotonicity',dict(**location,column=int(j)),reason=str(exc))
                    interrupted=True
                t=time.perf_counter()
                if not interrupted:
                    location['current_profile_unconverged_columns']=current_profile_unconverged
                    state,safe_delta,drift,scaled=recovery.finish_sweep(G,W,K,anchor,limit,
                        'base_sweep_monotonicity',location)
            except (ComparisonUnresolved,NumericalUnresolved) as exc:
                action,state,safe_delta=recovery.rejected_sweep(G,anchor,location,exc,
                    rejection_retry_pending or sweep==cc['max_sweeps'],cc['outer_prox_resid_tol'],
                    cc['relative_tol']*max(abs(anchor.metrics['F']),1e-15))
                W,K=state.W,state.K;prev=state.metrics
                if action=='retry':
                    cur=state.metrics;change=0.
                    rejection_retry_pending=True
                    continue
                drift=0.;scaled=False;interrupted=False
            else:rejection_retry_pending=False
            inversions+=1;inv_time+=time.perf_counter()-t
            cur,K=state.metrics,state.K;W=state.W
            if drift is not None:max_drift=max(max_drift,float(drift))
            if interrupted or scaled:
                no_progress_recoveries=(no_progress_recoveries+1 if np.array_equal(W,anchor.W) else 0)
                if no_progress_recoveries>context.recovery_no_progress_retries:
                    require_certified(state)
                    recovery.record_soft('recovery_no_accepted_change',location,
                        rollback_domain_valid=state_reliable(state),stop_inferred_from_rejection=False)
            else:no_progress_recoveries=0
            change=-safe_delta/max(abs(prev['F']),1e-15)
            if scheduler.audit: trace.append(dict(stage=stage,sweep=sweep,seconds=time.perf_counter()-start,mu=mu,s=s,
                relative_decrease=change,total_column_updates=updates,**cur))
            stop=False
            # Practical branch: a reliable accepted full sweep may finish this
            # fixed-parameter stage by relative objective change. The original
            # residual remains a diagnostic, not a second completion gate.
            if not interrupted and not scaled and state_reliable(state):
                stop,_=comparator.full(G,W_previous,W,mu,s,lam,safe_delta,
                    cc['relative_tol']*max(abs(prev['F']),1e-15),'abs_le',abs(change)<=cc['relative_tol'],
                    'base_relative_stopping',max(abs(prev['F']),abs(cur['F'])),K_previous)
            prev=cur
            if stop:
                converged=True;break
        # `cur` already carries stop_residual / residual_normalized / kkt from
        # NumericalContext.evaluate; only the explicit decisions are added here.
        stages.append(dict(stage=stage,mu=mu,s=s,sweeps=sweep,converged=converged,
            current_profile_unconverged_columns=current_profile_unconverged,
            profile_warning_only=bool(current_profile_unconverged and converged),
            stop_metric='objective_relative_change',stop_tolerance=cc['relative_tol'],
            residual_reference_tolerance=cc['outer_prox_resid_tol'],
            objective_relative_change=abs(change),
            legacy_kkt_at_stop=float(cur['kkt']),
            had_inner_warning=bool(inner_warnings>warn_before),
            inner_warnings=inner_warnings-warn_before,
            inner_passes=total_passes-stage_pass0,**cur))
        if scheduler.audit:stage_weights.append(W.copy())
        if not converged: status='outer_cap'
    # rc6-fix2: only a column whose CURRENT state missed the profile-level
    # requirement is a warning when a reliable current global stop passed.
    # An actually incomplete global stop still retains the cap status.
    if current_profile_unconverged and not converged: status='profile_unconverged' if status=='completed' else status+'+profile_unconverged'
    return W,dict(status=status,seconds=time.perf_counter()-start,trace=trace,stages=stages,
        stop_metric='objective_relative_change',stop_tolerance=cc['relative_tol'],
        residual_reference_tolerance=cc['outer_prox_resid_tol'],
        inner_failures=inner_failures,inner_failure_worst=inner_failure_worst,
        inner_warnings=inner_warnings,had_inner_warning=had_inner_warning,
        worst_inner_residual=worst_inner_residual,profile_unconverged_columns=profile_unconverged,
        current_profile_unconverged_columns=current_profile_unconverged,
        inversions=inversions,column_updates=updates,inner_passes=total_passes,root_iterations=roots,
        inner_caps=inner_caps,max_inner_residual=max_inner_res,roundoff_negative_c_clamps=clamp_count,
        solve_seconds=solve_time,inverse_seconds=inv_time,update_seconds=update_time,
        inverse_max_relative_drift=max_drift,max_block_objective_increase=max_increase,
        effective_s=list(config['s'])),stage_weights


def polish_columns(G,W,mu,s,lam,max_sweeps=500,inner_tol=1e-9,outer_tol=1e-8,
                   min_sweeps=0,time_budget=None,stop_on_convergence=True,relative_tol=None, *, context=None, initial=None, audit=False, comparator=None, kernel=profile_column):
    """Unchanged exact column solver and rank-one update, arbitrary fixed start."""
    if relative_tol is None:relative_tol=1e-6
    start=time.perf_counter();W=W.copy()
    cur,K=context.reuse(W,mu,s,lam,initial) if initial is not None else context.evaluate(G,W,mu,s,lam)
    trace=[dict(sweep=0,seconds=0.,**cur)] if audit else []
    caps=0;max_drift=0.;converged=False;n_updates=0;inversions=1
    recovery=recovery_controller(context,comparator);no_progress_recoveries=0
    inner_failures=0;inner_failure_worst=None
    inner_warnings=0;had_inner_warning=False;worst_inner_residual=0.;profile_unconverged=0
    context.float64_c2_tolerance=outer_tol
    recovery.start_phase(G,W,(mu,s,lam),'polish')
    rejection_retry_pending=False
    for sweep in range(1,max_sweeps+1):
        current_profile_unconverged=0
        anchor=context.checkpoint();W_previous=anchor.W;K_previous=anchor.K
        location=dict(phase='polish',sweep=sweep);interrupted=False
        recovery.begin(G,anchor,location)
        try:
            try:
                for j in range(len(W)):
                    context.check_deadline()
                    old=W[:,j].copy();c=context.coefficients(K,old,j)
                    if np.min(c)<-1e-12:raise FloatingPointError('negative c')
                    try:
                        with context.ledger.measure('column_solve'):
                            new,passes,nroots,resid,ok,warned,warn_n,warn_worst=column_status_kernel(
                                G,G[:,j].copy(),np.maximum(c,0.),old,mu,lam,j,inner_tol,2000)
                    except HardNumericalFailure:raise
                    except RECOVERABLE as exc:
                        recovery.reject_column_exception(G,old,j,exc)
                        continue
                    # Same conditional objective and existing C1 allowance;
                    # no new numerical constant or alternative column solve.
                    old_block=block_value(G,G[:,j],np.maximum(c,0.),old,mu,lam)
                    inc=block_value(G,G[:,j],np.maximum(c,0.),new,mu,lam)-old_block
                    if not recovery.certify_or_reject_column(G,G[:,j],np.maximum(c,0.),old,new,j,mu,lam,
                            old_block,inc,1e-8*(1.+abs(old_block)),'polish_conditional_column_monotonicity'):
                        continue
                    caps+=int(not ok);K,_=context.commit_column(K,old,new,j);n_updates+=1
                    if warned:
                        had_inner_warning=True;inner_warnings+=int(warn_n)
                        worst_inner_residual=max(worst_inner_residual,float(warn_worst))
                    if not ok:
                        current_profile_unconverged+=1
                        profile_unconverged+=1
                        inner_failures+=1
                        record=dict(sweep=sweep,column=int(j),**classify_inner_failure(passes,nroots,resid,2000))
                        worst=inner_failure_worst
                        if worst is None or (record['local_residual'] is None) or (
                                worst['local_residual'] is not None and record['local_residual']>worst['local_residual']):
                            inner_failure_worst=record
            except HardNumericalFailure:raise
            except (ComparisonUnresolved,TimeoutError):
                recovery.rollback(anchor,'column_unresolved_or_timeout')
                raise
            except RECOVERABLE as exc:
                state,safe_delta,drift,scaled=recovery.restore(G,W,anchor,1e-11,
                    'polish_column_increase',dict(**location,column=int(j)),reason=str(exc))
                interrupted=True
            prev=cur
            if not interrupted:
                location['current_profile_unconverged_columns']=current_profile_unconverged
                state,safe_delta,drift,scaled=recovery.finish_sweep(G,W,K,anchor,1e-11,
                    'polish_column_increase',location)
        except (ComparisonUnresolved,NumericalUnresolved) as exc:
            action,state,safe_delta=recovery.rejected_sweep(G,anchor,location,exc,
                rejection_retry_pending or sweep==max_sweeps,outer_tol,
                (relative_tol or 0.)*max(abs(anchor.metrics['F']),1e-15),
                can_stop=sweep>=min_sweeps and (time_budget is None or time.perf_counter()-start>=time_budget))
            W,K=state.W,state.K;cur=state.metrics;prev=cur
            if action=='retry':
                rejection_retry_pending=True
                continue
            drift=0.;scaled=False;interrupted=False
        else:rejection_retry_pending=False
        cur,K=state.metrics,state.K;W=state.W;inversions+=1
        if drift is not None:max_drift=max(max_drift,float(drift))
        if interrupted or scaled:
            no_progress_recoveries=(no_progress_recoveries+1 if np.array_equal(W,anchor.W) else 0)
            if no_progress_recoveries>context.recovery_no_progress_retries:
                require_certified(state)
                recovery.record_soft('recovery_no_accepted_change',location,
                    rollback_domain_valid=state_reliable(state),stop_inferred_from_rejection=False)
        else:no_progress_recoveries=0
        if audit:trace.append(dict(sweep=sweep,seconds=time.perf_counter()-start,**cur))
        # State safety plus the accepted full-sweep relative objective stop.
        converged=state_reliable(state)
        if interrupted or scaled:converged=False
        if converged and relative_tol is not None:
            limit=relative_tol*max(abs(prev['F']),1e-15)
            converged,_=comparator.full(G,W_previous,W,mu,s,lam,safe_delta,limit,'abs_le',
                abs(safe_delta)<=limit,'polish_relative_stopping',max(abs(prev['F']),abs(cur['F'])),K_previous)
        enough_time=time_budget is None or time.perf_counter()-start>=time_budget
        observer=getattr(context,'float64_observer',None)
        if observer is not None and not interrupted and not scaled and sweep>=min_sweeps and enough_time:
            relative_ok=relative_tol is None or recovery.stopping_ok(safe_delta,relative_tol*max(abs(prev['F']),1e-15))
            if relative_ok and cur['residual_normalized']<=1e-3 and state_reliable(state):
                observer(context,G,W,mu,s,lam,sweep,cur)
        if not interrupted and not scaled and sweep>=min_sweeps and enough_time and (converged or not stop_on_convergence):break
    return W,dict(seconds=time.perf_counter()-start,sweeps=sweep,
                  converged=bool(converged),
                  profile_warning_only=bool(converged and current_profile_unconverged),
                  outer_converged=bool(converged),
                  current_profile_unconverged_columns=current_profile_unconverged,
                  inner_caps=caps,max_inverse_drift=max_drift,column_updates=n_updates,inversions=inversions,trace=trace,
                  stop_metric='objective_relative_change',stop_tolerance=relative_tol,
                  residual_reference_tolerance=outer_tol,
                  objective_relative_change=abs(float(safe_delta))/max(abs(prev['F']),1e-15),
                  stop_residual=float(cur['residual_normalized']),legacy_kkt_at_stop=float(cur['kkt']),
                  inner_failures=inner_failures,inner_failure_worst=inner_failure_worst,
                  inner_warnings=inner_warnings,had_inner_warning=had_inner_warning,
                  worst_inner_residual=worst_inner_residual,
                  profile_unconverged_columns=profile_unconverged,
                  _state=context.handoff(cur,K,mu,s,lam) if context is not None else None,final=cur)

"""Public fit/refine entry points: one explicit CRC pass, no global function patching.

One active numerical fit per process. Use process-level parallelism (BLAS=1).
Validation is done at boundaries; inner loops use their inherited numerical gates.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
import signal, threading, time, traceback
import numpy as np
import scipy.linalg as sla
from threadpoolctl import threadpool_limits
from . import __version__, guards
from .config import Config
from .numerics import NumericalContext, bounded_inverse, rank_practical, positive_m_inverse
from .float64_acceptance import FloatComparator as Comparator, set_phase
from .c2_comparison import ComparisonUnresolved
from .soft_events import HardNumericalFailure
from ._kernel.profile import profile_column, profile_column_status
from .column import block_value, prepare
from .controller import ProductionBridge
from .admission import joint_admission
from .natural_r_search import repair_endpoint_natural
from .ranking import warmup as warmup_ranking
from .finite_policy import state_reliable, POLICY_ID
from .objective_stopping import output_qualified, POLICY_ID as STOPPING_POLICY_ID

_LOCK=threading.RLock()
_WARMED=False


def warmup():
    global _WARMED
    start=time.perf_counter()
    if not _WARMED:
        G=np.eye(4);u=np.zeros(4)
        positive_m_inverse(G*0,1.)
        rank_practical(G,u,u,0);rank_practical(np.asfortranarray(G),u,u,0)
        block_value(G,G[:,0],u,u,1.,.03);block_value(G,u,u,u,1.,.03)
        profile_column(G,u,u,u,1.,.03,0,1e-6,2000)
        profile_column_status(G,u,u,u,1.,.03,0,1e-6,2000)
        warmup_ranking()
        _WARMED=True
    return time.perf_counter()-start


def graph_is_dag(graph):
    a=np.asarray(graph,dtype=bool)
    if np.diag(a).any():return False
    indegree=a.sum(axis=0).astype(int);queue=list(np.flatnonzero(indegree==0));count=0
    while queue:
        i=queue.pop();count+=1
        for j in np.flatnonzero(a[i]):
            indegree[j]-=1
            if indegree[j]==0:queue.append(int(j))
    return count==len(a)


def independent_check(X,W,mu,s,lam=.03,threshold=1e-4,*,ledger=None):
    """One fresh sample-space audit. It does not change W or stopping tolerances."""
    n,d=X.shape;Y=np.array(X,dtype=np.float64,copy=True);Y-=Y.mean(axis=0,keepdims=True)
    I=np.eye(d);M=s*I-W*W
    raw=sla.solve(M,I,check_finite=False)
    K,ld,_,num=bounded_inverse(W,s,mu,K=raw,ledger=ledger)
    E=Y@W-Y;loss=float(np.sum(E*E)/(2*n));l1=float(lam*np.abs(W).sum());h=float(d*np.log(s)-ld)
    grad=mu*(Y.T@E/n)+2*W*K.T
    z=W-grad;p=np.sign(z)*np.maximum(np.abs(z)-mu*lam,0.);r=W-p;np.fill_diagonal(r,0.)
    residual=float(np.max(abs(r)))/mu
    return dict(F=mu*(loss+l1)+h,loss=loss,l1=l1,h=h,mu=mu,s=s,
        residual_normalized=residual,residual_absolute=float(np.max(abs(r))),
        legal=bool(num['domain']['legal']),reliable=bool(num['accuracy']['reliable']),
        domain_lower=num['domain']['lower'],inverse_error_bound=num['accuracy']['inverse_relative_bound'],
        gradient_error_bound=num['accuracy']['gradient_scaled_bound'],backend=num['backend'],
        threshold=threshold,passed=bool(residual<=threshold),independent=True,
        source='sample-space tau=1 prox mapping / mu; fresh solve')


def _input(X):
    a=np.asarray(X,dtype=np.float64)
    if a.ndim!=2 or min(a.shape)<2 or not np.isfinite(a).all():
        raise ValueError('X must be finite n x d with n,d >= 2')
    return a


def _configuration(config,hot_cap_seconds,max_corrections):
    cfg=Config() if config is None else config
    if not isinstance(cfg,Config):raise TypeError('config must be Config')
    changes={}
    if hot_cap_seconds is not None:changes['hot_cap_seconds']=hot_cap_seconds
    if max_corrections is not None:
        raise ValueError('Natural R has no correction-count cap; Top-8 is unchanged')
    return replace(cfg,**changes)


@contextmanager
def _budget(seconds,state=None):
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('Use separate processes: the budget is enforced on the main thread')
    if signal.getitimer(signal.ITIMER_REAL)[0]>0:
        raise RuntimeError('An existing process alarm is active; use a separate worker')
    old=signal.getsignal(signal.SIGALRM)
    if state is None:state={'triggered':False,'deadline':time.monotonic()+seconds}
    def timeout(*_):
        state['triggered']=True
        raise TimeoutError(f'Prespecified {seconds:g}-second hot-fit cap')
    signal.signal(signal.SIGALRM,timeout)
    try:
        signal.setitimer(signal.ITIMER_REAL,seconds)
        yield state
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)


def _repair_status(repair,all_stages_ok=True):
    reason=repair['stop_reason']
    if reason in ('polish_cap','terminal_polish_rejected'):return reason
    if reason=='candidate_budget_numerically_inconclusive':return 'joint_numerically_inconclusive'
    return 'completed' if all_stages_ok else 'completed_with_warm_caps'


COMPLETED_STATUSES=('completed','completed_with_inner_warnings','completed_with_warm_caps')


def trusted_partial(ctx, threshold):
    """Export an already checked, owned state; never perform a new fit/audit.

    This is deliberately separate from W/graph and from an independent final
    residual. No stale in-flight state or threshold-only DAG is certified here.
    """
    try:
        gate=ctx.recovery
        if gate is not None and hasattr(gate,'G_reference'):
            if not np.array_equal(gate.G_reference,gate.G_basis):return None
        state=ctx.checkpoint()
        if not state_reliable(state):return None
    except (RuntimeError,guards.NumericalUnresolved,ValueError):return None
    graph=(abs(state.W)>=threshold).astype(np.uint8)
    return dict(W=state.W,graph=graph,mu=state.params[0],s=state.params[1],
        lambda1=state.params[2],state_reliable=True,domain_legal=True,
        dag=graph_is_dag(graph),residual_normalized=state.metrics['residual_normalized'],
        independent_residual=False,metrics=state.metrics,
        location=getattr(ctx,'last_accepted_location',None),state_version=ctx.state_version,
        source='owned current checked state; not a complete CRC result')


def _stage_inner_warnings(stages):
    """Historical inner active-set warnings, per stage and in total.

    These are diagnostics: they never make a current column unacceptable.
    """
    flags=[bool(z.get('had_inner_warning')) for z in stages]
    total=sum(int(z.get('inner_warnings') or 0) for z in stages)
    worst=0.
    for z in stages:
        value=(z.get('inner_failure_worst') or {}).get('local_residual')
        value2=z.get('worst_inner_residual')
        for candidate in (value,value2):
            if candidate is not None and np.isfinite(candidate):worst=max(worst,float(candidate))
    return flags,total,worst


def _execute(X,cfg,mode,checkpoints,endpoint=None,run_reconstruction=True,exploration=None):
    if mode not in ('production','audit'):raise ValueError('mode must be production or audit')
    if mode=='audit' and checkpoints is None:raise ValueError('audit requires checkpoints/output path')
    if threading.current_thread() is not threading.main_thread():raise RuntimeError('use isolated processes for fits')
    c2_threshold=(exploration or {}).get('C2_final_residual',cfg.outer_prox_resid_tol)
    if exploration and 'C2_final_residual' in exploration and c2_threshold not in (1e-4,1e-3):
        raise ValueError('This exploration permits only A=1e-4 or B=1e-3 for C2')
    path=Path(checkpoints) if checkpoints else None
    if path:path.mkdir(parents=True,exist_ok=True)
    with _LOCK,threadpool_limits(1):
        warm_seconds=warmup();guards.reset_stats()
        ctx=NumericalContext(audit=mode=='audit',recovery_alphas=cfg.recovery_alphas,
                             recovery_no_progress_retries=cfg.recovery_no_progress_retries)
        if path:
            ctx.recovery_event_path=path/'recovery_events.jsonl'
            ctx.float64_progress_path=path/'float64_progress.jsonl'
        if exploration:
            ctx.float64_observer=exploration.get('observer')
        cmp=Comparator(path/'comparison_events' if mode=='audit' else None,
                       path/'anomalies' if mode=='audit' else None)
        cmp.ledger=ctx.ledger
        if exploration and exploration.get('context_hook'):exploration['context_hook'](ctx,cmp)
        set_phase(ctx,'C1')
        model=None;repair=None;phase='column' if endpoint is None else 'endpoint'
        timers=dict(column_prefix=0.,joint_inclusive=0.,final_independent_check=0.)
        result=dict(method_id=__version__,mode=mode,final_returned=False,status='incomplete',
            termination='running',n=X.shape[0],d=X.shape[1],truth_input=False,
            cap_seconds=cfg.hot_cap_seconds,warmup_seconds=warm_seconds,
            native_converged=False,numerically_qualified=False,ready_for_joint=False,
            joint_executed=False,joint_skip_reason='continuation_not_finished',
            all_stages_native_completed=False,final_stage_completed=False,
            W=None,W_raw=None,graph=None,W_column=None,
            resolved_configuration=dict(**asdict(cfg),framework='single CRC',column_order='fixed forward',
                pair_ranking='predicted_J_gain',correction_count_cap=None,
                R_stopping='current Top-8 exhausted without certified acceptance; uncertain candidates rejected',
                profile_gate=(exploration or {}).get('R_acceptance_policy','BLOCK-FULL'),
                pair_eligibility='i<j and max(abs(W[i,j]),abs(W[j,i]))>=edge_threshold',per_accept_polish=False,terminal_polish=True,
                preparation='center only',kernel='Cholesky with LU fallback',fastmath=False),
            endpoint_input=endpoint is not None)
        result['C2_final_residual_threshold']=c2_threshold
        start=time.perf_counter();cmp.hot_deadline=time.monotonic()+cfg.hot_cap_seconds
        budget_state={'triggered':False,'deadline':cmp.hot_deadline}
        ctx.budget_state=budget_state
        try:
            with _budget(cfg.hot_cap_seconds,budget_state):
                if endpoint is None:
                    model=ProductionBridge(cfg.algorithm(),ctx,audit=mode=='audit',checkpoints=path,
                                           comparator=cmp,kernel=profile_column)
                    t=time.perf_counter()
                    try:
                        model.fit(X.copy(),s=list(cfg.initial_s),lambda1=cfg.lambda1,w_threshold=cfg.threshold,
                            T=cfg.T,mu_init=cfg.mu_init,mu_factor=cfg.mu_factor,warm_iter=30000,max_iter=60000,
                            lr=.0003,checkpoint=1000,beta_1=.99,beta_2=.999,
                            exclude_edges=tuple((i,i) for i in range(X.shape[1])))
                    finally:timers['column_prefix']=time.perf_counter()-t
                    W,mu,s=model.raw_final_state()
                    result['W_column']=W.copy()
                    info=joint_admission(model,ctx,W,mu,s,model.last_lambda,cmp.hot_deadline)
                    result.update(info);native_ok=info['all_stages_native_completed']
                    ctx.float64_C1_prefix=dict(info)
                    initial=model.last_evaluation
                    if exploration and exploration.get('endpoint_observer'):
                        exploration['endpoint_observer']('C',ctx,model.cov,W,mu,s,cfg.lambda1)
                else:
                    W,mu,s=endpoint;W=W.copy()
                    G_endpoint=prepare(X)
                    ctx.evaluate(G_endpoint,W,mu,s,cfg.lambda1)
                    initial=ctx.handoff(ctx.cached[1],ctx.K,mu,s,cfg.lambda1)
                    native_ok=True
                    result.update(ready_for_joint=True,joint_skip_reason=None,final_stage_completed=None,
                                  all_stages_native_completed=None)
                    ctx.float64_C1_prefix=dict(ready_for_joint=True,endpoint_input=True)
                if result['ready_for_joint'] and run_reconstruction:
                    phase='reconstruction';set_phase(ctx,'R');result['joint_executed']=True;t=time.perf_counter()
                    try:
                        G=model.cov if model else G_endpoint
                        terminal_config=dict(cfg.algorithm()['column'],outer_prox_resid_tol=c2_threshold)
                        W,repair,jumps=repair_endpoint_natural(G,W,mu,s,cfg.lambda1,terminal_config,cfg.search(),
                            context=ctx,comparator=cmp,audit=mode=='audit',initial=initial,
                            kernel=(exploration or {}).get('R_kernel',profile_column_status),
                            observer=(exploration or {}).get('R_observer'),retain_snapshots=False,
                            acceptance_policy=(exploration or {}).get('R_acceptance_policy','BLOCK-FULL'),
                            dag_checker=(exploration or {}).get('R_dag_checker',
                                lambda weights: graph_is_dag((abs(weights)>=cfg.threshold).astype(np.uint8))))
                    finally:timers['joint_inclusive']=time.perf_counter()-t
                    status=_repair_status(repair,native_ok)
                    if mode=='audit' and jumps:np.savez(path/'accepted_jumps.npz',W=np.asarray(jumps))
                elif not run_reconstruction:
                    status='completed' if native_ok else 'native_cap'
                    result['joint_skip_reason']='column_only_requested'
                else:status='native_cap'
                phase='final_check';set_phase(ctx,'final_check');t=time.perf_counter()
                final_threshold=(c2_threshold if result['joint_executed'] and exploration
                    and 'C2_final_residual' in exploration else cfg.evaluation_residual_threshold)
                try:check=independent_check(X,W,mu,s,cfg.lambda1,final_threshold,ledger=ctx.ledger)
                finally:timers['final_independent_check']=time.perf_counter()-t
                graph=(abs(W)>=cfg.threshold);dag=graph_is_dag(graph)
                warning_flags,warning_total,warning_worst=_stage_inner_warnings(model.stage_results if model else [])
                result['stage_inner_warning_flags']=warning_flags
                result['inner_warning_total']=warning_total
                result['worst_inner_warning_residual']=warning_worst
                result['stage_inner_warning']=bool(any(warning_flags))
                if status=='completed' and any(warning_flags):
                    status='completed_with_inner_warnings'
                completed=status in COMPLETED_STATUSES
                check['required_for_completion']=False
                result.update(final_returned=True,status=status,termination='returned',native_converged=completed,
                    residual_check=check,dag=bool(dag),numerically_qualified=output_qualified(completed,check,dag),
                    stopping_policy=STOPPING_POLICY_ID,residual_required_for_completion=False,
                    residual_passed_1e4=bool(check['residual_normalized']<=1e-4),
                    residual_passed_1e3=bool(check['residual_normalized']<=1e-3),
                    mu=mu,s=s,final_gram_objective=float(repair['final']['F'] if repair else model.last_evaluation[0]['F']),
                    W=W,W_raw=W,graph=graph.astype(np.uint8))
        except Exception as exc:
            timed_out=isinstance(exc,TimeoutError) or budget_state['triggered']
            result.update(status='incomplete',termination='timeout' if timed_out else 'hard_numerical_failure' if isinstance(exc,HardNumericalFailure) else 'comparison_unresolved' if isinstance(exc,ComparisonUnresolved) else 'numerical_unresolved' if isinstance(exc,guards.NumericalUnresolved) else type(exc).__name__,
                exception_type=type(exc).__name__,exception=str(exc),failure_phase=phase,traceback=traceback.format_exc())
            if not result['joint_executed']:result['joint_skip_reason']='global_budget_exhausted' if timed_out else 'unresolved_failure: '+str(exc)
            if path and ctx.latest is not None:
                w,m,s0=ctx.latest;np.savez(path/'partial_not_a_result.npz',W=w,mu=m,s=s0)
        finally:
            result['finite_precision_policy']=POLICY_ID
            if result.get('numerically_qualified'):
                result['result_class']='fully_qualified'
            else:
                partial=trusted_partial(ctx,cfg.threshold)
                result['partial_result']=partial
                result['result_class']='trustworthy_partial' if partial is not None else 'no_trustworthy_result'
            set_phase(ctx,None)
            result['phase_seconds']=dict(ctx.float64_phase_seconds)
            result['deadline_triggered']=budget_state['triggered']
            result['recovery']=ctx.recovery.summary() if ctx.recovery is not None else {'counts':{},'seconds':0.,'events':[]}
            result.update(seconds=time.perf_counter()-start,repair=repair,comparison=cmp.summary(),inverse=ctx.summary(),
                guard={k:v for k,v in guards.STATS.items() if not isinstance(v,(dict,list,np.ndarray))})
            if model is not None:
                result.update(actual_s=model.actual_s,actual_parameter_path=[dict(stage=e['stage'],mu=e['mu'],s=e['s']) for e in model.attempts if e.get('success') is True],
                    retries=model.start_rejections,attempts=model.attempts,stages=model.stage_results,
                    all_stages_native_completed=bool(len(model.stage_results)==cfg.T and all(z['status']=='completed' for z in model.stage_results)),
                    final_stage_completed=bool(len(model.stage_results)==cfg.T and model.stage_results[-1]['status']=='completed'))
            elif endpoint is not None:result.update(actual_s=[endpoint[2]],actual_parameter_path=[dict(stage='external_endpoint',mu=endpoint[1],s=endpoint[2])],stages=[],attempts=[],retries=0)
            else:result.update(actual_s=[],actual_parameter_path=[],stages=[],attempts=[],retries=0)
            timers.update(joint_ranking=repair['ranking_seconds'] if repair else 0.,joint_candidates=repair['candidate_seconds'] if repair else 0.,
                polish=repair['polish_seconds'] if repair else 0.,final_check=timers['final_independent_check'],
                checkpoint_io=model.io_seconds if model else 0.,comparison_native=cmp.seconds['native_total'],
                comparison_direct=cmp._stats()['direct_seconds'],inverse_maintenance=ctx.seconds['rank_total'],inverse_refresh=ctx.seconds['inverse_refresh'],
                note='joint includes polish; numerical timers overlap phases; do not add both axes')
            result['timing']=timers
            result['exclusive_timing']=ctx.ledger.summary()
        return result


def fit(X, *, config=None, mode='production', checkpoints=None, hot_cap_seconds=None,
        max_corrections=None, column_only=False, exploration=None):
    """Run one CRC. ``column_only=True`` exposes the unchanged first stage.

    Returned W is unthresholded; graph is a separate 0/1 output. Failed fits have
    W=None; a checked partial_result may be returned separately. A returned cap
    is labelled, never silently described as converged.
    """
    X=_input(X);cfg=_configuration(config,hot_cap_seconds,max_corrections)
    return _execute(X,cfg,mode,checkpoints,run_reconstruction=not column_only,exploration=exploration)


def refine(X,W, *, mu,s,config=None,mode='production',checkpoints=None,
           hot_cap_seconds=None,max_corrections=None,exploration=None):
    """Run only RC from a saved raw endpoint. Does not claim prefix convergence."""
    X=_input(X);cfg=_configuration(config,hot_cap_seconds,max_corrections)
    W=np.asarray(W,dtype=np.float64)
    if W.shape!=(X.shape[1],X.shape[1]) or not np.isfinite(W).all() or np.any(np.diag(W)):
        raise ValueError('W must be a finite d x d matrix with zero diagonal')
    if not np.isfinite(mu) or not np.isfinite(s) or mu<=0 or s<=0:raise ValueError('mu/s must be positive and finite')
    return _execute(X,cfg,mode,checkpoints,endpoint=(W,float(mu),float(s)),exploration=exploration)

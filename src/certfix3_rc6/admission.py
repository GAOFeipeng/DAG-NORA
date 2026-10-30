"""Joint admission from the last solved state, separate from stage history."""
import time
import numpy as np
from . import guards as ns

def joint_admission(model,context,W,mu,s,lam,hot_deadline):
    stages=model.stage_results
    full_path=len(stages)==model.config['T']
    final_completed=full_path and stages[-1]['status']=='completed'
    info=dict(all_stages_native_completed=bool(full_path and all(z['status']=='completed' for z in stages)),
              final_stage_completed=bool(final_completed),ready_for_joint=False,
              joint_executed=False,joint_skip_reason=None)
    if not full_path:info['joint_skip_reason']='continuation_path_incomplete';return info
    if not final_completed:info['joint_skip_reason']='final_stage_not_completed';return info
    if time.monotonic()>=hot_deadline:raise TimeoutError('Prespecified hot-fit cap exhausted')
    # Earlier finite, checked iterates may have missed a stopping condition.
    # Only the completed final stage determines current eligibility; unknown
    # numerical failures and unfinished attempts still invalidate the path.
    allowed_history=('completed','outer_cap','profile_unconverged','outer_cap+profile_unconverged')
    if model.current_attempt is not None or any(z['status'] not in allowed_history for z in stages):
        raise ns.NumericalUnresolved('joint_admission_unresolved_stage')
    successes=[e for e in model.attempts if e.get('success') is True]
    if len(successes)!=model.config['T'] or [e['stage'] for e in successes]!=list(range(model.config['T'])):
        raise RuntimeError('joint_admission_path_binding_invalid')
    if model.last_params!=(mu,s) or model.last_lambda!=lam or model.lambda1!=lam:
        raise RuntimeError('joint_admission_parameter_binding_invalid')
    if not np.array_equal(W,model.last_raw) or context.latest is None:
        raise RuntimeError('joint_admission_weight_binding_invalid')
    _,cm,cs=context.latest
    cur,K,B,params,version=model.last_evaluation
    # Upstream thresholds its returned W in place. The raw copy and handoff
    # were bound inside minimize, before that mutation of context.latest[0].
    bw,bparams,blam,bversion=model.last_raw_binding
    if bw is not model.last_raw or bparams!=(mu,s) or blam!=lam or bversion!=version:
        raise RuntimeError('joint_admission_raw_token_invalid')
    if (cm,cs)!=(mu,s) or params!=(mu,s,lam):
        raise RuntimeError('joint_admission_cached_binding_invalid')
    if context.cached is None or version!=context.state_version or context.cached[0]!=version or K is not context.K:
        raise RuntimeError('joint_admission_stale_inverse')
    if W.ndim!=2 or W.shape[0]!=W.shape[1] or not np.isfinite(W).all() or np.any(np.diag(W)):
        raise ns.NumericalUnresolved('joint_admission_invalid_weights')
    if not all(np.isfinite(v) for v in (mu,s,lam)) or mu<=0 or s<=0:
        raise ns.NumericalUnresolved('joint_admission_invalid_parameters')
    # Check the existing inverse in scratch space; never update algorithm K/C.
    diag,_=ns.check_existing(W,K,s,mu)
    if not diag['domain']['legal'] or not diag['accuracy']['reliable']:
        raise ns.NumericalUnresolved('joint_admission_domain_or_inverse_unresolved')
    if time.monotonic()>=hot_deadline:raise TimeoutError('Prespecified hot-fit cap exhausted')
    info.update(ready_for_joint=True,joint_bound_parameters=dict(mu=mu,s=s,lambda1=lam),
                joint_domain_legal=True,joint_inverse_reliable=True)
    return info

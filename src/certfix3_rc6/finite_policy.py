"""One bounded engineering policy; no new numerical evidence or backend."""
import math

POLICY_ID = 'practical_finite_precision_v1'
SINGLE_RELATIVE_F = 1e-8
STAGE_POSITIVE_CAP = 1e-6


def objective_scale(anchor, trial):
    mu = anchor.params[0]
    return max(1., *(abs(mu*s.metrics['J'])+abs(s.metrics['h'])
                     for s in (anchor, trial)))


def decide(delta, error, reliable, scale):
    limit = SINGLE_RELATIVE_F*scale
    out = dict(policy_id=POLICY_ID, scale_F=scale, single_limit_F=limit,
               error_F=error, delta_F=delta, accepted=False,
               normalized_positive_charge=None)
    if (not reliable or not all(math.isfinite(v) for v in (delta,error,scale))
            or error < 0 or scale < 1):
        return dict(out, action='unresolved', reason='unreliable_evidence')
    upper = delta+error
    out.update(upper_F=upper, lower_F=delta-error)
    if upper <= 0:
        return dict(out, accepted=True, action='accept',
                    reason='nonpositive_interval', normalized_positive_charge=0.)
    if error <= limit and upper <= limit:
        return dict(out, accepted=True, action='accept', reason='bounded_finite_error',
                    normalized_positive_charge=upper/scale)
    return dict(out, action='reject' if delta-error > limit else 'unresolved',
                reason='outside_finite_error_allowance')


def state_reliable(state):
    import numpy as np
    m=state.metrics
    return bool(all(np.isfinite(a).all() for a in (state.W,state.K,state.B))
        and not np.any(np.diag(state.W)) and m.get('domain_lower',0)>0
        and m.get('inverse_error_bound',float('inf'))<=1e-8
        and m.get('gradient_error_bound',float('inf'))<=1e-8
        and all(math.isfinite(m.get(k,float('inf'))) for k in ('F','J','h'))
        and math.isfinite(m.get('residual_normalized',float('inf'))))


def stop_change(delta, original_bound, scale, error=None):
    # The ordinary recorded path retains its existing engineering check;
    # absence of an endpoint envelope is not a claim of zero rounding error.
    magnitude=abs(float(delta))+(error if error is not None else 0.)
    return bool(math.isfinite(magnitude) and magnitude <=
                max(original_bound,SINGLE_RELATIVE_F*scale))

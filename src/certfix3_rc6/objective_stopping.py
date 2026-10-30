"""Practical stopping only; candidate acceptance and state safety are unchanged."""
import math

POLICY_ID = 'crc_objective_relative_stop_v1'
RELATIVE_TOLERANCE = 1e-6
OBJECTIVE_FLOOR = 1e-15


def within_bound(delta, bound):
    """Called only with an accepted, complete, state-bound sweep difference."""
    return bool(math.isfinite(float(delta)) and math.isfinite(float(bound))
                and bound >= 0 and abs(float(delta)) <= bound)


def relative_change(delta, objective_before):
    if not all(math.isfinite(float(v)) for v in (delta, objective_before)):
        return float('inf')
    return abs(float(delta)) / max(abs(float(objective_before)), OBJECTIVE_FLOOR)


def output_qualified(completed, check, dag):
    """Residual attainment is reported separately from this practical endpoint."""
    return bool(completed and check.get('legal') and check.get('reliable') and dag
                and math.isfinite(check.get('residual_normalized', float('nan'))))

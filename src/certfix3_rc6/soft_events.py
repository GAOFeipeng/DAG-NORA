"""Classification only: no objective, threshold, solver or search policy."""
import numpy as np
from .c2_comparison import ComparisonUnresolved
from .finite_policy import state_reliable


class HardNumericalFailure(ComparisonUnresolved):
    """The incumbent/data/transaction is unusable, not an undecided proposal."""


def require_finite(*values):
    if any(not np.isfinite(v).all() for v in values):
        raise HardNumericalFailure('nonfinite_current_state_or_data')


def require_certified(state):
    if not state_reliable(state):
        raise HardNumericalFailure('last_certified_state_failed_existing_safety_checks')
    return state

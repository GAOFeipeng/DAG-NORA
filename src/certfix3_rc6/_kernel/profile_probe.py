"""Diagnostic-only variant of :func:`profile_column` that reports its exit branch.

The outer loop is a line-for-line copy of the production kernel and calls the
SAME `_active_solve`; the only additions are writes into two caller-supplied
arrays. ``tests/test_stop_metric_unification.py`` asserts bitwise equality of
every returned value against ``profile_column`` on random and failing inputs, so
running this probe cannot change a trajectory - it only explains why a column
solve did not reach its tolerance.

Exit codes written to ``diag_i[0]``:
  0 converged (local KKT residual reached the tolerance)
  1 profile outer loop consumed its 80-iteration budget
  2 outer Newton step stagnated (``newt == t``) before the budget
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .kkt import kkt_vec
from .lapack_backend import solve_spd
from .profile import _active_solve


@njit(cache=True)
def profile_column_probe(G, b, c, u0, mu, lam, skip, tol, max_pass, diag_i, diag_f):
    """Same arithmetic as ``profile_column`` plus exit-branch diagnostics.

    diag_i: int64[8]  = [code, lasso_ok_last, lasso_iter_last, total_active,
                         outer_used, q_ge_1_count, nonzero_last, 0]
    diag_f: float64[3] = [err, q_last, t_last]
    """
    u = u0.copy()
    u[skip] = 0.
    q0 = np.dot(c, u * u)
    t = 1. / (1. - q0)
    low = 1.
    high = np.inf
    total_active = 0
    good = True
    err = np.inf
    code = 1
    used = 0
    q_ge_1 = 0
    lasso_ok_last = 1
    lasso_iter_last = 0
    q_last = q0
    nz_last = 0
    for outer in range(80):
        used = outer + 1
        diagonal = (2. * t / mu) * c
        u, na, lasso_err, ok = _active_solve(
            G, b, diagonal, u, lam, skip, min(tol * .01, 1e-9), max_pass
        )
        total_active += na
        good = good and ok
        lasso_ok_last = 1 if ok else 0
        lasso_iter_last = na
        q = np.dot(c, u * u)
        q_last = q
        nz_last = np.sum(u != 0)
        if q < 1.:
            g = G @ u - b + (2. / mu) * c * u / (1. - q)
            err = kkt_vec(u, g, lam, skip)
            if err <= tol:
                code = 0
                diag_i[0] = code
                diag_i[1] = lasso_ok_last
                diag_i[2] = lasso_iter_last
                diag_i[3] = total_active
                diag_i[4] = used
                diag_i[5] = q_ge_1
                diag_i[6] = nz_last
                diag_f[0] = err
                diag_f[1] = q
                diag_f[2] = t
                return u, total_active, outer + 1, err, good
        else:
            q_ge_1 += 1
        residual = q - 1. + 1. / t
        if residual > 0:
            low = t
        else:
            high = t
        ids = np.where(u != 0)[0]
        p = len(ids)
        deriv = -1. / (t * t)
        if p:
            A = np.empty((p, p))
            v = np.empty(p)
            for a in range(p):
                i = ids[a]
                v[a] = c[i] * u[i]
                for z in range(p):
                    A[a, z] = G[i, ids[z]]
                A[a, a] += diagonal[i]
            dv = solve_spd(A, v)
            deriv -= (4. / mu) * np.dot(v, dv)
        newt = t - residual / deriv
        if not np.isfinite(high):
            newt = min(max(newt, low * (1. + 1e-8)), 2. * low)
        elif newt <= low or newt >= high:
            newt = .5 * (low + high)
        if newt == t:
            code = 2
            break
        t = newt
    diag_i[0] = code
    diag_i[1] = lasso_ok_last
    diag_i[2] = lasso_iter_last
    diag_i[3] = total_active
    diag_i[4] = used
    diag_i[5] = q_ge_1
    diag_i[6] = nz_last
    diag_f[0] = err
    diag_f[1] = q_last
    diag_f[2] = t
    return u, total_active, used, err, False

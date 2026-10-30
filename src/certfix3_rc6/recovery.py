"""Transactional recovery on the convex ray of the unchanged CRC objective.

This is bounded feasibility/descent backtracking, not a claim of exact scalar
minimization. Exact conditional-column arithmetic remains in _kernel/profile.py.
"""
from __future__ import annotations
import json
import math
import time
from collections import Counter
import numpy as np
from .guards import NumericalUnresolved
from .c2_comparison import Verdict, ComparisonUnresolved
from .timing import metered

DEFAULT_ALPHAS = (1.-1e-6, 1.-1e-5, 1.-1e-4, 1.-1e-3, 1.-1e-2, .9, .5, .25)
RECOVERABLE = (FloatingPointError, np.linalg.LinAlgError, OverflowError)


def _finite(value):
    if isinstance(value, dict): return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [_finite(v) for v in value]
    if isinstance(value, np.generic): return _finite(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


class Recovery:
    def __init__(self, context, comparator):
        self.context = context
        self.comparator = comparator
        self.ledger = context.ledger
        comparator.ledger = self.ledger
        self.events = []
        self.counts = Counter()
        self.seconds = 0.

    def record(self, event):
        event = _finite(event)
        self.events.append(event)
        self.counts[event['action']] += 1
        self.seconds += event['seconds']
        if self.context.recovery_event_path is not None:
            with self.context.recovery_event_path.open('a') as handle:
                handle.write(json.dumps(event, allow_nan=False) + '\n')

    @metered('comparison_screen_and_bookkeeping')
    def compare(self, G, anchor, trial, limit, site):
        if site == 'polish_column_increase':
            answer = self.comparator.c2.decide(G, anchor, trial, limit, site)
            if answer.verdict is Verdict.UNRESOLVED:
                raise ComparisonUnresolved('C2 sweep comparison unresolved: '+str(answer.evidence))
            return answer.verdict is Verdict.ACCEPT, answer.delta
        mu, s, lam = anchor.params
        delta = trial.metrics['F'] - anchor.metrics['F']
        return self.comparator.full(G, anchor.W, trial.W, mu, s, lam,
            delta, limit, 'le', delta <= limit, site,
            max(abs(anchor.metrics['F']), abs(trial.metrics['F'])), anchor.K)

    def rollback_comparison(self, W, anchor, location, exc, start):
        """Keep both exact states as evidence; only the anchor is an iterate."""
        ctx = self.context
        # Copy BEFORE adopting the anchor: W can alias the in-flight context.
        candidate = np.array(W, copy=True)
        ctx.adopt(anchor)
        if ctx.recovery_event_path is not None:
            path = ctx.recovery_event_path.parent/'c2_unresolved_candidate.npz'
            np.savez(path, candidate=candidate, anchor=anchor.W,
                     mu=anchor.params[0], s=anchor.params[1], lam=anchor.params[2])
        self.record(dict(**location, action='comparison_rollback',
            reason='global_budget_exhausted' if isinstance(exc, TimeoutError) else 'comparison_unresolved',
            exception=str(exc), terminal_exception=type(exc).__name__,
            alpha=None, candidate_preserved=True, accepted=False, converged=False,
            seconds=time.perf_counter()-start))

    @metered('sweep_validation')
    def finish_sweep(self, G, W, K, anchor, limit, site, location):
        """Detached validation followed by a single atomic adoption."""
        ctx = self.context
        start = time.perf_counter()
        try:
            trial = ctx.evaluate(G, W, *anchor.params, commit=False)
        except TimeoutError as exc:
            if site == 'polish_column_increase':
                self.rollback_comparison(W, anchor, location, exc, start)
            raise
        except RECOVERABLE as exc:
            return self.restore(G, W, anchor, limit, site, location,
                                reason=str(exc), already_tried=True, start=start)
        drift = float(np.linalg.norm(K-trial.K, np.inf) /
                      max(1., np.linalg.norm(trial.K, np.inf)))
        try:
            monotone, delta = self.compare(G, anchor, trial, limit, site)
        except (ComparisonUnresolved, TimeoutError) as exc:
            if site == 'polish_column_increase':
                self.rollback_comparison(W, anchor, location, exc, start)
            raise
        except RECOVERABLE as exc:
            return self.restore(G, W, anchor, limit, site, location,
                                reason=str(exc), already_tried=True, start=start, drift=drift)
        if monotone:
            ctx.adopt(trial)
            if not np.isfinite(drift) or drift > 1e-6:
                self.record(dict(**location, action='inverse_rebase', reason='inverse_drift',
                    alpha=1., drift=drift, delta_F=delta, attempts=1,
                    F_before=anchor.metrics['F'], F_after=trial.metrics['F'],
                    inverse_error_bound=trial.metrics['inverse_error_bound'],
                    gradient_error_bound=trial.metrics['gradient_error_bound'],
                    seconds=time.perf_counter()-start))
            return trial, delta, drift, False
        return self.restore(G, W, anchor, limit, site, location,
                            reason='full_objective_not_monotone', already_tried=True,
                            start=start, drift=drift)

    @metered('recovery')
    def restore(self, G, W, anchor, limit, site, location, *, reason,
                already_tried=False, start=None, drift=None):
        ctx = self.context
        start = time.perf_counter() if start is None else start
        candidate = np.array(W, copy=True, order='K')
        trials = []
        try:
            if (not np.isfinite(candidate).all() or np.any(np.diag(candidate)) or
                    candidate.shape != anchor.W.shape):
                raise NumericalUnresolved('invalid_recovery_candidate')
            alphas = (() if already_tried else (1.,)) + tuple(ctx.recovery_alphas)
            for alpha in alphas:
                ctx.check_deadline()
                V = candidate.copy() if alpha == 1. else alpha*candidate
                try:
                    trial = ctx.evaluate(G, V, *anchor.params, commit=False)
                    monotone, delta = self.compare(G, anchor, trial, limit, site)
                    trials.append(dict(alpha=alpha, certified=True,
                                       monotone=bool(monotone), delta_F=delta))
                    if not monotone:
                        continue
                except ComparisonUnresolved as exc:
                    # An unknown verdict cannot authorize trying a smaller W.
                    self.rollback_comparison(V, anchor, location, exc, start)
                    raise
                except RECOVERABLE as exc:
                    trials.append(dict(alpha=alpha, certified=False, reason=str(exc)))
                    continue
                ctx.adopt(trial)
                self.record(dict(**location, action='inverse_rebase' if alpha == 1. else 'radial_restore',
                    reason=reason, alpha=alpha, drift=drift, delta_F=delta,
                    F_before=anchor.metrics['F'], F_after=trial.metrics['F'],
                    attempts=len(trials)+int(already_tried), trials=trials,
                    inverse_error_bound=trial.metrics['inverse_error_bound'],
                    gradient_error_bound=trial.metrics['gradient_error_bound'],
                    seconds=time.perf_counter()-start))
                return trial, delta, drift, alpha != 1.
            raise NumericalUnresolved('bounded_radial_recovery_exhausted: '+reason)
        except Exception as exc:
            # Timeout is propagated, never treated as a new numerical retry.
            ctx.adopt(anchor)
            self.record(dict(**location, action='rollback', reason=reason,
                terminal_exception=type(exc).__name__, exception=str(exc),
                trials=trials, attempts=len(trials)+int(already_tried),
                seconds=time.perf_counter()-start))
            raise

    def summary(self):
        return dict(counts=dict(self.counts), seconds=self.seconds, events=self.events,
                    alphas=list(self.context.recovery_alphas),
                    exact_scalar_minimization=False,
                    checkpoints_are_results=False)


def controller(context, comparator):
    if context.recovery is None:
        from .float64_acceptance import FloatRecovery
        context.recovery = FloatRecovery(context, comparator)
    return context.recovery

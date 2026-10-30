"""C2 transaction decisions around the unchanged LD/binary128 comparison.

No new objective-difference expression and no new acceptance tolerance.
Content-addressed, single-entry caches are bounded and detect in-place changes.
"""
from collections import Counter
from dataclasses import dataclass
from enum import Enum
import hashlib
import time
import numpy as np
from .guards import NumericalUnresolved


class Verdict(str, Enum):
    ACCEPT = 'accept_original_predicate'
    REJECT = 'confirmed_reject'
    UNRESOLVED = 'unresolved'


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    delta: object = None
    evidence: object = None


class ComparisonUnresolved(NumericalUnresolved):
    """Neither rejection nor convergence; the certified anchor must be retained."""


def fingerprint(*values):
    h = hashlib.sha256()
    for v in values:
        if isinstance(v, np.ndarray):
            a = np.ascontiguousarray(v)
            h.update(repr((a.shape, a.dtype.str)).encode())
            h.update(memoryview(a).cast('B'))
        else:
            h.update(repr(v).encode())
        h.update(b'\0')
    return h.hexdigest()


# Observed full d=1000 comparisons took 48--58 s with only two workers.
# Restore the existing cubic size allowance (187.5 s at d=1000) instead
# of clipping it at 60 s; one larger retry is allowed for a LOCAL timeout.
# Both attempts share the fit deadline. Arithmetic uncertainty is not
# treated as a reason to repeat the very same precision calculation.
POLICY = dict(size_coefficient_seconds=12., reference_dimension=400,
              minimum_seconds=2., first_cap_seconds=240.,
              retry_multiplier=2., retry_cap_seconds=480.,
              max_retries=1, retry_reason='comparison_budget_exhausted')


def allowance(d, attempt, hot_deadline, now=None):
    now = time.monotonic() if now is None else now
    remaining = hot_deadline-now
    if remaining <= 0:
        raise TimeoutError('Prespecified hot-fit cap exhausted')
    base = max(POLICY['minimum_seconds'], POLICY['size_coefficient_seconds']*(d/400.)**3)
    cap = POLICY['first_cap_seconds'] if attempt == 0 else POLICY['retry_cap_seconds']
    budget = min(remaining, cap, base*(1. if attempt == 0 else POLICY['retry_multiplier']))
    return min(hot_deadline, now+budget), budget


class C2Gate:
    def __init__(self, comparator):
        self.comparator = comparator
        self.active = None
        self.counts = Counter()
        self.events = []
        self.native_cache = None
        self.direct_cache = None

    def key(self, *values):
        self.counts['matrix_hashes'] += sum(isinstance(v, np.ndarray) for v in values)
        return fingerprint(*values)

    def domain_certified(self, old, new, mu, s, lam):
        return self.active is not None and self.active == self.key(old, new, mu, s, lam)

    def decide(self, G, anchor, trial, limit, site):
        """The caller supplies a checked checkpoint and a detached evaluation.

        They are not mutated while this synchronous transaction is active.
        Their independent domain/inverse gates have already passed. Check
        the content and parameters again at each use of that certification.
        """
        if self.active is not None:
            raise RuntimeError('nested C2 comparison transaction')
        if anchor.params != trial.params:
            raise ValueError('C2 state parameters differ')
        mu, s, lam = anchor.params
        delta = trial.metrics['F']-anchor.metrics['F']
        self.active = self.key(anchor.W, trial.W, mu, s, lam)
        cmp = self.comparator
        cmp.last = None
        try:
            ok, value = cmp.full(G, anchor.W, trial.W, mu, s, lam,
                delta, limit, 'le', delta <= limit, site,
                max(abs(anchor.metrics['F']), abs(trial.metrics['F'])), anchor.K)
            answer = Decision(Verdict.ACCEPT if ok else Verdict.REJECT, value, cmp.last)
        except NumericalUnresolved as exc:
            answer = Decision(Verdict.UNRESOLVED, evidence=dict(
                reason=str(exc), native=cmp.last, site=site))
        finally:
            self.active = None
        self.counts[answer.verdict.value] += 1
        return answer

    def direct(self, G, old, new, mu, s, lam, K_old, compute):
        # K participates because it is used in the inherited screen formula.
        key = self.key(G, old, new, mu, s, lam, K_old)
        if self.direct_cache is not None and self.direct_cache[0] == key:
            self.counts['direct_cache_hits'] += 1
            return self.direct_cache[1]
        self.counts['direct_cache_misses'] += 1
        value = compute()
        self.direct_cache = (key, value)
        return value

    def native(self, G, old, new, mu, s, lam, bound, kind):
        cmp = self.comparator
        # Include every mathematical input, the predicate and its bound.
        # Unknown answers are never cached, nor reused under a different gate.
        key = self.key(G, old, new, mu, s, lam, bound, kind)
        if self.native_cache is not None and self.native_cache[0] == key:
            self.counts['native_cache_hits'] += 1
            return dict(self.native_cache[1])
        self.counts['native_cache_misses'] += 1
        for attempt in range(POLICY['max_retries']+1):
            deadline, budget = allowance(len(G), attempt, cmp.hot_deadline)
            if attempt:
                self.counts['native_retries'] += 1
            started = time.perf_counter()
            event = dict(attempt=attempt, dimension=len(G), budget_seconds=budget,
                         comparison_key=key, candidate_key=self.key(new),
                         bound=float(bound), kind=kind)
            try:
                ans = cmp._native_once(G, old, new, mu, s, lam, bound, kind,
                                       event_window=(deadline, budget))
                event.update(ans)
            except TimeoutError:
                event.update(resolved=False, reason='global_budget_exhausted')
                self.counts['global_timeouts'] += 1
                raise
            finally:
                event['seconds'] = time.perf_counter()-started
                self.events.append(event)
                self.counts['high_precision_calls'] += 1
            if ans.get('resolved'):
                self.native_cache = (key, dict(ans))
                return ans
            if ans.get('reason') != POLICY['retry_reason']:
                return ans
            self.counts['local_comparison_timeouts'] += 1
        return ans

    def summary(self):
        return dict(policy=POLICY, counts=dict(self.counts), events=self.events,
                    high_precision_seconds=sum(e['seconds'] for e in self.events),
                    cache='single entry; exact content SHA256; all predicate inputs',
                    unchanged_native_formula=True, unchanged_acceptance_threshold=True)

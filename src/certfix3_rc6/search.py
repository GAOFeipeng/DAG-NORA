"""Single, bounded target-reset reconstruction pass and one terminal Column.

Candidates are evaluated in private snapshots. Only accepted states are adopted.
No candidate evaluation can mutate the incumbent's inverse or its version token.
"""
from __future__ import annotations
import time
import numpy as np
from .pair import PairProblem
from .compare import CandidateUncertain
from .column import polish_columns
from ._kernel.profile import profile_column_status
from .ranking import rank_pairs
def target_only_candidate(p, mode='fast', max_sweeps=100, tol=1e-6, inner_tol=1e-8):
    """Reset only the current target column and rebuild source before target."""
    if mode != 'fast':
        raise ValueError('latest candidate defines only production fast mode')
    # p.J contains the unordered pair. z indexes the current dominant source,
    # t indexes the current dominant target, under W[i,j] == i->j.
    z = 0 if abs(p.old[p.J[0], 1]) >= abs(p.old[p.J[1], 0]) else 1
    t = 1 - z
    V = p.old.copy()
    V[:, t] = 0.0
    order = (z, t)

    start = time.perf_counter()
    inner0, caps0, inv0 = p.inner_calls, p.inner_caps, p.inversions
    V, backtracks = p.valid_seed(V)
    seed_delta = p.delta(V)
    V, info = p.local_columns(V, order, max_sweeps, tol, inner_tol)
    delta = p.delta(V)
    rec = dict(
        seed='target_only_reset',
        delta_F=delta,
        seed_delta_F=seed_delta,
        seconds=time.perf_counter() - start,
        legal=True,
        backtracks=backtracks,
        accepted=False,
        inner_calls=p.inner_calls - inner0,
        inner_caps=p.inner_caps - caps0,
        inversions=p.inversions - inv0,
        source=int(p.J[z]),
        target=int(p.J[t]),
        order=[int(p.J[z]), int(p.J[t])],
        **info,
    )
    if delta < 0.0:
        return V.copy(), delta, [rec]
    return p.old.copy(), 0.0, [rec]


def _terminal_polish(G, W, mu, s, lam, column_config, context, cur, K, comparator, kernel, audit=False):
    """One final all-column coordination using the unchanged rc4-fix solver."""
    state = context.handoff(cur, K, mu, s, lam)
    return polish_columns(
        G, W, mu, s, lam,
        max_sweeps=column_config['max_sweeps'],
        inner_tol=column_config['inner_kkt_tol'],
        outer_tol=column_config['outer_prox_resid_tol'],
        relative_tol=column_config['relative_tol'],
        context=context,
        initial=state,
        audit=audit, comparator=comparator, kernel=kernel,
    )


def repair_endpoint_latest(G, W, mu, s, lam, column_config, rule, backend='woodbury', *, context, comparator, audit=False, initial=None, kernel=profile_column_status, ranker=rank_pairs):
    """Ranked finite local restarts, then exactly one terminal full-column polish."""
    start = time.perf_counter()
    W = W.copy()
    cur, K = context.reuse(W, mu, s, lam, initial) if initial is not None else context.evaluate(G, W, mu, s, lam)

    trace = [dict(phase='repair_start', seconds=time.perf_counter() - start, **cur)] if audit else []
    attempts, rounds, snapshots, accepted_pairs = [], [], [], []
    accepted = tested = all_inner_caps = all_local_caps = 0
    rank_time = candidate_time = numtime = polish_time = 0.0
    polish_sweeps = 0
    inversions = int(initial is None)
    positive_fallbacks = 0
    max_drift = 0.0
    max_value_error = 0.0
    max_witness = cur['witness_res']
    max_actual_inner_res = 0.0
    uncertain_total = 0
    last_reason = 'correction_budget_exhausted'

    # Local restart phase: no per-accept full-column polish.
    for turn in range(rule['max_corrections']):
        ranked, trank, _ = ranker(G, W, K, mu, s, lam, rule['edge_threshold'])
        rank_time += trank
        accepted_turn = False
        uncertain_turn = 0
        roundrec = dict(turn=turn, candidate_pool=len(ranked), ranking_seconds=trank, accepted=False)
        if audit:
            roundrec['ranks'] = ranked[:rule['max_pairs_per_trigger']]

        for rank, item in enumerate(ranked[:rule['max_pairs_per_trigger']]):
            t0 = time.perf_counter()
            J = [item['i'], item['j']]
            tested += 1
            before_unresolved = comparator.counts['native_unresolved']
            p = PairProblem(G, W, K, mu, s, lam, J, backend, comparator=comparator, kernel=kernel)
            try:
                V, delta, records = target_only_candidate(
                    p, 'fast', rule['max_local_sweeps'],
                    rule['local_prox_tol'], rule['local_inner_kkt_tol']
                )
            except CandidateUncertain as exc:
                candidate_time += time.perf_counter() - t0
                uncertain_turn += 1
                uncertain_total += 1
                all_inner_caps += p.inner_caps
                all_local_caps += p.local_caps
                if audit:
                    attempts.append(dict(turn=turn, rank=rank + 1, pair=J, accepted=False, reason=str(exc)))
                continue

            candidate_time += time.perf_counter() - t0
            all_inner_caps += p.inner_caps
            all_local_caps += p.local_caps
            max_actual_inner_res = max(max_actual_inner_res, p.max_inner_res)
            required = max(rule['min_absolute_decrease'], rule['min_relative_decrease'] * abs(cur['F']))
            accept = False
            uncertain_this = False
            check_error = None
            initial_check_error = None
            crosscheck_recheck = None
            delta_exact = None
            reason = 'not_strictly_improving'
            promising = False

            if not p.current_inner_caps:
                promising, _ = comparator.compare_pair(
                    p, p.old, V, delta, -required, 'lt', delta < -required,
                    'joint_strict_screen'
                )
            if promising and not p.current_inner_caps:
                tn = time.perf_counter()
                trial = W.copy()
                trial[:, p.J] = V
                try:
                    trial_state = context.evaluate(G, trial, mu, s, lam, commit=False)
                except FloatingPointError as exc:
                    uncertain_turn += 1
                    uncertain_total += 1
                    if audit:
                        attempts.append(dict(turn=turn,rank=rank+1,pair=J,accepted=False,
                                             reason='candidate_numerically_unresolved: '+str(exc)))
                    numtime += time.perf_counter()-tn
                    continue
                exact, Kexact = trial_state.metrics, trial_state.K
                inversions += 1
                final_descent, delta_exact = comparator.full(
                    G, W, trial, mu, s, lam,
                    exact['F'] - cur['F'], -required, 'lt',
                    exact['F'] - cur['F'] < -required,
                    'joint_strict_final_acceptance',
                    max(abs(cur['F']), abs(exact['F'])), K
                )
                check_error = abs(delta_exact - delta)
                initial_check_error = check_error
                max_value_error = max(max_value_error, check_error)
                max_witness = max(max_witness, exact['witness_res'])
                if not np.isfinite(check_error) or check_error > 1e-9:
                    crosscheck_recheck = comparator.recheck_joint_difference(
                        G,W,trial,mu,s,lam,-required,delta)
                    if crosscheck_recheck.get('resolved'):
                        delta_exact = crosscheck_recheck['delta_F']
                        check_error = crosscheck_recheck['crosscheck_error']
                        final_descent = bool(crosscheck_recheck['accepted'])
                    else:
                        final_descent = False
                        uncertain_this = True
                if not np.isfinite(check_error) or check_error > 1e-9:
                    reason = 'reduced_full_difference_disagreement'
                    uncertain_this = True
                elif final_descent:
                    accept = True
                    W = trial
                    K = Kexact
                    cur = exact
                    context.adopt(trial_state)
                    reason = 'accepted'
                    if audit:
                        trace.append(dict(phase='local_restart_accept', turn=turn, pair=J, seconds=time.perf_counter() - start, **cur))
                        snapshots.append(W.copy())
                numtime += time.perf_counter() - tn

            if comparator.counts['native_unresolved'] > before_unresolved:
                uncertain_this = True
                reason = 'bounded_comparison_unresolved'
            if uncertain_this:
                uncertain_turn += 1
                uncertain_total += 1

            if audit:
                selected = int(np.argmin([z['delta_F'] for z in records]))
                for k, rec in enumerate(records):
                    rec.update(
                        turn=turn,
                        rank=rank + 1,
                        i=J[0],
                        j=J[1],
                        accepted=bool(accept and k == selected),
                        selected_for_pair=(k == selected),
                        predicted_J_gain=item['predicted_J_gain'],
                        delta_F_exact_for_selected=(delta_exact if k == selected else None),
                        objective_crosscheck_error=(check_error if k == selected else None),
                        initial_objective_crosscheck_error=(initial_check_error if k == selected else None),
                        crosscheck_recheck=(crosscheck_recheck if k == selected else None),
                        reason=reason,
                    )
                    attempts.append(rec)

            inversions += p.inversions
            positive_fallbacks += p.positive_fallbacks

            if accept:
                accepted += 1
                accepted_pairs.append(J)
                accepted_turn = True
                roundrec.update(accepted=True, pair=J, delta_F=delta_exact)
                break

        roundrec['numerically_unresolved_candidates'] = uncertain_turn
        rounds.append(roundrec)
        if not accepted_turn:
            last_reason = 'candidate_budget_numerically_inconclusive' if uncertain_turn else 'no_improving_pair_in_budget'
            break

    # Exactly one terminal full-column coordination.  This provides a full-column
    # endpoint without paying P after every accepted local restart.
    # Retain this completed prefix before C2, so a saved B endpoint remains
    # independently assessable if the subsequent A refinement times out.
    context.float64_R_prefix=dict(stop_reason=last_reason,rounds=len(rounds),
        accepted_jumps=accepted,complete=last_reason!='candidate_budget_numerically_inconclusive',
        last_round=dict(rounds[-1]) if rounds else None)
    from .float64_acceptance import set_phase
    set_phase(context,'C2')
    pre_tail_W = W.copy()
    pre_tail_cur = cur
    pre_tail_K = K
    pre_tail_state = context.checkpoint()
    tp = time.perf_counter()
    tail_accepted = False
    tail_converged = True
    tail_delta = 0.0
    try:
        Vtail, pol = _terminal_polish(G, W, mu, s, lam, column_config, context, cur, K, comparator, kernel, audit=audit)
        polish_time += pol['seconds']
        polish_sweeps += pol['sweeps']
        all_inner_caps += pol['inner_caps']
        inversions += pol['inversions'] - 1
        max_drift = max(max_drift, pol['max_inverse_drift'])
        tail_cur, tail_K, _, _, _ = pol['_state']
        max_witness = max(max_witness, tail_cur['witness_res'])
        good, tail_delta = comparator.full(
            G, pre_tail_W, Vtail, mu, s, lam,
            tail_cur['F'] - pre_tail_cur['F'], 1e-11, 'le',
            tail_cur['F'] - pre_tail_cur['F'] <= 1e-11,
            'terminal_polish_monotonicity',
            max(abs(pre_tail_cur['F']), abs(tail_cur['F'])), pre_tail_K
        )
        if good:
            W = Vtail.copy()
            cur = tail_cur
            K = tail_K
            tail_accepted = True
        else:
            W = pre_tail_W
            cur = pre_tail_cur
            K = pre_tail_K
            context.adopt(pre_tail_state)
        tail_converged = bool(pol['converged'])
        if audit:
            for row in pol['trace'][1:]:
                trace.append(dict(phase='terminal_polish', **row))
        if not good:
            last_reason = 'terminal_polish_rejected'
        elif not tail_converged:
            last_reason = 'polish_cap'
    except Exception:
        # Preserve the latest fully accepted local-restart incumbent; propagate the
        # exception so the parent API records the incomplete numerical status.
        raise

    final = cur
    if audit:
        trace.append(dict(phase='repair_end', seconds=time.perf_counter() - start, **final))

    repair = dict(
        seconds=time.perf_counter() - start,
        trace=trace,
        attempts=attempts,
        rounds=rounds,
        accepted_jumps=accepted,
        accepted_pairs=accepted_pairs,
        tested_pairs=tested,
        inner_caps=all_inner_caps,
        local_caps=all_local_caps,
        ranking_seconds=rank_time,
        candidate_seconds=candidate_time,
        polish_seconds=polish_time,
        inversions=inversions,
        positive_fallbacks=positive_fallbacks,
        numerical_check_seconds=numtime,
        max_inverse_drift=max_drift,
        max_objective_crosscheck_error=max_value_error,
        max_witness_residual=max_witness,
        max_actual_inner_residual=max_actual_inner_res,
        polish_sweeps=polish_sweeps,
        stop_reason=last_reason,
        final=final,
        numerically_unresolved_candidates=uncertain_total,
        terminal_polish=dict(
            attempted=True,
            accepted=bool(tail_accepted),
            converged=bool(tail_converged),
            delta_F=float(tail_delta),
            seconds=float(polish_time),
            sweeps=int(polish_sweeps),
        ),
        algorithm_note='target-column reset + source-first two-column restart; Ranked-8/K8/C(max_corrections); one terminal polish',
    )
    return W, repair, snapshots

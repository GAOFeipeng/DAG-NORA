"""Reporting helpers: separate counts, one explicit conjunction, no solver effect.

Conventions fixed here (and used by the delivery report):

* ``final_returned``  -- the fit produced a weight matrix at all.
* ``native_completed`` -- status is ``completed`` or ``completed_with_warm_caps``.
* ``residual_passed`` -- the independent sample-space proximal residual is
  ``<= evaluation_residual_threshold``.
* ``domain_legal`` / ``inverse_reliable`` / ``graph_dag`` -- the other three
  conditions of the composite flag.
* ``numerically_qualified`` -- legacy records retain their original conjunction.
  Explicit objective-stop records require completion and state/graph safety;
  their independently computed residual is diagnostic, not a completion veto.

Structure metrics are offline only and must be reported separately:
``skeleton_f1`` / ``directed_f1`` are different quantities, and ``skeleton_shd``
counts an edge reversal once while ``dag_shd`` compares the directed edge sets.
Entries whose numeric flag is false are never dropped from a quality average
and then presented as the whole-set result.
"""
from __future__ import annotations


CONDITION_KEYS = (
    "final_returned",
    "native_completed",
    "residual_passed",
    "domain_legal",
    "inverse_reliable",
    "graph_dag",
)

COMPLETED_STATUS_VALUES = ("completed", "completed_with_inner_warnings", "completed_with_warm_caps")


def conditions(record: dict) -> dict:
    """Extract the five conditions from a result/record dict."""
    check = record.get("residual_check") or {}
    status = record.get("status")
    if record.get("native_status") is not None:
        status = record.get("native_status")
    returned = record.get("final_returned")
    if returned is None:
        returned = bool(record.get("w_available", False))
    return {
        "final_returned": bool(returned),
        "native_completed": bool(status in COMPLETED_STATUS_VALUES),
        "residual_passed": bool(check.get("passed", record.get("residual_passed", False))),
        "domain_legal": bool(check.get("legal", record.get("domain_legal", False))),
        "inverse_reliable": bool(check.get("reliable", record.get("inverse_reliable", False))),
        "graph_dag": bool(record.get("dag", record.get("graph_dag", False))),
    }


def composite(record: dict) -> bool:
    """The composite flag, recomputed as the conjunction (never copied blindly)."""
    flags = conditions(record)
    from .objective_stopping import POLICY_ID, output_qualified
    if record.get('stopping_policy') == POLICY_ID:
        return bool(flags['final_returned'] and output_qualified(
            flags['native_completed'],record.get('residual_check') or {},flags['graph_dag']))
    return all(flags[key] for key in CONDITION_KEYS)


def qualification_counts(records) -> dict:
    """Separate counts for each condition plus the conjunction."""
    rows = list(records)
    total = len(rows)
    counts = {key: 0 for key in CONDITION_KEYS}
    conj = 0
    reported = 0
    for row in rows:
        flags = conditions(row)
        for key in CONDITION_KEYS:
            counts[key] += int(flags[key])
        conj += int(composite(row))
        reported += int(bool(row.get("numerically_qualified", False)))
    return {
        "tasks": total,
        "final_returned": counts["final_returned"],
        "native_completed": counts["native_completed"],
        "residual_passed_only": counts["residual_passed"],
        "domain_legal": counts["domain_legal"],
        "inverse_reliable": counts["inverse_reliable"],
        "graph_dag": counts["graph_dag"],
        "numerically_qualified": conj,
        "numerically_qualified_as_reported": reported,
        "conjunction_disagreements": reported - conj,
        "returned_but_not_completed": counts["final_returned"] - counts["native_completed"],
        "residual_passed_but_not_qualified": sum(int(conditions(r)['residual_passed'] and not composite(r)) for r in rows),
        "completed_with_inner_warnings": int(
            sum(1 for r in rows if r.get("status") == "completed_with_inner_warnings")
        ),
        "had_inner_warning": int(sum(1 for r in rows if r.get("stage_inner_warning"))),
        "inner_warning_total": int(sum(int(r.get("inner_warning_total") or 0) for r in rows)),
    }


def structure_metric_note() -> dict:
    return {
        "skeleton_f1": "undirected skeleton precision/recall; a one-way prediction of a reversed edge counts as correct",
        "directed_f1": "directed edge-set F1 over the same thresholded graph",
        "skeleton_shd": "skeleton SHD; an edge reversal counts once",
        "dag_shd": "directed SHD; an edge reversal counts twice (one deletion + one insertion)",
        "usage": "offline evaluation only; truth never enters the fit, candidate selection, tolerance or fallback logic",
    }

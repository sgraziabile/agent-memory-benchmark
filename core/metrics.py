"""core.metrics — Pure, dependency-free metric helpers.

Shared by the runner (per-run aggregates exported to ``results.json`` /
``summary.csv`` / ``summary.md``) and by the reporting scripts under
``scripts/``, so one definition of the interval and of the retrieval-recall
denominator exists in the codebase.

Nothing here performs I/O or imports a framework: every function is
deterministic and unit-testable in isolation.

Retrieval recall
----------------
``retrieval_recall = passed positive retrieval pins / declared positive pins``

Three policy decisions are encoded below and must not drift from SPEC §5.8:

1. **Declared denominator (D1).** The denominator is counted from the
   scenario YAML, not from the evaluated assertions. Error turns never reach
   ``evaluate_assertions`` (``BenchmarkRunner.run_single``), so an evaluated
   denominator would silently shrink and a blank/truncated response would
   *improve* recall — the "pass_rate = 1.0 over 9 of 11 assertions" artefact
   documented in ``notes/2026-11-06_l2_authority_vs_recency.md`` (F7).
2. **Rule-level unit (D2).** One pin on one turn is one unit. Six scenarios
   deliberately pin the same gold document on several turns
   (``doc_addendum_1`` ×3 in ``document_addendum_revision_01``), so a
   unique-gold dedup would drop turns instead of measuring them. The metric is
   therefore "probe recall", not corpus-level IR recall.
3. **``must_retrieve_chunk`` only (D3).** ``must_not_retrieve_*`` are
   *precision* guards (counting them would let an amnesic agent score 100% by
   retrieving nothing), and ``must_retrieve_source`` pins a *channel*
   (``document`` / ``chat``) rather than a gold item, so it is reported
   separately and never enters this numerator/denominator.

Because ``must_retrieve_chunk`` matches a substring of ``chunk_id``, the
resulting figure is **document-level** recall: a sibling sub-chunk of the gold
document satisfies it. The stricter chunk-level ``Recall@k`` (the retrieved
chunk must actually *contain* the answer token) is computed by
``scripts/lr_report.py``; the two are not interchangeable.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

Z_95 = 1.959963984540054

#: Positive retrieval-assertion types entering the recall denominator.
#: Deliberately narrower than ``core.runner._RETRIEVAL_ASSERTION_TYPES`` —
#: see the module docstring (D3).
RETRIEVAL_RECALL_RULE_TYPES = frozenset({"must_retrieve_chunk"})


def wilson_ci(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (no scipy needed).

    Preferred over the normal approximation because benchmark cells are
    small (the D16 corpus-size probes run at ``n = 8``, where the Wald
    interval collapses to a point at 8/8).
    """
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def assertion_recall(
    details: Iterable[dict[str, Any]],
    rule_types: frozenset[str] = RETRIEVAL_RECALL_RULE_TYPES,
) -> tuple[int, int]:
    """Count ``(passed, total)`` positive retrieval pins in assertion details.

    Args:
        details: ``TurnResult.assertion_details`` dicts of one or more turns
            (each carries ``rule_type`` and ``passed``).
        rule_types: Rule types entering the count. Defaults to
            :data:`RETRIEVAL_RECALL_RULE_TYPES`.

    Returns:
        ``(passed, total)`` — both 0 when the details declare no pin.
    """
    passed = 0
    total = 0
    for detail in details:
        if detail.get("rule_type") in rule_types:
            total += 1
            if detail.get("passed"):
                passed += 1
    return passed, total


def declared_retrieval_assertions(
    turns: Iterable[Any],
    rule_types: frozenset[str] = RETRIEVAL_RECALL_RULE_TYPES,
) -> int:
    """Count the retrieval pins a scenario *declares* (the D1 denominator).

    Only ``role == "user"`` turns can assert anything: ``system_event`` turns
    are ingestion records and never reach ``evaluate_assertions``. Counting
    the declaration rather than the evaluation keeps a probe lost to an
    infrastructure error in the denominator, where it counts as a miss.

    Args:
        turns: ``ConversationTurn`` objects (or any objects exposing
            ``role`` and ``assertions``).
        rule_types: Rule types entering the count.

    Returns:
        Number of declared positive retrieval assertions.
    """
    total = 0
    for turn in turns:
        if getattr(turn, "role", "user") != "user":
            continue
        for rule in getattr(turn, "assertions", None) or []:
            if getattr(rule, "type", None) in rule_types:
                total += 1
    return total


def recall_with_ci(
    passed: int, total: int
) -> tuple[float | None, float | None, float | None]:
    """Return ``(recall, ci_low, ci_high)``; all ``None`` with no declaration.

    ``None`` (instead of ``0.0``) lets a text-only scenario be excluded from
    cross-scenario aggregates instead of dragging their mean to zero.
    """
    if total <= 0:
        return (None, None, None)
    low, high = wilson_ci(passed, total)
    return (passed / total, low, high)

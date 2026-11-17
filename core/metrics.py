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


# ── L2b agentic retrieval ───────────────────────────────────────────────────
#
# TODOs/l2b-telemetry-contract-todo.md §2.4. Three policy decisions:
#
# 1. **The match set is the union of every call in the turn.** L2b issues up to
#    `max_tool_calls` retrievals before answering; scoring only the last one
#    would let a wasted or mis-ordered call hide the evidence it did retrieve.
#    The union is de-duplicated by `chunk_id` keeping the FIRST occurrence, so
#    the same chunk arriving twice does not inflate anything.
# 2. **Calls are counted, never collapsed.** De-duplicating the *calls* would
#    hide wasted retrievals, which are exactly what an agentic loop can make
#    worse than a one-shot. The match set is deduped; the call count is not.
# 3. **Efficiency is `None` with no calls**, not `0.0`, so a no-tool run is
#    excluded from cross-arm means instead of scoring zero efficiency.


def union_retrieved_episodes(
    retrieved_memories: Iterable[dict[str, Any]] | None,
    retrieval_calls: Iterable[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build the retrieval-assertion match set from every call in a turn.

    Args:
        retrieved_memories: The turn's flat episode list, as L2a reports it.
        retrieval_calls: One dict per tool call (L2b). Each entry may carry an
            ``episodes`` list of the same episode dicts. When absent (or empty)
            the function falls back to ``retrieved_memories``, so L2a telemetry
            keeps working unchanged.

    Returns:
        Episodes from every call, de-duplicated by ``chunk_id`` (first wins).
        Entries without a ``chunk_id`` are kept as-is: dropping them would
        silently weaken ``must_retrieve_source``, which matches ``source``.
    """
    calls = [call for call in (retrieval_calls or []) if isinstance(call, dict)]
    if not calls:
        return [ep for ep in (retrieved_memories or []) if isinstance(ep, dict)]

    union: list[dict[str, Any]] = []
    seen: set[str] = set()
    for call in calls:
        episodes = call.get("episodes")
        if not isinstance(episodes, list):
            # Tolerant fallback: a call that only recorded chunk ids still
            # contributes to `must_retrieve_chunk` (not to `*_source`).
            episodes = [
                {"chunk_id": chunk_id}
                for chunk_id in (call.get("chunk_ids") or [])
                if isinstance(chunk_id, str)
            ]
        for episode in episodes:
            if not isinstance(episode, dict):
                continue
            chunk_id = episode.get("chunk_id")
            if isinstance(chunk_id, str) and chunk_id:
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
            union.append(episode)
    return union


def pins_per_call(passed: int, calls: int) -> float | None:
    """Retrieval-call efficiency: positive pins satisfied per call made.

    Args:
        passed: Declared positive pins satisfied this run.
        calls: Tool calls made this run.

    Returns:
        ``passed / calls``, or ``None`` when no call was made — a no-tool run
        has undefined efficiency, not zero.
    """
    if calls <= 0:
        return None
    return passed / calls


def mean_calls_per_probe(call_counts: Iterable[int]) -> float | None:
    """Average tool calls per probe.

    Args:
        call_counts: One entry per graded probe (a turn declaring at least one
            positive assertion), holding that turn's ``retrieval_call_count``.

    Returns:
        The arithmetic mean, or ``None`` when there are no probes.
    """
    counts = [int(count) for count in call_counts]
    if not counts:
        return None
    return sum(counts) / len(counts)


def gross_context_tokens(
    retrieval_calls: Iterable[dict[str, Any]] | None,
) -> int:
    """Token estimate summed over every retrieval call, duplicates included.

    ``retrieved_context_tokens`` measures the de-duplicated union actually
    injected into the prompt; this measures what the retriever *produced*.
    The difference is the price of re-querying, and it is the quantity an
    agentic loop can make worse than a one-shot call without any change in
    accuracy (``l2b-sequencing-plan`` §1.1.5).

    Per-call cost comes from the call's ``context_tokens`` field — the token
    estimate of that call's own block. Malformed entries contribute nothing
    instead of raising, so a partially-instrumented graph still reports a
    usable (if under-counted) figure.

    Args:
        retrieval_calls: One dict per tool call issued this turn.

    Returns:
        Sum of per-call ``context_tokens``; ``0`` when no call recorded one.
    """
    total = 0
    for call in retrieval_calls or []:
        if not isinstance(call, dict):
            continue
        try:
            total += int(call.get("context_tokens") or 0)
        except (TypeError, ValueError):
            continue
    return total


# ── Five-class failure taxonomy (TODOs/l2b-sequencing-plan.md §1.2) ─────────
#
# Without F0–F3 an L2b regression is indistinguishable from a retriever
# regression: "the agent answered wrong" has four different causes and each
# one calls for a different fix.
#
#   F0  no retrieval attempted            (pins fail, retrieval_call_count == 0)
#   F1  call made, query malformed        (query failed `query_must_contain`)
#   F2  query fine, gold absent from union (pins fail, calls non-empty)
#   F3  gold present, answer wrong        (pins pass, text rules fail)
#   F4  gold present, answer right        (both pass)

#: Class identifiers in severity order, so a distribution can be rendered
#: deterministically without sorting at every call site.
FAILURE_CLASSES: tuple[str, ...] = ("F0", "F1", "F2", "F3", "F4")


def classify_failure_class(
    *,
    pins_total: int,
    pins_passed: int,
    text_total: int,
    text_passed: int,
    retrieval_call_count: int,
    query_checked: bool = False,
    query_matched: bool = True,
) -> str | None:
    """Classify one probe turn into exactly one of ``F0``–``F4``.

    A turn that declares **no positive retrieval pin** is not a retrieval
    probe and gets ``None``: the taxonomy describes why retrieval (or the
    reading of what it returned) failed, and a text-only turn has nothing to
    say about that.

    Args:
        pins_total: Positive pins (``must_retrieve_chunk``) declared.
        pins_passed: Positive pins satisfied by the union of the turn's calls.
        text_total: Text rules (``must_contain`` / ``must_not_contain``) scored.
        text_passed: Text rules satisfied.
        retrieval_call_count: Tool calls the turn actually issued.
        query_checked: Whether the probe declared ``query_must_contain`` (so a
            malformed query is *detectable*; without it F1 cannot be claimed).
        query_matched: Whether at least one call's query contained every
            declared token.

    Returns:
        ``"F0"``–``"F4"``, or ``None`` when the turn is not a retrieval probe.

    Note:
        ``F1`` requires ``query_checked``: an unguarded probe whose gold is
        absent is ``F2`` by definition — claiming "malformed query" without a
        declaration would be an inference, not a measurement.

    Note:
        The gold is checked **before** the call count on purpose. ``F0`` means
        "nothing was retrieved, so the answer cannot be grounded", and call
        count is only the L2b evidence for it: L2a retrieves implicitly and
        always reports ``retrieval_call_count == 0``, so classifying a
        satisfied L2a pin as ``F0`` would mislabel its baseline row.
    """
    if pins_total <= 0:
        return None
    if pins_passed >= pins_total:
        return "F4" if text_passed >= text_total else "F3"
    if retrieval_call_count <= 0:
        return "F0"
    if query_checked and not query_matched:
        return "F1"
    return "F2"


def count_failure_classes(
    classes: Iterable[str | None],
) -> dict[str, int]:
    """Build the per-run distribution of the failure taxonomy.

    Args:
        classes: One entry per graded probe (``None`` for non-probes).

    Returns:
        ``{class: count}`` for the classes that actually occurred, in
        :data:`FAILURE_CLASSES` order. Empty when no probe was classified,
        which keeps every pre-L2b ``results.json`` unchanged.
    """
    observed = [cls for cls in classes if cls]
    counts = {cls: 0 for cls in FAILURE_CLASSES if cls in observed}
    for cls in observed:
        counts[cls] = counts.get(cls, 0) + 1
    return counts

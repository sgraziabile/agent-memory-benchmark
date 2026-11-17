#!/usr/bin/env python3
"""Extraction-difficulty report: FEB-30 tiers (D1/D2/D3) per scenario.

Two different things are called ``D1``/``D2``/``D3`` in this repository and this
script only speaks about **one** of them:

* **This script — FEB-30 extraction tiers.** ``D1`` explicit/literal, ``D2``
  contextual/embedded, ``D3`` inferential/hybrid injection. They describe how
  hard it is to *find and extract* a fact from the scenario's context.
* **Not this script — SPEC §5.8 retrieval-recall policy tags.** ``core.metrics``
  uses ``D1``/``D2``/``D3`` as docstring labels for the *denominator*, *unit*
  and *rule-type* decisions of the recall metric. See ``DIFFICULTY_AUDIT.md``
  §L2: renaming those to ``P1``/``P2``/``P3`` is still open.

Bands (declared in :data:`TIER_BANDS`, from the FEB-30 definition quoted in
``DIFFICULTY_AUDIT.md`` §A1)::

    D1  explicit / literal     1-2 docs     200-400 words   2-3 states   0 noise
    D2  contextual / embedded  3-4 docs     400-800 words   3-4 states   2-3 noise
    D3  inferential / injected 4-5 docs    600-1000 words   4-5 states   4-5 noise

The four axes are measured, never estimated:

``documents``
    ``len(scenario.documents)`` — every ``path`` entry is resolved with
    ``core.documents.load_document`` (the runner's own resolver), so an
    unresolvable fixture is reported instead of silently counted as zero.
``words``
    ``\\S+`` over every ``turns[].content`` **plus** the full text of every
    resolved document. This is the unit the FEB-30 bands are written in.
``states``
    Distinct ``session_id`` values across turns — the same quantity
    ``core.dataset_layout.thread_dir_for_sessions`` uses to classify a scenario
    as ``multi_thread``, so the two cannot disagree.
``noise``
    Documents that **no** ``must_retrieve_chunk`` pin names (irrelevant corpus)
    **plus** ``role: user`` turns that assert nothing (filler). Both are context
    the probe does not need.

    Caveat, load-bearing: a filler turn competes for *attention*, not for the
    answer, so it is **not** a distractor claim in the FEB-30 sense
    (``DIFFICULTY_AUDIT.md`` §A2). ``noise_documents`` is the honest half of this
    axis, and the verbose output prints both components so the split is never
    hidden.

Classification policy (:func:`classify`): a scenario sits *in band* ``Dk`` when
it satisfies **at least :data:`MIN_AXES_IN_BAND` of the 4 axes** of that tier.
Ties go to the lower tier. When no tier reaches the threshold the scenario is
out of band and ``violations`` names, axis by axis, where it lands::

    docs>D3      above the D3 ceiling      docs~D2   inside the D2 band
    docs<D1      below the D1 floor

Empirical columns reuse the existing metric definitions instead of inventing
new ones:

``pass``     ``sum(passed_assertions) / sum(total_assertions)`` pooled per
             ``scenario_id`` (never a mean of per-run ratios).
``recall``   ``scripts.retrieval_recall.run_stats`` — declared denominator,
             rule-level unit, ``must_retrieve_chunk`` only. A ``!`` marks runs
             whose scenario gained pins *after* the run (denominator drift), so
             the pooled figure is a lower bound over today's larger denominator.
``tokens``   ``sum(total_tokens)`` per run, as persisted by ``core.runner``.

Usage::

    python scripts/difficulty_report.py                       # every scenario
    python scripts/difficulty_report.py --group multi_thread  # the multihilo suite
    python scripts/difficulty_report.py --group multi_thread \
        --level level_2_episodic_rag
    python scripts/difficulty_report.py --with-legacy --sort tier
    python scripts/difficulty_report.py --csv outputs/difficulty_report.csv

A scenario with no ``results.json`` keeps its axis columns (those come from the
YAML alone) and reports ``-`` for every empirical column.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allow `python scripts/difficulty_report.py`
    sys.path.insert(0, str(REPO_ROOT))

from core.dataset_layout import (  # noqa: E402
    dataset_root,
    theme_category_from_parts,
    thread_dir_from_parts,
)
from core.documents import load_document  # noqa: E402
from core.metrics import recall_with_ci  # noqa: E402
from core.schemas import BenchmarkScenario  # noqa: E402
# Same denominator policy as the recall report: declared > recorded > evaluated,
# with drift flagged. Reused, not re-derived, so the two reports cannot diverge.
from scripts.retrieval_recall import run_stats  # noqa: E402

#: FEB-30 extraction tiers, low to high.
TIER_ORDER: tuple[str, ...] = ("D1", "D2", "D3")

#: The four measured axes, in reporting order.
TIER_AXES: tuple[str, ...] = ("documents", "words", "states", "noise")

#: ``tier -> axis -> (low, high)``, both inclusive.
TIER_BANDS: dict[str, dict[str, tuple[int, int]]] = {
    "D1": {
        "documents": (1, 2),
        "words": (200, 400),
        "states": (2, 3),
        "noise": (0, 0),
    },
    "D2": {
        "documents": (3, 4),
        "words": (400, 800),
        "states": (3, 4),
        "noise": (2, 3),
    },
    "D3": {
        "documents": (4, 5),
        "words": (600, 1000),
        "states": (4, 5),
        "noise": (4, 5),
    },
}

#: Axes a scenario must satisfy before it is declared to sit in a tier.
MIN_AXES_IN_BAND = 3

_WORD_RE = re.compile(r"\S+")


# ── Scenario measurement (YAML only, no results needed) ─────────────────────


def count_words(text: str | None) -> int:
    """``\\S+`` token count — the unit the FEB-30 bands are written in."""
    return len(_WORD_RE.findall(text or ""))


@dataclass(frozen=True)
class ScenarioAxes:
    """The four measured FEB-30 axes plus their components.

    Attributes:
        documents: Declared ``documents[]`` entries.
        document_words: Words contributed by the document corpus.
        turn_words: Words contributed by the chat turns.
        states: Distinct ``session_id`` values.
        noise_documents: Documents no ``must_retrieve_chunk`` pin names.
        filler_turns: ``role: user`` turns that assert nothing.
        unresolved_documents: ``path`` entries that could not be loaded.
    """

    documents: int
    document_words: int
    turn_words: int
    states: int
    noise_documents: int
    filler_turns: int
    unresolved_documents: int = 0

    @property
    def words(self) -> int:
        """Full context length in words (corpus + chat)."""
        return self.document_words + self.turn_words

    @property
    def noise(self) -> int:
        """Irrelevant corpus documents plus assertion-free user turns."""
        return self.noise_documents + self.filler_turns

    def axis(self, name: str) -> int:
        """Look up one axis by its reporting name."""
        return {
            "documents": self.documents,
            "words": self.words,
            "states": self.states,
            "noise": self.noise,
        }[name]


# ── Tier classification (pure, unit-tested) ─────────────────────────────────


@dataclass(frozen=True)
class TierVerdict:
    """Result of :func:`classify`.

    Attributes:
        tier: ``D1``/``D2``/``D3`` when in band, ``None`` when not.
        in_band: How many of the 4 axes the decisive tier satisfies.
        violations: One ``"<axis><marker>"`` string per failing axis, e.g.
            ``"documents>D3"`` (above the highest ceiling), ``"words~D2"``
            (lands inside D2 but not inside the decisive tier), ``"noise<D1"``
            (below the lowest floor).
    """

    tier: str | None
    in_band: int
    violations: tuple[str, ...]

    @property
    def label(self) -> str:
        """Short cell label: the tier, or ``"-"`` when out of band."""
        return self.tier or "-"

    @property
    def detail(self) -> str:
        """``"-"`` in band, otherwise the comma-joined violations."""
        return ",".join(self.violations) if self.violations else "-"


def _in_band(value: int, low: int, high: int) -> bool:
    return low <= value <= high


def axis_marker(axis: str, value: int) -> str:
    """Describe where one axis lands relative to the three bands.

    Returns ``"<axis>~Dk"`` when the value falls inside some band (a *straddle*,
    not a violation), otherwise ``"<axis>>D3"`` when it exceeds the highest
    ceiling or ``"<axis><D1"`` when it is under the lowest floor.
    """
    for tier in TIER_ORDER:
        low, high = TIER_BANDS[tier][axis]
        if _in_band(value, low, high):
            return f"{axis}~{tier}"
    if value > TIER_BANDS[TIER_ORDER[-1]][axis][1]:
        return f"{axis}>{TIER_ORDER[-1]}"
    return f"{axis}<{TIER_ORDER[0]}"


def classify(axes: ScenarioAxes) -> TierVerdict:
    """Assign a FEB-30 tier, or explain why none fits.

    The decisive tier is the one satisfying the most axes; ties go to the lower
    tier, so a scenario that is simultaneously D1-ish and D2-ish is reported as
    D1 and the harder tier is never claimed for free. A tier is only returned
    when it satisfies :data:`MIN_AXES_IN_BAND` axes.

    Args:
        axes: The measured axes.

    Returns:
        A :class:`TierVerdict`; ``tier`` is ``None`` when out of band.
    """
    values = {axis: axes.axis(axis) for axis in TIER_AXES}
    counts = {
        tier: sum(
            1
            for axis in TIER_AXES
            if _in_band(values[axis], *TIER_BANDS[tier][axis])
        )
        for tier in TIER_ORDER
    }
    best = max(
        TIER_ORDER, key=lambda tier: (counts[tier], -TIER_ORDER.index(tier))
    )
    violations = tuple(
        axis_marker(axis, values[axis])
        for axis in TIER_AXES
        if not _in_band(values[axis], *TIER_BANDS[best][axis])
    )
    return TierVerdict(
        tier=best if counts[best] >= MIN_AXES_IN_BAND else None,
        in_band=counts[best],
        violations=violations,
    )


def measure_scenario(path: Path) -> tuple[BenchmarkScenario, ScenarioAxes]:
    """Load one scenario YAML and measure its four axes.

    Args:
        path: Absolute path to a ``test_<scenario_id>.yaml`` file.

    Returns:
        ``(scenario, axes)`` with every document resolved and word-counted.

    Raises:
        ValueError: When the file is not a valid ``BenchmarkScenario``.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path}: expected a YAML mapping, got {type(raw).__name__}"
        )
    scenario = BenchmarkScenario.model_validate(raw)

    document_words = 0
    unresolved = 0
    for spec in scenario.documents:
        try:
            document_words += count_words(
                load_document(
                    spec,
                    scenario_id=scenario.id,
                    scenario_dir=path.parent,
                ).text
            )
        except (FileNotFoundError, ValueError, OSError):
            # A fixture that does not resolve must not be silently counted as
            # an empty document: that would deflate the `words` axis.
            unresolved += 1

    pins = [
        rule.value
        for turn in scenario.turns
        if turn.role == "user"
        for rule in turn.assertions
        if rule.type == "must_retrieve_chunk"
    ]
    noise_documents = sum(
        1
        for spec in scenario.documents
        if not any(spec.doc_id in pin for pin in pins)
    )

    axes = ScenarioAxes(
        documents=len(scenario.documents),
        document_words=document_words,
        turn_words=sum(count_words(turn.content) for turn in scenario.turns),
        states=len({turn.session_id or "default" for turn in scenario.turns}),
        noise_documents=noise_documents,
        filler_turns=sum(
            1
            for turn in scenario.turns
            if turn.role == "user" and not turn.assertions
        ),
        unresolved_documents=unresolved,
    )
    return scenario, axes


@dataclass(frozen=True)
class ScenarioRecord:
    """One scenario's structural identity: path, group, theme and axes."""

    scenario: BenchmarkScenario
    path: Path
    group: str
    theme: str
    category: str
    axes: ScenarioAxes
    verdict: TierVerdict

    @property
    def id(self) -> str:
        return self.scenario.id


def load_scenarios(root: Path, group: str = "all") -> dict[str, ScenarioRecord]:
    """Measure every scenario YAML under ``root``.

    Args:
        root: Scenario tree root (``datasets/conversations``).
        group: ``all``, ``single_thread`` or ``multi_thread``.

    Returns:
        ``scenario_id -> ScenarioRecord``, sorted by id.
    """
    records: dict[str, ScenarioRecord] = {}
    for path in sorted(root.rglob("*.yaml")):
        rel_parts = path.relative_to(root).parts
        thread_dir = thread_dir_from_parts(rel_parts) or "single_thread"
        if group != "all" and thread_dir != group:
            continue
        try:
            theme, category = theme_category_from_parts(rel_parts)
        except ValueError:
            theme, category = "", ""
        try:
            scenario, axes = measure_scenario(path)
        except (ValueError, yaml.YAMLError) as exc:
            print(f"warning: skipping {path.name}: {exc}", file=sys.stderr)
            continue
        records[scenario.id] = ScenarioRecord(
            scenario=scenario,
            path=path,
            group=thread_dir,
            theme=theme,
            category=category,
            axes=axes,
            verdict=classify(axes),
        )
    return dict(sorted(records.items()))


# ── Empirical aggregation (results.json) ────────────────────────────────────


@dataclass
class RunTotals:
    """Pooled telemetry for one ``scenario_id`` across every matching run."""

    runs: int = 0
    assertions_passed: int = 0
    assertions_total: int = 0
    pins_passed: int = 0
    pins_declared: int = 0
    pins_evaluated: int = 0
    drift_runs: int = 0
    tokens: int = 0
    models: set[str] = field(default_factory=set)
    levels: Counter = field(default_factory=Counter)

    def add(self, other: "RunTotals") -> None:
        """Accumulate another (single-run) totals object in place."""
        self.runs += other.runs
        self.assertions_passed += other.assertions_passed
        self.assertions_total += other.assertions_total
        self.pins_passed += other.pins_passed
        self.pins_declared += other.pins_declared
        self.pins_evaluated += other.pins_evaluated
        self.drift_runs += other.drift_runs
        self.tokens += other.tokens
        self.models |= other.models
        self.levels.update(other.levels)

    @property
    def pass_rate(self) -> float | None:
        """Pooled ``passed / total`` assertions, or ``None`` with no runs."""
        if not self.assertions_total:
            return None
        return self.assertions_passed / self.assertions_total

    @property
    def recall(self) -> float | None:
        """Pooled retrieval recall under the declared denominator."""
        if not self.pins_declared:
            return None
        return self.pins_passed / self.pins_declared


def declared_pin_counts(
    scenarios: dict[str, ScenarioRecord],
) -> dict[str, int]:
    """``scenario_id -> declared must_retrieve_chunk pins`` from the YAML.

    ``system_event`` turns are ingestion records and never assert, so they are
    skipped — identical to ``scripts.retrieval_recall.load_declared_counts``.
    """
    counts: dict[str, int] = {}
    for scenario_id, record in scenarios.items():
        counts[scenario_id] = sum(
            1
            for turn in record.scenario.turns
            if turn.role == "user"
            for rule in turn.assertions
            if rule.type == "must_retrieve_chunk"
        )
    return counts


def iter_runs(
    results_roots: Iterable[Path],
) -> Iterable[dict[str, Any]]:
    """Yield every run dict under ``results_roots``, de-duplicated by ``run_id``.

    ``outputs/by_theme/`` is a legacy layout that shares runs with
    ``outputs/runs/``; the same ``run_id`` must not be counted twice.
    """
    seen: set[str] = set()
    for root in results_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("results.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"warning: unreadable {path}: {exc}", file=sys.stderr)
                continue
            runs = payload if isinstance(payload, list) else [payload]
            for run in runs:
                if not isinstance(run, dict):
                    continue
                run_id = str(run.get("run_id") or "")
                if run_id and run_id in seen:
                    continue
                if run_id:
                    seen.add(run_id)
                yield run


def collect(
    results_roots: Iterable[Path],
    declared: dict[str, int],
    level: str | None = None,
) -> dict[str, RunTotals]:
    """Pool every matching run per ``scenario_id``.

    Args:
        results_roots: Directories scanned recursively for ``results.json``.
        declared: Declared pin counts from :func:`declared_pin_counts`.
        level: Only count runs of this ``agent_level`` (``None`` = all).

    Returns:
        ``scenario_id -> RunTotals`` (absent = no run matched).
    """
    totals: dict[str, RunTotals] = defaultdict(RunTotals)
    for run in iter_runs(results_roots):
        if level is not None and run.get("agent_level") != level:
            continue
        stats = run_stats(run, declared)
        scenario_id = stats["scenario_id"]
        if not scenario_id:
            continue
        total = totals[scenario_id]
        total.runs += 1
        total.assertions_passed += int(run.get("passed_assertions") or 0)
        total.assertions_total += int(run.get("total_assertions") or 0)
        total.pins_passed += int(stats["passed"])
        total.pins_declared += int(stats["declared"])
        total.pins_evaluated += int(stats["evaluated"])
        total.drift_runs += 1 if stats["denominator_drift"] else 0
        total.tokens += int(run.get("total_tokens") or 0)
        total.models.add(str(run.get("model_id") or "?"))
        total.levels[str(run.get("agent_level") or "?")] += 1
    return dict(totals)


# ── Presentation ────────────────────────────────────────────────────────────


def _pct(numerator: int, denominator: int) -> str:
    """``"37/40  92.5%"`` or ``"-"`` when the denominator is empty."""
    if not denominator:
        return "-"
    return f"{numerator}/{denominator} {numerator / denominator * 100:5.1f}%"


def _thousands(value: int) -> str:
    """Group digits with a thin space so wide tables stay readable."""
    return f"{value:,}".replace(",", " ")


@dataclass(frozen=True)
class Row:
    """One printable line: a scenario record joined with its run totals."""

    id: str
    group: str
    theme: str
    category: str
    axes: ScenarioAxes
    verdict: TierVerdict
    totals: RunTotals | None

    @property
    def tier_key(self) -> str:
        """Sort key that keeps D1 < D2 < D3 < off-band."""
        return self.verdict.tier or "Z"


def build_rows(
    scenarios: dict[str, ScenarioRecord],
    totals: dict[str, RunTotals],
) -> list[Row]:
    """Join scenario structure with its pooled telemetry."""
    return [
        Row(
            id=scenario_id,
            group=record.group,
            theme=record.theme,
            category=record.category,
            axes=record.axes,
            verdict=record.verdict,
            totals=totals.get(scenario_id),
        )
        for scenario_id, record in scenarios.items()
    ]


def sort_rows(rows: list[Row], key: str) -> list[Row]:
    """Order rows by ``key``: ``scenario``, ``tier``, ``words``, ``runs`` or ``tokens``."""
    if key == "tier":
        return sorted(rows, key=lambda r: (r.tier_key, r.id))
    if key == "words":
        return sorted(rows, key=lambda r: -r.axes.words)
    if key == "runs":
        return sorted(rows, key=lambda r: -((r.totals.runs) if r.totals else 0))
    if key == "tokens":
        return sorted(rows, key=lambda r: -((r.totals.tokens) if r.totals else 0))
    return sorted(rows, key=lambda r: r.id)


def print_table(rows: list[Row], verbose: bool = False) -> None:
    """Print the per-scenario table followed by the census and aggregates."""
    header = (
        f"{'scenario':44s} {'group':13s} {'tier':4s} {'in':>4s} "
        f"{'docs':>4s} {'words':>7s} {'states':>6s} {'noise':>5s} "
        f"{'runs':>4s} {'pass':>13s} {'recall':>13s} {'tokens':>10s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        totals = row.totals
        recall = _pct(totals.pins_passed, totals.pins_declared) if totals else "-"
        # `!` = the scenario declares more pins today than these runs evaluated
        # (and no turn errored), so the figure is a lower bound.
        if totals and totals.drift_runs:
            recall += "!"
        passed = _pct(totals.assertions_passed, totals.assertions_total) if totals else "-"
        print(
            f"{row.id[:44]:44s} {row.group[:13]:13s} {row.verdict.label:4s} "
            f"{row.verdict.in_band:>3d}/4 "
            f"{row.axes.documents:>4d} {row.axes.words:>7d} "
            f"{row.axes.states:>6d} {row.axes.noise:>5d} "
            f"{(totals.runs if totals else 0):>4d} "
            f"{passed:>13s} "
            f"{recall:>13s} "
            f"{(_thousands(totals.tokens) if totals else '-'):>10s}"
        )
        if verbose:
            print(
                f"    theme={row.theme}/{row.category} "
                f"doc_words={row.axes.document_words} "
                f"turn_words={row.axes.turn_words} "
                f"noise_docs={row.axes.noise_documents} "
                f"filler_turns={row.axes.filler_turns} "
                f"unresolved={row.axes.unresolved_documents} "
                f"violations={row.verdict.detail}"
            )
            if totals:
                print(
                    "    levels="
                    + ", ".join(f"{k}:{v}" for k, v in sorted(totals.levels.items()))
                    + f" models={len(totals.models)}"
                    + f" drift_runs={totals.drift_runs}"
                )
    print(
        "\ntier = FEB-30 extraction band met on >=3 of the 4 axes "
        "(`-` = out of band, see `in` and the violations in --verbose). "
        "noise = corpus documents no pin names + assertion-free user turns. "
        "pass = pooled passed/total assertions; recall = declared-denominator "
        "`must_retrieve_chunk` recall (`!` marks denominator drift, so the "
        "figure is a lower bound); tokens = sum(total_tokens) over the runs."
    )
    print_tier_census(rows)
    print_aggregates(rows)


def print_tier_census(rows: list[Row]) -> None:
    """Print how many scenarios sit in each tier versus out of band."""
    print("\ntier census (in-band = >=3/4 axes)")
    print("  " + "-" * 34)
    for tier in TIER_ORDER:
        members = [r for r in rows if r.verdict.tier == tier]
        print(f"  {tier}  {len(members):>3d} scenario(s)")
    off = [r for r in rows if r.verdict.tier is None]
    print(f"  -   {len(off):>3d} scenario(s) out of band")
    if off and len(off) <= 8:
        for row in off:
            print(f"        {row.id[:44]:44s} {row.verdict.detail}")


def aggregate(rows: Iterable[Row], key: str) -> dict[str, RunTotals]:
    """Pool run totals per group key (``tier`` or ``group``)."""
    pooled: dict[str, RunTotals] = defaultdict(RunTotals)
    for row in rows:
        if row.totals is None:
            continue
        bucket = row.verdict.tier if key == "tier" else row.group
        pooled[bucket or "off-band"].add(row.totals)
    return dict(pooled)


def print_aggregates(rows: list[Row]) -> None:
    """Print pooled pass/recall/tokens per tier and per thread group."""
    for key, title in (("tier", "by tier"), ("group", "by thread group")):
        pooled = aggregate(rows, key)
        if not pooled:
            continue
        counts: Counter = Counter(
            (row.verdict.tier or "off-band") if key == "tier" else row.group
            for row in rows
        )
        header = (
            f"  {'bucket':10s} {'scenario':>8s} {'runs':>5s} "
            f"{'pass':>13s} {'recall':>13s} {'tokens':>11s}"
        )
        print(f"\n{title}")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for bucket in sorted(pooled):
            totals = pooled[bucket]
            recall = _pct(totals.pins_passed, totals.pins_declared)
            if totals.drift_runs:
                recall += "!"
            print(
                f"  {bucket[:10]:10s} {counts[bucket]:>8d} {totals.runs:>5d} "
                f"{_pct(totals.assertions_passed, totals.assertions_total):>13s} "
                f"{recall:>13s} {_thousands(totals.tokens):>11s}"
            )


def write_csv(path: Path, rows: list[Row]) -> None:
    """Write the same table as the printed one, plus the Wilson interval."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario_id",
        "thread_group",
        "theme",
        "category",
        "tier",
        "axes_in_band",
        "violations",
        "documents",
        "context_words",
        "document_words",
        "turn_words",
        "states",
        "noise",
        "noise_documents",
        "filler_turns",
        "unresolved_documents",
        "runs",
        "models",
        "assertions_passed",
        "assertions_total",
        "pass_rate",
        "pins_passed",
        "pins_declared",
        "pins_evaluated",
        "retrieval_recall",
        "ci95_low",
        "ci95_high",
        "denominator_drift_runs",
        "total_tokens",
        "tokens_per_run",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            totals = row.totals or RunTotals()
            recall, low, high = recall_with_ci(
                totals.pins_passed, totals.pins_declared
            )
            writer.writerow(
                {
                    "scenario_id": row.id,
                    "thread_group": row.group,
                    "theme": row.theme,
                    "category": row.category,
                    "tier": row.verdict.label,
                    "axes_in_band": f"{row.verdict.in_band}/4",
                    "violations": row.verdict.detail,
                    "documents": row.axes.documents,
                    "context_words": row.axes.words,
                    "document_words": row.axes.document_words,
                    "turn_words": row.axes.turn_words,
                    "states": row.axes.states,
                    "noise": row.axes.noise,
                    "noise_documents": row.axes.noise_documents,
                    "filler_turns": row.axes.filler_turns,
                    "unresolved_documents": row.axes.unresolved_documents,
                    "runs": totals.runs,
                    "models": len(totals.models),
                    "assertions_passed": totals.assertions_passed,
                    "assertions_total": totals.assertions_total,
                    "pass_rate": (
                        "" if totals.pass_rate is None
                        else f"{totals.pass_rate:.4f}"
                    ),
                    "pins_passed": totals.pins_passed,
                    "pins_declared": totals.pins_declared,
                    "pins_evaluated": totals.pins_evaluated,
                    "retrieval_recall": "" if recall is None else f"{recall:.4f}",
                    "ci95_low": "" if low is None else f"{low:.4f}",
                    "ci95_high": "" if high is None else f"{high:.4f}",
                    "denominator_drift_runs": totals.drift_runs,
                    "total_tokens": totals.tokens,
                    "tokens_per_run": (
                        round(totals.tokens / totals.runs) if totals.runs else 0
                    ),
                }
            )
    print(f"\nCSV written to {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify every scenario into the FEB-30 extraction tiers (D1/D2/D3) "
            "from four measured axes, and join the result with pooled pass rate, "
            "retrieval recall and token totals."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=dataset_root(),
        help="Scenario YAML root",
    )
    parser.add_argument(
        "--results",
        type=Path,
        nargs="+",
        default=[REPO_ROOT / "outputs" / "runs"],
        help="Directory (or directories) scanned recursively for results.json",
    )
    parser.add_argument(
        "--with-legacy",
        action="store_true",
        help=(
            "Also scan outputs/by_theme (legacy layout). Runs are de-duplicated "
            "by run_id, so only genuinely missing runs are added."
        ),
    )
    parser.add_argument(
        "--group",
        choices=("all", "single_thread", "multi_thread"),
        default="all",
        help="Restrict to one thread class",
    )
    parser.add_argument(
        "--level",
        default=None,
        help="Only count runs of this agent_level (e.g. level_2_episodic_rag)",
    )
    parser.add_argument(
        "--sort",
        choices=("scenario", "tier", "words", "runs", "tokens"),
        default="scenario",
        help="Row order",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the axis components and the per-level run split",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.dataset.exists():
        print(f"Error: dataset root {args.dataset} does not exist.")
        return 2

    scenarios = load_scenarios(args.dataset, args.group)
    if not scenarios:
        print(f"No scenarios found under {args.dataset} (group={args.group}).")
        return 1

    results_roots = list(args.results)
    if args.with_legacy:
        results_roots.append(REPO_ROOT / "outputs" / "by_theme")

    totals = collect(results_roots, declared_pin_counts(scenarios), args.level)
    rows = sort_rows(build_rows(scenarios, totals), args.sort)

    print(
        f"{len(scenarios)} scenario(s) from {args.dataset} | group={args.group} | "
        f"level={args.level or 'all'} | results="
        + ", ".join(str(root) for root in results_roots)
    )
    print_table(rows, verbose=args.verbose)
    if args.csv:
        write_csv(args.csv, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())

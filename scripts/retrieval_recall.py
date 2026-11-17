#!/usr/bin/env python3
"""Assertion-level retrieval recall report: passed / declared pins.

The metric is ``sum(must_retrieve_chunk)``: ``recall = passed / declared``.

This script is the *post-hoc* half of the recall metric (SPEC §5.8): it reads
``results.json`` (any run, including archived ones that predate the
``retrieval_recall`` summary field) and recomputes the ratio from
``turn_results[*].assertion_details``. The runner exports the same figure per
run as ``retrieval_recall`` under the identical policy:

* **Denominator = declared** pins, counted from the scenario YAML when the run
  does not carry ``retrieval_assertions_total``. A probe lost to an
  infrastructure error stays in the denominator and counts as a miss
  (``notes/2026-11-06_l2_authority_vs_recency.md`` F7).
* **Unit = one pin on one turn** (rule-level). Six scenarios pin the same
  gold document on several turns on purpose.
* **Only ``must_retrieve_chunk``** counts. ``must_not_retrieve_*`` are
  precision guards; ``must_retrieve_source`` pins a channel, not a gold item.

The ratio is **document-level**: ``must_retrieve_chunk`` matches a substring
of ``chunk_id``, so a sibling sub-chunk of the gold document satisfies it.
Chunk-level ``Recall@k`` (the retrieved chunk must contain the answer token)
is a different metric: ``scripts/lr_report.py``.

Usage::

    python scripts/retrieval_recall.py                     # every run under outputs/runs
    python scripts/retrieval_recall.py --scenario-prefix document_ --verbose
    python scripts/retrieval_recall.py --csv outputs/retrieval_recall.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allow `python scripts/retrieval_recall.py`
    sys.path.insert(0, str(REPO_ROOT))

from core.metrics import (  # noqa: E402
    RETRIEVAL_RECALL_RULE_TYPES,
    recall_with_ci,
)

DEFAULT_RESULTS = REPO_ROOT / "outputs" / "runs"
DEFAULT_DATASET = REPO_ROOT / "datasets" / "conversations"


# ── Loaders ─────────────────────────────────────────────────────────────────


def load_declared_counts(dataset_root: Path) -> dict[str, int]:
    """Map ``scenario_id -> declared must_retrieve_chunk pins`` from the YAMLs.

    Only ``role == "user"`` turns can assert anything: ``system_event`` turns
    are ingestion records and never reach ``evaluate_assertions``.
    """
    counts: dict[str, int] = {}
    for path in sorted(dataset_root.rglob("*.yaml")):
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(raw, dict) or "id" not in raw:
            continue
        total = 0
        for turn in raw.get("turns") or []:
            if not isinstance(turn, dict):
                continue
            if (turn.get("role") or "user") != "user":
                continue
            for rule in turn.get("assertions") or []:
                if not isinstance(rule, dict):
                    continue
                if (rule.get("type") or "") in RETRIEVAL_RECALL_RULE_TYPES:
                    total += 1
        counts[str(raw["id"])] = total
    return counts


def iter_results_files(root: Path) -> Iterable[Path]:
    """Every ``results.json`` under ``root``, oldest directory first."""
    return sorted(root.rglob("results.json"))


# ── Per-run extraction (pure) ───────────────────────────────────────────────


def run_stats(
    run: dict[str, Any], declared_counts: dict[str, int] | None = None
) -> dict[str, Any]:
    """Extract one run's retrieval counts, misses and denominator source."""
    declared_counts = declared_counts or {}
    scenario_id = str(run.get("scenario_id") or "")

    passed = 0
    evaluated = 0
    misses: list[tuple[Any, Any]] = []
    for turn in run.get("turn_results") or []:
        if not isinstance(turn, dict):
            continue
        # system_event turns assert nothing; error turns were never scored.
        if (turn.get("role") or "user") != "user":
            continue
        if turn.get("status") != "ok":
            continue
        for detail in turn.get("assertion_details") or []:
            if not isinstance(detail, dict):
                continue
            if detail.get("rule_type") not in RETRIEVAL_RECALL_RULE_TYPES:
                continue
            evaluated += 1
            if detail.get("passed"):
                passed += 1
            else:
                misses.append((turn.get("turn_number"), detail.get("value")))

    recorded = run.get("retrieval_assertions_total")
    if isinstance(recorded, int) and recorded > 0:
        declared, source = recorded, "run"
    elif declared_counts.get(scenario_id):
        declared, source = declared_counts[scenario_id], "yaml"
    else:
        declared, source = evaluated, "evaluated"
    # A legacy run can carry more evaluated pins than the YAML lookup found
    # (the scenario changed since the run); never report above 100 %.
    if evaluated > declared:
        declared = evaluated

    error_turns = int(run.get("error_turns") or 0)
    # Denominator drift: the scenario declares MORE pins today than this run
    # evaluated, and no turn errored — so the scenario gained a pin after the
    # run (e.g. document_dilution_01 gained ``doc_power_audit_scope`` on
    # 2026-11-02). The as-run denominator is unrecoverable from the record;
    # scoring against today's declaration makes the figure a lower bound.
    drift = error_turns == 0 and declared > evaluated

    return {
        "scenario_id": scenario_id,
        "model_id": str(run.get("model_id") or ""),
        "agent_level": str(run.get("agent_level") or ""),
        "memory_backend": str(run.get("memory_backend") or "none"),
        "top_k": run.get("top_k"),
        "passed": passed,
        "declared": declared,
        "evaluated": evaluated,
        "denominator_source": source,
        "denominator_drift": drift,
        "misses": misses,
        "error_turns": error_turns,
    }


def collect(
    results_root: Path,
    declared_counts: dict[str, int] | None = None,
    scenario_prefix: str | None = None,
) -> dict[tuple, dict[str, Any]]:
    """Bucket every matching run by experimental identity."""
    buckets: dict[tuple, dict[str, Any]] = {}
    for path in iter_results_files(results_root):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        runs = payload if isinstance(payload, list) else [payload]
        for run in runs:
            if not isinstance(run, dict):
                continue
            stats = run_stats(run, declared_counts)
            if scenario_prefix and not stats["scenario_id"].startswith(
                scenario_prefix
            ):
                continue
            if stats["declared"] <= 0:
                continue  # no retrieval pins -> recall is undefined, not zero
            key = (
                stats["scenario_id"],
                stats["agent_level"],
                stats["memory_backend"],
                stats["top_k"],
                stats["model_id"],
            )
            bucket = buckets.setdefault(
                key,
                {
                    "runs": 0,
                    "passed": 0,
                    "declared": 0,
                    "evaluated": 0,
                    "errors": 0,
                    "drift": 0,
                    "sources": set(),
                    "misses": [],
                },
            )
            bucket["runs"] += 1
            bucket["passed"] += stats["passed"]
            bucket["declared"] += stats["declared"]
            bucket["evaluated"] += stats["evaluated"]
            bucket["errors"] += stats["error_turns"]
            bucket["drift"] += 1 if stats["denominator_drift"] else 0
            bucket["sources"].add(stats["denominator_source"])
            for turn_number, value in stats["misses"]:
                bucket["misses"].append((path.parent.name, turn_number, value))
    return buckets


def summarize(bucket: dict[str, Any]) -> dict[str, Any]:
    """Pool a bucket's runs into one recall figure + Wilson interval."""
    recall, ci_low, ci_high = recall_with_ci(bucket["passed"], bucket["declared"])
    sources = sorted(bucket["sources"])
    return {
        "recall": recall,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "denominator_source": sources[0] if len(sources) == 1 else "mixed",
        "denominator_drift": bucket["drift"],
        "misses": bucket["misses"],
    }


# ── Presentation ────────────────────────────────────────────────────────────


def _fmt_ci(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return "-"
    return f"[{low * 100:.0f}, {high * 100:.0f}]"


def print_table(
    rows: list[tuple[tuple, dict[str, Any]]], verbose: bool = False
) -> None:
    header = (
        f"{'scenario':38s} {'level':22s} {'backend':7s} {'k':>3s} "
        f"{'runs':>4s} {'hit':>4s} {'pins':>4s} {'recall':>7s} "
        f"{'ci95':>11s} {'eval':>4s} {'err':>3s} {'denom':>9s}"
    )
    print(header)
    print("-" * len(header))
    for (scenario, level, backend, top_k, model), bucket in sorted(rows):
        s = summarize(bucket)
        recall = "-" if s["recall"] is None else f"{s['recall'] * 100:.1f}%"
        # `!` = the scenario declares more pins today than this run evaluated
        # (and no turn errored), so the figure is a lower bound over today's
        # larger denominator: the as-run count is not recoverable.
        if s["denominator_drift"]:
            recall += "!"
        print(
            f"{scenario[:38]:38s} {level[:22]:22s} {backend[:7]:7s} "
            f"{str(top_k if top_k is not None else '-'):>3s} "
            f"{bucket['runs']:>4d} {bucket['passed']:>4d} {bucket['declared']:>4d} "
            f"{recall:>7s} {_fmt_ci(s['ci_low'], s['ci_high']):>11s} "
            f"{bucket['evaluated']:>4d} {bucket['errors']:>3d} "
            f"{s['denominator_source']:>9s}"
        )
        if verbose:
            print(f"    model={model}")
            if not s["misses"]:
                print("    (no missed pins)")
            for run_dir, turn_number, value in s["misses"]:
                print(f"    MISS {run_dir} T{turn_number}: {value}")
    print(
        "\nrecall = passed / declared `must_retrieve_chunk` pins "
        "(rule-level, document-level). `eval` < `pins` means at least one "
        "declared probe was never evaluated (infrastructure error) and "
        "counted as a miss. `denom` says where the denominator came from "
        "(run field / scenario YAML / evaluated details); `!` after the recall "
        "marks runs whose scenario gained pins after the run, making that "
        "figure a lower bound."
    )
    print(
        "Chunk-level Recall@k (the gold chunk must contain the answer token) "
        "is a different metric: scripts/lr_report.py."
    )


def write_csv(path: Path, rows: list[tuple[tuple, dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario_id",
        "agent_level",
        "model_id",
        "memory_backend",
        "top_k",
        "runs",
        "retrieval_assertions_passed",
        "retrieval_assertions_total",
        "retrieval_assertions_evaluated",
        "retrieval_recall",
        "ci95_low",
        "ci95_high",
        "error_turns",
        "denominator_source",
        "denominator_drift_runs",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for (scenario, level, backend, top_k, model), bucket in sorted(rows):
            s = summarize(bucket)
            writer.writerow(
                {
                    "scenario_id": scenario,
                    "agent_level": level,
                    "model_id": model,
                    "memory_backend": backend,
                    "top_k": "" if top_k is None else top_k,
                    "runs": bucket["runs"],
                    "retrieval_assertions_passed": bucket["passed"],
                    "retrieval_assertions_total": bucket["declared"],
                    "retrieval_assertions_evaluated": bucket["evaluated"],
                    "retrieval_recall": (
                        "" if s["recall"] is None else f"{s['recall']:.4f}"
                    ),
                    "ci95_low": "" if s["ci_low"] is None else f"{s['ci_low']:.4f}",
                    "ci95_high": "" if s["ci_high"] is None else f"{s['ci_high']:.4f}",
                    "error_turns": bucket["errors"],
                    "denominator_source": s["denominator_source"],
                    "denominator_drift_runs": s["denominator_drift"],
                }
            )
    print(f"\nCSV written to {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieval recall report: passed / declared must_retrieve_chunk pins."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Root directory scanned recursively for results.json",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Scenario YAML root, used to recover the declared denominator for "
        "runs whose results.json predates the retrieval_recall fields",
    )
    parser.add_argument(
        "--scenario-prefix",
        type=str,
        default=None,
        help="Only report scenarios whose id starts with this prefix",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every missed pin (run directory, turn, expected chunk id)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.results.exists():
        print(f"Error: results root {args.results} does not exist.")
        return 2

    declared_counts = (
        load_declared_counts(args.dataset) if args.dataset.exists() else {}
    )
    buckets = collect(args.results, declared_counts, args.scenario_prefix)
    if not buckets:
        print(
            "No runs with positive `must_retrieve_chunk` pins found under "
            f"{args.results}"
            + (f" (prefix {args.scenario_prefix!r})" if args.scenario_prefix else "")
            + "."
        )
        return 1

    rows = list(buckets.items())
    print_table(rows, verbose=args.verbose)
    if args.csv:
        write_csv(args.csv, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())



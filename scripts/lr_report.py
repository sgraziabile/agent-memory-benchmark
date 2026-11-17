#!/usr/bin/env python3
"""L-R retrieval report: Recall@k, MRR, Δcos margin and coverage.

The D16 deliverable is a *curve*, not a pass rate: ``Recall@k`` per corpus
size with a Wilson interval. This script recomputes retrieval metrics from the
``retrieved_memories`` telemetry that ``core.runner`` already persists, so no
run-time code has to change and the metric is chunk-level.

Two levels of "hit" are reported on purpose:

* ``doc_hit`` — a chunk of the gold *document* reached the top-k. That is what
  the scenario's ``must_retrieve_chunk: corpus_0NN`` assertion measures (it is
  a substring match on ``chunk_id``, so sibling sub-chunks satisfy it).
* ``chunk_hit`` — the chunk that actually **contains the answer token**
  reached the top-k. This is the honest Recall@k for §5.3: a sibling chunk of
  the right document does not let the model answer.

Usage::

    python scripts/lr_report.py                       # every result under outputs/runs
    python scripts/lr_report.py --scenario-prefix document_corpus_size --verbose
    python scripts/lr_report.py --csv outputs/lr_recall.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allow `python scripts/lr_report.py`
    sys.path.insert(0, str(REPO_ROOT))

# Interval lives in ``core.metrics`` so the runner's per-run retrieval recall
# and this script's Recall@k share one definition (and one test).
from core.metrics import Z_95, wilson_ci  # noqa: E402,F401  (re-exported)


# ── Pure metric helpers (unit-tested) ───────────────────────────────────────


def _rank_of(episodes: Iterable[dict[str, Any]], predicate) -> int | None:
    """Best (lowest) rank satisfying ``predicate``, or ``None``."""
    ranks = [
        int(ep.get("rank") or index)
        for index, ep in enumerate(episodes, start=1)
        if predicate(ep)
    ]
    return min(ranks) if ranks else None


def gold_chunk_rank(episodes: list[dict[str, Any]], answer: str) -> int | None:
    """Rank of the chunk whose text contains the corpus-exclusive answer."""
    if not answer:
        return None
    needle = answer.casefold()
    return _rank_of(
        episodes, lambda ep: needle in str(ep.get("content", "")).casefold()
    )


def gold_doc_rank(episodes: list[dict[str, Any]], doc_prefix: str) -> int | None:
    """Rank of the first chunk belonging to the gold document."""
    if not doc_prefix:
        return None
    return _rank_of(
        episodes, lambda ep: doc_prefix.casefold() in str(ep.get("chunk_id", "")).casefold()
    )


def delta_cos(episodes: list[dict[str, Any]], answer: str) -> float | None:
    """``cos(query, gold) − max(cos(query, best non-gold))``.

    Reported as a sign + magnitude, never as a rank: the stress plan measured
    margins of 0.001–0.002 between near-duplicate chunks (finding 1.14), so a
    "win" is often a coin flip and the margin is the honest evidence.
    """
    if not episodes or not answer:
        return None
    needle = answer.casefold()
    gold_scores = [
        float(ep.get("score") or 0.0)
        for ep in episodes
        if needle in str(ep.get("content", "")).casefold()
    ]
    other_scores = [
        float(ep.get("score") or 0.0)
        for ep in episodes
        if needle not in str(ep.get("content", "")).casefold()
    ]
    if not gold_scores:
        return None
    return max(gold_scores) - (max(other_scores) if other_scores else 0.0)


# ── Scenario gold map + run scanning ────────────────────────────────────────


def load_gold_map(scenario_prefix: str | None = None) -> dict[str, dict[int, dict[str, str]]]:
    """Map ``scenario_id → turn_number → {answer, doc}`` from scenario YAMLs.

    A probe is a turn that pins both a corpus-exclusive ``must_contain`` token
    and a ``must_retrieve_chunk`` document prefix. Scenarios without that pair
    are ignored, so ordinary L0/L1/L2 suites never pollute the report.
    """
    from core.runner import BenchmarkRunner

    gold: dict[str, dict[int, dict[str, str]]] = {}
    for scenario in BenchmarkRunner().load_scenarios():
        if scenario_prefix and not scenario.id.startswith(scenario_prefix):
            continue
        per_turn: dict[int, dict[str, str]] = {}
        for turn in scenario.turns:
            answer = next(
                (a.value for a in turn.assertions if a.type == "must_contain"), None
            )
            doc = next(
                (a.value for a in turn.assertions if a.type == "must_retrieve_chunk"),
                None,
            )
            if answer and doc:
                per_turn[turn.turn_number] = {"answer": answer, "doc": doc}
        if per_turn:
            gold[scenario.id] = per_turn
    return gold


def iter_runs(results_root: Path) -> Iterable[dict[str, Any]]:
    """Yield every run dict from every ``results.json`` under ``results_root``."""
    pattern = os.path.join(str(results_root), "**", "results.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover
            print(f"warning: skipping {path}: {exc}", file=sys.stderr)
            continue
        for run in data if isinstance(data, list) else [data]:
            if isinstance(run, dict):
                run.setdefault("_source", path)
                yield run


def _new_bucket() -> dict[str, Any]:
    return {
        "probes": 0,
        "doc_hits": 0,
        "chunk_hits": 0,
        "mrr_sum": 0.0,
        "margins": [],
        "hit_and_correct": 0,
        "correct": 0,
        "errors": 0,
        "corpus_chunks": None,
        "documents_ingested": None,
        "top_k": None,
        "runs": 0,
        "ingestion_ms": 0.0,
    }


def collect(
    gold: dict[str, dict[int, dict[str, str]]],
    results_root: Path,
    *,
    verbose: bool = False,
) -> dict[tuple[str, int | None, int | None, str], dict[str, Any]]:
    """Aggregate per-probe metrics into ``(scenario, chunk_size, top_k, model)``."""
    buckets: dict[tuple[str, int | None, int | None, str], dict[str, Any]] = defaultdict(
        _new_bucket
    )
    for run in iter_runs(results_root):
        scenario_id = run.get("scenario_id")
        per_turn = gold.get(str(scenario_id))
        if not per_turn:
            continue
        key = (
            str(scenario_id),
            run.get("chunk_size"),
            run.get("top_k"),
            str(run.get("model_id", "?")),
        )
        bucket = buckets[key]
        bucket["corpus_chunks"] = run.get("corpus_chunks")
        bucket["documents_ingested"] = run.get("documents_ingested")
        bucket["top_k"] = run.get("top_k")
        bucket["runs"] += 1
        bucket["ingestion_ms"] += float(run.get("ingestion_latency_ms") or 0.0)

        for turn in run.get("turn_results") or []:
            if turn.get("role") != "user":
                continue
            spec = per_turn.get(int(turn.get("turn_number") or -1))
            if not spec:
                continue
            if turn.get("status") == "error":
                bucket["errors"] += 1
                continue
            episodes = turn.get("retrieved_memories") or []
            chunk_rank = gold_chunk_rank(episodes, spec["answer"])
            doc_rank = gold_doc_rank(episodes, spec["doc"])
            details = turn.get("assertion_details") or []
            correct = all(
                d.get("passed")
                for d in details
                if d.get("rule_type") == "must_contain"
            ) and any(d.get("rule_type") == "must_contain" for d in details)

            bucket["probes"] += 1
            bucket["chunk_hits"] += int(chunk_rank is not None)
            bucket["doc_hits"] += int(doc_rank is not None)
            bucket["mrr_sum"] += (1.0 / chunk_rank) if chunk_rank else 0.0
            bucket["correct"] += int(correct)
            bucket["hit_and_correct"] += int(correct and chunk_rank is not None)
            margin = delta_cos(episodes, spec["answer"])
            if margin is not None:
                bucket["margins"].append(margin)

            if verbose:
                print(
                    f"{scenario_id} T{turn.get('turn_number')} "
                    f"chunk_rank={chunk_rank} doc_rank={doc_rank} "
                    f"correct={correct} margin={margin if margin is None else f'{margin:+.4f}'}"
                )
    return dict(buckets)


def summarize(bucket: dict[str, Any]) -> dict[str, Any]:
    """Derive display metrics (recall, CI, MRR, margins, coverage) from a bucket."""
    probes = int(bucket["probes"])
    recall = bucket["chunk_hits"] / probes if probes else 0.0
    low, high = wilson_ci(int(bucket["chunk_hits"]), probes)
    margins = bucket["margins"]
    coverage = None
    if bucket["top_k"] and bucket["corpus_chunks"]:
        coverage = bucket["top_k"] / float(bucket["corpus_chunks"])
    return {
        "probes": probes,
        "recall": recall,
        "ci_low": low,
        "ci_high": high,
        "doc_hit_rate": (bucket["doc_hits"] / probes) if probes else 0.0,
        "mrr": (bucket["mrr_sum"] / probes) if probes else 0.0,
        "mean_margin": (sum(margins) / len(margins)) if margins else None,
        "p_correct_given_hit": (
            bucket["hit_and_correct"] / bucket["chunk_hits"]
            if bucket["chunk_hits"]
            else None
        ),
        "coverage": coverage,
        # Mean per run: D16 legitimately has several runs per cell (cold and
        # warm cache, different chat models), so summing would be misleading.
        "ingest_s": (
            bucket["ingestion_ms"] / 1000.0 / bucket["runs"] if bucket["runs"] else 0.0
        ),
        "runs": bucket["runs"],
        "errors": bucket["errors"],
    }


HEADER = (
    f"{'scenario':<30} {'docs':>4} {'chk':>5} {'k':>3} {'model':<22} "
    f"{'probes':>7} {'recall@k':>9} {'95% CI':>15} {'doc-hit':>8} "
    f"{'MRR':>6} {'dcos':>9} {'P(c|hit)':>9} {'cover':>8} {'ingest':>8}"
)


def _fmt_pct(value: float | None) -> str:
    return "  n/a" if value is None else f"{value * 100:5.1f}%"


def print_table(rows: list[tuple[tuple[Any, ...], dict[str, Any]]]) -> None:
    print(HEADER)
    print("-" * len(HEADER))
    for (scenario, chunk_size, top_k, model), bucket in sorted(rows):
        s = summarize(bucket)
        ci = f"[{s['ci_low'] * 100:4.1f},{s['ci_high'] * 100:5.1f}]"
        margin = "     n/a" if s["mean_margin"] is None else f"{s['mean_margin']:+.4f}"
        coverage = "     n/a" if s["coverage"] is None else f"{s['coverage'] * 100:5.2f}%"
        docs = bucket.get("documents_ingested") or 0
        print(
            f"{scenario:<30} {docs:>4} {str(chunk_size):>5} {str(top_k):>3} "
            f"{model:<22} {s['probes']:>7} {_fmt_pct(s['recall']):>9} {ci:>15} "
            f"{_fmt_pct(s['doc_hit_rate']):>8} {s['mrr']:>6.2f} {margin:>9} "
            f"{_fmt_pct(s['p_correct_given_hit']):>9} {coverage:>8} "
            f"{s['ingest_s']:>7.0f}s"
        )


def write_csv(
    path: Path, rows: list[tuple[tuple[Any, ...], dict[str, Any]]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "scenario",
                "corpus_docs",
                "chunk_size",
                "top_k",
                "model_id",
                "probes",
                "runs",
                "chunk_recall_at_k",
                "ci95_low",
                "ci95_high",
                "doc_hit_rate",
                "mrr",
                "mean_delta_cos",
                "p_correct_given_chunk_hit",
                "coverage_ratio",
                "ingest_s",
                "error_turns",
            ]
        )
        for (scenario, chunk_size, top_k, model), bucket in sorted(rows):
            s = summarize(bucket)
            writer.writerow(
                [
                    scenario,
                    bucket.get("documents_ingested") or 0,
                    chunk_size,
                    top_k,
                    model,
                    s["probes"],
                    s["runs"],
                    f"{s['recall']:.4f}",
                    f"{s['ci_low']:.4f}",
                    f"{s['ci_high']:.4f}",
                    f"{s['doc_hit_rate']:.4f}",
                    f"{s['mrr']:.4f}",
                    "" if s["mean_margin"] is None else f"{s['mean_margin']:.6f}",
                    (
                        ""
                        if s["p_correct_given_hit"] is None
                        else f"{s['p_correct_given_hit']:.4f}"
                    ),
                    "" if s["coverage"] is None else f"{s['coverage']:.6f}",
                    f"{s['ingest_s']:.1f}",
                    s["errors"],
                ]
            )
    print(f"\nCSV written to {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recall@k / MRR / Δcos report over benchmark results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=REPO_ROOT / "outputs" / "runs",
        help="Root directory scanned recursively for results.json",
    )
    parser.add_argument(
        "--scenario-prefix",
        type=str,
        default="document_corpus_size",
        help="Only report scenarios whose id starts with this prefix",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print one line per probe (rank, correctness, Δcos)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.results.exists():
        print(f"Error: results root {args.results} does not exist.")
        return 2

    gold = load_gold_map(args.scenario_prefix or None)
    if not gold:
        print(
            "No L-R probes found. Generate the D16 scenarios first "
            "(python scripts/build_d16_scenarios.py)."
        )
        return 1

    buckets = collect(gold, args.results, verbose=args.verbose)
    if not buckets:
        print(
            f"No runs for the {len(gold)} scenario(s) with probes under "
            f"{args.results}."
        )
        return 1

    rows = list(buckets.items())
    print_table(rows)
    if args.csv:
        write_csv(args.csv, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())


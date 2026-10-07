#!/usr/bin/env python3
"""Reorganize benchmark run outputs by research theme and scenario.

Reads every ``outputs/runs/<timestamp>/results.json`` run record, maps its
``scenario_id`` to a broader research ``theme`` and a specific
``test_category``, then:

1. Copies (never moves) each run folder into
   ``outputs/by_theme/<theme>/<test_category>/<timestamp>/`` so the original
   timestamped folders under ``outputs/runs/`` stay untouched.
2. Scaffolds the empty ``robustness_security`` and ``computational_cost``
   themes with a ``.gitkeep`` file each so version control tracks them.
3. Generates ``outputs/consolidated_summary.csv`` and
   ``outputs/consolidated_summary.md`` aggregating every run.

The script is idempotent: run folders already present under ``by_theme/``
are skipped (never duplicated), the summary files are fully regenerated on
each run, and re-running never crashes or leaves stale duplicates.

Usage::

    python scripts/reorganize_outputs.py
"""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "outputs" / "runs"
BY_THEME_DIR = REPO_ROOT / "outputs" / "by_theme"
CSV_PATH = REPO_ROOT / "outputs" / "consolidated_summary.csv"
MD_PATH = REPO_ROOT / "outputs" / "consolidated_summary.md"

# ── scenario_id -> (theme, test_category) mapping ───────────────────────────
SCENARIO_MAP: dict[str, tuple[str, str]] = {
    "belief_revision_01": ("logical_consistency", "belief_revision"),
    "explicit_forget_01": ("logical_consistency", "explicit_forget"),
    "in_context_control_01": ("logical_consistency", "in_context_control"),
    "attrition_01": ("retention_persistence", "attrition"),
    "needle_haystack_01": ("retention_persistence", "needle_haystack"),
    "temporal_multi_hop_01": ("compositional_reasoning", "temporal_multi_hop"),
}

# Fixed ordering of themes in the directory tree and consolidated markdown.
THEME_ORDER: list[str] = [
    "logical_consistency",
    "retention_persistence",
    "compositional_reasoning",
    "robustness_security",
    "computational_cost",
    "others",  # fallback for unknown scenario_ids
]

# Themes with no scenarios yet — scaffolded with a .gitkeep file so version
# control tracks them, preparing the structure for future tests.
EMPTY_THEMES: tuple[str, ...] = ("robustness_security", "computational_cost")

# Human-readable titles for the markdown headings.
THEME_TITLES: dict[str, str] = {
    "logical_consistency": "Logical Consistency",
    "retention_persistence": "Retention & Persistence",
    "compositional_reasoning": "Compositional Reasoning",
    "robustness_security": "Robustness & Security",
    "computational_cost": "Computational Cost",
    "others": "Others",
}

# ── Consolidated summary columns (exact order required by the spec) ──────────
CSV_COLUMNS: list[str] = [
    "timestamp",
    "theme",
    "test_category",
    "scenario_id",
    "model_id",
    "prompt_id",
    "agent_level",
    "turns",
    "pass_rate",
    "avg_latency_ms",
    "total_tokens",
    "errors",
]


def _extract_errors(record: dict) -> int:
    """Return the run's error-turn count.

    Newer runs carry ``error_turns`` (added in the audit §3.3 resolution);
    older ones only have ``turn_results`` with a ``status`` field. Fall back
    gracefully in both directions, defaulting to 0.
    """
    if record.get("error_turns") is not None:
        return int(record["error_turns"])
    turns = record.get("turn_results") or []
    return sum(
        1 for turn in turns
        if isinstance(turn, dict) and turn.get("status") == "error"
    )


def load_run_records() -> list[dict]:
    """Load every run record under ``outputs/runs/`` and enrich it.

    Returns a list of dicts, one per run, each carrying the required
    consolidated-summary columns plus the source folder path (``_run_dir``).
    """
    records: list[dict] = []
    for results_path in sorted(RUNS_DIR.glob("*/results.json")):
        run_dir = results_path.parent
        try:
            data = json.loads(results_path.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  ! Skipping {run_dir.name}: could not read results.json ({exc})")
            continue

        # results.json is a list of run records (normally exactly one).
        run_entries = data if isinstance(data, list) else [data]
        for entry in run_entries:
            if not isinstance(entry, dict):
                continue
            scenario_id = entry.get("scenario_id", "unknown")
            theme, test_category = SCENARIO_MAP.get(
                scenario_id, ("others", scenario_id)
            )
            records.append(
                {
                    "timestamp": run_dir.name,
                    "theme": theme,
                    "test_category": test_category,
                    "scenario_id": scenario_id,
                    "model_id": entry.get("model_id", ""),
                    "prompt_id": entry.get("prompt_id", ""),
                    "agent_level": entry.get("agent_level", ""),
                    "turns": entry.get("total_turns", 0),
                    "pass_rate": entry.get("pass_rate", 0.0),
                    "avg_latency_ms": entry.get("avg_latency_per_turn_ms", 0.0),
                    "total_tokens": entry.get("total_tokens", 0),
                    "errors": _extract_errors(entry),
                    "_run_dir": run_dir,
                }
            )
    return records

def reorganize(records: list[dict]) -> tuple[int, int, int]:
    """Build the ``by_theme/`` tree and copy run folders into it.

    Returns ``(copied, skipped, unknown)`` — folders copied now, folders
    already present from a previous run, and runs mapped to ``others``.
    Original folders under ``outputs/runs/`` are never modified or moved.
    """
    copied = skipped = 0
    seen_dirs: set[Path] = set()

    # Scaffold the empty themes (idempotent: only created if missing).
    for theme in EMPTY_THEMES:
        theme_dir = BY_THEME_DIR / theme
        theme_dir.mkdir(parents=True, exist_ok=True)
        gitkeep = theme_dir / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.write_bytes(b"")

    for record in records:
        run_dir: Path = record["_run_dir"]
        if run_dir in seen_dirs:
            continue  # multiple records in one folder -> copy the folder once
        seen_dirs.add(run_dir)

        dest = (
            BY_THEME_DIR
            / record["theme"]
            / record["test_category"]
            / record["timestamp"]
        )
        if (dest / "results.json").exists():
            skipped += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(run_dir, dest)
        copied += 1

    unknown = sum(1 for record in records if record["theme"] == "others")
    return copied, skipped, unknown


def write_csv(records: list[dict]) -> Path:
    """Write the flat consolidated CSV, sorted by theme/category/timestamp."""
    theme_rank = {theme: idx for idx, theme in enumerate(THEME_ORDER)}
    ordered = sorted(
        records,
        key=lambda r: (
            theme_rank.get(r["theme"], len(THEME_ORDER)),
            r["test_category"],
            r["timestamp"],
        ),
    )
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in ordered:
            writer.writerow({col: record[col] for col in CSV_COLUMNS})
    return CSV_PATH


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def write_markdown(records: list[dict]) -> Path:
    """Write the consolidated markdown, grouped by theme with an Overall row."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines: list[str] = [
        "# Consolidated Benchmark Summary",
        "",
        f"- **Generated at:** {now}",
        f"- **Total Runs:** {len(records)}",
        f"- **Models:** {', '.join(sorted({r['model_id'] for r in records})) or 'n/a'}",
        f"- **Prompts:** {', '.join(sorted({r['prompt_id'] for r in records})) or 'n/a'}",
        f"- **Agent Levels:** {', '.join(sorted({r['agent_level'] for r in records})) or 'n/a'}",
        "",
    ]

    for theme in THEME_ORDER:
        has_runs = any(r["theme"] == theme for r in records)
        if not has_runs and theme not in EMPTY_THEMES:
            continue
        lines.append(f"## {THEME_TITLES[theme]}")
        lines.append("")
        theme_records = sorted(
            [r for r in records if r["theme"] == theme],
            key=lambda r: (r["test_category"], r["timestamp"]),
        )
        if not theme_records:
            lines.extend(
                [
                    "_No runs yet — scaffolding prepared for future tests._",
                    "",
                ]
            )
            continue
        lines.append(
            "| Test Category | Timestamp | Scenario | Model | Prompt | Level"
            " | Turns | Pass Rate | Avg Latency (ms) | Total Tokens | Errors |"
        )
        lines.append(
            "| :-- | :-- | :-- | :-- | :-- | :-- | --: | --: | --: | --: | --: |"
        )
        for r in theme_records:
            lines.append(
                f"| {r['test_category']} | {r['timestamp']} | {r['scenario_id']}"
                f" | {r['model_id']} | {r['prompt_id']} | {r['agent_level']}"
                f" | {r['turns']} | {_fmt_pct(r['pass_rate'])}"
                f" | {r['avg_latency_ms']:.0f} | {r['total_tokens']}"
                f" | {r['errors']} |"
            )
        lines.append("")

    # ── Overall averages across every run ───────────────────────────────
    n = len(records)
    avg_pass = sum(r["pass_rate"] for r in records) / n if n else 0.0
    avg_latency = sum(r["avg_latency_ms"] for r in records) / n if n else 0.0
    total_turns = sum(r["turns"] for r in records)
    total_tokens = sum(r["total_tokens"] for r in records)
    total_errors = sum(r["errors"] for r in records)

    lines.extend(
        [
            "## Overall",
            "",
            "| Runs | Avg Pass Rate | Avg Latency (ms) | Total Turns | Total Tokens | Total Errors |",
            "| --: | --: | --: | --: | --: | --: |",
            f"| {n} | {_fmt_pct(avg_pass)} | {avg_latency:.0f} | {total_turns} | {total_tokens} | {total_errors} |",
            "",
        ]
    )

    MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    return MD_PATH


def main() -> None:
    if not RUNS_DIR.is_dir():
        raise SystemExit(f"Runs directory not found: {RUNS_DIR}")

    print(f"Scanning {RUNS_DIR} ...")
    records = load_run_records()
    if not records:
        raise SystemExit("No run records found — nothing to do.")
    print(
        f"Found {len(records)} run record(s) across "
        f"{len({r['_run_dir'] for r in records})} run folder(s)."
    )

    print(f"\nReorganizing into {BY_THEME_DIR} ...")
    copied, skipped, unknown = reorganize(records)
    for theme in EMPTY_THEMES:
        print(f"  + ensured empty theme '{theme}/' with .gitkeep")
    print(f"  = {copied} folder(s) copied, {skipped} already present (skipped)")
    if unknown:
        print(f"  ! {unknown} run(s) mapped to 'others' (unknown scenario_id)")

    print("\nRun distribution by theme / test_category:")
    counts: dict[tuple[str, str], int] = {}
    for record in records:
        key = (record["theme"], record["test_category"])
        counts[key] = counts.get(key, 0) + 1
    for (theme, category), count in sorted(counts.items()):
        print(f"  - {theme}/{category}: {count} run(s)")

    csv_path = write_csv(records)
    md_path = write_markdown(records)
    print("\nConsolidated summaries written:")
    print(f"  - {csv_path}")
    print(f"  - {md_path}")
    print("\nDone. Original folders under outputs/runs/ were not modified.")


if __name__ == "__main__":
    main()

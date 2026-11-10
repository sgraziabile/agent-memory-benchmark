"""Tests for the retrieval-recall metric (SPEC §5.8).

Covers the three policies that define the number:

* **D1 declared denominator** — an unevaluated probe (error turn) stays in the
  denominator and counts as a miss.
* **D2 rule-level unit** — one pin on one turn is one unit.
* **D3 ``must_retrieve_chunk`` only** — negative guards and
  ``must_retrieve_source`` never enter numerator or denominator.

Plus the post-hoc report (``scripts/retrieval_recall.py``) for runs whose
``results.json`` predates the summary fields.

No network, no models: pure helpers + synthetic result files.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from core.metrics import (
    RETRIEVAL_RECALL_RULE_TYPES,
    assertion_recall,
    declared_retrieval_assertions,
    recall_with_ci,
    wilson_ci,
)
from scripts import backfill_retrieval_recall as backfill, retrieval_recall


class _Rule:
    """Minimal ``AssertionRule`` double (only ``type`` is inspected)."""

    def __init__(self, type_: str):
        self.type = type_


class _Turn:
    """Minimal ``ConversationTurn`` double (``role`` + ``assertions``)."""

    def __init__(self, role: str, assertions: list[_Rule]):
        self.role = role
        self.assertions = assertions


# ── core.metrics ────────────────────────────────────────────────────────────


class TestAssertionRecall:
    def test_ignores_negative_and_source_rules(self):
        details = [
            {"rule_type": "must_retrieve_chunk", "value": "doc_a", "passed": True},
            {"rule_type": "must_retrieve_chunk", "value": "doc_b", "passed": False},
            {"rule_type": "must_not_retrieve_chunk", "value": "doc_x", "passed": True},
            {"rule_type": "must_retrieve_source", "value": "document", "passed": True},
            {"rule_type": "must_contain", "value": "answer", "passed": True},
        ]
        assert assertion_recall(details) == (1, 2)

    def test_empty_details_are_zero(self):
        assert assertion_recall([]) == (0, 0)


class TestDeclaredDenominator:
    def test_skips_system_event_turns(self):
        """``system_event`` turns are ingestion records, not probes."""
        turns = [
            _Turn("user", [_Rule("must_retrieve_chunk"), _Rule("must_contain")]),
            _Turn("system_event", [_Rule("must_retrieve_chunk")]),
        ]
        assert declared_retrieval_assertions(turns) == 1

    def test_counts_each_pin_once_per_turn(self):
        """D2: the same gold document pinned twice on two turns counts twice."""
        turns = [
            _Turn("user", [_Rule("must_retrieve_chunk")]),
            _Turn("user", [_Rule("must_retrieve_chunk")]),
        ]
        assert declared_retrieval_assertions(turns) == 2


class TestRecallWithCi:
    def test_none_when_nothing_declared(self):
        assert recall_with_ci(0, 0) == (None, None, None)

    def test_matches_plain_ratio_and_sandwiches_it(self):
        recall, low, high = recall_with_ci(3, 4)
        assert recall == pytest.approx(0.75)
        assert low < recall < high
        assert high == pytest.approx(0.954, abs=0.01)

    def test_lr_report_reuses_the_shared_interval(self):
        from scripts import lr_report

        assert lr_report.wilson_ci is wilson_ci
        assert lr_report.Z_95 == pytest.approx(1.959963984540054)


# ── scripts.retrieval_recall (turns a run dict into counts) ─────────────────


class TestRunStats:
    def test_error_turn_stays_in_the_declared_denominator(self):
        """D1: an unscored probe counts as a miss, never disappears."""
        run = {
            "scenario_id": "s",
            "turn_results": [
                {
                    "turn_number": 1,
                    "role": "user",
                    "status": "ok",
                    "assertion_details": [
                        {"rule_type": "must_retrieve_chunk", "passed": True}
                    ],
                },
                {"turn_number": 2, "role": "user", "status": "error"},
            ],
        }
        stats = retrieval_recall.run_stats(run, {"s": 2})
        assert stats["passed"] == 1
        assert stats["declared"] == 2
        assert stats["evaluated"] == 1
        assert stats["denominator_source"] == "yaml"

    def test_recorded_field_wins_over_yaml(self):
        run = {"scenario_id": "s", "retrieval_assertions_total": 5, "turn_results": []}
        stats = retrieval_recall.run_stats(run, {"s": 9})
        assert stats["declared"] == 5
        assert stats["denominator_source"] == "run"

    def test_falls_back_to_evaluated_when_scenario_is_unknown(self):
        run = {
            "scenario_id": "not_in_dataset",
            "turn_results": [
                {
                    "turn_number": 1,
                    "role": "user",
                    "status": "ok",
                    "assertion_details": [
                        {"rule_type": "must_retrieve_chunk", "passed": True},
                        {"rule_type": "must_retrieve_chunk", "passed": False},
                    ],
                }
            ],
        }
        stats = retrieval_recall.run_stats(run, {})
        assert stats["declared"] == 2
        assert stats["denominator_source"] == "evaluated"
        assert stats["denominator_drift"] is False

    def test_denominator_drift_flagged_when_scenario_grew_since_run(self):
        """No errors + fewer pins evaluated than declared today = drift."""
        run = {
            "scenario_id": "doc_dilution",
            "error_turns": 0,
            "turn_results": [
                {
                    "turn_number": 1,
                    "role": "user",
                    "status": "ok",
                    "assertion_details": [
                        {"rule_type": "must_retrieve_chunk", "passed": True},
                        {"rule_type": "must_retrieve_chunk", "passed": True},
                        {"rule_type": "must_retrieve_chunk", "passed": True},
                    ],
                }
            ],
        }
        stats = retrieval_recall.run_stats(run, {"doc_dilution": 4})
        assert stats["declared"] == 4
        assert stats["evaluated"] == 3
        assert stats["denominator_drift"] is True

    def test_no_drift_flag_when_the_gap_is_explained_by_error_turns(self):
        run = {
            "scenario_id": "s",
            "error_turns": 2,
            "turn_results": [
                {
                    "turn_number": 1,
                    "role": "user",
                    "status": "ok",
                    "assertion_details": [
                        {"rule_type": "must_retrieve_chunk", "passed": True}
                    ],
                }
            ],
        }
        stats = retrieval_recall.run_stats(run, {"s": 4})
        assert stats["denominator_drift"] is False


# ── scripts.retrieval_recall (end-to-end over synthetic artifacts) ──────────


def _write_scenario(dataset_root: Path, scenario_id: str, pins_per_turn: int) -> None:
    dataset_root.mkdir(parents=True, exist_ok=True)
    turns = []
    for turn_number in (1, 2):
        turns.append(
            {
                "turn_number": turn_number,
                "role": "user",
                "content": f"q{turn_number}",
                "assertions": [
                    {"type": "must_retrieve_chunk", "value": f"doc_{turn_number}"}
                    for _ in range(pins_per_turn)
                ]
                + [{"type": "must_contain", "value": "answer"}],
            }
        )
    (dataset_root / f"test_{scenario_id}.yaml").write_text(
        yaml.safe_dump(
            {
                "id": scenario_id,
                "name": "Retrieval recall double",
                "description": "synthetic scenario for the report test",
                "category": "unit_test",
                "turns": turns,
            }
        ),
        encoding="utf-8",
    )


def _write_run(runs_root: Path, scenario_id: str) -> None:
    """One run: T1 hits both pins, T2 hits one of two -> 3/4 declared."""
    run = {
        "run_id": "run-1",
        "scenario_id": scenario_id,
        "model_id": "fake-model",
        "prompt_id": "baseline_react",
        "agent_level": "level_2_episodic_rag",
        "memory_backend": "vector",
        "top_k": 5,
        "error_turns": 0,
        "turn_results": [
            {
                "turn_number": 1,
                "role": "user",
                "status": "ok",
                "assertion_details": [
                    {
                        "rule_type": "must_retrieve_chunk",
                        "value": "doc_1",
                        "passed": True,
                    },
                    {
                        "rule_type": "must_retrieve_chunk",
                        "value": "doc_1",
                        "passed": True,
                    },
                    {"rule_type": "must_contain", "value": "answer", "passed": True},
                ],
            },
            {
                "turn_number": 2,
                "role": "user",
                "status": "ok",
                "assertion_details": [
                    {
                        "rule_type": "must_retrieve_chunk",
                        "value": "doc_2",
                        "passed": True,
                    },
                    {
                        "rule_type": "must_retrieve_chunk",
                        "value": "doc_3",
                        "passed": False,
                    },
                ],
            },
        ],
    }
    run_dir = runs_root / "20260101_000000"
    run_dir.mkdir(parents=True)
    (run_dir / "results.json").write_text(json.dumps([run]), encoding="utf-8")


class TestReportEndToEnd:
    def test_yaml_denominator_and_csv(self, tmp_path, capsys):
        dataset = tmp_path / "conversations"
        _write_scenario(dataset, "probe_scenario", pins_per_turn=2)
        runs_root = tmp_path / "runs"
        _write_run(runs_root, "probe_scenario")
        csv_path = tmp_path / "recall.csv"

        code = retrieval_recall.main(
            [
                "--results",
                str(runs_root),
                "--dataset",
                str(dataset),
                "--csv",
                str(csv_path),
                "--verbose",
            ]
        )
        out = capsys.readouterr().out

        assert code == 0
        assert "75.0%" in out
        assert "MISS 20260101_000000 T2: doc_3" in out

        with open(csv_path, encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        row = rows[0]
        assert row["scenario_id"] == "probe_scenario"
        assert row["retrieval_assertions_passed"] == "3"
        assert row["retrieval_assertions_total"] == "4"
        assert row["retrieval_assertions_evaluated"] == "4"
        assert row["retrieval_recall"] == "0.7500"
        assert row["denominator_source"] == "yaml"
        assert row["denominator_drift_runs"] == "0"

    def test_scenario_prefix_filters_out_other_scenarios(self, tmp_path, capsys):
        dataset = tmp_path / "conversations"
        _write_scenario(dataset, "probe_scenario", pins_per_turn=1)
        runs_root = tmp_path / "runs"
        _write_run(runs_root, "probe_scenario")

        code = retrieval_recall.main(
            [
                "--results",
                str(runs_root),
                "--dataset",
                str(dataset),
                "--scenario-prefix",
                "other_",
            ]
        )
        out = capsys.readouterr().out
        assert code == 1
        assert "No runs with positive" in out

    def test_missing_results_root_is_an_error(self, tmp_path, capsys):
        code = retrieval_recall.main(
            ["--results", str(tmp_path / "nope"), "--dataset", str(tmp_path)]
        )
        assert code == 2


# ── scripts.backfill_retrieval_recall (retro-fitting old run dirs) ──────────

OLD_HEADER = [
    "run_id",
    "scenario_id",
    "model_id",
    "prompt_id",
    "agent_level",
    "memory_backend",
    "total_turns",
    "total_assertions",
    "passed_assertions",
    "failed_assertions",
    "pass_rate",
    "error_turns",
    "timestamp",
]


def _old_run(scenario_id: str = "probe_scenario", pins: bool = True) -> dict:
    retr = "must_retrieve_chunk"
    text = "must_contain"
    t1 = [
        {"rule_type": retr if pins else text, "value": "doc_1", "passed": True},
        {"rule_type": retr if pins else text, "value": "doc_1", "passed": True},
    ]
    t2 = [
        {"rule_type": retr if pins else text, "value": "doc_2", "passed": True},
        {"rule_type": retr if pins else text, "value": "doc_3", "passed": False},
    ]
    return {
        "run_id": "run-1",
        "scenario_id": scenario_id,
        "model_id": "fake-model",
        "prompt_id": "baseline_react",
        "agent_level": "level_2_episodic_rag",
        "memory_backend": "vector",
        "total_turns": 2,
        "total_assertions": 4,
        "passed_assertions": 3,
        "failed_assertions": 1,
        "pass_rate": 0.75,
        "error_turns": 0,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "turn_results": [
            {
                "turn_number": 1,
                "status": "ok",
                "assertion_details": t1,
            },
            {
                "turn_number": 2,
                "status": "ok",
                "assertion_details": t2,
            },
        ],
    }


def _make_old_run_dir(
    tmp_path: Path, scenario_id: str = "probe_scenario", pins: bool = True
) -> Path:
    run_dir = tmp_path / "runs" / "20260101_000000"
    run_dir.mkdir(parents=True)
    run = _old_run(scenario_id, pins=pins)
    (run_dir / "results.json").write_text(
        json.dumps([run], indent=2), encoding="utf-8"
    )
    with open(run_dir / "summary.csv", "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OLD_HEADER)
        writer.writeheader()
        writer.writerow({k: run.get(k, "") for k in OLD_HEADER})
    return run_dir



class TestBackfill:
    def test_default_fields_match_a_fresh_export_header(self, tmp_path):
        """The fallback column set cannot drift from the runner's export."""
        from core.runner import BenchmarkRunner
        from core.schemas import BenchmarkRunSummary

        summary = BenchmarkRunSummary(
            run_id="r",
            scenario_id="s",
            model_id="m",
            prompt_id="p",
            agent_level="level_2_episodic_rag",
        )
        run_dir = BenchmarkRunner(output_dir=tmp_path / "out")._export_results([summary])
        with open(run_dir / "summary.csv", encoding="utf-8-sig", newline="") as fh:
            header = next(csv.reader(fh))
        assert header == backfill.DEFAULT_FIELDS

    def test_adds_fields_and_rewrites_csv(self, tmp_path):
        run_dir = _make_old_run_dir(tmp_path)

        info = backfill.backfill_dir(run_dir, {"probe_scenario": 4})

        assert info["runs_with_pins"] == 1
        assert (info["passed"], info["declared"]) == (3, 4)
        # Backup kept, so the pre-backfill record survives.
        assert (run_dir / "results.json.bak").is_file()
        assert (run_dir / "summary.csv.bak").is_file()

        runs = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
        run = runs[0]
        assert run["retrieval_assertions_total"] == 4
        assert run["retrieval_assertions_evaluated"] == 4
        assert run["retrieval_assertions_passed"] == 3
        assert run["retrieval_recall"] == pytest.approx(0.75)
        assert (
            run["retrieval_recall_ci_low"]
            < run["retrieval_recall"]
            < run["retrieval_recall_ci_high"]
        )
        # Everything else untouched, and the fields sit right after pass_rate.
        assert run["pass_rate"] == 0.75
        assert run["turn_results"][0]["assertion_details"][0]["value"] == "doc_1"
        keys = list(run.keys())
        assert keys[keys.index("pass_rate") + 1] == "retrieval_assertions_total"

        with open(run_dir / "summary.csv", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            header = reader.fieldnames
            rows = list(reader)
        assert header == backfill.csv_fieldnames(OLD_HEADER)
        assert header.index("retrieval_recall") == header.index("pass_rate") + 4
        assert rows[0]["retrieval_recall"] == "0.75"
        assert (run_dir / backfill.MANIFEST_NAME).is_file()

    def test_dry_run_writes_nothing(self, tmp_path):
        run_dir = _make_old_run_dir(tmp_path)
        before = (run_dir / "results.json").read_bytes()

        info = backfill.backfill_dir(run_dir, {"probe_scenario": 4}, dry_run=True)

        assert info["passed"] == 3
        assert (run_dir / "results.json").read_bytes() == before
        assert not (run_dir / "results.json.bak").exists()
        assert not (run_dir / backfill.MANIFEST_NAME).exists()

    def test_runs_without_pins_are_left_untouched(self, tmp_path):
        run_dir = _make_old_run_dir(tmp_path, pins=False)
        before = (run_dir / "results.json").read_bytes()

        info = backfill.backfill_dir(run_dir, {"probe_scenario": 0})

        assert info["runs_with_pins"] == 0
        assert (run_dir / "results.json").read_bytes() == before
        assert not (run_dir / backfill.MANIFEST_NAME).exists()

    def test_cli_reports_and_is_idempotent(self, tmp_path, capsys):
        run_dir = _make_old_run_dir(tmp_path)
        dataset = tmp_path / "conversations"
        _write_scenario(dataset, "probe_scenario", pins_per_turn=2)

        first = backfill.main(
            ["--results", str(tmp_path / "runs"), "--dataset", str(dataset)]
        )
        out_first = capsys.readouterr().out
        assert first == 0
        assert "patched 1 run dir(s)" in out_first

        second = backfill.main(
            ["--results", str(tmp_path / "runs"), "--dataset", str(dataset)]
        )
        capsys.readouterr()
        assert second == 0
        # Re-running is safe: the backup keeps the FIRST original and the
        # values are recomputed identically.
        assert (run_dir / "results.json.bak").is_file()
        runs = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
        assert runs[0]["retrieval_recall"] == pytest.approx(0.75)

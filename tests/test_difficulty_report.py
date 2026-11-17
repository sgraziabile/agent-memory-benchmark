"""Tests for the extraction-difficulty report (``scripts/difficulty_report.py``).

Covers the two contracts that make the D1/D2/D3 classification quotable:

* **The bands and the 3-of-4 policy** — a scenario is only declared in band when
  it satisfies at least three of the four measured axes, and a tie between two
  tiers resolves to the *lower* tier so the harder tier is never claimed free.
* **The violation markers** — an out-of-band scenario must say where it lands
  (``documents>D3`` / ``words~D2`` / ``noise<D1``), so the reader never has to
  guess whether "out of band" means too much or too little context.

Plus the cheap contracts the note depends on: the noise split (corpus documents
no pin names + assertion-free user turns), the declared-pin count, the pooled
pass/recall arithmetic and the denominator-drift flag.

No network, no models: pure helpers plus a synthetic scenario tree in ``tmp_path``.
"""

from __future__ import annotations

import json

import pytest
import yaml

from scripts import difficulty_report as report


def axes(documents: int, words: int, states: int, noise: int) -> report.ScenarioAxes:
    """Build ``ScenarioAxes`` from the four reported values.

    ``document_words`` carries the whole length and ``turn_words`` is zero, so
    the ``words`` axis equals the requested value without touching fixtures.
    """
    return report.ScenarioAxes(
        documents=documents,
        document_words=words,
        turn_words=0,
        states=states,
        noise_documents=noise,
        filler_turns=0,
    )


# ── Bands and the 3-of-4 policy ─────────────────────────────────────────────


class TestClassify:
    @pytest.mark.parametrize(
        ("tier", "values"),
        [
            ("D1", (2, 300, 3, 0)),
            ("D2", (4, 600, 4, 3)),
            ("D3", (5, 800, 5, 4)),
        ],
    )
    def test_perfect_fit_lands_in_band(self, tier, values):
        verdict = report.classify(axes(*values))
        assert verdict.tier == tier
        assert verdict.in_band == 4
        assert verdict.violations == ()

    def test_three_of_four_axes_is_enough(self):
        # D1 on documents, states and noise; words 29 % over the D1 ceiling.
        verdict = report.classify(axes(documents=1, words=517, states=2, noise=0))
        assert verdict.tier == "D1"
        assert verdict.in_band == 3
        assert verdict.violations == ("words~D2",)

    def test_two_of_four_is_out_of_band(self):
        verdict = report.classify(axes(documents=0, words=204, states=1, noise=0))
        assert verdict.tier is None
        assert verdict.in_band == 2
        assert "documents<D1" in verdict.violations

    def test_tie_resolves_to_the_lower_tier(self):
        # 1 doc + 2 states (D1) versus 579 words + 2 noise items (D2): both
        # tiers satisfy exactly 2 of 4 axes, so the decisive tier is the lower
        # one and the scenario stays out of band. This is the real
        # `document_cross_thread_01` shape.
        verdict = report.classify(axes(documents=1, words=579, states=2, noise=2))
        assert verdict.tier is None
        assert verdict.in_band == 2
        assert verdict.violations == ("words~D2", "noise~D2")

    def test_d2_can_win_when_it_satisfies_three_axes(self):
        # Same shape but D2 on states as well (3), so D2 reaches 3/4 and wins.
        verdict = report.classify(axes(documents=2, words=600, states=3, noise=2))
        assert verdict.tier == "D2"
        assert verdict.in_band == 3
        assert verdict.violations == ("documents~D1",)

    def test_huge_corpus_is_marked_above_d3(self):
        verdict = report.classify(
            axes(documents=60, words=52_064, states=3, noise=61)
        )
        assert verdict.tier is None
        assert verdict.violations == ("documents>D3", "words>D3", "noise>D3")

    def test_empty_chat_scenario_is_marked_below_d1(self):
        verdict = report.classify(axes(documents=0, words=198, states=1, noise=8))
        assert verdict.tier is None
        assert verdict.violations == (
            "documents<D1",
            "words<D1",
            "states<D1",
            "noise>D3",
        )


class TestAxisMarker:
    @pytest.mark.parametrize(
        ("axis", "value", "expected"),
        [
            ("documents", 60, "documents>D3"),
            ("documents", 0, "documents<D1"),
            ("documents", 4, "documents~D2"),
            ("words", 420, "words~D2"),
            ("words", 1_037, "words>D3"),
            ("noise", 0, "noise~D1"),
            ("states", 4, "states~D2"),
        ],
    )
    def test_marker(self, axis, value, expected):
        assert report.axis_marker(axis, value) == expected


# ── Scenario measurement ────────────────────────────────────────────────────


SCENARIO = {
    "id": "demo_01",
    "name": "Demo",
    "description": "Synthetic scenario for the report test.",
    "category": "demo",
    "documents": [
        {"doc_id": "gold", "content": "The launch code is ALPHA-1."},
        {"doc_id": "filler", "content": "Unrelated scheduling note."},
    ],
    "turns": [
        {"turn_number": 1, "session_id": "s1", "role": "user", "content": "hi there"},
        {"turn_number": 2, "session_id": "s1", "role": "user", "content": "ok thanks"},
        {
            "turn_number": 3,
            "session_id": "s2",
            "role": "user",
            "content": "what is the launch code now",
            "assertions": [
                {"type": "must_contain", "value": "ALPHA-1"},
                {"type": "must_retrieve_chunk", "value": "gold"},
            ],
        },
    ],
}


def write_scenario(root, payload: dict, group: str = "multi_thread"):
    """Materialize ``payload`` as ``<root>/<group>/theme/category/test_<id>.yaml``."""
    path = root / group / "theme" / "category" / f"test_{payload['id']}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


class TestMeasureScenario:
    def test_axes_and_noise_split(self, tmp_path):
        path = write_scenario(tmp_path, SCENARIO)
        scenario, measured = report.measure_scenario(path)

        assert scenario.id == "demo_01"
        assert measured.documents == 2
        assert measured.states == 2
        # "The launch code is ALPHA-1." (5) + "Unrelated scheduling note." (3)
        assert measured.document_words == 8
        # "hi there" (2) + "ok thanks" (2) + "what is the launch code now" (6)
        assert measured.turn_words == 10
        assert measured.words == 18
        # `filler` is never pinned, `gold` is.
        assert measured.noise_documents == 1
        # Turns 1 and 2 assert nothing; turn 3 is the probe.
        assert measured.filler_turns == 2
        assert measured.noise == 3

    def test_system_event_turns_never_declare_pins(self, tmp_path):
        payload = json.loads(json.dumps(SCENARIO))
        payload["turns"].append(
            {
                "turn_number": 4,
                "session_id": "s2",
                "role": "system_event",
                "content": "ingesting the revision",
                "assertions": [
                    {"type": "must_retrieve_chunk", "value": "ignored"},
                ],
            }
        )
        write_scenario(tmp_path, payload)

        scenarios = report.load_scenarios(tmp_path, "multi_thread")

        # Only `role == "user"` turns can declare a retrieval pin: the ingestion
        # turn's `ignored` pin never enters the denominator.
        assert report.declared_pin_counts(scenarios) == {"demo_01": 1}

    def test_group_filter_skips_other_thread_classes(self, tmp_path):
        write_scenario(tmp_path, SCENARIO, group="multi_thread")
        write_scenario(tmp_path, {**SCENARIO, "id": "single_01"}, group="single_thread")

        assert set(report.load_scenarios(tmp_path, "multi_thread")) == {"demo_01"}
        assert set(report.load_scenarios(tmp_path, "single_thread")) == {"single_01"}
        assert len(report.load_scenarios(tmp_path, "all")) == 2


# ── Empirical aggregation ──────────────────────────────────────────────────


def write_run(root, run_id: str, **overrides):
    """Write a one-run ``results.json`` under ``<root>/<run_id>/``."""
    run = {
        "run_id": run_id,
        "scenario_id": "demo_01",
        "agent_level": "level_2_episodic_rag",
        "model_id": "test-model",
        "passed_assertions": 3,
        "total_assertions": 4,
        "total_tokens": 1000,
        "error_turns": 0,
        "turn_results": [
            {
                "turn_number": 3,
                "role": "user",
                "status": "ok",
                "assertion_details": [
                    {
                        "rule_type": "must_retrieve_chunk",
                        "value": "gold",
                        "passed": True,
                    }
                ],
            }
        ],
    }
    run.update(overrides)
    path = root / run_id / "results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([run]), encoding="utf-8")
    return path


class TestCollect:
    def test_pools_runs_and_dedups_by_run_id(self, tmp_path):
        canonical = tmp_path / "runs"
        legacy = tmp_path / "by_theme"
        write_run(canonical, "run_a")
        write_run(canonical, "run_b")
        write_run(legacy, "run_a")  # same run in the legacy layout: count once

        totals = report.collect([canonical, legacy], {"demo_01": 1})["demo_01"]

        assert totals.runs == 2
        assert totals.assertions_passed == 6
        assert totals.assertions_total == 8
        assert totals.pass_rate == pytest.approx(0.75)
        assert totals.pins_declared == 2
        assert totals.recall == pytest.approx(1.0)
        assert totals.tokens == 2000
        assert totals.models == {"test-model"}

    def test_level_filter(self, tmp_path):
        write_run(tmp_path, "run_a")
        write_run(tmp_path, "run_b", agent_level="level_0_reactive")

        assert report.collect([tmp_path], {}, "level_0_reactive")["demo_01"].runs == 1
        assert report.collect([tmp_path], {}, "level_2_episodic_rag")["demo_01"].runs == 1
        assert report.collect([tmp_path], {})["demo_01"].runs == 2

    def test_denominator_drift_is_flagged(self, tmp_path):
        # The scenario declares a pin this run never evaluated and no turn
        # errored -> the recall figure is a lower bound over today's YAML.
        path = write_run(tmp_path, "run_a")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload[0]["turn_results"][0]["assertion_details"] = []
        path.write_text(json.dumps(payload), encoding="utf-8")

        totals = report.collect([tmp_path], {"demo_01": 1})["demo_01"]

        assert totals.drift_runs == 1
        assert totals.pins_declared == 1
        assert totals.pins_passed == 0
        assert totals.recall == 0.0


class TestAggregate:
    def test_aggregate_by_tier_pools_across_scenarios(self, tmp_path):
        measured = axes(documents=1, words=300, states=2, noise=0)
        scenarios = {
            "demo_01": report.ScenarioRecord(
                scenario=report.BenchmarkScenario.model_validate(SCENARIO),
                path=tmp_path / "test_demo_01.yaml",
                group="multi_thread",
                theme="theme",
                category="category",
                axes=measured,
                verdict=report.classify(measured),
            )
        }
        totals = {
            "demo_01": report.RunTotals(
                runs=2,
                assertions_passed=3,
                assertions_total=4,
                pins_passed=1,
                pins_declared=2,
                tokens=100,
            )
        }
        rows = report.build_rows(scenarios, totals)
        pooled = report.aggregate(rows, "tier")

        assert set(pooled) == {"D1"}
        assert pooled["D1"].runs == 2
        assert pooled["D1"].tokens == 100
        assert pooled["D1"].recall == pytest.approx(0.5)

    def test_scenario_without_runs_has_no_totals(self, tmp_path):
        write_scenario(tmp_path, SCENARIO)
        scenarios = report.load_scenarios(tmp_path, "multi_thread")
        rows = report.build_rows(scenarios, {})

        assert rows[0].totals is None
        assert report.aggregate(rows, "tier") == {}


# ── Formatting and CLI ─────────────────────────────────────────────────────


class TestFormatting:
    @pytest.mark.parametrize(
        ("numerator", "denominator", "expected"),
        [
            (37, 40, "37/40  92.5%"),
            (0, 4, "0/4   0.0%"),
            (5, 5, "5/5 100.0%"),
            (1, 0, "-"),
        ],
    )
    def test_pct(self, numerator, denominator, expected):
        assert report._pct(numerator, denominator) == expected

    def test_thousands_uses_a_space_separator(self):
        assert report._thousands(501417) == "501 417"
        assert report._thousands(0) == "0"


class TestMain:
    def test_returns_2_when_dataset_root_is_missing(self, tmp_path):
        assert report.main(["--dataset", str(tmp_path / "nope")]) == 2

    def test_end_to_end_writes_csv(self, tmp_path, capsys):
        dataset = tmp_path / "conversations"
        write_scenario(dataset, SCENARIO)
        results = tmp_path / "runs"
        write_run(results, "run_a")
        csv_path = tmp_path / "report.csv"

        code = report.main(
            [
                "--dataset",
                str(dataset),
                "--results",
                str(results),
                "--group",
                "multi_thread",
                "--csv",
                str(csv_path),
            ]
        )

        assert code == 0
        output = capsys.readouterr().out
        assert "demo_01" in output
        assert "tier census" in output
        assert csv_path.exists()
        header = csv_path.read_text(encoding="utf-8-sig").splitlines()[0]
        assert "context_words" in header
        assert "retrieval_recall" in header

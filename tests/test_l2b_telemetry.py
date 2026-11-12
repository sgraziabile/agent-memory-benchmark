"""Tests for the L2b retrieval-telemetry contract.

Covers ``TODOs/l2b-telemetry-contract-todo.md`` §3 (acceptance criteria). The
contract exists so that a single ``results.json`` can explain an L2b failure:

* **Guarded reads** — L0/L1/L2a never write the new channels, so they must
  default to ``[]`` / ``0`` / ``False`` instead of raising.
* **Union match set** — L2b issues up to ``max_tool_calls`` retrievals per turn.
  Scoring only the last one would let a wasted or mis-ordered call hide the
  evidence it did retrieve, so ``must_retrieve_chunk`` must match across
  ``calls[0]`` and ``calls[-1]`` alike.
* **Calls are counted, never collapsed** — the match set is de-duplicated, the
  call count is not, because wasted retrievals are exactly what an agentic loop
  can make worse.
* **No-tool is a state, not an error** — a turn that never retrieves reports
  ``retrieval_call_count == 0`` and an efficiency of ``None`` (undefined), not
  ``0.0``.
* **``--level 2b``** resolves through the registry, so it fails cleanly today
  and works the moment the agent is registered.

No network, no models.
"""

from __future__ import annotations

import pytest

from agents import AGENT_REGISTRY, LEVEL_ALIASES, resolve_level_ids
from core.metrics import (
    mean_calls_per_probe,
    pins_per_call,
    union_retrieved_episodes,
)
from core.runner import BenchmarkRunner, _read_l2_telemetry
from core.schemas import BenchmarkRunSummary, L2AgentState, TurnResult


def episode(chunk_id: str, source: str = "document", rank: int = 1) -> dict:
    """Minimal retrieved-episode dict as the backend emits it."""
    return {"chunk_id": chunk_id, "source": source, "rank": rank}


def call(index: int, *chunk_ids: str) -> dict:
    """One tool call carrying its episodes, as an L2b graph reports it."""
    return {
        "call_index": index,
        "query": f"query-{index}",
        "k": len(chunk_ids),
        "episodes": [episode(chunk_id) for chunk_id in chunk_ids],
    }


# ── Guarded reads ──────────────────────────────────────────────────────────


class TestGuardedReads:
    def test_absent_channels_default_to_zero(self):
        """An L0/L1/L2a invoke result carries none of the new keys."""
        telemetry = _read_l2_telemetry({"retrieved_memories": []})

        assert telemetry["retrieval_calls"] == []
        assert telemetry["retrieval_call_count"] == 0
        assert telemetry["retrieval_skipped"] is False
        assert telemetry["tool_call_cap_hit"] is False

    def test_non_dict_result_is_fully_zeroed(self):
        telemetry = _read_l2_telemetry(None)

        assert telemetry["retrieval_calls"] == []
        assert telemetry["retrieval_call_count"] == 0

    def test_channels_are_read_when_present(self):
        result = {
            "retrieval_calls": [call(1, "doc_a_c0")],
            "retrieval_call_count": 3,
            "retrieval_skipped": True,
            "tool_call_cap_hit": True,
        }
        telemetry = _read_l2_telemetry(result)

        assert telemetry["retrieval_call_count"] == 3
        assert telemetry["retrieval_skipped"] is True
        assert telemetry["tool_call_cap_hit"] is True
        assert telemetry["retrieval_calls"][0]["query"] == "query-1"

    def test_turn_and_summary_defaults_keep_l0_l1_l2_constructible(self):
        """The new fields must not make the old constructors invalid."""
        turn = TurnResult(turn_number=1, user_input="q", agent_response="a")
        summary = BenchmarkRunSummary(
            run_id="r",
            scenario_id="s",
            model_id="m",
            prompt_id="baseline_react",
            agent_level="level_0_reactive",
        )

        assert turn.retrieval_call_count == 0
        assert turn.retrieval_calls == []
        assert turn.retrieval_skipped is False
        assert summary.total_tool_calls == 0
        assert summary.avg_calls_per_retrieval_turn == 0.0
        assert summary.calls_per_probe is None
        assert summary.retrieval_call_efficiency is None

    def test_state_typeddict_declares_the_channels(self):
        hints = L2AgentState.__annotations__

        for channel in (
            "retrieval_calls",
            "retrieval_call_count",
            "retrieval_skipped",
            "tool_call_cap_hit",
        ):
            assert channel in hints, channel


# ── Union match set ────────────────────────────────────────────────────────


class TestUnionMatchSet:
    def test_falls_back_to_retrieved_memories_without_calls(self):
        """L2a telemetry must behave exactly as before."""
        memories = [episode("doc_a_c0"), episode("ep_s1_t1", source="chat")]

        assert union_retrieved_episodes(memories, []) == memories
        assert union_retrieved_episodes(memories, None) == memories

    def test_union_covers_first_and_last_call(self):
        """The whole point: call 1's evidence is not shadowed by call 3."""
        calls = [call(1, "doc_rows_c2"), call(2, "doc_noise_c0"), call(3, "doc_legend_c0")]

        union = union_retrieved_episodes([], calls)
        chunk_ids = [ep["chunk_id"] for ep in union]

        assert chunk_ids == ["doc_rows_c2", "doc_noise_c0", "doc_legend_c0"]

    def test_union_dedups_by_chunk_id_keeping_first(self):
        calls = [call(1, "doc_a_c0", "doc_b_c0"), call(2, "doc_b_c0", "doc_c_c0")]

        union = union_retrieved_episodes([], calls)

        assert [ep["chunk_id"] for ep in union] == [
            "doc_a_c0",
            "doc_b_c0",
            "doc_c_c0",
        ]
        # First occurrence wins, so the rank reported is call 1's.
        assert next(ep for ep in union if ep["chunk_id"] == "doc_b_c0")["rank"] == 1

    def test_call_with_only_chunk_ids_still_counts(self):
        """A call that recorded ids but not episodes is not silently dropped."""
        calls = [{"call_index": 1, "chunk_ids": ["doc_a_c0"]}]

        union = union_retrieved_episodes([], calls)

        assert [ep["chunk_id"] for ep in union] == ["doc_a_c0"]

    def test_entries_without_chunk_id_are_kept_for_source_rules(self):
        calls = [{"call_index": 1, "episodes": [{"source": "document"}]}]

        union = union_retrieved_episodes([], calls)

        assert union == [{"source": "document"}]

    def test_malformed_input_is_ignored(self):
        calls = [None, "nope", {"call_index": 1, "episodes": [None, 7, episode("d")]}]

        assert union_retrieved_episodes([], calls) == [episode("d")]


class TestAssertionsSeeTheUnion:
    """End-to-end through ``evaluate_assertions``: the pin matches any call."""

    @staticmethod
    def _rules(*values: str):
        from core.schemas import AssertionRule

        return [
            AssertionRule(type="must_retrieve_chunk", value=value)
            for value in values
        ]

    def test_pin_satisfied_by_an_earlier_call(self):
        calls = [call(1, "doc_legend_c0"), call(2, "doc_rows_c2")]
        match_set = union_retrieved_episodes([], calls)

        details = BenchmarkRunner.evaluate_assertions(
            "", self._rules("doc_legend_c0", "doc_rows_c2"), retrieved_memories=match_set
        )

        assert [d["passed"] for d in details] == [True, True]

    def test_last_call_alone_would_have_failed(self):
        """Regression guard: scoring only the last call loses the first."""
        calls = [call(1, "doc_legend_c0"), call(2, "doc_rows_c2")]

        details = BenchmarkRunner.evaluate_assertions(
            "",
            self._rules("doc_legend_c0"),
            retrieved_memories=calls[-1]["episodes"],
        )

        assert [d["passed"] for d in details] == [False]

    def test_negative_pin_also_sees_every_call(self):
        from core.schemas import AssertionRule

        calls = [call(1, "doc_a_c0"), call(2, "bg_02_c0")]
        match_set = union_retrieved_episodes([], calls)

        details = BenchmarkRunner.evaluate_assertions(
            "",
            [AssertionRule(type="must_not_retrieve_chunk", value="bg_02")],
            retrieved_memories=match_set,
        )

        assert [d["passed"] for d in details] == [False]


# ── Derived metrics ────────────────────────────────────────────────────────


class TestDerivedMetrics:
    def test_pins_per_call(self):
        assert pins_per_call(2, 3) == pytest.approx(2 / 3)
        assert pins_per_call(0, 1) == 0.0

    def test_efficiency_is_none_without_calls(self):
        """A no-tool run has undefined efficiency, not zero."""
        assert pins_per_call(5, 0) is None

    def test_mean_calls_per_probe(self):
        assert mean_calls_per_probe([1, 2, 3]) == pytest.approx(2.0)
        assert mean_calls_per_probe([0, 0]) == 0.0

    def test_mean_calls_is_none_without_probes(self):
        assert mean_calls_per_probe([]) is None


# ── End-to-end: the L2a path must be unchanged ─────────────────────────────


class _ScriptedGraph:
    """Minimal stand-in for a compiled graph: scripted replies + telemetry."""

    def __init__(self, responses: list[str], channels: dict):
        self._responses = list(responses)
        self._channels = channels

    def invoke(self, state, config=None):  # noqa: ARG002 - runner passes config
        from langchain_core.messages import AIMessage

        return {
            "messages": [AIMessage(content=self._responses.pop(0))],
            **self._channels,
        }


def _scenario_with_pin(*values: str):
    """One probe turn that pins ``values`` and asserts a reply."""
    from core.schemas import AssertionRule, BenchmarkScenario, ConversationTurn

    rules = [AssertionRule(type="must_contain", value="OK")]
    rules += [
        AssertionRule(type="must_retrieve_chunk", value=value) for value in values
    ]
    return BenchmarkScenario(
        id="l2b_contract_probe",
        name="L2b contract probe",
        description="Synthetic scenario for the telemetry contract.",
        category="unit_test",
        turns=[
            ConversationTurn(
                turn_number=1,
                role="user",
                content="what is the figure",
                session_id="s1",
                assertions=rules,
            )
        ],
    )


class TestL2aPathUnchanged:
    """The whole contract hinges on L2a producing identical science."""

    @staticmethod
    def _run(channels: dict, *pins: str):
        return BenchmarkRunner().run_single(
            _scenario_with_pin(*pins),
            "baseline_react",
            "You are a test agent.",
            "fake-model",
            _ScriptedGraph(["OK"], channels),
            "level_2_episodic_rag",
        )

    def test_read_then_union_returns_the_same_list(self):
        memories = [
            episode("doc_policy_v1_c2", rank=1),
            episode("ep_s1_t1", source="chat", rank=2),
        ]

        telemetry = _read_l2_telemetry({"retrieved_memories": list(memories)})
        match_set = union_retrieved_episodes(
            telemetry["retrieved_memories"], telemetry["retrieval_calls"]
        )

        assert match_set == memories

    def test_l2a_run_scores_exactly_as_before(self):
        """Same pins, same verdicts, same recall — only the new fields appear."""
        summary = self._run(
            {
                "retrieved_memories": [
                    episode("doc_policy_v1_c2"),
                    episode("ep_s1_t1", source="chat"),
                ],
            },
            "doc_policy_v1",
            "ep_s1_t1",
        )

        # Science unchanged.
        assert summary.total_assertions == 3
        assert summary.passed_assertions == 3
        assert summary.pass_rate == 1.0
        assert summary.retrieval_assertions_total == 2
        assert summary.retrieval_assertions_passed == 2
        assert summary.retrieval_recall == 1.0
        # New fields: zero, and zero is not None.
        assert summary.total_tool_calls == 0
        assert summary.avg_calls_per_retrieval_turn == 0.0
        assert summary.tool_call_cap_hits == 0
        assert summary.calls_per_probe == 0.0
        assert summary.retrieval_call_efficiency is None
        turn = summary.turn_results[0]
        assert turn.retrieval_calls == []
        assert turn.retrieval_call_count == 0
        assert turn.tool_call_cap_hit is False

    def test_l2b_three_call_turn_reports_three_calls_and_the_union(self):
        summary = self._run(
            {
                "retrieved_memories": [],
                "retrieval_calls": [
                    call(1, "doc_legend_c0"),
                    call(2, "doc_noise_c0"),
                    call(3, "doc_rows_c2"),
                ],
                "retrieval_call_count": 3,
            },
            "doc_legend_c0",
            "doc_rows_c2",
        )

        turn = summary.turn_results[0]
        assert turn.retrieval_call_count == 3
        assert len(turn.retrieval_calls) == 3
        # Both pins satisfied from the union, not from the last call alone.
        assert summary.retrieval_assertions_passed == 2
        assert summary.retrieval_recall == 1.0
        assert summary.total_tool_calls == 3
        assert summary.avg_calls_per_retrieval_turn == 3.0
        assert summary.calls_per_probe == 3.0
        assert summary.retrieval_call_efficiency == pytest.approx(2 / 3)
        assert summary.pass_rate == 1.0

    def test_cap_hit_is_counted_separately_from_accuracy(self):
        channels = {
            "retrieved_memories": [],
            "retrieval_calls": [call(1, "doc_a_c0")],
            "retrieval_call_count": 3,
            "tool_call_cap_hit": True,
        }

        summary = self._run(channels, "doc_a_c0")

        assert summary.tool_call_cap_hits == 1
        assert summary.turn_results[0].tool_call_cap_hit is True


# ── CLI alias ──────────────────────────────────────────────────────────────


class TestLevelAlias:
    def test_alias_points_at_the_agentic_level(self):
        assert LEVEL_ALIASES["2b"] == "level_2b_agentic_rag"

    def test_bare_2_still_resolves_to_l2a_only(self):
        """``--level 2`` must not drag ``level_2b`` along."""
        assert resolve_level_ids("2") == ["level_2_episodic_rag"]

    def test_unregistered_alias_fails_cleanly(self):
        """Until the L2b agent exists, ``2b`` reports the normal error."""
        if "level_2b_agentic_rag" in AGENT_REGISTRY:
            pytest.skip("L2b is registered; the alias now resolves")
        with pytest.raises(ValueError, match="Available levels"):
            resolve_level_ids("2b")

    def test_alias_resolves_once_registered(self, monkeypatch):
        """The alias needs no second edit when the agent lands."""
        from agents import _available_levels

        monkeypatch.setitem(
            AGENT_REGISTRY, "level_2b_agentic_rag", AGENT_REGISTRY["level_2_episodic_rag"]
        )

        assert resolve_level_ids("2b") == ["level_2b_agentic_rag"]
        assert resolve_level_ids("2") == ["level_2_episodic_rag"]
        # `_available_levels` keeps the short form readable for the new level.
        assert "2b (level_2b_agentic_rag)" in _available_levels()

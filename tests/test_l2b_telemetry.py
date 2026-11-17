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
* **Query-guarded pins (F1)** — a probe may declare ``query_must_contain``;
  only calls whose query carried those tokens can satisfy the pin, which is
  what separates "searched for the wrong thing" from "the retriever missed it".
* **Five-class taxonomy (F0–F4)** — one class per graded probe, plus the
  per-run distribution, so a regression is attributable instead of merely
  visible.
* **Gross re-query cost** — ``retrieval_context_tokens_gross`` counts every
  call, duplicates included; the union alone hides what re-querying costs.

No network, no models.
"""

from __future__ import annotations

import pytest

from agents import AGENT_REGISTRY, LEVEL_ALIASES, resolve_level_ids
from core.metrics import (
    classify_failure_class,
    count_failure_classes,
    gross_context_tokens,
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

    def __init__(self, responses: list[str], channels: dict, captured: list | None = None):
        self._responses = list(responses)
        self._channels = channels
        #: Optional sink for the ``config`` of every invocation, so tests can
        #: assert what the runner injected into ``configurable``.
        self._captured = captured if captured is not None else []

    def invoke(self, state, config=None):  # noqa: ARG002 - runner passes config
        from langchain_core.messages import AIMessage

        self._captured.append(config or {})
        return {
            "messages": [AIMessage(content=self._responses.pop(0))],
            **self._channels,
        }


def _scenario_with_pin(
    *values: str,
    query_must_contain: list[str] | None = None,
):
    """One probe turn that pins ``values`` and asserts a reply."""
    from core.schemas import AssertionRule, BenchmarkScenario, ConversationTurn

    rules = [AssertionRule(type="must_contain", value="OK")]
    rules += [
        AssertionRule(
            type="must_retrieve_chunk",
            value=value,
            query_must_contain=query_must_contain or [],
        )
        for value in values
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


# ── Query-guarded pins (F1: the agent searched for the wrong thing) ─────────


class TestQueryScopedPins:
    """``query_must_contain`` scopes a pin to the calls that asked for it."""

    @staticmethod
    def _rule(tokens):
        from core.schemas import AssertionRule

        return AssertionRule(
            type="must_retrieve_chunk",
            value="doc_rows",
            query_must_contain=tokens,
        )

    def test_gold_behind_a_malformed_query_does_not_satisfy_the_pin(self):
        calls = [
            {
                "call_index": 1,
                "query": "pressure code",
                "episodes": [episode("doc_rows_c2")],
            }
        ]

        details = BenchmarkRunner.evaluate_assertions(
            "", [self._rule(["VLV-320"])], retrieval_calls=calls
        )

        assert details[0]["passed"] is False
        assert details[0]["query_matched"] is False
        assert details[0]["query_must_contain"] == ["VLV-320"]

    def test_a_later_matching_call_satisfies_the_pin(self):
        calls = [
            {
                "call_index": 1,
                "query": "pressure code",
                "episodes": [episode("doc_noise_c0")],
            },
            {
                "call_index": 2,
                "query": "certified value for VLV-320",
                "episodes": [episode("doc_rows_c2")],
            },
        ]

        details = BenchmarkRunner.evaluate_assertions(
            "", [self._rule(["VLV-320"])], retrieval_calls=calls
        )

        assert details[0]["passed"] is True
        assert details[0]["query_matched"] is True

    def test_query_matching_is_case_insensitive(self):
        calls = [
            {
                "call_index": 1,
                "query": "what is nbl-420",
                "episodes": [episode("doc_rows_c2")],
            },
        ]

        details = BenchmarkRunner.evaluate_assertions(
            "", [self._rule(["NBL-420"])], retrieval_calls=calls
        )

        assert details[0]["passed"] is True

    def test_every_declared_token_is_required(self):
        calls = [
            {
                "call_index": 1,
                "query": "calibration code for LOOP-420",
                "episodes": [episode("doc_rows_c2")],
            }
        ]

        details = BenchmarkRunner.evaluate_assertions(
            "", [self._rule(["LOOP-420", "NBL-420"])], retrieval_calls=calls
        )

        assert details[0]["passed"] is False
        assert details[0]["query_matched"] is False

    def test_guard_without_call_telemetry_fails_honestly(self):
        """The guard cannot be verified, so the pin is not credited."""
        details = BenchmarkRunner.evaluate_assertions(
            "",
            [self._rule(["VLV-320"])],
            retrieved_memories=[episode("doc_rows_c2")],
        )

        assert details[0]["passed"] is False
        assert details[0]["query_matched"] is False

    def test_unguarded_pin_still_reads_the_union(self):
        details = BenchmarkRunner.evaluate_assertions(
            "",
            [self._rule([])],
            retrieved_memories=[episode("doc_rows_c2")],
        )

        assert details[0]["passed"] is True
        assert "query_matched" not in details[0]

    def test_single_string_is_accepted_and_normalized(self):
        """YAML authors may write ``query_must_contain: NBL-420``."""
        rule = self._rule("NBL-420")

        assert rule.query_must_contain == ["NBL-420"]

    def test_call_index_selects_only_that_call(self):
        """A probe can require the *second* query to find the gold."""
        from core.schemas import AssertionRule

        calls = [
            {
                "call_index": 1,
                "query": "q1",
                "episodes": [episode("doc_rows_c2")],
            },
            {
                "call_index": 2,
                "query": "q2",
                "episodes": [episode("doc_noise_c0")],
            },
        ]
        rule = AssertionRule(
            type="must_retrieve_chunk", value="doc_rows", call_index=2
        )

        details = BenchmarkRunner.evaluate_assertions(
            "", [rule], retrieval_calls=calls
        )

        assert details[0]["passed"] is False
        assert details[0]["call_index"] == 2

    def test_the_indexed_call_passing_is_enough(self):
        from core.schemas import AssertionRule

        calls = [
            {"call_index": 1, "query": "q1", "episodes": [episode("doc_noise_c0")]},
            {"call_index": 2, "query": "q2", "episodes": [episode("doc_rows_c2")]},
        ]
        rule = AssertionRule(
            type="must_retrieve_chunk", value="doc_rows", call_index=2
        )

        details = BenchmarkRunner.evaluate_assertions(
            "", [rule], retrieval_calls=calls
        )

        assert details[0]["passed"] is True

    def test_index_and_query_guard_can_be_combined(self):
        from core.schemas import AssertionRule

        calls = [
            {
                "call_index": 1,
                "query": "VLV-320 certified value",
                "episodes": [episode("doc_rows_c2")],
            },
            {
                "call_index": 2,
                "query": "pressure code",
                "episodes": [episode("doc_rows_c2")],
            },
        ]
        rule = AssertionRule(
            type="must_retrieve_chunk",
            value="doc_rows",
            call_index=2,
            query_must_contain=["VLV-320"],
        )

        details = BenchmarkRunner.evaluate_assertions(
            "", [rule], retrieval_calls=calls
        )

        # The query was right somewhere (no F1), but not on call 2.
        assert details[0]["query_matched"] is True
        assert details[0]["passed"] is False

    def test_call_index_must_be_positive(self):
        from core.schemas import AssertionRule

        with pytest.raises(ValueError):
            AssertionRule(
                type="must_retrieve_chunk", value="doc_rows", call_index=0
            )


# ── Five-class failure taxonomy ────────────────────────────────────────────


class TestFailureClassification:
    @staticmethod
    def _classify(**overrides):
        base = {
            "pins_total": 1,
            "pins_passed": 0,
            "text_total": 1,
            "text_passed": 0,
            "retrieval_call_count": 1,
        }
        base.update(overrides)
        return classify_failure_class(**base)

    def test_text_only_turn_is_not_a_probe(self):
        assert (
            classify_failure_class(
                pins_total=0,
                pins_passed=0,
                text_total=2,
                text_passed=2,
                retrieval_call_count=0,
            )
            is None
        )

    def test_f0_no_call_made(self):
        assert self._classify(retrieval_call_count=0) == "F0"

    def test_f1_declared_token_missing_from_every_query(self):
        assert self._classify(query_checked=True, query_matched=False) == "F1"

    def test_undetectable_malformed_query_cannot_be_claimed(self):
        """Without a ``query_must_contain`` declaration this is F2, not F1."""
        assert self._classify(query_checked=False) == "F2"

    def test_f2_gold_absent_from_the_union(self):
        assert self._classify(query_checked=True, query_matched=True) == "F2"

    def test_f3_gold_present_answer_wrong(self):
        assert self._classify(pins_passed=1, text_passed=0) == "F3"

    def test_f4_gold_present_answer_right(self):
        assert self._classify(pins_passed=1, text_passed=1) == "F4"

    def test_text_free_probe_can_still_be_f4(self):
        assert self._classify(pins_passed=1, text_total=0, text_passed=0) == "F4"

    def test_one_missing_pin_out_of_two_is_not_f4(self):
        assert self._classify(pins_total=2, pins_passed=1, text_passed=1) == "F2"

    def test_distribution_is_ordered_and_drops_non_probes(self):
        counts = count_failure_classes(["F2", None, "F0", "F2"])

        assert counts == {"F0": 1, "F2": 2}
        assert list(counts) == ["F0", "F2"]

    def test_distribution_is_empty_without_probes(self):
        assert count_failure_classes([None, None]) == {}


class TestFailureClassEndToEnd:
    """The class reaches ``results.json`` through ``run_single``."""

    @staticmethod
    def _run(channels, *pins, answer="OK", query_must_contain=None):
        return BenchmarkRunner().run_single(
            _scenario_with_pin(*pins, query_must_contain=query_must_contain),
            "baseline_react",
            "You are a test agent.",
            "fake-model",
            _ScriptedGraph([answer], channels),
            "level_2_episodic_rag",
        )

    def test_f4_when_gold_and_answer_are_right(self):
        summary = self._run(
            {"retrieved_memories": [episode("doc_a_c0")]}, "doc_a_c0"
        )

        assert summary.turn_results[0].failure_class == "F4"
        assert summary.failure_class_counts == {"F4": 1}

    def test_f0_when_the_turn_made_no_call(self):
        summary = self._run({}, "doc_a_c0", answer="no")

        assert summary.turn_results[0].failure_class == "F0"
        assert summary.failure_class_counts == {"F0": 1}

    def test_f1_when_the_query_missed_the_declared_token(self):
        summary = self._run(
            {
                "retrieved_memories": [],
                "retrieval_calls": [
                    {
                        "call_index": 1,
                        "query": "pressure code",
                        "episodes": [episode("doc_b_c0")],
                    }
                ],
                "retrieval_call_count": 1,
            },
            "doc_a_c0",
            answer="no",
            query_must_contain=["NBL-420"],
        )

        assert summary.turn_results[0].failure_class == "F1"

    def test_l2a_probes_keep_their_science_and_are_classified(self):
        """L2a never calls a tool, so a satisfied pin is F4 — not a bug, the
        baseline the L2b arm has to beat."""
        summary = self._run(
            {"retrieved_memories": [episode("doc_a_c0")]}, "doc_a_c0"
        )

        assert summary.turn_results[0].retrieval_call_count == 0
        assert summary.turn_results[0].failure_class == "F4"
        assert summary.retrieval_recall == 1.0


# ── Gross re-query cost ────────────────────────────────────────────────────


class TestGrossContextTokens:
    def test_duplicates_are_counted_twice(self):
        calls = [{"context_tokens": 100}, {"context_tokens": 100}]

        assert gross_context_tokens(calls) == 200

    def test_malformed_entries_contribute_nothing(self):
        calls = [None, "nope", {"context_tokens": "n/a"}, {"context_tokens": 7}]

        assert gross_context_tokens(calls) == 7

    def test_no_calls_is_zero(self):
        assert gross_context_tokens(None) == 0
        assert gross_context_tokens([]) == 0

    def test_telemetry_prefers_the_explicit_channel(self):
        telemetry = _read_l2_telemetry(
            {
                "retrieval_context_tokens_gross": 42,
                "retrieval_calls": [{"context_tokens": 1}],
            }
        )

        assert telemetry["retrieval_context_tokens_gross"] == 42

    def test_telemetry_falls_back_to_per_call_cost(self):
        telemetry = _read_l2_telemetry(
            {
                "retrieval_calls": [
                    {"context_tokens": 10},
                    {"context_tokens": 10},
                    {"context_tokens": 5},
                ]
            }
        )

        assert telemetry["retrieval_context_tokens_gross"] == 25

    def test_summary_aggregates_the_gross_cost(self):
        summary = BenchmarkRunner().run_single(
            _scenario_with_pin("doc_a_c0"),
            "baseline_react",
            "You are a test agent.",
            "fake-model",
            _ScriptedGraph(
                ["OK"],
                {
                    "retrieved_memories": [episode("doc_a_c0")],
                    "retrieval_calls": [
                        {"call_index": 1, "query": "q1", "context_tokens": 30},
                        {"call_index": 2, "query": "q2", "context_tokens": 30},
                    ],
                    "retrieval_call_count": 2,
                },
            ),
            "level_2_episodic_rag",
        )

        assert summary.turn_results[0].retrieval_context_tokens_gross == 60
        assert summary.retrieval_context_tokens_gross == 60


# ── Tool-loop budget (--max-tool-calls) ────────────────────────────────────


class TestMaxToolCallsConfig:
    def test_cli_default_is_three(self):
        from run import build_parser

        assert build_parser().parse_args([]).max_tool_calls == 3

    def test_cli_flag_is_parsed(self):
        from run import build_parser

        parsed = build_parser().parse_args(["--max-tool-calls", "5"])

        assert parsed.max_tool_calls == 5

    def test_base_class_declares_the_loop_flag(self):
        from agents.base import BaseBenchmarkAgent

        assert BaseBenchmarkAgent.supports_tool_calls is False

    def test_cap_reaches_the_graph_configurable(self):
        captured: list = []

        BenchmarkRunner().run_single(
            _scenario_with_pin("doc_a_c0"),
            "baseline_react",
            "You are a test agent.",
            "fake-model",
            _ScriptedGraph(["OK"], {}, captured=captured),
            "level_2b_agentic_rag",
            max_tool_calls=4,
        )

        assert captured[0]["configurable"]["max_tool_calls"] == 4

    def test_cap_is_absent_when_undeclared(self):
        """L0/L1/L2a must keep a byte-identical configurable."""
        captured: list = []

        BenchmarkRunner().run_single(
            _scenario_with_pin("doc_a_c0"),
            "baseline_react",
            "You are a test agent.",
            "fake-model",
            _ScriptedGraph(["OK"], {}, captured=captured),
            "level_2_episodic_rag",
        )

        assert "max_tool_calls" not in captured[0]["configurable"]

    def test_summary_records_the_cap(self):
        summary = BenchmarkRunner().run_single(
            _scenario_with_pin("doc_a_c0"),
            "baseline_react",
            "You are a test agent.",
            "fake-model",
            _ScriptedGraph(["OK"], {}),
            "level_2b_agentic_rag",
            max_tool_calls=3,
        )

        assert summary.max_tool_calls == 3

    def test_summary_defaults_the_cap_to_none(self):
        summary = BenchmarkRunSummary(
            run_id="r",
            scenario_id="s",
            model_id="m",
            prompt_id="baseline_react",
            agent_level="level_2_episodic_rag",
        )

        assert summary.max_tool_calls is None
        assert summary.failure_class_counts == {}
        assert summary.retrieval_context_tokens_gross == 0

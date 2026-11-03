"""Offline tests for L2 Milestone 3 harness extensions (plan §4.2/§4.3).

Covers the backward-compatible runner/schema extensions that Level 2 needs:
- ``_slice_token_usage`` — §4.2 token accounting buckets
- ``_read_l2_telemetry`` — guarded L2 channel reads (L0/L1 stay zeroed)
- per-session ``thread_id`` rotation and stable ``HumanMessage.id``
- telemetry capture into every ``TurnResult`` + §4.3 summary aggregation
- memory-backend snapshot export (before caller reset) and identity fields

All tests run against a ``FakeGraph`` — no network, no real models.
"""

import copy
import json

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage

from core.runner import BenchmarkRunner, _read_l2_telemetry, _slice_token_usage
from core.schemas import BenchmarkScenario, ConversationTurn


# ── Test doubles ─────────────────────────────────────────────────────────────


class FakeGraph:
    """Records every invoke; returns scripted responses + L2 channels."""

    def __init__(self, responses, l2_channels=None):
        self.responses = list(responses)
        self.l2_channels = l2_channels or {}
        self.calls: list[tuple[list, dict | None]] = []

    def invoke(self, state, config=None):
        # deepcopy: the runner mutates its config dict in place between
        # turns (session rotation), so a bare reference would show every
        # recorded call the FINAL state.
        self.calls.append((state["messages"], copy.deepcopy(config)))
        return {
            "messages": [AIMessage(content=self.responses.pop(0))],
            **self.l2_channels,
        }


class StubBackend:
    """EpisodicMemoryBackend test double recording snapshot exports."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.export_calls: list[tuple[object, dict]] = []
        self.reset_calls = 0

    def export_snapshot(self, dest_dir, run_metadata):
        if self.fail:
            raise RuntimeError("simulated export failure")
        self.export_calls.append((dest_dir, run_metadata))
        return dest_dir

    def reset(self):
        # Milestone 5: run_matrix resets the backend after every tuple for
        # isolation — the double records the calls so tests can assert them.
        self.reset_calls += 1


def make_turn(n, content, session_id="default", role="user"):
    return ConversationTurn(turn_number=n, role=role, content=content, session_id=session_id)


def make_scenario(turns, scenario_id="test_scenario"):
    return BenchmarkScenario(
        id=scenario_id,
        name="Test scenario",
        description="Milestone 3 harness test",
        category="unit_test",
        turns=turns,
    )


def run_scenario(graph, scenario, **kwargs):
    return BenchmarkRunner().run_single(
        scenario, "baseline_react", "You are a test agent.",
        "fake-model", graph, "1", **kwargs,
    )


L2_CHANNELS = {
    "retrieved_context_chars": 42,
    "retrieved_context_tokens": 10,
    "retrieval_latency_ms": 1.5,
    "append_latency_ms": 2.5,
    "embedding_latency_ms": 3.5,
    "embedding_tokens": 15,
    "retrieved_memories": [{"chunk_id": "ep_s1_t1", "rank": 1}],
    "appended_chunk_ids": ["ep_s2_t1"],
}

USAGE_WITH_DETAILS = {
    "input_tokens": 100,
    "output_tokens": 50,
    "total_tokens": 150,
    "input_token_details": {"cache_read": 80},
    "output_token_details": {"reasoning": 20},
}


# ── _slice_token_usage (§4.2) ─────────────────────────────────────────────────


class TestSliceTokenUsage:
    def test_full_detail_buckets(self):
        sliced = _slice_token_usage(USAGE_WITH_DETAILS)
        assert sliced == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "reasoning_tokens": 20,
            "visible_output_tokens": 30,  # 50 billed − 20 hidden reasoning
            "cached_prompt_tokens": 80,
        }

    def test_openai_style_aliases(self):
        sliced = _slice_token_usage(
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )
        assert sliced["prompt_tokens"] == 10
        assert sliced["completion_tokens"] == 5
        assert sliced["total_tokens"] == 15
        assert sliced["cached_prompt_tokens"] == 0

    def test_empty_usage_defaults_zero(self):
        assert _slice_token_usage({}) == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
            "visible_output_tokens": 0,
            "cached_prompt_tokens": 0,
        }

    def test_visible_output_floored_at_zero(self):
        sliced = _slice_token_usage(
            {"output_tokens": 10, "output_token_details": {"reasoning": 25}}
        )
        assert sliced["visible_output_tokens"] == 0


# ── _read_l2_telemetry (§4.2 guarded read) ────────────────────────────────────


class TestReadL2Telemetry:
    def test_reads_all_channels(self):
        tel = _read_l2_telemetry(dict(L2_CHANNELS))
        assert tel["retrieved_context_chars"] == 42
        assert tel["retrieved_context_tokens"] == 10
        assert tel["retrieval_latency_ms"] == pytest.approx(1.5)
        assert tel["append_latency_ms"] == pytest.approx(2.5)
        assert tel["embedding_latency_ms"] == pytest.approx(3.5)
        assert tel["embedding_tokens"] == 15
        assert tel["retrieved_memories"] == [{"chunk_id": "ep_s1_t1", "rank": 1}]
        assert tel["appended_chunk_ids"] == ["ep_s2_t1"]

    def test_missing_channels_default_zero(self):
        """L0/L1 graphs never write L2 channels — the read must not crash."""
        tel = _read_l2_telemetry({"messages": []})
        assert tel == {
            "retrieved_context_chars": 0,
            "retrieved_context_tokens": 0,
            "retrieval_latency_ms": 0.0,
            "append_latency_ms": 0.0,
            "embedding_latency_ms": 0.0,
            "embedding_tokens": 0,
            "retrieved_memories": [],
            "appended_chunk_ids": [],
        }

    def test_non_dict_result_zeroed(self):
        tel = _read_l2_telemetry(None)
        assert tel["retrieved_context_tokens"] == 0
        assert tel["appended_chunk_ids"] == []


# ── Per-session thread_id rotation + stable message ids ──────────────────────


class TestSessionRotation:
    def test_thread_id_rotates_per_session(self):
        graph = FakeGraph(["a", "b", "c"])
        scenario = make_scenario(
            [
                make_turn(1, "hello s1", session_id="s1"),
                make_turn(2, "more s1", session_id="s1"),
                make_turn(3, "hello s2", session_id="s2"),
            ]
        )
        run_scenario(graph, scenario)

        assert len(graph.calls) == 3
        thread_ids = [cfg["configurable"]["thread_id"] for _, cfg in graph.calls]
        sessions = [cfg["configurable"]["session_id"] for _, cfg in graph.calls]

        # Same session → same thread; session boundary → new thread.
        assert sessions == ["s1", "s1", "s2"]
        assert thread_ids[0] == thread_ids[1]
        assert thread_ids[2] != thread_ids[0]
        assert thread_ids[0].endswith("_test_scenario_s1")
        assert thread_ids[2].endswith("_test_scenario_s2")

    def test_default_session_keeps_legacy_thread_id(self):
        """Legacy single-session YAMLs must produce the pre-L2 thread_id."""
        graph = FakeGraph(["a"])
        run_scenario(graph, make_scenario([make_turn(1, "hello")]))
        _, cfg = graph.calls[0]
        assert cfg["configurable"]["session_id"] == "default"
        # Legacy form: run_id + scenario_id, WITHOUT a session suffix.
        assert cfg["configurable"]["thread_id"].endswith("_test_scenario")
        assert not cfg["configurable"]["thread_id"].endswith("_default")

    def test_user_message_ids_are_stable_per_turn(self):
        graph = FakeGraph(["a", "b"])
        run_scenario(
            graph,
            make_scenario([make_turn(1, "first"), make_turn(2, "second")]),
        )
        ids = [msgs[0].id for msgs, _ in graph.calls]
        assert ids == ["1_user", "2_user"]
        assert all(isinstance(msgs[0], HumanMessage) for msgs, _ in graph.calls)

    def test_system_event_turns_never_invoke_graph(self):
        graph = FakeGraph(["a", "b"])
        scenario = make_scenario(
            [
                make_turn(1, "fact drop"),
                make_turn(2, "[document attached]", role="system_event"),
                make_turn(3, "probe"),
            ]
        )
        summary = run_scenario(graph, scenario)
        assert len(graph.calls) == 2  # system_event skipped
        assert [tr.turn_number for tr in summary.turn_results] == [1, 3]


# ── Telemetry capture + §4.3 aggregation ─────────────────────────────────────


class UsageGraph(FakeGraph):
    """FakeGraph variant attaching usage metadata to every AI message."""

    def invoke(self, state, config=None):
        self.calls.append((state["messages"], config))
        return {
            "messages": [
                AIMessage(
                    content=self.responses.pop(0),
                    usage_metadata=dict(USAGE_WITH_DETAILS),
                )
            ],
            **self.l2_channels,
        }


class TestTelemetryCapture:
    def test_l2_channels_captured_in_turn_results(self):
        graph = UsageGraph(["answer 1", "answer 2"], l2_channels=dict(L2_CHANNELS))
        summary = run_scenario(
            graph, make_scenario([make_turn(1, "q1"), make_turn(2, "q2")])
        )

        assert len(summary.turn_results) == 2
        for tr in summary.turn_results:
            assert tr.retrieved_context_chars == 42
            assert tr.retrieved_context_tokens == 10
            assert tr.retrieval_latency_ms == pytest.approx(1.5)
            assert tr.append_latency_ms == pytest.approx(2.5)
            assert tr.embedding_latency_ms == pytest.approx(3.5)
            assert tr.embedding_tokens == 15
            assert tr.retrieved_memories == [{"chunk_id": "ep_s1_t1", "rank": 1}]
            assert tr.appended_chunk_ids == ["ep_s2_t1"]
            # §4.2 token slices
            assert tr.prompt_tokens == 100
            assert tr.completion_tokens == 50
            assert tr.total_tokens == 150
            assert tr.reasoning_tokens == 20
            assert tr.visible_output_tokens == 30  # 50 billed − 20 hidden
            assert tr.cached_prompt_tokens == 80
            assert tr.session_id == "default"
            assert tr.role == "user"

        # §4.3 aggregation: token-cost slices over ok turns (doubled)…
        assert summary.reasoning_tokens == 40
        assert summary.visible_output_tokens == 60
        assert summary.cached_prompt_tokens == 160
        assert summary.prompt_tokens == 200
        assert summary.completion_tokens == 100
        assert summary.total_tokens == 300
        # …while L2 telemetry channels sum over ALL turns.
        assert summary.retrieved_context_tokens == 20
        assert summary.embedding_tokens == 30
        assert summary.total_retrieval_latency_ms == pytest.approx(3.0)
        assert summary.total_append_latency_ms == pytest.approx(5.0)
        assert summary.total_embedding_latency_ms == pytest.approx(7.0)
        assert summary.ingestion_latency_ms == 0.0  # doc ingestion deferred

    def test_l0_style_graph_keeps_telemetry_zeroed(self):
        graph = FakeGraph(["plain answer"])
        summary = run_scenario(graph, make_scenario([make_turn(1, "q")]))
        tr = summary.turn_results[0]
        assert tr.retrieved_context_tokens == 0
        assert tr.retrieval_latency_ms == 0.0
        assert tr.append_latency_ms == 0.0
        assert tr.embedding_tokens == 0
        assert tr.retrieved_memories == []
        assert tr.appended_chunk_ids == []
        assert tr.reasoning_tokens == 0
        assert tr.visible_output_tokens == 0
        assert tr.cached_prompt_tokens == 0
        assert summary.memory_backend == "none"
        assert summary.memory_snapshot_path is None
        assert summary.embedding_model_id is None
        assert summary.total_retrieval_latency_ms == 0.0


# ── Memory-backend integration (snapshot export + §4.3 identity) ─────────────


class TestMemoryBackendIntegration:
    def test_snapshot_exported_and_path_recorded(self, tmp_path):
        backend = StubBackend()
        graph = FakeGraph(["answer 1", "answer 2"])
        summary = run_scenario(
            graph,
            make_scenario([make_turn(1, "q1"), make_turn(2, "q2")]),
            memory_backend=backend,
            memory_backend_config={
                "backend": "vector",
                "embedding_model_id": "qwen3-embedding-8b",
                "top_k": 5,
                "chunk_size": 1200,
                "chunk_overlap": 200,
            },
            snapshot_dir=tmp_path / "memory",
        )

        # Exactly one export, into snapshot_dir/<run_id>/ with full identity.
        assert len(backend.export_calls) == 1
        dest, metadata = backend.export_calls[0]
        assert dest == tmp_path / "memory" / summary.run_id
        assert metadata["run_id"] == summary.run_id
        assert metadata["scenario_id"] == "test_scenario"
        assert metadata["model_id"] == "fake-model"
        assert metadata["prompt_id"] == "baseline_react"
        assert metadata["agent_level"] == "1"

        # Summary records the relative path + full §4.3 identity.
        assert summary.memory_snapshot_path == f"memory/{summary.run_id}"
        assert summary.memory_backend == "vector"
        assert summary.embedding_model_id == "qwen3-embedding-8b"
        assert summary.top_k == 5
        assert summary.chunk_size == 1200
        assert summary.chunk_overlap == 200

    def test_backend_without_config_gets_unknown_name(self, tmp_path):
        summary = run_scenario(
            FakeGraph(["ok"]),
            make_scenario([make_turn(1, "q")]),
            memory_backend=StubBackend(),
            snapshot_dir=tmp_path / "memory",
        )
        assert summary.memory_backend == "unknown"
        assert summary.embedding_model_id is None
        assert summary.top_k is None

    def test_backend_without_snapshot_dir_skips_export(self):
        backend = StubBackend()
        summary = run_scenario(
            FakeGraph(["ok"]),
            make_scenario([make_turn(1, "q")]),
            memory_backend=backend,
        )
        assert backend.export_calls == []
        assert summary.memory_snapshot_path is None
        # Identity is still recorded — export and identity are independent.
        assert summary.memory_backend == "unknown"

    def test_export_failure_does_not_crash_run(self, tmp_path):
        summary = run_scenario(
            FakeGraph(["answer 1", "answer 2"]),
            make_scenario([make_turn(1, "q1"), make_turn(2, "q2")]),
            memory_backend=StubBackend(fail=True),
            snapshot_dir=tmp_path / "memory",
        )
        # The run completes and stays scorable; only the path is dropped.
        assert len(summary.turn_results) == 2
        assert all(tr.status == "ok" for tr in summary.turn_results)
        assert summary.memory_snapshot_path is None


# ── Snapshot ordering with the real M2 backend ──────────────────────────────


class _TinyFakeEmbedding(Embeddings):
    """Minimal deterministic embedding (2-D) — no ranking semantics needed."""

    def embed_documents(self, texts):
        return [[1.0, 0.0]] * len(texts)

    def embed_query(self, text):
        return [1.0, 0.0]


class TestSnapshotOrderingRealBackend:
    def test_snapshot_captures_state_before_caller_reset(self, tmp_path):
        """§4.3 ordering: export happens inside run_single, so the episodes
        on disk always reflect the run's final state — even when the caller
        resets the backend immediately after the call returns."""
        from agents.level_2_episodic_rag.backends import VectorMemoryBackend
        from agents.level_2_episodic_rag.backends.base import EpisodeRecord

        backend = VectorMemoryBackend(_TinyFakeEmbedding(), 2, embedding_id="tiny")
        backend.append(
            [
                EpisodeRecord.from_content(
                    chunk_id="ep_s1_t1",
                    session_id="s1",
                    turn_number=1,
                    source="chat",
                    content="the reactor uses helium cooling",
                )
            ]
        )

        summary = run_scenario(
            FakeGraph(["answer"]),
            make_scenario([make_turn(1, "q")]),
            memory_backend=backend,
            memory_backend_config={"backend": "vector"},
            snapshot_dir=tmp_path / "memory",
        )

        # Caller-side reset AFTER the run — the §4.3 policy.
        backend.reset()

        # The on-disk snapshot still holds the pre-reset episode state.
        snap_dir = tmp_path / "memory" / summary.run_id
        episodes = json.loads((snap_dir / "episodes.json").read_text(encoding="utf-8"))
        assert [e["chunk_id"] for e in episodes] == ["ep_s1_t1"]
        assert episodes[0]["content"] == "the reactor uses helium cooling"


# ── run_matrix: shared run directory for snapshots ───────────────────────────


class TestRunMatrixSharedRunDir:
    @staticmethod
    def _patched_runner(tmp_path, monkeypatch):
        runner = BenchmarkRunner(output_dir=tmp_path / "outputs")
        scenario = make_scenario([make_turn(1, "q1"), make_turn(2, "q2")])
        monkeypatch.setattr(
            runner, "load_scenarios", lambda filter_ids=None: [scenario]
        )
        monkeypatch.setattr(
            runner, "load_prompts", lambda: {"baseline_react": {"content": "sys"}}
        )
        monkeypatch.setattr(runner, "load_model_ids", lambda: ["fake-model"])
        return runner

    def test_shared_run_dir_created_and_snapshots_colocated(
        self, tmp_path, monkeypatch
    ):
        runner = self._patched_runner(tmp_path, monkeypatch)
        backend = StubBackend()

        results = runner.run_matrix(
            FakeGraph(["answer 1", "answer 2"]),
            "2",
            export=False,  # CLI batches levels — export comes later
            memory_backend=backend,
            memory_backend_config={"backend": "vector"},
        )

        assert len(results) == 1
        run_dir = runner._current_run_dir
        assert run_dir is not None
        assert run_dir.exists()
        assert run_dir.parent == tmp_path / "outputs"

        # The snapshot landed inside the shared run dir, next to where the
        # final results export will be written by _export_results.
        dest, metadata = backend.export_calls[0]
        assert dest == run_dir / "memory" / results[0].run_id
        assert results[0].memory_snapshot_path == f"memory/{results[0].run_id}"

    def test_no_backend_leaves_run_dir_unset(self, tmp_path, monkeypatch):
        runner = self._patched_runner(tmp_path, monkeypatch)
        results = runner.run_matrix(FakeGraph(["ok", "ok"]), "0", export=False)
        assert len(results) == 1
        assert runner._current_run_dir is None
        assert results[0].memory_backend == "none"





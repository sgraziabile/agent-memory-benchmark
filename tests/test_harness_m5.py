"""Offline tests for L2 Milestone 5 — runner + CLI integration (plan §4.3).

Covers the wiring that makes the L2 vector backend reachable end to end:
- ``run_single`` injecting ``memory_backend`` + ``top_k`` into
  ``config["configurable"]`` (L0/L1 configurable untouched — backward compat)
- ``run_matrix`` resetting the backend after EVERY tuple, with the snapshot
  export (inside ``run_single``) always happening BEFORE the reset
- ``memory_backend`` / ``embedding_model_id`` columns in the summary CSV
- ``requires_memory_backend`` class attribute (True on L2, False default)
- CLI: ``--backend`` / ``--embedding-model`` flags, backend-choice
  validation, and ``VectorMemoryBackend`` construction via
  ``create_embeddings`` / ``get_embedding_dims``

All tests run against doubles — no network, no real models, no API keys.
"""

import copy
import csv
import importlib.util
import sys
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage

from agents import AGENT_REGISTRY
from agents.base import BaseBenchmarkAgent
from agents.level_2_episodic_rag.backends.vector import VectorMemoryBackend
from agents.level_2_episodic_rag.graph import Level2EpisodicRAGAgent
from core.runner import BenchmarkRunner
from core.schemas import BenchmarkScenario, ConversationTurn

# ── Test doubles ─────────────────────────────────────────────────────────────


class FakeGraph:
    """Records every invoke; returns scripted responses.

    The runner hands the SAME backend instance to the graph by reference
    (never a copy) — a ``deepcopy`` of the recorded config would clone it,
    so backend identity is recorded on a separate list.
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[tuple[list, dict | None]] = []
        self.backends: list[object] = []
        self.top_ks: list[object] = []

    def invoke(self, state, config=None):
        # deepcopy: the runner mutates its config dict in place between
        # turns, so a bare reference would show every recorded call the
        # FINAL state.
        self.calls.append((state["messages"], copy.deepcopy(config)))
        configurable = (config or {}).get("configurable", {})
        self.backends.append(configurable.get("memory_backend"))
        self.top_ks.append(configurable.get("top_k"))
        return {"messages": [AIMessage(content=self.responses.pop(0))]}


class EventsBackend:
    """Backend double logging export/reset ordering into a shared list."""

    def __init__(self, events):
        self.events = events

    def export_snapshot(self, dest_dir, run_metadata):
        self.events.append(("export", str(dest_dir)))
        return dest_dir

    def reset(self):
        self.events.append("reset")


class FakeEmbeddings(Embeddings):
    """Deterministic offline embeddings double."""

    def embed_documents(self, texts):
        return [[float(len(t)), 1.0] for t in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0]


def make_turn(n, content, session_id="default"):
    return ConversationTurn(turn_number=n, role="user", content=content, session_id=session_id)


def make_scenario(turns, scenario_id="test_scenario"):
    return BenchmarkScenario(
        id=scenario_id,
        name="Test scenario",
        description="Milestone 5 integration test",
        category="unit_test",
        turns=turns,
    )


def run_scenario(graph, scenario, **kwargs):
    return BenchmarkRunner().run_single(
        scenario, "baseline_react", "You are a test agent.",
        "fake-model", graph, "2", **kwargs,
    )


def patched_runner(tmp_path, monkeypatch, model_ids=("fake-model",)):
    runner = BenchmarkRunner(output_dir=tmp_path / "outputs")
    scenario = make_scenario([make_turn(1, "q1"), make_turn(2, "q2")])
    monkeypatch.setattr(runner, "load_scenarios", lambda filter_ids=None: [scenario])
    monkeypatch.setattr(
        runner, "load_prompts", lambda: {"baseline_react": {"content": "sys"}}
    )
    monkeypatch.setattr(runner, "load_model_ids", lambda: list(model_ids))
    return runner


# ── run_single: backend + top_k injection into configurable ─────────────────


class TestBackendInjection:
    def test_backend_and_top_k_injected_into_configurable(self):
        backend = EventsBackend(events=[])
        graph = FakeGraph(["answer 1", "answer 2"])
        run_scenario(
            graph,
            make_scenario([make_turn(1, "q1"), make_turn(2, "q2")]),
            memory_backend=backend,
            memory_backend_config={"backend": "vector", "top_k": 3},
        )

        for recorded_backend, recorded_top_k in zip(graph.backends, graph.top_ks):
            assert recorded_backend is backend
            assert recorded_top_k == 3

    def test_no_top_k_in_configurable_when_config_omits_it(self):
        # The graph then falls back to its own DEFAULT_TOP_K — the runner
        # never fabricates a value the caller did not configure.
        backend = EventsBackend(events=[])
        graph = FakeGraph(["answer"])
        run_scenario(
            graph,
            make_scenario([make_turn(1, "q")]),
            memory_backend=backend,
            memory_backend_config={"backend": "vector"},
        )
        assert "top_k" not in graph.calls[0][1]["configurable"]
        assert all(tk is None for tk in graph.top_ks)

    def test_l0_l1_configurable_unchanged_without_backend(self):
        # Backward compatibility: no backend → no memory keys at all.
        graph = FakeGraph(["answer"])
        run_scenario(graph, make_scenario([make_turn(1, "q")]))
        configurable = graph.calls[0][1]["configurable"]
        assert "memory_backend" not in configurable
        assert "top_k" not in configurable


# ── run_matrix: reset-after-tuple isolation with export-before-reset order ───


class TestRunMatrixReset:
    def test_reset_after_each_tuple_and_export_before_reset(self, tmp_path, monkeypatch):
        runner = patched_runner(tmp_path, monkeypatch, model_ids=["m1", "m2"])
        events: list = []
        backend = EventsBackend(events)

        results = runner.run_matrix(
            FakeGraph(["a", "b", "c", "d"]),
            "2",
            export=False,
            memory_backend=backend,
            memory_backend_config={"backend": "vector", "top_k": 5},
        )

        assert len(results) == 2
        # Exact ordering: each tuple's snapshot export (inside run_single,
        # before return) precedes the caller-side reset.
        assert [e if isinstance(e, str) else e[0] for e in events] == [
            "export",
            "reset",
            "export",
            "reset",
        ]
        assert all(r.memory_backend == "vector" for r in results)

    def test_failed_tuple_still_resets_backend(self, tmp_path, monkeypatch):
        runner = patched_runner(tmp_path, monkeypatch)
        events: list = []
        backend = EventsBackend(events)

        # A tuple-level failure (outside run_single's per-turn error
        # handling) — e.g. an unexpected loader/export bug — must still
        # reset the backend before the next tuple.
        def exploding_run_single(**kwargs):
            raise RuntimeError("simulated tuple failure")

        monkeypatch.setattr(runner, "run_single", exploding_run_single)

        results = runner.run_matrix(
            FakeGraph([]),
            "2",
            export=False,
            memory_backend=backend,
            memory_backend_config={"backend": "vector"},
        )

        # The tuple failed → no summary, but the backend was still reset
        # (finally-block) so the next tuple starts from a clean store.
        assert results == []
        assert events == ["reset"]

    def test_no_backend_no_reset_needed(self, tmp_path, monkeypatch):
        # L0/L1 behavior is untouched — nothing to reset, identity "none".
        runner = patched_runner(tmp_path, monkeypatch)
        results = runner.run_matrix(FakeGraph(["a", "b"]), "0", export=False)
        assert len(results) == 1
        assert results[0].memory_backend == "none"


# ── ensure_shared_run_dir: idempotent, consumed by _export_results ───────────


class TestEnsureSharedRunDir:
    def test_idempotent_and_consumed_by_export(self, tmp_path):
        runner = BenchmarkRunner(output_dir=tmp_path / "outputs")
        first = runner.ensure_shared_run_dir()
        assert runner.ensure_shared_run_dir() == first  # idempotent
        assert first.exists()

        summary = run_scenario(FakeGraph(["ok"]), make_scenario([make_turn(1, "q")]))
        run_dir = runner._export_results([summary])
        assert run_dir == first  # export lands in the shared directory
        assert runner._current_run_dir is None  # consumed

        # After consumption a fresh call re-arms the shared directory (same
        # second-resolution timestamp is possible, so only the state machine
        # is asserted — None → set → consumed → set again).
        second = runner.ensure_shared_run_dir()
        assert runner._current_run_dir == second
        assert second.exists()


# ── CSV export: memory_backend merge-identity columns ───────────────────────


class TestCsvExportColumns:
    def test_summary_csv_has_memory_identity_columns(self, tmp_path):
        runner = BenchmarkRunner(output_dir=tmp_path / "outputs")

        l2 = run_scenario(
            FakeGraph(["ok", "ok"]),
            make_scenario([make_turn(1, "q1"), make_turn(2, "q2")]),
            memory_backend=EventsBackend(events=[]),
            memory_backend_config={
                "backend": "vector",
                "embedding_model_id": "qwen3-embedding-8b",
                "top_k": 5,
            },
        )
        l0 = run_scenario(FakeGraph(["ok"]), make_scenario([make_turn(1, "q")]))

        run_dir = runner._export_results([l2, l0])
        with open(run_dir / "summary.csv", encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))

        assert len(rows) == 2
        l2_row, l0_row = rows
        assert l2_row["memory_backend"] == "vector"
        assert l2_row["embedding_model_id"] == "qwen3-embedding-8b"
        assert l0_row["memory_backend"] == "none"
        assert l0_row["embedding_model_id"] == ""

    def test_memory_backend_column_directly_after_agent_level(self, tmp_path):
        runner = BenchmarkRunner(output_dir=tmp_path / "outputs")
        summary = run_scenario(FakeGraph(["ok"]), make_scenario([make_turn(1, "q")]))
        run_dir = runner._export_results([summary])
        with open(run_dir / "summary.csv", encoding="utf-8-sig", newline="") as fh:
            header = next(csv.reader(fh))
        idx = header.index("agent_level")
        assert header[idx + 1] == "memory_backend"
        assert header[idx + 2] == "embedding_model_id"


# ── requires_memory_backend class attribute ──────────────────────────────────


class TestRequiresMemoryBackendFlag:
    def test_base_default_false_and_l2_true(self):
        assert BaseBenchmarkAgent.requires_memory_backend is False
        assert Level2EpisodicRAGAgent.requires_memory_backend is True

    def test_only_l2_requires_backend_in_registry(self):
        l2_keys = {
            key
            for key, cls in AGENT_REGISTRY.items()
            if cls is Level2EpisodicRAGAgent
        }
        assert l2_keys, "L2 agent must be registered"
        for key, cls in AGENT_REGISTRY.items():
            assert cls.requires_memory_backend is (key in l2_keys), key

    def test_l2_instance_inherits_the_flag(self):
        assert AGENT_REGISTRY["level_2_episodic_rag"]().requires_memory_backend is True


# ── CLI: --backend / --embedding-model flags and helpers ─────────────────────


@pytest.fixture(scope="module")
def run_cli():
    """Load run.py as a module (it is a script, not an installed package)."""
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location(
        "benchmark_run_cli", root / "run.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestResolveBackendChoice:
    def test_vector_is_valid(self, run_cli):
        assert run_cli._resolve_backend_choice("vector") == "vector"

    def test_whitespace_tolerated(self, run_cli):
        assert run_cli._resolve_backend_choice("  vector ") == "vector"

    @pytest.mark.parametrize("raw", ["", "   ", ",", " , "])
    def test_empty_is_rejected(self, run_cli, raw):
        with pytest.raises(ValueError):
            run_cli._resolve_backend_choice(raw)

    def test_multi_backend_reserved_rejected(self, run_cli):
        with pytest.raises(ValueError, match="not supported yet"):
            run_cli._resolve_backend_choice("vector,markdown")

    def test_unknown_backend_rejected(self, run_cli):
        with pytest.raises(ValueError, match="not implemented yet"):
            run_cli._resolve_backend_choice("markdown")


class TestCliFlags:
    def test_backend_defaults_to_vector(self, run_cli):
        assert run_cli.build_parser().parse_args([]).backend == "vector"

    def test_embedding_model_defaults_to_qwen(self, run_cli):
        args = run_cli.build_parser().parse_args([])
        assert args.embedding_model == "qwen3-embedding-8b"

    def test_backend_short_flag(self, run_cli):
        args = run_cli.build_parser().parse_args(["-b", "vector"])
        assert args.backend == "vector"

    def test_embedding_model_flag(self, run_cli):
        args = run_cli.build_parser().parse_args(["--embedding-model", "bge-m3"])
        assert args.embedding_model == "bge-m3"


class TestBuildMemoryBackend:
    def test_constructs_vector_backend_with_registry_embeddings(
        self, run_cli, monkeypatch
    ):
        fake_embeddings = FakeEmbeddings()
        seen: dict = {}

        def fake_create_embeddings(embedding_id):
            seen["create"] = embedding_id
            return fake_embeddings

        def fake_get_dims(embedding_id):
            seen["dims"] = embedding_id
            return 8

        monkeypatch.setattr(run_cli, "create_embeddings", fake_create_embeddings)
        monkeypatch.setattr(run_cli, "get_embedding_dims", fake_get_dims)

        backend = run_cli._build_memory_backend("vector", "qwen3-embedding-8b")

        assert isinstance(backend, VectorMemoryBackend)
        assert seen == {"create": "qwen3-embedding-8b", "dims": "qwen3-embedding-8b"}
        # Provenance identity recorded for snapshot manifests.
        assert backend._embedding_id == "qwen3-embedding-8b"
        assert backend._dims == 8




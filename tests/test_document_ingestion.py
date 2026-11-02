"""Offline tests for plan WS-D — documents as a memory-ingestion channel.

Covers the harness half of document memory:
- ``DocumentSpec`` validation (exactly one of ``path`` / ``content``)
- path resolution (scenario directory → ``datasets/documents/`` → error)
- scenario-level corpus preload (once, before turn 1, ``session="corpus"``,
  ``turn_number=0``, ``source="document"``)
- ``system_event`` ingestion (attachments + inline content, chat model never
  invoked, telemetry-only ``TurnResult``)
- metric aggregation (``system_event`` turns excluded from assertion
  denominators and per-turn latency; ``ingestion_latency_ms`` reported)
- L0/L1 behaviour (no backend → documents unavailable, run still scorable)
- ``--no-documents`` control path
- snapshot provenance (``documents.json``, ``by_source``, ``sha256``) and
  ``corpus_stats()`` dilution reporting

All tests run against doubles or ``FakeEmbeddings`` — no network, no models,
no API keys.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage

from agents.level_2_episodic_rag.backends.base import AppendTelemetry
from agents.level_2_episodic_rag.backends.vector import VectorMemoryBackend
from core.documents import (
    CORPUS_SESSION_ID,
    build_episode_records,
    load_document,
    resolve_document_path,
)
from core.runner import BenchmarkRunner
from core.schemas import (
    AssertionRule,
    BenchmarkScenario,
    ConversationTurn,
    DocumentSpec,
)

# ── Test doubles ─────────────────────────────────────────────────────────────


class FakeEmbeddings(Embeddings):
    """Deterministic 2-D embeddings double (no length bias in ranking)."""

    def embed_documents(self, texts):
        return [[1.0, 0.0]] * len(texts)

    def embed_query(self, text):
        return [1.0, 0.0]


class RecordingBackend:
    """Backend double recording appends/exports and returning telemetry.

    ``events`` (optional) is shared with the graph double so tests can assert
    ordering — e.g. that the corpus is appended BEFORE the first chat turn.
    """

    def __init__(self, events: list | None = None):
        self.events = events if events is not None else []
        self.appended: list = []
        self.append_latencies: list[float] = []
        self.export_calls: list[tuple[object, dict]] = []
        self.reset_calls = 0

    def append(self, records):
        chunk_ids = [record.chunk_id for record in records]
        self.appended.extend(records)
        self.append_latencies.append(5.0)
        self.events.append(("append", chunk_ids))
        return AppendTelemetry(
            chunk_ids=chunk_ids,
            append_latency_ms=5.0,
            embedding_latency_ms=3.0,
            embedding_tokens=7 * len(records),
        )

    def export_snapshot(self, dest_dir, run_metadata):
        self.export_calls.append((dest_dir, run_metadata))
        return dest_dir

    def reset(self):
        self.reset_calls += 1


class FakeGraph:
    """Records every invoke and returns scripted responses."""

    def __init__(self, responses, events: list | None = None):
        self.responses = list(responses)
        self.events = events if events is not None else []
        self.calls: list[tuple[list, dict | None]] = []

    def invoke(self, state, config=None):
        self.calls.append((state["messages"], config))
        self.events.append(("invoke", state["messages"][0].content))
        return {"messages": [AIMessage(content=self.responses.pop(0))]}


class RetrievalGraph(FakeGraph):
    """FakeGraph variant emitting L2 retrieval telemetry."""

    def __init__(self, responses, memories):
        super().__init__(responses)
        self.memories = memories

    def invoke(self, state, config=None):
        self.calls.append((state["messages"], config))
        return {
            "messages": [AIMessage(content=self.responses.pop(0))],
            "retrieved_memories": self.memories,
        }


def make_turn(
    n,
    content,
    session_id="default",
    role="user",
    attachments=None,
    assertions=None,
):
    return ConversationTurn(
        turn_number=n,
        role=role,
        content=content,
        session_id=session_id,
        attachments=attachments or [],
        assertions=assertions or [],
    )


def make_scenario(turns, documents=None, scenario_id="doc_test"):
    return BenchmarkScenario(
        id=scenario_id,
        name="Document ingestion test",
        description="WS-D harness test",
        category="unit_test",
        turns=turns,
        documents=documents or [],
    )


def run_scenario(graph, scenario, **kwargs):
    return BenchmarkRunner(documents_dir=kwargs.pop("documents_dir", ".")).run_single(
        scenario=scenario,
        prompt_id="baseline_react",
        prompt_content="You are a test agent.",
        model_id="fake-model",
        agent_graph=graph,
        agent_level="level_2_episodic_rag",
        **kwargs,
    )


# ── DocumentSpec contract ────────────────────────────────────────────────────


class TestDocumentSpec:
    def test_path_only_is_valid(self):
        spec = DocumentSpec(doc_id="d1", path="policy.md")
        assert spec.path == "policy.md"
        assert spec.content is None

    def test_content_only_is_valid(self):
        spec = DocumentSpec(doc_id="d1", content="hello")
        assert spec.content == "hello"

    def test_path_and_content_rejected(self):
        with pytest.raises(ValueError, match="exactly one"):
            DocumentSpec(doc_id="d1", path="policy.md", content="hello")

    def test_neither_rejected(self):
        with pytest.raises(ValueError, match="exactly one"):
            DocumentSpec(doc_id="d1")

    def test_blank_content_rejected(self):
        with pytest.raises(ValueError, match="exactly one"):
            DocumentSpec(doc_id="d1", content="   ")

    def test_attachments_rejected_on_user_turns(self):
        with pytest.raises(ValueError, match="only valid on"):
            ConversationTurn(
                turn_number=1,
                role="user",
                content="hi",
                attachments=["policy_v1.md"],
            )

    def test_attachments_allowed_on_system_events(self):
        turn = ConversationTurn(
            turn_number=2,
            role="system_event",
            content="notice",
            attachments=["addendum_1.md"],
        )
        assert turn.attachments == ["addendum_1.md"]

    def test_scenario_documents_default_empty(self):
        scenario = make_scenario([make_turn(1, "hi")])
        assert scenario.documents == []
        assert scenario.source_path is None

    def test_source_path_excluded_from_dump(self):
        scenario = make_scenario([make_turn(1, "hi")])
        scenario.source_path = "C:/secret/path/scenario.yaml"
        assert "source_path" not in scenario.model_dump()


# ── Resolution / loading ─────────────────────────────────────────────────────


class TestDocumentResolution:
    def test_resolves_relative_to_scenario_dir_first(self, tmp_path):
        scenario_dir = tmp_path / "scenario"
        docs_dir = tmp_path / "documents"
        scenario_dir.mkdir()
        docs_dir.mkdir()
        (scenario_dir / "policy.md").write_text("local copy", encoding="utf-8")
        (docs_dir / "policy.md").write_text("shared copy", encoding="utf-8")

        resolved = resolve_document_path(
            "policy.md", scenario_dir=scenario_dir, documents_dir=docs_dir
        )
        assert resolved.read_text(encoding="utf-8") == "local copy"

    def test_falls_back_to_shared_documents_dir(self, tmp_path):
        scenario_dir = tmp_path / "scenario"
        docs_dir = tmp_path / "documents"
        scenario_dir.mkdir()
        docs_dir.mkdir()
        (docs_dir / "policy.md").write_text("shared copy", encoding="utf-8")

        resolved = resolve_document_path(
            "policy.md", scenario_dir=scenario_dir, documents_dir=docs_dir
        )
        assert resolved.read_text(encoding="utf-8") == "shared copy"

    def test_absolute_path_used_verbatim(self, tmp_path):
        target = tmp_path / "policy.md"
        target.write_text("absolute", encoding="utf-8")
        resolved = resolve_document_path(
            str(target), scenario_dir=tmp_path, documents_dir=tmp_path
        )
        assert resolved == target

    def test_missing_document_lists_searched_locations(self, tmp_path):
        with pytest.raises(FileNotFoundError) as excinfo:
            resolve_document_path(
                "ghost.md", scenario_dir=tmp_path, documents_dir=tmp_path / "documents"
            )
        message = str(excinfo.value)
        assert "ghost.md" in message
        assert "datasets/documents" in message

    def test_load_inline_document_has_no_source_path(self):
        loaded = load_document(DocumentSpec(doc_id="d1", content="inline text"))
        assert loaded.text == "inline text"
        assert loaded.source_path is None
        assert loaded.char_count == len("inline text")
        assert len(loaded.sha256) == 64

    def test_load_blank_file_rejected(self, tmp_path):
        (tmp_path / "blank.md").write_text("   \n", encoding="utf-8")
        with pytest.raises(ValueError, match="is empty"):
            load_document(
                DocumentSpec(doc_id="blank", path="blank.md"),
                scenario_id="scenario_x",
                documents_dir=tmp_path,
            )

    def test_committed_fixture_resolves(self):
        """The repository's own corpus must resolve out of the box."""
        loaded = load_document(
            DocumentSpec(doc_id="policy_v1", path="policy_v1.md"),
            scenario_id="document_needle_01",
        )
        assert "iris-breakglass-77" in loaded.text
        assert loaded.source_path is not None
        assert loaded.source_path.name == "policy_v1.md"


# ── Record building ─────────────────────────────────────────────────────────


class TestRecordBuilding:
    def test_preload_ids_and_precedence(self):
        documents = [
            load_document(DocumentSpec(doc_id="policy_v1", content="policy text")),
            load_document(DocumentSpec(doc_id="addendum", content="addendum text")),
        ]
        records = build_episode_records(
            documents, session_id=CORPUS_SESSION_ID, turn_number=0
        )
        assert [r.chunk_id for r in records] == ["doc_policy_v1", "doc_addendum"]
        assert all(r.source == "document" for r in records)
        assert all(r.session_id == "corpus" for r in records)
        assert all(r.turn_number == 0 for r in records)

    def test_event_ids_include_turn_number(self):
        records = build_episode_records(
            [load_document(DocumentSpec(doc_id="addendum_1", content="x"))],
            session_id="s1",
            turn_number=2,
        )
        assert records[0].chunk_id == "doc_addendum_1_t2"
        assert records[0].turn_number == 2
        assert records[0].session_id == "s1"

    def test_duplicate_doc_ids_get_deterministic_suffix(self):
        documents = [
            load_document(DocumentSpec(doc_id="dup", content="first")),
            load_document(DocumentSpec(doc_id="dup", content="second")),
        ]
        records = build_episode_records(
            documents, session_id=CORPUS_SESSION_ID, turn_number=0
        )
        assert [r.chunk_id for r in records] == ["doc_dup", "doc_dup_2"]


# ── Scenario-level preload ───────────────────────────────────────────────────


class TestCorpusPreload:
    def test_preload_happens_before_first_turn(self):
        events: list = []
        backend = RecordingBackend(events)
        graph = FakeGraph(["answer"], events=events)
        scenario = make_scenario(
            [make_turn(1, "probe")],
            documents=[DocumentSpec(doc_id="policy", content="the policy text")],
        )

        run_scenario(graph, scenario, memory_backend=backend)

        assert events[0] == ("append", ["doc_policy"])
        assert events[1][0] == "invoke"

    def test_preload_metrics_reported_separately(self):
        backend = RecordingBackend()
        graph = FakeGraph(["answer"])
        scenario = make_scenario(
            [make_turn(1, "probe")],
            documents=[
                DocumentSpec(doc_id="policy", content="the policy text"),
                DocumentSpec(doc_id="handbook", content="the handbook text"),
            ],
        )

        summary = run_scenario(graph, scenario, memory_backend=backend)

        assert summary.documents_ingested == 2
        assert summary.ingested_chars == len("the policy text") + len("the handbook text")
        assert summary.ingested_chunks == 2
        assert summary.ingestion_latency_ms == 5.0
        # Preload embedding cost is folded into the run totals.
        assert summary.total_embedding_latency_ms == 3.0
        assert summary.embedding_tokens == 14
        # ...but it never enters the per-turn chat latency average.
        chat_latency = sum(tr.latency_ms for tr in summary.turn_results)
        assert summary.total_latency_ms == pytest.approx(chat_latency)

    def test_preload_records_are_document_sourced(self):
        backend = RecordingBackend()
        scenario = make_scenario(
            [make_turn(1, "probe")],
            documents=[DocumentSpec(doc_id="policy", content="text")],
        )

        run_scenario(FakeGraph(["a"]), scenario, memory_backend=backend)

        record = backend.appended[0]
        assert record.source == "document"
        assert record.session_id == CORPUS_SESSION_ID
        assert record.turn_number == 0
        assert record.metadata["sha256"]

    def test_missing_document_fails_the_run(self, tmp_path):
        scenario = make_scenario(
            [make_turn(1, "probe")],
            documents=[DocumentSpec(doc_id="ghost", path="ghost.md")],
        )
        with pytest.raises(FileNotFoundError, match="ghost.md"):
            run_scenario(
                FakeGraph(["a"]),
                scenario,
                memory_backend=RecordingBackend(),
                documents_dir=str(tmp_path),
            )


# ── system_event ingestion ───────────────────────────────────────────────────


class TestSystemEventIngestion:
    def _scenario(self):
        return make_scenario(
            [
                make_turn(1, "before the addendum", session_id="s1"),
                make_turn(
                    2,
                    "Addendum 1 supersedes section 3.2.",
                    session_id="s1",
                    role="system_event",
                    attachments=["addendum.md"],
                ),
                make_turn(3, "after the addendum", session_id="s1"),
            ]
        )

    def test_ingests_attachments_and_inline_content_without_invoking_graph(
        self, tmp_path
    ):
        (tmp_path / "addendum.md").write_text(
            "Section 3.2 is superseded.", encoding="utf-8"
        )
        backend = RecordingBackend()
        graph = FakeGraph(["a", "b"])

        summary = run_scenario(
            graph,
            self._scenario(),
            memory_backend=backend,
            documents_dir=str(tmp_path),
        )

        # The chat model is never called for the event turn.
        assert len(graph.calls) == 2
        event_results = [tr for tr in summary.turn_results if tr.role == "system_event"]
        assert len(event_results) == 1
        event_result = event_results[0]
        assert event_result.turn_number == 2
        assert event_result.status == "ok"
        assert event_result.ingested_documents == ["evt_s1", "addendum"]
        assert event_result.ingested_chunk_ids == ["doc_evt_s1_t2", "doc_addendum_t2"]
        assert event_result.embedding_tokens == 14
        assert event_result.append_latency_ms == 5.0
        # Provenance: one record per file, one for the inline notice.
        assert [r.source for r in backend.appended] == ["document", "document"]
        assert {r.turn_number for r in backend.appended} == {2}

    def test_event_turns_excluded_from_assertion_denominators(self, tmp_path):
        (tmp_path / "addendum.md").write_text("superseded", encoding="utf-8")
        rules = [{"type": "must_contain", "value": "a"}]
        scenario = make_scenario(
            [
                make_turn(1, "q1", assertions=rules),
                make_turn(
                    2,
                    "notice",
                    role="system_event",
                    attachments=["addendum.md"],
                ),
                make_turn(3, "q2", assertions=rules),
            ]
        )

        summary = run_scenario(
            FakeGraph(["answer one", "answer two"]),
            scenario,
            memory_backend=RecordingBackend(),
            documents_dir=str(tmp_path),
        )

        assert summary.total_turns == 3  # the event is recorded for observability
        assert summary.total_assertions == 2  # ...but asserts nothing
        assert summary.passed_assertions == 2
        assert summary.ingestion_latency_ms == 5.0
        chat_latency = sum(
            tr.latency_ms for tr in summary.turn_results if tr.role == "user"
        )
        assert summary.total_latency_ms == pytest.approx(chat_latency)


# ── L0/L1 behaviour and the --no-documents control path ──────────────────────


class TestLevelsWithoutBackend:
    def test_declared_documents_are_ignored_without_a_backend(self):
        graph = FakeGraph(["answer"])
        scenario = make_scenario(
            [make_turn(1, "probe")],
            documents=[DocumentSpec(doc_id="policy", content="policy text")],
        )

        summary = run_scenario(graph, scenario)  # L0/L1: no backend

        assert summary.memory_backend == "none"
        assert summary.documents_ingested == 0
        assert summary.ingested_chars == 0
        assert summary.ingested_chunks == 0
        assert summary.ingestion_latency_ms == 0.0
        # The conversation still runs and stays scorable (probes fail by design).
        assert len(graph.calls) == 1
        assert summary.total_turns == 1

    def test_system_event_is_a_no_op_without_a_backend(self):
        graph = FakeGraph(["answer"])
        scenario = make_scenario(
            [
                make_turn(1, "q"),
                make_turn(
                    2,
                    "notice",
                    role="system_event",
                    attachments=["policy_v1.md"],
                ),
                make_turn(3, "q2"),
            ]
        )

        summary = run_scenario(graph, scenario)

        # Historical behaviour preserved: no TurnResult, no graph call.
        assert [tr.turn_number for tr in summary.turn_results] == [1, 3]
        assert len(graph.calls) == 2
        assert summary.documents_ingested == 0

    def test_documents_enabled_false_skips_both_channels(self):
        backend = RecordingBackend()
        graph = FakeGraph(["answer"])
        scenario = make_scenario(
            [
                make_turn(1, "q"),
                make_turn(2, "notice", role="system_event"),
            ],
            documents=[DocumentSpec(doc_id="policy", content="policy text")],
        )

        summary = run_scenario(
            graph, scenario, memory_backend=backend, documents_enabled=False
        )

        assert backend.append_latencies == []
        assert summary.documents_ingested == 0
        assert [tr.turn_number for tr in summary.turn_results] == [1]


# ── Real vector backend: snapshots, provenance, dilution stats ───────────────


class TestVectorBackendDocuments:
    def _backend(self) -> VectorMemoryBackend:
        return VectorMemoryBackend(
            FakeEmbeddings(), 2, embedding_id="fake", chunk_size=40, chunk_overlap=5
        )

    def test_snapshot_exports_documents_and_provenance(self, tmp_path):
        backend = self._backend()
        long_text = "telemetry port 7443. " * 10
        scenario = make_scenario(
            [make_turn(1, "which port?")],
            documents=[
                DocumentSpec(
                    doc_id="handbook",
                    content=long_text,
                    metadata={"title": "Ops handbook"},
                )
            ],
        )

        summary = run_scenario(
            FakeGraph(["port 7443"]),
            scenario,
            memory_backend=backend,
            memory_backend_config={"backend": "vector", "top_k": 5},
            snapshot_dir=tmp_path / "memory",
        )

        snapshot_dir = tmp_path / "memory" / summary.run_id
        documents = json.loads(
            (snapshot_dir / "documents.json").read_text(encoding="utf-8")
        )
        assert documents
        assert all(d["source"] == "document" for d in documents)
        assert all(d["metadata"]["doc_id"] == "handbook" for d in documents)
        assert all(len(d["metadata"]["sha256"]) == 64 for d in documents)

        manifest = json.loads(
            (snapshot_dir / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["by_source"]["document"] == len(documents)
        assert manifest["document_chunks"] == len(documents)
        assert manifest["run_metadata"]["documents"][0]["doc_id"] == "handbook"
        assert manifest["run_metadata"]["document_ingestion"]["documents_ingested"] == 1

        # Summary mirrors the store: the long document was split into chunks.
        assert summary.ingested_chunks == len(documents) > 1
        assert summary.documents_ingested == 1
        assert summary.corpus_chunks == len(documents)
        assert summary.corpus_chars is not None
        assert summary.corpus_chars >= len(long_text)

    def test_corpus_stats_reports_sources(self):
        backend = self._backend()
        scenario = make_scenario(
            [make_turn(1, "q")],
            documents=[DocumentSpec(doc_id="handbook", content="short doc")],
        )

        run_scenario(FakeGraph(["a"]), scenario, memory_backend=backend)

        stats = backend.corpus_stats()
        assert stats["total_chunks"] == 1
        assert stats["document_chunks"] == 1
        assert stats["chat_chunks"] == 0
        assert stats["document_chars"] == len("short doc")

    def test_stub_backend_yields_none_corpus_stats(self):
        summary = run_scenario(
            FakeGraph(["a"]),
            make_scenario(
                [make_turn(1, "q")],
                documents=[DocumentSpec(doc_id="d", content="text")],
            ),
            memory_backend=RecordingBackend(),
        )
        assert summary.corpus_chunks is None
        assert summary.corpus_chars is None


# ── Retrieval assertions (plan WS-D §Q4) ─────────────────────────────────────


class TestRetrievalAssertions:
    @staticmethod
    def _episodes(*chunk_sources):
        return [
            {"chunk_id": chunk_id, "source": source}
            for chunk_id, source in chunk_sources
        ]

    def test_chunk_assertion_matches_subchunks(self):
        rules = [AssertionRule(type="must_retrieve_chunk", value="doc_policy_v1")]
        details = BenchmarkRunner.evaluate_assertions(
            "",
            rules,
            retrieved_memories=self._episodes(("doc_policy_v1_c2", "document")),
        )
        assert details[0]["passed"] is True
        assert details[0]["matched_chunks"] == ["doc_policy_v1_c2"]

    def test_chunk_assertion_fails_when_absent(self):
        rules = [AssertionRule(type="must_retrieve_chunk", value="doc_needle")]
        details = BenchmarkRunner.evaluate_assertions(
            "",
            rules,
            retrieved_memories=self._episodes(("ep_s1_t1", "chat")),
        )
        assert details[0]["passed"] is False
        assert details[0]["matched_chunks"] == []

    def test_negative_chunk_assertion(self):
        rules = [AssertionRule(type="must_not_retrieve_chunk", value="doc_secret")]
        details = BenchmarkRunner.evaluate_assertions(
            "answer",
            rules,
            retrieved_memories=self._episodes(("ep_s1_t1", "chat")),
        )
        assert details[0]["passed"] is True

    def test_source_assertions(self):
        rules = [
            AssertionRule(type="must_retrieve_source", value="document"),
            AssertionRule(type="must_not_retrieve_source", value="chat"),
        ]
        details = BenchmarkRunner.evaluate_assertions(
            "answer",
            rules,
            retrieved_memories=self._episodes(("doc_a", "document")),
        )
        assert [d["passed"] for d in details] == [True, True]

    def test_missing_telemetry_is_deterministic(self):
        """L0/L1 retrieve nothing: positive rules fail, negative rules pass."""
        rules = [
            AssertionRule(type="must_retrieve_chunk", value="doc_a"),
            AssertionRule(type="must_not_retrieve_chunk", value="doc_a"),
        ]
        details = BenchmarkRunner.evaluate_assertions("answer", rules)
        assert [d["passed"] for d in details] == [False, True]

    def test_retrieval_assertion_end_to_end(self):
        rules = [
            AssertionRule(type="must_contain", value="iris-breakglass-77"),
            AssertionRule(type="must_retrieve_chunk", value="doc_policy_v1"),
        ]
        scenario = make_scenario(
            [make_turn(1, "break-glass id?", assertions=rules)]
        )
        memories = [{"chunk_id": "doc_policy_v1_c3", "source": "document"}]

        summary = run_scenario(
            RetrievalGraph(["iris-breakglass-77"], memories), scenario
        )

        assert summary.total_assertions == 2
        assert summary.passed_assertions == 2
        assert summary.turn_results[0].assertion_details[1]["matched_chunks"] == [
            "doc_policy_v1_c3"
        ]

    def test_retrieval_assertion_fails_when_nothing_retrieved(self):
        rules = [AssertionRule(type="must_retrieve_chunk", value="doc_policy_v1")]
        scenario = make_scenario([make_turn(1, "q", assertions=rules)])

        summary = run_scenario(RetrievalGraph(["answer"], []), scenario)

        assert summary.passed_assertions == 0
        assert summary.failed_assertions == 1


# ── Committed scenarios ──────────────────────────────────────────────────────


class TestDocumentScenarios:
    """Shape checks for the four committed WS-D scenarios."""

    @staticmethod
    def _load():
        runner = BenchmarkRunner()
        return {sc.id: sc for sc in runner.load_scenarios()}

    def test_all_document_scenarios_load(self):
        scenarios = self._load()
        for scenario_id in (
            "document_needle_01",
            "document_addendum_revision_01",
            "document_dilution_01",
            "document_cross_thread_01",
        ):
            assert scenario_id in scenarios, f"{scenario_id} not found in datasets/"
            assert scenarios[scenario_id].source_path is not None

    def test_document_needle_shape(self):
        scenario = self._load()["document_needle_01"]
        # X5: the document corpus must exceed the default top_k, otherwise
        # recall is guaranteed by construction and the scenario measures
        # ingestion plumbing rather than retrieval.
        assert [d.doc_id for d in scenario.documents] == ["policy_v1", "ops_handbook"]
        assert scenario.documents[0].path == "policy_v1.md"
        assert scenario.metadata["default_top_k"] < scenario.metadata["document_chunks_approx"]
        # Probes live in a later session so L1 cannot answer them.
        graded = [t for t in scenario.turns if t.assertions]
        assert graded and all(t.session_id == "s2" for t in graded)
        # X5/X8: every fact turn pins the policy chunks; the precision turn inverts.
        fact_turns = [t for t in graded if any(a.type == "must_contain" for a in t.assertions)]
        assert fact_turns
        for turn in fact_turns:
            assert any(
                a.type == "must_retrieve_chunk" and a.value == "doc_policy_v1"
                for a in turn.assertions
            ), f"turn {turn.turn_number} does not pin the policy chunk"
        assert any(
            a.type == "must_not_contain" and a.value == "iris-breakglass-77"
            for t in graded
            for a in t.assertions
        )

    def test_document_addendum_revision_shape(self):
        scenario = self._load()["document_addendum_revision_01"]
        events = [t for t in scenario.turns if t.role == "system_event"]
        assert len(events) == 1
        assert events[0].attachments == ["addendum_1.md"]
        # Graded turns exist both before and after the ingestion event.
        assert any(t.assertions and t.turn_number < events[0].turn_number for t in scenario.turns)
        assert any(t.assertions and t.turn_number > events[0].turn_number for t in scenario.turns)
        # The event must not be the final turn (that would grade nothing).
        assert scenario.turns[-1].role != "system_event"

    def test_document_dilution_shape(self):
        scenario = self._load()["document_dilution_01"]
        assert len(scenario.documents) == 9
        assert all(d.content and d.content.strip() for d in scenario.documents)
        target = next(d for d in scenario.documents if d.doc_id == scenario.metadata["target_document"]["doc_id"])
        assert scenario.metadata["target_document"]["serial"] in target.content
        assert scenario.metadata["default_top_k"] < len(scenario.documents)

    def test_document_cross_thread_shape(self):
        scenario = self._load()["document_cross_thread_01"]
        assert [d.doc_id for d in scenario.documents] == ["ops_handbook"]
        # s1 must contain no facts at all: every assertion is in s2.
        assert all(not t.assertions for t in scenario.turns if t.session_id == "s1")
        assert all(t.assertions for t in scenario.turns if t.session_id == "s2")
        # X8: at least one precision probe (identifiers must not leak unprompted).
        assert any(
            a.type == "must_not_contain" for t in scenario.turns for a in t.assertions
        ), "X8 precision probe missing"

    # ── Regression tests for the X1-X8 defect fixes (stress plan §4) ────

    def test_policy_v1_reboot_level_has_a_single_source_of_truth(self):
        """X1: §2 must not restate the reboot level.

        With the roles table also asserting an approval level for reboots, T1
        ("Level-1") and T3 ("Level-2") were both satisfiable without the
        addendum, which made the supersession test unmeasurable.
        """
        text = (
            Path(__file__).resolve().parents[1]
            / "datasets"
            / "documents"
            / "policy_v1.md"
        ).read_text(encoding="utf-8")
        assert "Required for production-tier reboots" not in text
        section_32 = text.split("### 3.2", 1)[1].split("### 3.3", 1)[0]
        assert "Level-1" in section_32
        assert "Level-2" not in section_32

    def test_document_addendum_pins_the_superseding_chunk_after_ingestion(self):
        """X2: every graded turn after the system_event retrieves the addendum."""
        scenario = self._load()["document_addendum_revision_01"]
        event_turn = next(t for t in scenario.turns if t.role == "system_event")
        after = [
            t
            for t in scenario.turns
            if t.assertions and t.turn_number > event_turn.turn_number
        ]
        assert after
        for turn in after:
            assert any(
                a.type == "must_retrieve_chunk" and "addendum" in a.value
                for a in turn.assertions
            ), f"turn {turn.turn_number} does not pin the addendum chunk"

    def test_document_addendum_has_a_negation_robust_probe(self):
        """X3/X4: one addendum-only fact, in s2, that no chat turn can supply."""
        scenario = self._load()["document_addendum_revision_01"]
        addendum_only = "2026-10-01"
        turns = [
            t
            for t in scenario.turns
            if any(a.value == addendum_only for a in t.assertions)
        ]
        assert len(turns) == 1
        turn = turns[0]
        assert turn.session_id == "s2"
        # No user turn may leak the fact: only the document can answer it.
        assert all(
            addendum_only not in t.content
            for t in scenario.turns
            if t.role == "user"
        )
        # Rule R1: never assert the absence of the superseded value.
        assert all(
            not (a.type == "must_not_contain" and "Level-1" in a.value)
            for t in scenario.turns
            for a in t.assertions
        )

    def test_document_dilution_decoys_share_the_probe_phrase(self):
        """X7: near-miss decoys carry the probe phrase with a different serial."""
        scenario = self._load()["document_dilution_01"]
        by_id = {d.doc_id: d for d in scenario.documents}
        phrase = scenario.metadata["target_document"]["probe_paraphrase"]
        serial = scenario.metadata["target_document"]["serial"]
        for doc_id in scenario.metadata["near_miss_decoys"]:
            assert phrase in by_id[doc_id].content, doc_id
            assert serial not in by_id[doc_id].content, doc_id
        assert phrase in by_id[scenario.metadata["target_document"]["doc_id"]].content

    def test_document_dilution_derived_probe_is_not_answerable_from_chat(self):
        """X9: a graded probe must not be answerable by the thread alone.

        T3 used to ask for the room of the unit T2 had just named, so the
        checkpointer answered it while `must_retrieve_chunk` failed (run
        20261102_151134: room3 absent from the top-5, answer still correct).
        The replacement probes a fact stated only in the Room 5 memo.
        """
        scenario = self._load()["document_dilution_01"]
        turn = next(t for t in scenario.turns if t.turn_number == 3)
        earlier = " ".join(
            t.content for t in scenario.turns if t.turn_number < turn.turn_number
        ).lower()
        near_miss = scenario.metadata["near_miss_probe"]
        assert "redundant conditioning loop" not in earlier
        assert near_miss["serial"].lower() not in earlier
        assert near_miss["room"].lower() not in earlier
        assert any(
            a.type == "must_retrieve_chunk" and near_miss["doc_id"] in a.value
            for a in turn.assertions
        ), "T3 must pin the near-miss memo chunk"

    def test_document_dilution_grades_every_probe_and_checks_abstention(self):
        """X6/X10: fact turns carry >=2 assertions, plus both absence probes."""
        scenario = self._load()["document_dilution_01"]
        graded = [t for t in scenario.turns if t.assertions]
        assert len(graded) >= 4
        fact_turns = [
            t for t in graded if any(a.type == "must_contain" for a in t.assertions)
        ]
        assert fact_turns
        for turn in fact_turns:
            assert len(turn.assertions) >= 2, f"turn {turn.turn_number} is under-asserted"
        # X10a: refusing must not dump inventory (negative probe).
        assert any(
            a.type == "must_not_contain" and a.value == "UPS-"
            for t in graded
            for a in t.assertions
        ), "over-disclosure probe missing"
        # X10b: the absence is also asserted positively, from a corpus-exclusive token.
        scope = scenario.metadata["scope_probe"]
        scope_turn = next(t for t in graded if t.turn_number == scope["turn"])
        assert any(
            a.type == "must_contain" and a.value == scope["token"]
            for a in scope_turn.assertions
        ), "scope probe must assert the corpus-exclusive ticket id"
        assert any(
            a.type == "must_retrieve_chunk" and scope["doc_id"] in a.value
            for a in scope_turn.assertions
        ), "scope probe must pin the scope-note chunk"

    def test_no_positive_assertion_echoes_its_own_probe(self):
        """R10: a `must_contain` token must not appear in its own user turn.

        An assertion satisfied by echoing the question measures nothing. This is
        why ``must_contain: "Room 11"`` was rejected as an abstention check
        (X10b): every answer, hallucinated or not, would have passed.
        """
        for scenario in self._load().values():
            if not scenario.id.startswith("document_"):
                continue
            for turn in scenario.turns:
                if turn.role != "user":
                    continue
                probe = turn.content.casefold()
                for rule in turn.assertions:
                    if rule.type != "must_contain":
                        continue
                    assert rule.value.casefold() not in probe, (
                        f"{scenario.id} T{turn.turn_number}: assertion "
                        f"{rule.value!r} is satisfied by the probe itself"
                    )

    def test_documents_are_not_chat_turns(self):
        """Documents must be ingested out-of-band, never as user turns."""
        for scenario in self._load().values():
            if not scenario.documents:
                continue
            for doc in scenario.documents:
                if doc.content is None:
                    continue
                assert not any(doc.content.strip() == t.content.strip() for t in scenario.turns)

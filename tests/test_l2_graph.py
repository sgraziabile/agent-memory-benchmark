"""Offline tests for L2 Milestone 4: the episodic Naive-RAG graph.

Covers TODOs/phase-2-l2-vector-implementation-plan.md Milestone 4:
- cross-session recall: a fact appended in session ``s1`` is retrieved and
  injected into the model payload of session ``s2`` (the core L2 capability)
- prefix-cache-preserving injection: the RAG SystemMessage is visible to
  the model but never checkpointed into thread state
- same-session chat exclusion: ``retrieve_node`` hides the active session's
  own chat episodes (already in-context via MemorySaver)
- append-only persistence: formatted turns land in the backend with stable
  ``ep_{session}_t{turn}`` chunk IDs; telemetry channels flow back
- no-backend degradation: L2 runs without a backend with zeroed telemetry
- Zero Hardcoding: model_id / top_k / session_id arrive via ``configurable``

Fully offline: a recording ``FakeChatModel`` stands in for the chat model
(monkeypatching ``create_model``), and a deterministic bag-of-words
embedding feeds a real ``VectorMemoryBackend``.
"""

import zlib

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agents import AGENT_REGISTRY
from agents.level_2_episodic_rag.backends.vector import VectorMemoryBackend
from agents.level_2_episodic_rag.graph import (
    DEFAULT_TOP_K,
    _MEMORY_SEPARATOR,
    _format_memory_block,
    Level2EpisodicRAGAgent,
)

DIMS = 32


class DeterministicFakeEmbedding(Embeddings):
    """Deterministic bag-of-words hashing embedding (stable across runs)."""

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * DIMS
        for word in text.lower().split():
            vec[zlib.crc32(word.encode("utf-8")) % DIMS] += 1.0
        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec] if norm else vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def injected_memory_blocks(messages: list[BaseMessage]) -> list[str]:
    """Return the injected memory block(s): ``SystemMessage``s after the prompt.

    ``model_node`` builds ``[system_prompt] + history + [memory block] + latest
    human``, so the block is every ``SystemMessage`` that is not the leading
    system prompt. Detection is positional on purpose: the formatter emits no
    header/prefix (``_MEMORY_SEPARATOR`` in the L2 graph).
    """
    return [
        str(message.content)
        for index, message in enumerate(messages)
        if index > 0 and isinstance(message, SystemMessage)
    ]


class FakeChatModel(BaseChatModel):
    """Records every payload and echoes the memory block it was shown.

    The response quotes any injected memory ``SystemMessage`` content so tests
    can assert exactly what the model saw: ``RECALL: <memory block>`` when a
    block was injected, ``ECHO: <latest human turn>`` otherwise.

    The block is identified **positionally** (a ``SystemMessage`` that is not
    the leading system prompt), never by a header prefix — the formatter emits
    no per-episode header on purpose (``_MEMORY_SEPARATOR`` in the graph).
    """

    payloads: list[list[BaseMessage]] = []
    model_ids: list[str] = []
    # Declared field (BaseChatModel is Pydantic — undeclared attrs are rejected).
    recorded_model_ids: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "fake-l2-test"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop=None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs,
    ) -> ChatResult:
        self.payloads.append(list(messages))

        memory_blocks = injected_memory_blocks(messages)
        latest_human = next(
            (m for m in reversed(messages) if isinstance(m, HumanMessage))
        )
        if memory_blocks:
            text = "RECALL: " + " || ".join(memory_blocks)
        else:
            text = "ECHO: " + str(latest_human.content)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=text))]
        )


@pytest.fixture
def fake_model(monkeypatch):
    """Patched create_model returning a fresh recording FakeChatModel."""
    model = FakeChatModel()
    model.payloads = []
    model.model_ids = []
    recorded_ids: list[str] = []

    def fake_create(model_id, **kwargs):
        recorded_ids.append(model_id)
        return model

    monkeypatch.setattr(
        "agents.level_2_episodic_rag.graph.create_model", fake_create
    )
    # Expose recorded ids for Zero-Hardcoding assertions.
    model.recorded_model_ids = recorded_ids  # type: ignore[attr-defined]
    return model


def make_backend() -> VectorMemoryBackend:
    return VectorMemoryBackend(
        DeterministicFakeEmbedding(), DIMS, embedding_id="fake-test"
    )


def make_config(backend, session: str, thread: str, top_k: int | None = None):
    configurable = {
        "model_id": "fake-model-id",
        "system_prompt": "You are a benchmark agent.",
        "thread_id": thread,
        "session_id": session,
        "memory_backend": backend,
    }
    if top_k is not None:
        configurable["top_k"] = top_k
    return {"configurable": configurable}


def last_ai_text(result: dict) -> str:
    return next(
        str(m.content) for m in reversed(result["messages"])
        if isinstance(m, AIMessage)
    )


# ── Cross-session recall (the core L2 capability) ────────────────────────────


class TestCrossSessionRecall:
    def test_fact_from_s1_recalled_in_s2(self, fake_model):
        backend = make_backend()
        graph = Level2EpisodicRAGAgent().build_graph()

        # Session 1: user states a fact.
        graph.invoke(
            {"messages": [HumanMessage(content="My favorite reactor is the DELTA-7749")]},
            make_config(backend, "s1", "thread_s1"),
        )
        assert "ep_s1_t1" in backend._chunks

        # Session 2 (different thread, same backend): probe for the fact.
        result = graph.invoke(
            {"messages": [HumanMessage(content="What is my favorite reactor?")]},
            make_config(backend, "s2", "thread_s2"),
        )

        # The model saw the s1 episode in an injected memory SystemMessage...
        payload_text = " ".join(str(m.content) for m in fake_model.payloads[-1])
        assert "DELTA-7749" in payload_text
        assert injected_memory_blocks(fake_model.payloads[-1])
        # ...and the AI response (echoing the block) contains it.
        assert "DELTA-7749" in last_ai_text(result)

        # Zero Hardcoding: the model_id from configurable reached create_model.
        assert fake_model.recorded_model_ids == ["fake-model-id", "fake-model-id"]

        # Telemetry channels flowed back through the runner-facing result.
        assert result["retrieved_memories"]
        assert result["retrieved_context_chars"] > 0
        assert result["retrieval_latency_ms"] >= 0.0
        assert result["appended_chunk_ids"] == ["ep_s2_t1"]

    def test_top_k_from_configurable_limits_retrieval(self, fake_model):
        backend = make_backend()
        graph = Level2EpisodicRAGAgent().build_graph()

        # Two distinct facts in s1.
        graph.invoke(
            {"messages": [HumanMessage(content="reactor fact alpha DELTA-7749")]},
            make_config(backend, "s1", "thread_s1"),
        )
        graph.invoke(
            {"messages": [HumanMessage(content="reactor fact beta GAMMA-2211")]},
            make_config(backend, "s1", "thread_s1"),
        )

        # s2 probes with top_k=1: exactly one memory may be injected.
        result = graph.invoke(
            {"messages": [HumanMessage(content="reactor fact?")]},
            make_config(backend, "s2", "thread_s2", top_k=1),
        )
        assert len(result["retrieved_memories"]) == 1

        # Default top_k matches the plan's reference config (§5.1).
        assert DEFAULT_TOP_K == 5


# ── Memory-block hygiene: no provenance may reach the prompt ─────────────────


class TestMemoryBlockHygiene:
    """The injected block must not hand the model the experiment's variables.

    Rank, ``chunk_id``, ``session_id``, ``turn_number``, ``source`` and ``score``
    stay in ``retrieved_memories`` telemetry and in the snapshots; the prompt gets
    content only (see ``_MEMORY_SEPARATOR`` in the L2 graph).
    """

    @staticmethod
    def _probe(fake_model):
        backend = make_backend()
        graph = Level2EpisodicRAGAgent().build_graph()
        graph.invoke(
            {"messages": [HumanMessage(content="My favorite reactor is the DELTA-7749")]},
            make_config(backend, "s1", "thread_s1"),
        )
        result = graph.invoke(
            {"messages": [HumanMessage(content="What is my favorite reactor?")]},
            make_config(backend, "s2", "thread_s2"),
        )
        blocks = injected_memory_blocks(fake_model.payloads[-1])
        assert len(blocks) == 1
        return blocks[0], result

    def test_block_is_content_only(self, fake_model):
        block, result = self._probe(fake_model)
        expected = _MEMORY_SEPARATOR.join(
            episode["content"] for episode in result["retrieved_memories"]
        )
        assert block == expected

    def test_no_provenance_fields_leak_into_the_prompt(self, fake_model):
        block, _ = self._probe(fake_model)
        for marker in ("score=", "session=", "turn=", "source=", "id=", "[Memory"):
            assert marker not in block, f"{marker!r} leaked into the prompt"
        # The chunk_id encodes session/turn as well (ep_s1_t1, doc_x_c2), so the
        # identifier itself must not appear either.
        assert "ep_s1_t1" not in block

    def test_formatter_emits_no_header(self):
        episodes = [
            {
                "rank": 1,
                "chunk_id": "doc_policy_v1_c2",
                "session_id": "corpus",
                "turn_number": 0,
                "source": "document",
                "score": 0.9123,
                "content": "alpha",
            },
            {
                "rank": 2,
                "chunk_id": "ep_s1_t2",
                "session_id": "s1",
                "turn_number": 2,
                "source": "chat",
                "score": 0.4567,
                "content": "beta",
            },
        ]
        assert _format_memory_block(episodes) == f"alpha{_MEMORY_SEPARATOR}beta"
        assert _format_memory_block([]) == ""


# ── Prefix-cache-preserving ephemeral injection ──────────────────────────────


class TestEphemeralInjection:
    def test_rag_system_message_not_checkpointed(self, fake_model):
        backend = make_backend()
        graph = Level2EpisodicRAGAgent().build_graph()
        cfg_s1 = make_config(backend, "s1", "thread_s1")

        graph.invoke(
            {"messages": [HumanMessage(content="reactor fact alpha DELTA-7749")]},
            cfg_s1,
        )
        graph.invoke(
            {"messages": [HumanMessage(content="reactor fact beta GAMMA-2211")]},
            cfg_s1,
        )

        # The checkpointer state must contain ONLY Human/AI messages —
        # the ephemeral memory SystemMessage never persists.
        checkpointed = graph.get_state(cfg_s1).values["messages"]
        assert injected_memory_blocks(checkpointed) == []
        # Thread state grew by exactly 4 messages (2 human + 2 AI).
        assert len(checkpointed) == 4

    def test_payload_layout_system_first_latest_human_last(self, fake_model):
        backend = make_backend()
        graph = Level2EpisodicRAGAgent().build_graph()

        # Seed one episode from s1 so s2 gets a memory block.
        graph.invoke(
            {"messages": [HumanMessage(content="reactor fact alpha DELTA-7749")]},
            make_config(backend, "s1", "thread_s1"),
        )
        graph.invoke(
            {"messages": [HumanMessage(content="What is the reactor fact?")]},
            make_config(backend, "s2", "thread_s2"),
        )

        payload = fake_model.payloads[-1]
        # First payload slot: the system prompt (stable prefix start).
        assert isinstance(payload[0], SystemMessage)
        assert payload[0].content == "You are a benchmark agent."
        # Last payload slot: the latest human turn.
        assert isinstance(payload[-1], HumanMessage)
        assert "reactor fact" in payload[-1].content


# ── Same-session chat exclusion ───────────────────────────────────────────────


class TestSessionExclusion:
    def test_own_session_episodes_not_injected(self, fake_model):
        backend = make_backend()
        graph = Level2EpisodicRAGAgent().build_graph()
        cfg_s1 = make_config(backend, "s1", "thread_s1")

        graph.invoke(
            {"messages": [HumanMessage(content="reactor secret alpha DELTA-7749")]},
            cfg_s1,
        )
        result = graph.invoke(
            {"messages": [HumanMessage(content="reactor secret?")]},
            cfg_s1,
        )

        # Telemetry is the ground truth: retrieval excluded s1's own chat
        # episodes, so nothing was retrieved at all (the backend holds only s1
        # chats) and no memory block may be injected. Checking the block by a
        # header prefix is no longer possible by design — the block has none.
        assert result["retrieved_memories"] == []
        assert injected_memory_blocks(fake_model.payloads[-1]) == []


# ── No-backend degradation ───────────────────────────────────────────────────


class TestNoBackendDegradation:
    """L2 must run without a backend attached (pure L1-style in-context)."""

    def test_no_backend_zeroed_telemetry_no_injection(self, fake_model):
        graph = Level2EpisodicRAGAgent().build_graph()
        configurable = {
            "model_id": "fake-model-id",
            "system_prompt": "You are a benchmark agent.",
            "thread_id": "thread_nb",
            "session_id": "s1",
        }
        result = graph.invoke(
            {"messages": [HumanMessage(content="reactor fact alpha DELTA-7749")]},
            {"configurable": configurable},
        )

        # The turn still completes — the model answered via echo.
        assert "DELTA-7749" in last_ai_text(result)

        # No memory block was injected anywhere (no backend attached).
        assert injected_memory_blocks(fake_model.payloads[-1]) == []

        # All telemetry channels are zeroed, not missing.
        assert result["retrieved_memories"] == []
        assert result["retrieved_context_chars"] == 0
        assert result["retrieved_context_tokens"] == 0
        assert result["retrieval_latency_ms"] == 0.0
        assert result["appended_chunk_ids"] == []
        assert result["append_latency_ms"] == 0.0
        assert result["embedding_latency_ms"] == 0.0
        assert result["embedding_tokens"] == 0
        assert result["embedding_cache_hits"] == 0
        assert result["embedding_cache_misses"] == 0


# ── Ingestion-cache telemetry channels ───────────────────────────────────────


class TestCacheTelemetryChannels:
    """The per-turn cache counters must be overwrite, never cumulative.

    ``embedding_cache_hits``/``_misses`` are reset by ``retrieve_node`` and
    accumulated by ``append_node`` (mirroring ``embedding_tokens``). If the
    reset were missing, the checkpointed thread state would carry turn N's
    counter into turn N+1 and a run total would be a triangular number
    (1 + 2 + … + n) instead of the real chunk count — measured once as 151
    instead of 123 on the 8-document D16 cell (2026-11-04).
    """

    def test_counters_do_not_accumulate_across_turns(self, fake_model, tmp_path):
        from core.embedding_cache import (
            CachedEmbeddings,
            EmbeddingCache,
            cache_path_for,
        )

        cache = EmbeddingCache(
            cache_path_for(tmp_path, "fake-test"), model_id="fake/model", dims=DIMS
        )
        client = CachedEmbeddings(
            DeterministicFakeEmbedding(), cache, model_id="fake/model", dims=DIMS
        )
        backend = VectorMemoryBackend(client, DIMS, embedding_id="fake-test")
        graph = Level2EpisodicRAGAgent().build_graph()

        counters = []
        for turn in range(1, 4):
            result = graph.invoke(
                {
                    "messages": [
                        HumanMessage(content=f"turn {turn}: my reactor is DELTA-{turn}")
                    ]
                },
                make_config(backend, "s1", "thread_cache"),
            )
            counters.append(
                (result["embedding_cache_hits"], result["embedding_cache_misses"])
            )

        # One new chat chunk per turn → exactly one miss, never a running sum.
        assert counters == [(0, 1), (0, 1), (0, 1)]





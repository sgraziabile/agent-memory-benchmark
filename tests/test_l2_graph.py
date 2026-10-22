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
from agents.level_2_episodic_rag.graph import DEFAULT_TOP_K, Level2EpisodicRAGAgent

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


class FakeChatModel(BaseChatModel):
    """Records every payload and echoes the memory block it was shown.

    The response quotes any ``[Memory ...]`` SystemMessage content so tests
    can assert exactly what the model saw: ``RECALL: <memory block>`` when
    a block was injected, ``ECHO: <latest human turn>`` otherwise.
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

        memory_blocks = [
            str(m.content) for m in messages
            if isinstance(m, SystemMessage) and str(m.content).startswith("[Memory")
        ]
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

        # The model saw the s1 episode in an injected [Memory] SystemMessage...
        payload_text = " ".join(str(m.content) for m in fake_model.payloads[-1])
        assert "DELTA-7749" in payload_text
        assert any(
            isinstance(m, SystemMessage) and str(m.content).startswith("[Memory")
            for m in fake_model.payloads[-1]
        )
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
        # the ephemeral [Memory] SystemMessage never persists.
        checkpointed = graph.get_state(cfg_s1).values["messages"]
        assert not any(
            isinstance(m, SystemMessage) and str(m.content).startswith("[Memory")
            for m in checkpointed
        )
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
        graph.invoke(
            {"messages": [HumanMessage(content="reactor secret?")]},
            cfg_s1,
        )

        # Turn 2's payload must NOT contain s1's own chat episodes.
        payload_text = " ".join(str(m.content) for m in fake_model.payloads[-1])
        assert "ep_s1_t1" not in payload_text
        # And no memory block was injected at all (backend holds only s1 chats).
        assert not any(
            isinstance(m, SystemMessage) and str(m.content).startswith("[Memory")
            for m in fake_model.payloads[-1]
        )


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

        # No [Memory] block was injected anywhere.
        assert not any(
            isinstance(m, SystemMessage) and str(m.content).startswith("[Memory")
            for m in fake_model.payloads[-1]
        )

        # All telemetry channels are zeroed, not missing.
        assert result["retrieved_memories"] == []
        assert result["retrieved_context_chars"] == 0
        assert result["retrieved_context_tokens"] == 0
        assert result["retrieval_latency_ms"] == 0.0
        assert result["appended_chunk_ids"] == []
        assert result["append_latency_ms"] == 0.0
        assert result["embedding_latency_ms"] == 0.0
        assert result["embedding_tokens"] == 0



"""agents.level_2_episodic_rag.graph — Level 2: Episodic Naive-RAG Agent.

Implements **Level 2** as a fixed LangGraph chain over a swappable
:class:`~agents.level_2_episodic_rag.backends.base.EpisodicMemoryBackend`::

    START ──► retrieve_node ──► model_node ──► append_node ──► END

- **retrieve_node** — queries the backend with the current user turn,
  excluding the current session's chat episodes (in-context history already
  covers them via the ``MemorySaver`` checkpointer) but never excluding
  ``source="document"`` chunks.
- **model_node** — prefix-cache-preserving injection:
  ``[SystemMessage(system_prompt), *prior_thread_history,
  SystemMessage(retrieved_memories_block), latest_human_message]``.
  The RAG ``SystemMessage`` is **ephemeral** — it is shown to the model but
  never written back into the thread state, so the checkpointer stays clean
  and the shared prefix ``S + H1 + A1 + …`` is byte-stable across turns
  (prefix-cache friendly).
- **append_node** — persists the formatted turn
  (``User: {input}\\nAssistant: {response}``) as a ``source="chat"``
  ``EpisodeRecord`` and overwrites the per-turn telemetry channels.

Memory topology (TODOs/phase-2-l2-plan.md §2):
- **YES** ``MemorySaver`` checkpointer — intra-session conversation history
  (per-session ``thread_id`` supplied by the runner).
- **YES** cross-thread long-term store — the episodic backend survives
  across sessions/threads (the very capability L2 is benchmarked on).

The backend instance is injected at runtime through
``config["configurable"]["memory_backend"]`` (Zero Hardcoding: no model,
prompt, backend, or retrieval parameter is bound at build time). Per-turn
telemetry flows back through the ``L2AgentState`` overwrite channels
(``retrieved_context_chars`` … ``appended_chunk_ids``), which
``BenchmarkRunner.run_single`` reads guardedly — L0/L1 never set them.

Routing is a fixed chain — no ``Command`` objects or
``add_conditional_edges()`` are needed; nodes return plain state updates
and the transitions are declared once in ``build_graph()``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.base import BaseBenchmarkAgent
from agents.level_2_episodic_rag.backends.base import EpisodeRecord, estimate_tokens
from core.model_factory import create_model
from core.schemas import L2AgentState

logger = logging.getLogger(__name__)

#: Default retrieval cut-off when ``configurable["top_k"]`` is absent
#: (matches the reference config in TODOs/phase-2-l2-plan.md §5.1).
DEFAULT_TOP_K = 5


# ── Memory-block formatting ──────────────────────────────────────────────────


def _format_memory_block(episodes: list[dict[str, Any]]) -> str:
    """Format retrieved episodes as the injected ``[Memory ...]`` block.

    Each episode becomes a provenance header followed by its content, per
    the plan's §3.3 format::

        [Memory #1 | id=ep_s1_t2 | session=s1 | turn=2 | source=chat | score=0.83]
        <episode content>

    An empty episode list formats to ``""`` — ``model_node`` then skips the
    ``SystemMessage`` injection entirely (no wasted tokens on empty blocks).

    The same formatter is used by ``retrieve_node`` (telemetry sizing) and
    ``model_node`` (injection), so the reported
    ``retrieved_context_chars`` is always the exact block length the model
    saw.
    """
    if not episodes:
        return ""

    parts: list[str] = []
    for episode in episodes:
        parts.append(
            "[Memory #{rank} | id={chunk_id} | session={session_id} | "
            "turn={turn_number} | source={source} | score={score:.4f}]\n"
            "{content}".format(
                rank=episode.get("rank", 0),
                chunk_id=episode.get("chunk_id", "?"),
                session_id=episode.get("session_id", "?"),
                turn_number=episode.get("turn_number", 0),
                source=episode.get("source", "?"),
                score=float(episode.get("score", 0.0)),
                content=episode.get("content", ""),
            )
        )
    return "\n\n".join(parts)


# ── Message helpers ───────────────────────────────────────────────────────────


def _message_text(message: BaseMessage) -> str:
    """Extract plain text from a message (str or content-block list)."""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and "text" in block
        ]
        return "\n".join(p for p in parts if p)
    return str(content)


def _zeroed_retrieval() -> dict[str, Any]:
    """Per-turn retrieval telemetry block for no-backend / no-query turns."""
    return {
        "retrieved_memories": [],
        "retrieved_context_chars": 0,
        "retrieved_context_tokens": 0,
        "retrieval_latency_ms": 0.0,
        "embedding_latency_ms": 0.0,
        "embedding_tokens": 0,
    }


# ── Graph Nodes ───────────────────────────────────────────────────────────────


def retrieve_node(state: L2AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Query the episodic backend with the current user turn.

    Reads ``memory_backend``, ``session_id`` and ``top_k`` from
    ``config["configurable"]`` and retrieves cross-session episodes with the
    current session's chat episodes excluded (the checkpointer already keeps
    those in-context; ``source="document"`` chunks are never excluded).

    The formatted ``[Memory ...]`` block is *sized here* (chars/tokens) but
    *re-formatted in* ``model_node`` — both call the same ``_format_memory_block``
    so the reported ``retrieved_context_chars`` is always the exact block
    length the model saw.
    """
    configurable = config.get("configurable", {})
    backend = configurable.get("memory_backend")

    # The retrieval query is the latest human turn.
    query = ""
    for message in reversed(list(state["messages"])):
        if isinstance(message, HumanMessage):
            query = _message_text(message)
            break

    if backend is None:
        logger.warning(
            "L2 retrieve_node: no memory_backend in configurable — "
            "retrieval skipped (telemetry zeroed)"
        )
        return _zeroed_retrieval()
    if not query.strip():
        logger.warning("L2 retrieve_node: no human turn found — retrieval skipped")
        return _zeroed_retrieval()

    session_id = configurable.get("session_id", "default")
    top_k = int(configurable.get("top_k", DEFAULT_TOP_K))

    telemetry = backend.retrieve(query, top_k, exclude_session_id=session_id)
    episodes = [episode.model_dump() for episode in telemetry.episodes]
    block = _format_memory_block(episodes)

    logger.debug(
        "L2 retrieve_node: %d/%d episodes (top_k=%d, exclude_session=%s, "
        "block_chars=%d)",
        len(episodes),
        top_k,
        top_k,
        session_id,
        len(block),
    )

    return {
        "retrieved_memories": episodes,
        "retrieved_context_chars": len(block),
        "retrieved_context_tokens": estimate_tokens(block),
        "retrieval_latency_ms": telemetry.retrieval_latency_ms,
        "embedding_latency_ms": telemetry.embedding_latency_ms,
        "embedding_tokens": telemetry.embedding_tokens,
    }


def model_node(state: L2AgentState, config: RunnableConfig) -> dict[str, Any]:
    """LLM invocation with ephemeral memory-block injection.

    Payload layout (prefix-cache-preserving)::

        [SystemMessage(system_prompt), *prior_thread_history,
         SystemMessage(memory_block)?, latest_human_message]

    The RAG ``SystemMessage`` is **ephemeral** — it is sent to the model but
    never returned in the state update, so the checkpointer never stores it
    and the byte-stable shared prefix ``S + H1 + A1 + …`` survives across
    turns (provider prefix-cache friendly, plan §3.3).

    The model/prompt are resolved dynamically from ``configurable``
    (Zero Hardcoding), exactly like L1's model node.
    """
    configurable = config.get("configurable", {})
    model_id: str = configurable["model_id"]
    system_prompt: str = configurable.get(
        "system_prompt", "You are a helpful assistant."
    )
    model_kwargs: dict[str, Any] = configurable.get("model_kwargs", {})

    messages = list(state["messages"])
    if not messages:
        logger.warning("L2 model_node: empty message list — nothing to answer")
        return {"messages": []}

    # ── Build the ephemeral payload ───────────────────────────────────
    memory_block = _format_memory_block(state.get("retrieved_memories") or [])

    payload: list[BaseMessage] = [SystemMessage(content=system_prompt)]
    payload.extend(messages[:-1])  # prior thread history (checkpointed)
    if memory_block:
        payload.append(SystemMessage(content=memory_block))
    payload.append(messages[-1])  # latest human turn

    logger.debug(
        "Invoking model=%s with %d messages (system + %d history + %s "
        "+ latest)",
        model_id,
        len(payload),
        len(messages) - 1,
        "memory-block" if memory_block else "no-memory-block",
    )

    model = create_model(model_id, **model_kwargs)
    response = model.invoke(payload)

    return {"messages": [response]}


def append_node(state: L2AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Persist the completed turn as a ``source="chat"`` episode.

    The formatted turn ``User: {input}\\nAssistant: {response}`` becomes one
    ``EpisodeRecord`` with a stable, deterministic ``chunk_id``
    (``ep_{session_id}_t{turn_number}``) — re-invoking the same logical turn
    hits the backend's idempotent upsert instead of duplicating.

    ``turn_number`` is the count of human turns in the thread's checkpointed
    history (1-based), so it does not depend on runner-side message IDs.
    Embedding latency/tokens are *accumulated* onto the values
    ``retrieve_node`` already wrote this turn, so the channel reflects the
    full per-turn embedding cost.
    """
    configurable = config.get("configurable", {})
    backend = configurable.get("memory_backend")

    if backend is None:
        logger.warning(
            "L2 append_node: no memory_backend in configurable — "
            "append skipped (telemetry zeroed)"
        )
        return {"appended_chunk_ids": [], "append_latency_ms": 0.0}

    messages = list(state["messages"])
    ai_message = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage)), None
    )
    human_message = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
    )

    if ai_message is None or human_message is None:
        logger.warning(
            "L2 append_node: no complete turn to persist — append skipped"
        )
        return {"appended_chunk_ids": [], "append_latency_ms": 0.0}

    session_id = configurable.get("session_id", "default")
    turn_number = sum(1 for m in messages if isinstance(m, HumanMessage))

    record = EpisodeRecord.from_content(
        chunk_id=f"ep_{session_id}_t{turn_number}",
        session_id=session_id,
        turn_number=turn_number,
        source="chat",
        content=(
            f"User: {_message_text(human_message)}\n"
            f"Assistant: {_message_text(ai_message)}"
        ),
    )

    telemetry = backend.append([record])

    logger.debug(
        "L2 append_node: persisted %s (%d chunks, %.1fms)",
        record.chunk_id,
        len(telemetry.chunk_ids),
        telemetry.append_latency_ms,
    )

    return {
        "appended_chunk_ids": telemetry.chunk_ids,
        "append_latency_ms": telemetry.append_latency_ms,
        "embedding_latency_ms": (state.get("embedding_latency_ms") or 0.0)
        + telemetry.embedding_latency_ms,
        "embedding_tokens": (state.get("embedding_tokens") or 0)
        + telemetry.embedding_tokens,
    }


# ── Agent Class ───────────────────────────────────────────────────────────────


class Level2EpisodicRAGAgent(BaseBenchmarkAgent):
    """Level 2: episodic Naive-RAG agent — cross-thread long-term memory.

    Combines both memory tiers (TODOs/phase-2-l2-plan.md §2):
    - **Intra-session**: a ``MemorySaver`` checkpointer keeps conversation
      history alive across ``invoke()`` calls within one thread (per-session
      ``thread_id`` supplied by the runner).
    - **Cross-session**: a swappable ``EpisodicMemoryBackend`` (injected via
      ``config["configurable"]["memory_backend"]``) accumulates every turn
      as an append-only episode and feeds top-k retrieved memories back
      into the prompt — the very capability L2 is benchmarked on.

    The topology is a fixed chain ``retrieve → model → append`` — no
    ``Command`` routing, no tools: a tool call could not be answered by the
    fixed chain, so ``build_graph(tools=...)`` logs a warning and ignores
    them (tools remain L1's extension point).
    """

    #: L2 consumes an ``EpisodicMemoryBackend`` injected at runtime through
    #: ``config["configurable"]["memory_backend"]`` — the CLI constructs the
    #: backend chosen via ``--backend`` for levels that require one.
    requires_memory_backend = True

    @property
    def level(self) -> str:
        return "level_2_episodic_rag"

    def build_graph(self, tools: list[Any] | None = None) -> Any:
        """Build and compile the Level 2 episodic RAG agent graph.

        Args:
            tools: Ignored (warning logged if non-empty) — the fixed chain
                   cannot route tool calls.

        Returns:
            A compiled ``StateGraph`` with a ``MemorySaver`` checkpointer
            (intra-session history) whose long-term store is supplied at
            runtime through ``configurable["memory_backend"]``.
        """
        if tools:
            logger.warning(
                "Level 2 is a fixed retrieve → model → append chain and "
                "does not route tool calls — ignoring %d provided tool(s)",
                len(tools),
            )

        builder = StateGraph(L2AgentState)

        builder.add_node("retrieve", retrieve_node)
        builder.add_node("model", model_node)
        builder.add_node("append", append_node)

        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "model")
        builder.add_edge("model", "append")
        builder.add_edge("append", END)

        memory = MemorySaver()
        compiled = builder.compile(checkpointer=memory)

        logger.info(
            "Level 2 episodic-RAG graph compiled (fixed chain "
            "retrieve → model → append, MemorySaver checkpointer)"
        )
        return compiled

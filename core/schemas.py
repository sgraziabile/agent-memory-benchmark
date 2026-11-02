"""core.schemas — Pydantic v2 data contracts and LangGraph state definitions.

Defines all structured types used across the benchmark harness:
- AgentState (TypedDict for LangGraph)
- AssertionRule, ConversationTurn, BenchmarkScenario (test case schemas)
- TurnResult, BenchmarkRunSummary (execution result schemas)
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, model_validator
from typing_extensions import TypedDict


# ── LangGraph State ──────────────────────────────────────────────────────────


class AgentState(TypedDict):
    """LangGraph agent state with message accumulation.

    The ``add_messages`` reducer appends new messages to the existing
    sequence rather than overwriting, enabling multi-turn conversations
    within a single graph invocation.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]


class L2AgentState(TypedDict):
    """LangGraph state for the Level 2 episodic RAG agent (plan §4.2).

    ``messages`` keeps the ``add_messages`` accumulator (MemorySaver-backed
    intra-session history). Every telemetry channel below is a **per-turn
    overwrite channel**: no reducer, so each graph invocation replaces the
    previous turn's value. ``BenchmarkRunner.run_single()`` reads these
    channels from the invoke result (guarded with ``.get()``) so the same
    runner code serves L0/L1 (channels absent) and L2 (channels present).

    Channels:
        retrieved_context_chars: Char length of the formatted
            ``[Memory ...]`` block injected into ``model_node``.
        retrieved_context_tokens: Token estimate of that block.
        retrieval_latency_ms: Time spent in ``retrieve_node``.
        append_latency_ms: Time spent in ``append_node`` (chat episodes).
        embedding_latency_ms: Embedding-API wait inside retrieve+append.
        embedding_tokens: Tokens sent to the embedding API this turn.
        retrieved_memories: Top-k ``RetrievedEpisode`` dicts (provenance).
        appended_chunk_ids: Chunk IDs written to the backend this turn.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    retrieved_context_chars: int
    retrieved_context_tokens: int
    retrieval_latency_ms: float
    append_latency_ms: float
    embedding_latency_ms: float
    embedding_tokens: int
    retrieved_memories: list[dict[str, Any]]
    appended_chunk_ids: list[str]


# ── Test Scenario Schemas ────────────────────────────────────────────────────


class AssertionRule(BaseModel):
    """A single assertion to evaluate against a turn's outcome.

    Two families (plan WS-D §Q4):

    - **Text assertions** — evaluated against the agent's response text:
      ``must_contain`` (presence) and ``must_not_contain`` (absence).
    - **Retrieval assertions** — evaluated against the turn's
      ``retrieved_memories`` telemetry (L2+ only; L0/L1 retrieve nothing, so
      positive retrieval assertions fail and negative ones pass):
      ``must_retrieve_chunk`` / ``must_not_retrieve_chunk`` match a
      case-insensitive substring of ``chunk_id`` (so ``doc_policy_v1`` also
      matches its sub-chunks ``doc_policy_v1_c2``);
      ``must_retrieve_source`` / ``must_not_retrieve_source`` match the exact
      ``source`` tag (``chat`` or ``document``).

    Retrieval assertions make retrieval failures causal: a top-k dilution test
    can prove the needle chunk never reached the model instead of inferring it
    from a wrong answer.

    Attributes:
        type: The assertion operator (see the two families above).
        value: The substring/keyword (text rules), chunk-ID fragment
            (``*_chunk``), or source tag (``*_source``).
        case_sensitive: Whether the match should be case-sensitive.
    """

    type: Literal[
        "must_contain",
        "must_not_contain",
        "must_retrieve_chunk",
        "must_not_retrieve_chunk",
        "must_retrieve_source",
        "must_not_retrieve_source",
    ]
    value: str
    case_sensitive: bool = False


class DocumentSpec(BaseModel):
    """A document to be ingested into the agent's memory (plan WS-D).

    Documents are the harness's second memory-ingestion channel, alongside
    chat turns: they are chunked and appended to the active
    ``EpisodicMemoryBackend`` with ``source="document"`` **without ever
    invoking the chat model** (embedding cost only).

    A spec is either *file-backed* (``path``) or *inline* (``content``) —
    never both. Paths are resolved by the runner relative to the scenario
    YAML file first, then ``datasets/documents/``.

    Attributes:
        doc_id: Stable document identifier. Deterministic chunk IDs derive
            from it (``doc_<doc_id>`` for scenario-level preloads,
            ``doc_<doc_id>_t<turn>`` for ``system_event`` ingestions), so a
            revision re-ingested under the same ``doc_id`` collides on
            purpose — append-only semantics for L2, mutation for L3.
        path: File path (absolute, or relative to the scenario YAML /
            ``datasets/documents/``).
        content: Inline document text (mutually exclusive with ``path``).
        metadata: Free-form provenance (``title``, ``version``, ...) copied
            into every ``EpisodeRecord`` chunk this document produces.
    """

    doc_id: str = Field(min_length=1)
    path: str | None = None
    content: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "DocumentSpec":
        has_path = bool(self.path and self.path.strip())
        has_content = bool(self.content and self.content.strip())
        if has_path == has_content:
            raise ValueError(
                f"DocumentSpec '{self.doc_id}' must define exactly one of "
                "'path' or 'content' (non-empty)."
            )
        return self


class ConversationTurn(BaseModel):
    """One turn in a benchmark conversation scenario.

    Attributes:
        turn_number: Sequential index of this turn (1-based).
        role: Who produces this turn — ``user`` for user messages,
              ``system_event`` for injected environmental changes.
        content: The text content of the turn.
        assertions: Rules to evaluate against the agent's response to this turn.
        session_id: Conversation session this turn belongs to (plan §5.2).
              Turns sharing a ``session_id`` share one LangGraph thread
              (``thread_id = run_id + scenario_id + session_id``), so
              checkpointer-backed agents (L1+) remember within a session
              but forget across sessions. Existing single-session YAMLs
              omit the key and default to ``"default"`` — 100% backward
              compatible.
        attachments: File paths ingested into memory on a ``system_event``
              turn (plan WS-D). Each file is read, chunked and appended to
              the active ``EpisodicMemoryBackend`` with
              ``source="document"`` — the chat model is never invoked.
              Resolved relative to the scenario YAML file, falling back to
              ``datasets/documents/``. Empty on ``user`` turns.
    """

    turn_number: int
    role: Literal["user", "system_event"]
    content: str
    assertions: list[AssertionRule] = Field(default_factory=list)
    session_id: str = "default"
    attachments: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _attachments_only_on_system_events(self) -> "ConversationTurn":
        """Document ingestion is bound to ``system_event`` turns.

        A ``user`` turn with attachments would be silently ignored at run
        time (the runner only ingests on ``system_event``), which is exactly
        the "contract you don't honor" failure mode the audit flagged. Fail
        at load time instead, naming the scenario author's mistake.
        """
        if self.role == "user" and self.attachments:
            raise ValueError(
                f"Turn {self.turn_number}: 'attachments' is only valid on "
                "role='system_event' turns. Declare documents under the "
                "scenario's top-level 'documents:' key for preloads."
            )
        return self


class BenchmarkScenario(BaseModel):
    """A complete benchmark test case loaded from a YAML file.

    Attributes:
        id: Unique identifier for this scenario.
        name: Human-readable scenario name.
        description: Detailed description of what this scenario tests.
        category: Test category (e.g., ``belief_revision``, ``attrition``,
                  ``needle_haystack``).
        turns: Ordered list of conversation turns.
        metadata: Optional extra metadata (difficulty, research question, etc.).
        documents: Scenario-level document corpus (plan WS-D). Every entry is
              ingested ONCE per benchmark tuple, before the first turn, with
              ``source="document"`` — the chat model is never invoked. This
              turns "context volume" into a declarative scenario parameter:
              a corpus of N documents competes with chat episodes for the
              retrieval ``top_k`` budget. Empty for scenarios without
              documents (100% backward compatible).
        source_path: Absolute path of the YAML file this scenario was loaded
              from (set by ``BenchmarkRunner.load_scenarios``). Used to
              resolve relative ``documents[].path`` / ``attachments``
              entries. Excluded from serialization so exports stay portable.
    """

    id: str
    name: str
    description: str
    category: str
    turns: list[ConversationTurn]
    metadata: dict[str, Any] = Field(default_factory=dict)
    documents: list[DocumentSpec] = Field(default_factory=list)
    source_path: str | None = Field(default=None, exclude=True)


# ── Execution Result Schemas ─────────────────────────────────────────────────


class TurnResult(BaseModel):
    """Result of evaluating a single conversation turn.

    Captures the agent's response, assertion outcomes, performance metrics,
    and token usage for one turn within a benchmark run.

    Attributes:
        turn_number: The turn index this result corresponds to.
        user_input: The user message that was sent.
        agent_response: The agent's textual response (empty for errored turns).
        assertions_passed: Count of assertions that passed.
        assertions_failed: Count of assertions that failed.
        assertion_details: Per-assertion breakdown with rule, expected, actual, passed.
        latency_ms: Wall-clock time for the agent to respond (milliseconds).
        token_usage: Token counts extracted from model response metadata.
        status: ``"ok"`` if the model responded; ``"error"`` if the turn
                failed at the infrastructure level (API outage, auth error).
        error: Exception description for errored turns; ``None`` otherwise.
                Errored turns are excluded from pass-rate and token metrics.
        session_id: Session this turn belongs to (``"default"``, ``"s1"``).
        role: Turn role — lets ``system_event`` document ingestions be
              logged in ``turn_results`` without entering the chat
              assertion denominators (plan §4.2).
        prompt_tokens: Chat LLM input tokens billed on this turn, summed
              across ALL ``AIMessage``s generated after the input
              ``HumanMessage`` (plan §4.1 slicing rule — never re-counts
              prior turns in checkpointer-backed graphs).
        completion_tokens: Chat LLM total output tokens billed (includes
              hidden reasoning tokens).
        total_tokens: ``prompt_tokens + completion_tokens``.
        reasoning_tokens: Hidden chain-of-thought tokens
              (``output_token_details.reasoning``).
        visible_output_tokens: ``max(0, completion_tokens - reasoning_tokens)``.
        cached_prompt_tokens: Prefix-cache hit tokens
              (``input_token_details.cache_read``).
        retrieved_context_chars: Char length of the ``[Memory ...]`` block
              injected by ``retrieve_node`` (0 for L0/L1).
        retrieved_context_tokens: Token estimate of that block.
        embedding_tokens: Tokens sent to the embedding API this turn
              (retrieve + append; 0 for L0/L1 and the markdown backend).
        retrieval_latency_ms: Time spent in ``retrieve_node``.
        append_latency_ms: Time spent in ``append_node`` (chat episodes) or
              ``system_event`` ingestion.
        embedding_latency_ms: Embedding-API wait within retrieve+append.
        retrieved_memories: Top-k retrieved episode dicts (provenance:
              ``chunk_id``, ``session_id``, ``turn_number``, ``source``,
              ``score``, ``rank``, ``content``, ``estimated_tokens``).
        appended_chunk_ids: Chunk IDs written to the backend this turn —
              links directly into ``memory/<run_id>/`` snapshots.
        ingested_documents: Document IDs ingested on this turn (``doc_id``
              values). Non-empty only for ``system_event`` turns with a
              memory backend attached.
        ingested_chars: Characters ingested on this turn (before chunking).
        ingested_chunk_ids: Chunk IDs produced by document ingestion this
              turn. Kept separate from ``appended_chunk_ids`` so chat
              persistence and document ingestion can be audited
              independently.
    """

    turn_number: int
    user_input: str
    agent_response: str
    assertions_passed: int = 0
    assertions_failed: int = 0
    assertion_details: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: float = 0.0
    token_usage: dict[str, Any] = Field(default_factory=dict)
    status: Literal["ok", "error"] = "ok"
    error: str | None = None
    # ── Plan §4.2 telemetry (defaults keep L0/L1 construction unchanged) ──
    session_id: str = "default"
    role: Literal["user", "system_event"] = "user"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    visible_output_tokens: int = 0
    cached_prompt_tokens: int = 0
    retrieved_context_chars: int = 0
    retrieved_context_tokens: int = 0
    embedding_tokens: int = 0
    retrieval_latency_ms: float = 0.0
    append_latency_ms: float = 0.0
    embedding_latency_ms: float = 0.0
    retrieved_memories: list[dict[str, Any]] = Field(default_factory=list)
    appended_chunk_ids: list[str] = Field(default_factory=list)
    # ── Plan WS-D document ingestion (0/[] for chat turns and L0/L1) ──────
    ingested_documents: list[str] = Field(default_factory=list)
    ingested_chars: int = 0
    ingested_chunk_ids: list[str] = Field(default_factory=list)


class BenchmarkRunSummary(BaseModel):
    """Summary of a complete benchmark run — one execution tuple.

    Represents the results of evaluating one
    ⟨Scenario, System Prompt, Model, Persistence Level⟩ combination.

    Attributes:
        run_id: Unique identifier for this run (UUID or timestamp-based).
        scenario_id: The benchmark scenario that was executed.
        model_id: The model identifier used.
        prompt_id: The system prompt identifier used.
        agent_level: The agent persistence level (e.g., ``level_0_reactive``).
        total_turns: Number of conversation turns executed.
        total_assertions: Total assertions evaluated across all turns.
        passed_assertions: Count of passed assertions.
        failed_assertions: Count of failed assertions.
        pass_rate: Fraction of assertions passed (0.0–1.0), computed over
                   successfully completed turns only.
        error_turns: Number of turns that failed at the infrastructure level
                     and were excluded from all metrics.
        error_rate: Fraction of turns that errored (0.0–1.0).
        total_latency_ms: Cumulative latency across successfully completed turns.
        avg_latency_per_turn_ms: Average latency per successfully completed turn.
        turn_results: Detailed per-turn results.
        timestamp: ISO 8601 timestamp of when the run started.
        memory_backend: Backend identity for reproducibility — ``"none"``
              (L0), ``"memory_saver"`` (L1), ``"vector"`` / ``"markdown"``
              (L2+, plan §4.3).
        embedding_model_id: Registry ID of the embedding model, or ``None``
              for backends/levels without embeddings.
        top_k: Retrieval cut-off configured for this run (``None`` = N/A).
        chunk_size: Chunking size configured for this run (``None`` = N/A).
        chunk_overlap: Chunking overlap configured for this run.
        memory_snapshot_path: Relative path ``memory/<run_id>`` of the
              exported backend snapshot inside the run folder (``None``
              when no backend was attached).
        reasoning_tokens: Hidden reasoning tokens summed over ok turns.
        visible_output_tokens: Visible output tokens summed over ok turns.
        cached_prompt_tokens: Prefix-cache hit tokens summed over ok turns.
        retrieved_context_tokens: RAG context tokens injected across the run.
        embedding_tokens: Embedding tokens across all turns, including
              ``system_event`` ingestions.
        total_retrieval_latency_ms: Cumulative ``retrieve_node`` time.
        total_append_latency_ms: Cumulative ``append_node`` time.
        total_embedding_latency_ms: Cumulative embedding-API wait.
        ingestion_latency_ms: Time spent ingesting documents (chunking +
              embedding + indexing) — scenario-level corpus preload AND
              ``system_event`` turns. Kept separate so large-file ingestion
              does not skew ``avg_latency_per_turn_ms``, which now averages
              over chat (``role="user"``) turns only.
        documents_ingested: Number of distinct documents ingested for this
              tuple (preload + ``system_event``), regardless of chunk count.
        ingested_chars: Characters ingested before chunking (corpus volume).
        ingested_chunks: Chunks produced by document ingestion.
        corpus_chunks: Total chunks present in the backend at snapshot time
              (chat + documents). ``None`` when the backend exposes no
              ``corpus_stats()`` (e.g. a test double).
        corpus_chars: Total characters stored in the backend at snapshot
              time; ``None`` when unavailable.
    """

    run_id: str
    scenario_id: str
    model_id: str
    prompt_id: str
    agent_level: str
    total_turns: int = 0
    total_assertions: int = 0
    passed_assertions: int = 0
    failed_assertions: int = 0
    pass_rate: float = 0.0
    error_turns: int = 0
    error_rate: float = 0.0
    total_latency_ms: float = 0.0
    avg_latency_per_turn_ms: float = 0.0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    turn_results: list[TurnResult] = Field(default_factory=list)
    timestamp: str = ""
    # ── Plan §4.3 memory configuration & reproducibility identity ──────────
    memory_backend: str = "none"
    embedding_model_id: str | None = None
    top_k: int | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    memory_snapshot_path: str | None = None
    # ── Plan §4.3 aggregated token metrics (over status == "ok" turns) ─────
    reasoning_tokens: int = 0
    visible_output_tokens: int = 0
    cached_prompt_tokens: int = 0
    retrieved_context_tokens: int = 0
    embedding_tokens: int = 0
    # ── Plan §4.3 aggregated latency metrics ───────────────────────────────
    total_retrieval_latency_ms: float = 0.0
    total_append_latency_ms: float = 0.0
    total_embedding_latency_ms: float = 0.0
    ingestion_latency_ms: float = 0.0
    # ── Plan WS-D document-ingestion metrics ───────────────────────────────
    documents_ingested: int = 0
    ingested_chars: int = 0
    ingested_chunks: int = 0
    corpus_chunks: int | None = None
    corpus_chars: int | None = None

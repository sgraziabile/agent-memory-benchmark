"""agents.level_2_episodic_rag.backends.base — L2 episodic memory contracts.

Level 2 models long-term memory as an **append-only (Naive RAG)** episodic
store. All storage backends — ``VectorMemoryBackend`` (LangGraph
``InMemoryStore`` with a dense vector index) and, later,
``MarkdownMemoryBackend`` (append-only ``.md`` sections + BM25) — implement
the unified :class:`EpisodicMemoryBackend` interface so the L2 agent graph
and :class:`~core.runner.BenchmarkRunner` can treat them polymorphically.

Active state mutation (``UPDATE`` / ``DELETE``) is intentionally excluded
from Level 2 and reserved for Level 3 (Reflective / CRUD Memory): L2 only
ever *appends* new episodes, empirically exposing stale/contradictory chunk
collision on belief revision before L3 introduces state mutation.

Data contracts (TODOs/phase-2-l2-plan.md §3.1):
- :class:`EpisodeRecord`       — one persisted chat/document chunk
- :class:`RetrievedEpisode`    — one top-k retrieved chunk with score + rank
- :class:`AppendTelemetry`     — per-append timing/embedding telemetry
- :class:`RetrievalTelemetry`  — per-retrieve timing/embedding telemetry
- :class:`EpisodicMemoryBackend` — the swappable backend interface
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

#: Provenance tag: conversational episode vs. ingested document chunk.
EpisodeSource = Literal["chat", "document"]


def estimate_tokens(text: str) -> int:
    """Cheap deterministic token estimate (~4 chars per token heuristic).

    Used for telemetry and snapshot manifests only — never for provider
    billing. A ``tiktoken`` ``cl100k_base`` count can be swapped in later
    without touching callers, per TODOs/phase-2-l2-plan.md §3.1.
    """
    return len(text) // 4


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string (episode timestamps)."""
    return datetime.now(timezone.utc).isoformat()


class EpisodeRecord(BaseModel):
    """A single persisted episodic chunk (chat turn or document section).

    Attributes:
        chunk_id: Stable chunk identifier, e.g. ``ep_s1_t2_0`` (episode,
                  session ``s1``, turn 2, chunk 0) or ``doc_spec01_c03``.
        session_id: Session the episode belongs to (``s1``, ``s2``, ...).
        turn_number: 1-based turn index within the session.
        source: ``chat`` for conversational turns, ``document`` for
                ingested large-file chunks.
        content: The chunk/section text itself.
        char_count: ``len(content)`` — kept explicit so snapshots stay
                    auditable without recomputation.
        estimated_tokens: Heuristic token estimate (see :func:`estimate_tokens`).
        timestamp: UTC ISO-8601 creation time.
        metadata: Free-form provenance (``doc_id``, ``section_title``, ...).
    """

    chunk_id: str
    session_id: str
    turn_number: int
    source: EpisodeSource
    content: str
    char_count: int
    estimated_tokens: int
    timestamp: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def _content_not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError(
                "EpisodeRecord.content must be non-empty — blank chunks "
                "would pollute top-k retrieval"
            )
        return value

    @field_validator("turn_number")
    @classmethod
    def _turn_number_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"turn_number is 1-based, got {value}")
        return value

    @classmethod
    def from_content(
        cls,
        *,
        chunk_id: str,
        session_id: str,
        turn_number: int,
        source: EpisodeSource,
        content: str,
        timestamp: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "EpisodeRecord":
        """Build a record from raw content, auto-computing size fields.

        Convenience factory used by the L2 graph's ``append_node`` and by
        ``BenchmarkRunner``'s ``system_event`` ingestion: ``char_count``,
        ``estimated_tokens`` and ``timestamp`` are derived so callers can
        never let them drift out of sync with ``content``.
        """
        return cls(
            chunk_id=chunk_id,
            session_id=session_id,
            turn_number=turn_number,
            source=source,
            content=content,
            char_count=len(content),
            estimated_tokens=estimate_tokens(content),
            timestamp=timestamp if timestamp is not None else utc_now_iso(),
            metadata=metadata if metadata is not None else {},
        )


class RetrievedEpisode(BaseModel):
    """One top-k retrieved chunk with provenance, score, and rank.

    ``score`` is cosine similarity for the vector backend and a BM25 score
    for the (deferred) markdown backend — both share this contract.
    """

    chunk_id: str
    session_id: str
    turn_number: int
    source: EpisodeSource
    content: str
    score: float
    rank: int
    char_count: int
    estimated_tokens: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class AppendTelemetry(BaseModel):
    """Per-append timing and embedding cost (plan §3.1)."""

    chunk_ids: list[str] = Field(default_factory=list)
    append_latency_ms: float = 0.0
    embedding_latency_ms: float = 0.0
    embedding_tokens: int = 0


class RetrievalTelemetry(BaseModel):
    """Per-retrieve timing and embedding cost (plan §3.1)."""

    episodes: list[RetrievedEpisode] = Field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    embedding_latency_ms: float = 0.0
    embedding_tokens: int = 0


class EpisodicMemoryBackend(ABC):
    """Abstract interface for swappable Level 2 episodic memory backends.

    Contract highlights (TODOs/phase-2-l2-plan.md §3.1):
    - **Append-only**: backends expose no update/delete primitives. State
      mutation is reserved for Level 3.
    - ``retrieve`` supports ``exclude_session_id`` so the L2 graph can hide
      *current-session chat episodes* (already covered by ``MemorySaver``
      in-context history) while always allowing ``source="document"``
      chunks regardless of session.
    - ``reset`` guarantees strict isolation between benchmark tuples.
    - ``export_snapshot`` writes human-auditable artifacts for the run.
    """

    @abstractmethod
    def append(self, records: list[EpisodeRecord]) -> AppendTelemetry:
        """Persist episode records, returning timing/embedding telemetry."""
        ...

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int,
        *,
        exclude_session_id: str | None = None,
    ) -> RetrievalTelemetry:
        """Return up to ``top_k`` episodes ranked by backend-specific score.

        Args:
            query: Query text (current user turn).
            top_k: Maximum number of episodes to return.
            exclude_session_id: If set, ``source="chat"`` episodes from this
                session are excluded (intra-session dedup guard);
                ``source="document"`` chunks are never excluded.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all stored state (strict run isolation between tuples)."""
        ...

    @abstractmethod
    def export_snapshot(self, dest_dir: Path, run_metadata: dict[str, Any]) -> Path:
        """Write human-auditable memory artifacts for the run.

        Args:
            dest_dir: Target directory (``outputs/runs/<ts>/memory/<run_id>/``).
            run_metadata: Run identity for the manifest (run_id, scenario_id,
                model_id, embedding model, top_k, chunk settings, ...).

        Returns:
            The directory the snapshot was written to.
        """
        ...

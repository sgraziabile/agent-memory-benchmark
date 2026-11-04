"""agents.level_2_episodic_rag.backends.vector — dense-retrieval L2 backend.

``VectorMemoryBackend`` stores every episodic chunk in a LangGraph
``InMemoryStore`` configured with a dense vector index
(``index={"embed": ..., "dims": ..., "fields": ["content"]}``). Retrieval
is exact top-k cosine similarity; long episode content is split into
overlapping sub-chunks with ``RecursiveCharacterTextSplitter``.

Telemetry design: the backend never calls the embedding model itself.
Instead the ``Embeddings`` client is wrapped in a :class:`_TimingEmbeddings`
proxy which is handed to the store — every embedding call the store makes
(at ``put`` and ``search`` time) passes through the proxy, so
``AppendTelemetry`` / ``RetrievalTelemetry`` report the *actual* embedding
latency/tokens with zero extra API calls per operation.

``exclude_session_id`` semantics (TODOs/phase-2-l2-plan.md §3.1): current
-session ``source="chat"`` episodes are hidden (they are already in-context
via the L2 graph's ``MemorySaver`` history) while ``source="document"``
chunks are NEVER excluded. The store's ``filter`` only supports equality,
so the backend over-fetches the full ranked list (benchmark scale: tens to
hundreds of chunks) and applies the exclusion client-side — the resulting
top-k is exact, not approximate.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings
from langgraph.store.memory import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:  # langgraph ships ``PutOp`` today; the import stays optional so an older
    # release degrades to per-chunk ``put()`` instead of failing at import.
    from langgraph.store.base import PutOp
except ImportError:  # pragma: no cover - exercised only on old langgraph
    PutOp = None  # type: ignore[assignment]

from agents.level_2_episodic_rag.backends.base import (
    AppendTelemetry,
    EpisodeRecord,
    EpisodicMemoryBackend,
    RetrievedEpisode,
    RetrievalTelemetry,
    estimate_tokens,
    utc_now_iso,
)

logger = logging.getLogger(__name__)

#: Value-field the vector index embeds (kept explicit, matches IndexConfig).
_EMBED_FIELD = "content"


def _is_timeout_exception(exc: BaseException) -> bool:
    """True for provider/client timeouts, without importing httpx here.

    ``httpx.ReadTimeout`` → ``"ReadTimeout"``; ``asyncio.TimeoutError`` and
    builtin ``TimeoutError`` are both ``TimeoutError``; some clients only
    surface the message (``"... timed out"``).
    """
    if isinstance(exc, TimeoutError):
        return True
    if "timeout" in type(exc).__name__.casefold():
        return True
    return "timed out" in str(exc).casefold()


def _split_duplicate_texts(
    window: list[tuple[str, dict[str, Any]]],
) -> tuple[list[tuple[str, dict[str, Any]]], list[tuple[str, dict[str, Any]]]]:
    """Partition a window into first-occurrence chunks and text duplicates.

    ``InMemoryStore`` embeds only the *unique* texts of a batch but then
    requires one vector per ``PutOp``, so duplicate content inside one batch
    raises a store-level ``ValueError``. The caller writes the duplicates one
    at a time instead.
    """
    unique: list[tuple[str, dict[str, Any]]] = []
    duplicates: list[tuple[str, dict[str, Any]]] = []
    seen_texts: set[str] = set()
    for chunk_id, value in window:
        text = value.get(_EMBED_FIELD)
        if isinstance(text, str) and text in seen_texts:
            duplicates.append((chunk_id, value))
            continue
        if isinstance(text, str):
            seen_texts.add(text)
        unique.append((chunk_id, value))
    return unique, duplicates


@dataclass
class _EmbeddingStats:
    """Cumulative embedding cost since the last reset (per graph-node op)."""

    documents_latency_ms: float = 0.0
    documents_tokens: int = 0
    query_latency_ms: float = 0.0
    query_calls: int = 0


class _TimingEmbeddings(Embeddings):
    """Transparent timing proxy around a LangChain ``Embeddings`` client.

    The L2 store embeds chunk text at ``put`` time and the query at
    ``search`` time; routing both through this proxy lets the backend
    report real embedding latency/tokens without issuing duplicate API
    calls (a second "telemetry" call per turn would double embedding cost
    and distort P3 latency metrics).
    """

    def __init__(self, inner: Embeddings) -> None:
        self.inner = inner
        self.stats = _EmbeddingStats()

    # ── Stats management ────────────────────────────────────────────
    def reset_stats(self) -> None:
        """Zero the counters before a backend ``append``/``retrieve``."""
        self.stats = _EmbeddingStats()

    def consume_stats(self) -> _EmbeddingStats:
        """Return the current counters and zero them for the next op."""
        stats = self.stats
        self.stats = _EmbeddingStats()
        return stats

    # ── Sync API (used by InMemoryStore's sync put/search path) ─────
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        t0 = time.perf_counter()
        vectors = self.inner.embed_documents(texts)
        self.stats.documents_latency_ms += (time.perf_counter() - t0) * 1000
        self.stats.documents_tokens += sum(estimate_tokens(t) for t in texts)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        t0 = time.perf_counter()
        vector = self.inner.embed_query(text)
        self.stats.query_latency_ms += (time.perf_counter() - t0) * 1000
        self.stats.query_calls += 1
        return vector

    # ── Async API (delegate + record, for completeness) ─────────────
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        t0 = time.perf_counter()
        vectors = await self.inner.aembed_documents(texts)
        self.stats.documents_latency_ms += (time.perf_counter() - t0) * 1000
        self.stats.documents_tokens += sum(estimate_tokens(t) for t in texts)
        return vectors

    async def aembed_query(self, text: str) -> list[float]:
        t0 = time.perf_counter()
        vector = await self.inner.aembed_query(text)
        self.stats.query_latency_ms += (time.perf_counter() - t0) * 1000
        self.stats.query_calls += 1
        return vector


class VectorMemoryBackend(EpisodicMemoryBackend):
    """Append-only episodic memory with dense (vector) retrieval.

    Every episode is stored in a LangGraph ``InMemoryStore`` vector index
    over the ``content`` field; retrieval is cosine-similarity top-k.
    Long content is split into overlapping sub-chunks
    (``RecursiveCharacterTextSplitter``); content that already fits in one
    chunk keeps its original ``chunk_id`` unchanged so IDs stay stable
    across tests and snapshots.

    The store is intentionally rebuilt on ``reset()`` rather than cleared
    item-by-item — guaranteeing strict isolation between benchmark tuples
    even if an internal index held residual vectors.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        dims: int,
        *,
        embedding_id: str | None = None,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        namespace: tuple[str, ...] = ("episodes",),
        embed_batch_size: int = 8,
    ) -> None:
        """Construct the backend.

        Args:
            embeddings: LangChain ``Embeddings`` client (e.g. from
                ``core.model_factory.create_embeddings``).
            dims: Vector dimensionality of ``embeddings`` — MUST match the
                model's real output size (see ``get_embedding_dims``); the
                store needs it at index-construction time.
            embedding_id: Optional registry ID, recorded in snapshot
                manifests for provenance.
            chunk_size: Max characters per chunk before splitting.
            chunk_overlap: Character overlap between adjacent chunks.
            namespace: Store namespace prefix for all episodes.
            embed_batch_size: How many chunks are handed to one
                ``InMemoryStore.batch()`` call. ``InMemoryStore`` embeds one
                provider call per batch, so batching is the difference
                between one HTTP round-trip per chunk and one per window
                (measured ~7.5× on local Ollama). ``1`` reproduces the
                historical per-chunk behaviour exactly.

                Keep the window small enough that one call fits the
                embedding client's request timeout: on CPU Ollama a 1,200-char
                chunk costs ~35 ms/token, so **8 chunks ≈ 17–22 s** while 16
                chunks ≈ 35–44 s (measured) — the latter only fits because
                ``configs/models.yaml`` raises the local embedder's timeout to
                240 s. Timed-out windows are halved and retried, so a slow
                request costs one retry rather than the whole ingestion.

        Raises:
            ValueError: If ``chunk_overlap >= chunk_size``, ``dims <= 0`` or
                ``embed_batch_size < 1``.
        """
        if dims <= 0:
            raise ValueError(f"dims must be a positive int, got {dims}")
        if embed_batch_size < 1:
            raise ValueError(
                f"embed_batch_size must be >= 1, got {embed_batch_size}"
            )
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})"
            )

        self._dims = dims
        self._embedding_id = embedding_id
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._namespace = tuple(namespace)
        self._embed_batch_size = embed_batch_size

        # Timing proxy — the store's only view of the embedding model.
        self._proxy = _TimingEmbeddings(embeddings)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Mirror of stored chunk values, keyed by chunk_id. The store owns
        # ranking; this mirror owns deterministic enumeration for exports
        # and idempotent chunk counting.
        self._chunks: dict[str, dict[str, Any]] = {}

        self._store = self._build_store()

    def _build_store(self) -> InMemoryStore:
        """(Re)create the underlying vector-indexed store."""
        return InMemoryStore(
            index={
                "embed": self._proxy,
                "dims": self._dims,
                "fields": [_EMBED_FIELD],
            }
        )

    # ── Chunking ─────────────────────────────────────────────────────

    def _chunk_record(self, record: EpisodeRecord) -> list[tuple[str, dict[str, Any]]]:
        """Split one record into ``(chunk_id, value)`` store entries.

        Content fitting in ``chunk_size`` becomes a single chunk carrying
        the record's own ``chunk_id``; longer content is split into
        ``{chunk_id}_c{i}`` sub-chunks whose metadata records the parent
        and chunk index. ``char_count`` / ``estimated_tokens`` are
        recomputed per chunk so they never drift from chunk content.
        """
        if len(record.content) <= self._chunk_size:
            return [(record.chunk_id, record.model_dump())]

        texts = [t for t in self._splitter.split_text(record.content) if t.strip()]
        base = record.model_dump(
            exclude={"chunk_id", "content", "char_count", "estimated_tokens"}
        )
        chunks: list[tuple[str, dict[str, Any]]] = []
        for index, text in enumerate(texts):
            metadata = dict(base["metadata"])
            metadata.update(
                {
                    "parent_chunk_id": record.chunk_id,
                    "chunk_index": index,
                    "chunk_count": len(texts),
                }
            )
            chunks.append(
                (
                    f"{record.chunk_id}_c{index}",
                    {
                        **base,
                        "chunk_id": f"{record.chunk_id}_c{index}",
                        "content": text,
                        "char_count": len(text),
                        "estimated_tokens": estimate_tokens(text),
                        "metadata": metadata,
                    },
                )
            )
        return chunks

    def _upsert_chunks(self, stored: list[tuple[str, dict[str, Any]]]) -> None:
        """Persist chunks, embedding one window per ``InMemoryStore`` batch.

        ``InMemoryStore.batch()`` collects the texts of *all* ``PutOp``s and
        issues a single ``embed_documents`` call, so the window size is the
        number of chunks per provider round-trip. Before this, ``append``
        called ``put()`` per chunk, i.e. one HTTP call per chunk — on local
        Ollama that was ~2.3 s/chunk instead of ~0.3 s.

        The ``_chunks`` mirror is updated only after its window succeeds, so a
        mid-ingestion provider failure leaves a consistent prefix (ids that
        the store actually received) rather than phantom entries.

        ``embed_batch_size=1`` reproduces the historical per-chunk behaviour.
        """
        if not stored:
            return

        store_batch = getattr(self._store, "batch", None)
        if PutOp is None or store_batch is None:
            for chunk_id, value in stored:
                self._store.put(self._namespace, chunk_id, value)
                self._chunks[chunk_id] = value
            return

        windows = [
            stored[start : start + self._embed_batch_size]
            for start in range(0, len(stored), self._embed_batch_size)
        ]
        started = time.perf_counter()
        total = len(windows)

        for index, window in enumerate(windows, start=1):
            unique, duplicates = _split_duplicate_texts(window)
            window_started = time.perf_counter()

            if unique:
                self._store_window(unique)
            # ``InMemoryStore`` embeds the *unique* texts of a batch but then
            # requires one vector per PutOp (``_insertinmem_store`` compares
            # ``len(indices)`` to ``len(embeddings)``), so two chunks with
            # identical content in one batch would trip a store-level
            # ValueError. They are written one at a time instead.
            for chunk_id, value in duplicates:
                self._store.put(self._namespace, chunk_id, value)
                self._chunks[chunk_id] = value

            elapsed = time.perf_counter() - started
            eta = (elapsed / index) * (total - index)
            logger.info(
                "embedding window %d/%d: %d chunk(s) embedded, %d written "
                "individually, %.1fs this window, %.0fs elapsed, ETA %.0fs",
                index,
                total,
                len(unique),
                len(duplicates),
                time.perf_counter() - window_started,
                elapsed,
                eta,
            )

    def _store_window(self, window: list[tuple[str, dict[str, Any]]]) -> None:
        """Embed and store one window, halving it if the request times out.

        A CPU-only embedder takes ~35–45 s per 16-chunk window, so a single
        slow request can cross the client timeout and otherwise abort an
        ingestion that is already minutes deep. On a timeout the window is
        split in half and retried; a single-chunk window re-raises (there is
        nothing left to shrink).

        ``_chunks`` is mirrored only after the window (or its halves) succeed,
        so a hard failure still leaves a consistent prefix in the mirror.
        """
        try:
            self._store.batch(
                [
                    PutOp(self._namespace, chunk_id, value)
                    for chunk_id, value in window
                ]
            )
        except (AttributeError, TypeError) as exc:
            # A store whose batch() contract differs (or a future langgraph
            # that drops it) must not take ingestion down: degrade to the
            # documented put() path for this window.
            logger.warning(
                "Store batch() unusable (%s) — falling back to put() for "
                "%d chunk(s)",
                exc,
                len(window),
            )
            for chunk_id, value in window:
                self._store.put(self._namespace, chunk_id, value)
        except Exception as exc:  # noqa: BLE001 — re-raised unless it is a timeout
            if not _is_timeout_exception(exc) or len(window) == 1:
                raise
            middle = len(window) // 2
            logger.warning(
                "Embedding request timed out on %d chunk(s) (%s) — retrying "
                "as %d + %d",
                len(window),
                exc,
                middle,
                len(window) - middle,
            )
            self._store_window(window[:middle])
            self._store_window(window[middle:])
            return

        for chunk_id, value in window:
            self._chunks[chunk_id] = value

    # ── EpisodicMemoryBackend: append ────────────────────────────────

    def append(self, records: list[EpisodeRecord]) -> AppendTelemetry:
        """Persist records, embedding chunk text via the store index.

        Re-appending an identical ``chunk_id`` with identical content is an
        idempotent upsert — L2 never accumulates duplicates.

        Raises:
            ValueError: If a chunk_id collides with different content
                (append-only guard against accidental mutation).
        """
        if not records:
            return AppendTelemetry()

        t0 = time.perf_counter()
        self._proxy.reset_stats()

        stored: list[tuple[str, dict[str, Any]]] = []
        for record in records:
            for chunk_id, value in self._chunk_record(record):
                existing = self._chunks.get(chunk_id)
                if existing is not None and existing["content"] != value["content"]:
                    raise ValueError(
                        f"chunk_id '{chunk_id}' already stored with different "
                        "content — L2 is append-only; mutation is L3 territory"
                    )
                stored.append((chunk_id, value))

        self._upsert_chunks(stored)

        stats = self._proxy.consume_stats()
        return AppendTelemetry(
            chunk_ids=[chunk_id for chunk_id, _ in stored],
            append_latency_ms=(time.perf_counter() - t0) * 1000,
            embedding_latency_ms=stats.documents_latency_ms,
            embedding_tokens=stats.documents_tokens,
        )

    # ── EpisodicMemoryBackend: retrieve ──────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int,
        *,
        exclude_session_id: str | None = None,
    ) -> RetrievalTelemetry:
        """Cosine-similarity top-k over all stored chunks.

        The ranked list is fetched in full (benchmark scale is tens to
        hundreds of chunks) so the ``exclude_session_id`` filter applied
        client-side yields an *exact* top-k, not an over-fetch
        approximation. The query embedding cost is captured by the
        ``_TimingEmbeddings`` proxy — the store embeds the query exactly
        once per search.
        """
        t0 = time.perf_counter()
        self._proxy.reset_stats()

        episodes: list[RetrievedEpisode] = []
        if self._chunks and top_k > 0:
            items = self._store.search(
                self._namespace, query=query, limit=len(self._chunks)
            )
            ranked = [
                item
                for item in items
                if not (
                    exclude_session_id is not None
                    and item.value["source"] == "chat"
                    and item.value["session_id"] == exclude_session_id
                )
            ]
            for rank, item in enumerate(ranked[:top_k], start=1):
                episodes.append(
                    RetrievedEpisode(
                        chunk_id=item.key,
                        session_id=item.value["session_id"],
                        turn_number=item.value["turn_number"],
                        source=item.value["source"],
                        content=item.value["content"],
                        score=item.score,
                        rank=rank,
                        char_count=item.value["char_count"],
                        estimated_tokens=item.value["estimated_tokens"],
                        metadata=item.value["metadata"],
                    )
                )

        stats = self._proxy.consume_stats()
        return RetrievalTelemetry(
            episodes=episodes,
            retrieval_latency_ms=(time.perf_counter() - t0) * 1000,
            embedding_latency_ms=stats.query_latency_ms,
            embedding_tokens=estimate_tokens(query) if self._chunks else 0,
        )

    # ── Introspection (plan WS-D) ────────────────────────────────────

    def corpus_stats(self) -> dict[str, Any]:
        """Store-size snapshot used for per-run dilution reporting.

        Context-heavy document tests are only interpretable alongside the
        size of the store the retriever had to search: a ``top_k`` of 5 over
        12 chunks is a different experiment from 5 over 300. Read from the
        in-memory mirror, so this never triggers an embedding call.

        Returns:
            ``total_chunks`` / ``total_chars`` / ``total_estimated_tokens``
            plus ``chat_chunks`` / ``document_chunks`` and their char totals.
            The runner reads ``total_chunks`` / ``total_chars`` guardedly, so
            any backend may implement this without breaking the harness.
        """
        chunks = list(self._chunks.values())
        document_chunks = [c for c in chunks if c.get("source") == "document"]
        chat_chunks = [c for c in chunks if c.get("source") == "chat"]
        return {
            "total_chunks": len(chunks),
            "total_chars": sum(c.get("char_count", 0) for c in chunks),
            "total_estimated_tokens": sum(
                c.get("estimated_tokens", 0) for c in chunks
            ),
            "chat_chunks": len(chat_chunks),
            "chat_chars": sum(c.get("char_count", 0) for c in chat_chunks),
            "document_chunks": len(document_chunks),
            "document_chars": sum(c.get("char_count", 0) for c in document_chunks),
        }

    # ── EpisodicMemoryBackend: reset ─────────────────────────────────

    def reset(self) -> None:
        """Rebuild the store from scratch (strict tuple isolation)."""
        self._store = self._build_store()
        self._chunks.clear()
        self._proxy.reset_stats()

    # ── EpisodicMemoryBackend: snapshot export ───────────────────────

    def _vector_l2_norm(self, chunk_id: str) -> float | None:
        """L2 norm of a chunk's stored vector, or ``None`` if unavailable.

        Reads the store's internal vector table; guarded with a defensive
        ``except`` because a LangGraph internals change must never crash a
        benchmark run at export time.
        """
        try:
            per_key = self._store._vectors.get(self._namespace, {}).get(chunk_id)
            if not per_key:
                return None
            vector = next(iter(per_key.values()))
            return float(sum(x * x for x in vector) ** 0.5)
        except Exception:  # noqa: BLE001 — snapshot must never crash the run
            return None

    def export_snapshot(
        self, dest_dir: Path, run_metadata: dict[str, Any]
    ) -> Path:
        """Write ``manifest.json`` + ``episodes.json`` + ``documents.json``.

        The episodes file is a human-auditable dump of every stored chunk
        (including per-chunk ``vector_l2_norm`` for index sanity checks);
        ``documents.json`` contains only ``source="document"`` chunks (plan
        WS-D); the manifest records run identity, backend configuration,
        per-source counts, and the ingested-document provenance supplied by
        the runner in ``run_metadata``.
        """
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        episodes = []
        for chunk_id in sorted(self._chunks):
            entry = dict(self._chunks[chunk_id])
            entry["vector_l2_norm"] = self._vector_l2_norm(chunk_id)
            episodes.append(entry)

        # Plan WS-D: documents are first-class evidence, not just chunks with
        # a tag — they get their own artifact so a run can be audited without
        # filtering ``episodes.json`` by hand.
        document_episodes = [e for e in episodes if e.get("source") == "document"]

        manifest = {
            "backend": "vector",
            "run_metadata": run_metadata,
            "embedding": {
                "id": self._embedding_id,
                "dims": self._dims,
            },
            "chunk_settings": {
                "chunk_size": self._chunk_size,
                "chunk_overlap": self._chunk_overlap,
            },
            "by_source": {
                "chat": sum(1 for e in episodes if e.get("source") == "chat"),
                "document": len(document_episodes),
            },
            "document_chunks": len(document_episodes),
            "document_chars": sum(e["char_count"] for e in document_episodes),
            "total_chunks": len(episodes),
            "total_chars": sum(e["char_count"] for e in episodes),
            "total_estimated_tokens": sum(e["estimated_tokens"] for e in episodes),
            "snapshot_timestamp": utc_now_iso(),
        }

        with open(dest_dir / "manifest.json", "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)
        with open(dest_dir / "episodes.json", "w", encoding="utf-8") as fh:
            json.dump(episodes, fh, indent=2, ensure_ascii=False)
        with open(dest_dir / "documents.json", "w", encoding="utf-8") as fh:
            json.dump(document_episodes, fh, indent=2, ensure_ascii=False)

        return dest_dir

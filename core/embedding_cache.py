"""core.embedding_cache — content-addressed cache for *ingestion* embeddings.

Level 2 ingestion is token-bound: a local ``qwen3-embedding:0.6b`` on CPU needs
~40–130 s for the 8-document D16 cell and ~50 min for the 200-document cell, and
every ``(top_k, chat-model)`` tuple of a sweep re-embeds the *same* chunks. This
module stores those chunk vectors on disk, keyed by a hash of
``(model, dims, text)``, so a warm run re-embeds nothing.

Design (TODOs/embedding-cache-todo.md §2, Option A):

* :class:`EmbeddingCache` — SQLite (stdlib) table
  ``embeddings(key PK, model, dims, vector BLOB, created_at)`` with float32
  vectors (4 KB per 1024-dim vector, ~12 MB for a 200-document corpus), WAL
  journalling and one writer.
* :class:`CachedEmbeddings` — a transparent ``langchain_core.embeddings.
  Embeddings`` wrapper caching **``embed_documents`` only**.

**Query embeddings are never cached.** ``embed_query`` always hits the provider:
it runs inside the measured ``retrieval_latency_ms`` window, so caching it would
turn the P3 latency metric into a measurement of the cache instead of the
embedder.

The wrapper also reports what it actually sent to the provider
(``last_call_stats``), which is what keeps ``embedding_tokens`` honest — see
:class:`~agents.level_2_episodic_rag.backends.vector._TimingEmbeddings`.

Example::

    cache = EmbeddingCache(Path("outputs/embed_cache/qwen3.sqlite"),
                           model_id="ollama/qwen3-embedding:0.6b", dims=1024)
    cached = CachedEmbeddings(client, cache,
                              model_id="ollama/qwen3-embedding:0.6b", dims=1024)
    cached.embed_documents(["chunk A", "chunk B"])   # cold: 2 provider vectors
    cached.embed_documents(["chunk A", "chunk B"])   # warm: 0 provider vectors
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
import time
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

#: ``IN (...)`` window size for lookups. 400 keeps every statement far below
#: SQLite's historical 999 host-parameter limit while still turning ~115 chunk
#: lookups into a single statement.
_LOOKUP_WINDOW = 400

#: File suffix used by :func:`cache_path_for`.
CACHE_FILE_SUFFIX = ".sqlite"


def estimate_tokens(text: str) -> int:
    """Cheap deterministic token estimate (~4 chars per token heuristic).

    Mirrors ``agents.level_2_episodic_rag.backends.base.estimate_tokens`` — the
    same heuristic the backend uses for telemetry — and is deliberately
    duplicated here so ``core`` never imports the ``agents`` package. Parity is
    pinned by ``tests/test_embedding_cache.py``, so the two cannot drift.
    """
    return len(text) // 4


def make_key(model_id: str, dims: int, text: str) -> str:
    """Content-address one chunk: ``sha256(f"{model_id}\\x00{dims}\\x00{text}")``.

    The NUL separators make the three fields unambiguous (no ``model="a"`` +
    ``text="bc"`` vs. ``model="ab"`` + ``text="c"`` collision), ``model_id``
    scopes vectors per embedder and ``dims`` scopes them per vector size, so a
    model upgrade can never serve a stale vector of the same text.
    """
    payload = f"{model_id}\x00{dims}\x00{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def cache_path_for(cache_dir: Path | str, model_id: str) -> Path:
    """Return ``<cache_dir>/<model_id>.sqlite`` (one file per embedding model).

    Colons/slashes in the id (``ollama:qwen3-embedding:0.6b``) are replaced so
    the path is valid on Windows; the key hash keeps the *real* identity.
    """
    safe = "".join(
        char if (char.isalnum() or char in "-_.") else "_" for char in model_id
    )
    return Path(cache_dir) / f"{safe}{CACHE_FILE_SUFFIX}"


# ── On-disk store ────────────────────────────────────────────────────────────


class EmbeddingCache:
    """Content-addressed float32 vector store backed by SQLite.

    One table, one writer, no eviction (a 200-document corpus is ~12 MB; see
    TODO C6). Every row carries ``model``/``dims`` so a stale or foreign cache
    file is *auditable* rather than silently trusted.

    Args:
        path: SQLite file path; parent directories are created on open.
        model_id: Identity of the embedder that produced the vectors
            (``"<provider>/<model_name>"``), recorded per row; callers scope
            keys with it via :func:`make_key`.
        dims: Expected vector dimensionality. Rows whose ``dims`` differ are
            treated as misses, never as errors.

    Raises:
        ValueError: If ``dims`` is not positive.
    """

    def __init__(self, path: Path | str, *, model_id: str, dims: int) -> None:
        if dims <= 0:
            raise ValueError(f"dims must be a positive int, got {dims}")
        self.path = Path(path)
        self.model_id = model_id
        self.dims = dims

        # ``check_same_thread=False``: LangGraph may hop threads between batch
        # windows; the lock serialises the single writer the runner needs (C5).
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path), check_same_thread=False)
        try:
            # WAL lets an inspection script read a cache being written by a
            # running sweep without blocking that sweep's writer.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    key        TEXT PRIMARY KEY,
                    model      TEXT NOT NULL,
                    dims       INTEGER NOT NULL,
                    vector     BLOB NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()
        except Exception:
            conn.close()
            raise
        self._conn = conn
        logger.debug(
            "Embedding cache open: %s (model=%s, dims=%d)",
            self.path,
            self.model_id,
            self.dims,
        )

    # ── Keys ────────────────────────────────────────────────────────

    @staticmethod
    def make_key(model_id: str, dims: int, text: str) -> str:
        """Content-address ``text`` under ``(model_id, dims)`` (see :func:`make_key`)."""
        return make_key(model_id, dims, text)

    # ── Reads ───────────────────────────────────────────────────────

    def get_many(self, keys: Iterable[str]) -> dict[str, list[float]]:
        """Return the cached vectors for ``keys`` (missing/invalid keys absent).

        Lookups are batched in windows of :data:`_LOOKUP_WINDOW`, so a 115-chunk
        ingestion is one statement instead of 115. A row that cannot be used
        (wrong ``dims``, corrupt blob) is simply not returned — a broken cache
        must degrade to a cold run, never crash one.
        """
        unique = list(dict.fromkeys(keys))
        hits: dict[str, list[float]] = {}
        if not unique:
            return hits
        with self._lock:
            conn = self._connection()
            for start in range(0, len(unique), _LOOKUP_WINDOW):
                window = unique[start : start + _LOOKUP_WINDOW]
                placeholders = ", ".join("?" for _ in window)
                rows = conn.execute(
                    "SELECT key, dims, vector FROM embeddings "
                    f"WHERE key IN ({placeholders})",
                    window,
                ).fetchall()
                for key, dims, blob in rows:
                    vector = self._decode(key, dims, blob)
                    if vector is not None:
                        hits[key] = vector
        return hits

    def count(self) -> int:
        """Number of rows currently stored (any model/dims — a file probe)."""
        with self._lock:
            (total,) = self._connection().execute(
                "SELECT COUNT(*) FROM embeddings"
            ).fetchone()
        return int(total)

    # ── Writes ──────────────────────────────────────────────────────

    def put_many(self, vectors: dict[str, list[float]]) -> int:
        """Persist ``{key: vector}`` (upsert) and return the number written.

        Raises:
            ValueError: If any vector's length differs from ``dims``. Every
                vector is validated before the first row is written, so a bad
                vector cannot leave the table half-updated.
        """
        if not vectors:
            return 0
        created_at = datetime.now(timezone.utc).isoformat()
        rows: list[tuple[str, str, int, bytes, str]] = []
        for key, vector in vectors.items():
            width = len(vector)
            if width != self.dims:
                raise ValueError(
                    f"vector for key {key} has {width} dimensions, cache "
                    f"expects {self.dims}"
                )
            # float32 halves the footprint vs. JSON and round-trips to ~1e-7
            # (TODO C4: cosine differs from an uncached run at that scale).
            rows.append(
                (key, self.model_id, self.dims, array("f", vector).tobytes(), created_at)
            )
        with self._lock:
            conn = self._connection()
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings "
                "(key, model, dims, vector, created_at) VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()
        logger.debug("Embedding cache: stored %d vector(s) in %s", len(rows), self.path)
        return len(rows)

    # ── Lifecycle ───────────────────────────────────────────────────

    def close(self) -> None:
        """Close the SQLite connection (idempotent)."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> "EmbeddingCache":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # ── Internals ───────────────────────────────────────────────────

    def _connection(self) -> sqlite3.Connection:
        """Return the open connection, raising if the cache was closed."""
        if self._conn is None:
            raise RuntimeError(f"Embedding cache at {self.path} is closed")
        return self._conn

    def _decode(self, key: str, dims: Any, blob: Any) -> list[float] | None:
        """Decode one row into floats, or return ``None`` for a usable miss.

        Wrong-``dims`` rows and undecodable blobs are MISSES (acceptance
        criterion §5): a cache written by an older or hand-edited build must
        cost a re-embedding, not a crash mid-sweep.
        """
        if dims != self.dims:
            logger.debug(
                "Embedding cache: ignoring %s — stored dims=%s, expected %s",
                key,
                dims,
                self.dims,
            )
            return None
        try:
            values = array("f")
            values.frombytes(blob)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Embedding cache: undecodable blob for key %s (%s) — treating "
                "it as a miss",
                key,
                exc,
            )
            return None
        if len(values) != self.dims:
            logger.warning(
                "Embedding cache: blob for key %s holds %d float(s), expected "
                "%d — treating it as a miss",
                key,
                len(values),
                self.dims,
            )
            return None
        return values.tolist()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"EmbeddingCache(path={str(self.path)!r}, model_id={self.model_id!r}, "
            f"dims={self.dims})"
        )


# ── Embeddings wrapper ───────────────────────────────────────────────────────


class CachedEmbeddings(Embeddings):
    """``Embeddings`` proxy that caches **ingestion** vectors on disk.

    ``embed_documents`` is cached and deduplicated inside the call: each
    distinct text reaches the provider at most once, and the returned list
    mirrors the input order exactly (duplicates are re-expanded from the single
    vector). ``embed_query`` is deliberately **not** cached — the query
    embedding happens inside the measured ``retrieval_latency_ms`` window, so
    serving it from disk would silently turn the P3 latency metric into a
    measurement of the cache.

    Args:
        inner: The real provider client (``OllamaEmbeddings``, ...).
        cache: Backing store (one file per embedding model).
        model_id: Identity of the embedder; scopes the cache keys.
        dims: Vector dimensionality; must equal ``cache.dims``.

    Raises:
        ValueError: If ``dims`` differs from ``cache.dims`` (a mismatched pair
            would silently mix vector spaces).
    """

    def __init__(
        self,
        inner: Embeddings,
        cache: EmbeddingCache,
        *,
        model_id: str,
        dims: int,
    ) -> None:
        if dims != cache.dims:
            raise ValueError(
                f"dims ({dims}) must match the cache's dims ({cache.dims})"
            )
        self.inner = inner
        self.cache = cache
        self.model_id = model_id
        self.dims = dims
        #: Cumulative counters since construction (hit-rate reporting; the
        #: backend reads them for ``manifest.json``).
        self._lifetime: dict[str, int] = {
            "hits": 0,
            "misses": 0,
            "tokens_sent": 0,
            "calls": 0,
        }
        #: Stats of the most recent ``embed_documents`` call — what telemetry
        #: needs to report *actually sent* tokens instead of processed text.
        self.last_call_stats: dict[str, Any] = {
            "hits": 0,
            "misses": 0,
            "tokens_sent": 0,
            "latency_ms": 0.0,
        }

    # ── Introspection ───────────────────────────────────────────────

    def lifetime_stats(self) -> dict[str, int]:
        """Return cumulative ``{hits, misses, tokens_sent, calls}``."""
        return dict(self._lifetime)

    @property
    def cache_path(self) -> Path:
        """Path of the backing SQLite file."""
        return self.cache.path

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"CachedEmbeddings(model_id={self.model_id!r}, dims={self.dims}, "
            f"cache={str(self.cache.path)!r})"
        )

    # ── Cached path: documents ──────────────────────────────────────

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed ingestion texts, serving cached vectors where possible.

        Steps: key every text → collapse duplicates within the call → look the
        distinct keys up → embed **only the misses in one provider call** →
        persist them → reassemble in the original order.
        """
        started = time.perf_counter()
        texts = list(texts)
        keys, first, cached, missing = self._lookup(texts)
        tokens_sent = 0
        if missing:
            payload = [texts[first[key]] for key in missing]
            vectors = self.inner.embed_documents(payload)
            tokens_sent = self._absorb(cached, first, texts, missing, vectors)
        self._record(
            hits=len(first) - len(missing),
            misses=len(missing),
            tokens_sent=tokens_sent,
            started=started,
        )
        return [cached[key] for key in keys]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Async mirror of :meth:`embed_documents`."""
        started = time.perf_counter()
        texts = list(texts)
        keys, first, cached, missing = self._lookup(texts)
        tokens_sent = 0
        if missing:
            payload = [texts[first[key]] for key in missing]
            vectors = await self.inner.aembed_documents(payload)
            tokens_sent = self._absorb(cached, first, texts, missing, vectors)
        self._record(
            hits=len(first) - len(missing),
            misses=len(missing),
            tokens_sent=tokens_sent,
            started=started,
        )
        return [cached[key] for key in keys]

    # ── Uncached path: queries (metric honesty, TODO §2.2) ──────────

    def embed_query(self, text: str) -> list[float]:
        """Embed a query — always the real provider call, never the cache."""
        return self.inner.embed_query(text)

    async def aembed_query(self, text: str) -> list[float]:
        """Async mirror of :meth:`embed_query` (still never cached)."""
        return await self.inner.aembed_query(text)

    # ── Internals ───────────────────────────────────────────────────

    def _lookup(
        self, texts: Sequence[str]
    ) -> tuple[list[str], dict[str, int], dict[str, list[float]], list[str]]:
        """Key ``texts``, dedupe in-call, and split hits from misses.

        Returns:
            ``(keys, first_index_by_key, cached_vectors, missing_keys)`` where
            ``first_index_by_key`` keeps the first position of every distinct
            text so the reassembly is order-exact.
        """
        keys = [self.cache.make_key(self.model_id, self.dims, text) for text in texts]
        first: dict[str, int] = {}
        for index, key in enumerate(keys):
            first.setdefault(key, index)
        cached = self.cache.get_many(first.keys())
        missing = [key for key in first if key not in cached]
        return keys, first, cached, missing

    def _absorb(
        self,
        cached: dict[str, list[float]],
        first: dict[str, int],
        texts: Sequence[str],
        missing: list[str],
        vectors: Sequence[Sequence[float]],
    ) -> int:
        """Persist freshly embedded vectors and return the tokens sent.

        ``tokens_sent`` counts only the *miss* texts — exactly what reached the
        provider — so a warm run reports ~0 billed ingestion tokens.
        """
        if len(vectors) != len(missing):
            raise ValueError(
                f"inner.embed_documents returned {len(vectors)} vector(s) for "
                f"{len(missing)} text(s)"
            )
        fresh = dict(zip(missing, vectors))
        self.cache.put_many(fresh)
        tokens_sent = 0
        for key, vector in fresh.items():
            cached[key] = list(vector)
            tokens_sent += estimate_tokens(texts[first[key]])
        return tokens_sent

    def _record(
        self, *, hits: int, misses: int, tokens_sent: int, started: float
    ) -> None:
        """Publish this call's stats (and accumulate the lifetime counters)."""
        latency_ms = (time.perf_counter() - started) * 1000
        self.last_call_stats = {
            "hits": hits,
            "misses": misses,
            "tokens_sent": tokens_sent,
            "latency_ms": latency_ms,
        }
        self._lifetime["hits"] += hits
        self._lifetime["misses"] += misses
        self._lifetime["tokens_sent"] += tokens_sent
        self._lifetime["calls"] += 1
        logger.debug(
            "embed_documents: %d cache hit(s), %d miss(es), %d token(s) sent, "
            "%.1f ms",
            hits,
            misses,
            tokens_sent,
            latency_ms,
        )


def wrap_with_cache(
    inner: Embeddings,
    *,
    cache_dir: Path | str,
    embedding_id: str,
    model_id: str,
    dims: int,
) -> CachedEmbeddings:
    """Attach a :class:`CachedEmbeddings` around ``inner`` for ``cache_dir``.

    The file name comes from the registry ``embedding_id`` (one file per
    registry entry, TODO C2) while the key scope uses ``model_id`` (the
    *resolved* ``<provider>/<model_name>``): re-pointing a registry alias at a
    different upstream model therefore invalidates its vectors instead of
    serving stale ones.
    """
    path = cache_path_for(cache_dir, embedding_id)
    cache = EmbeddingCache(path, model_id=model_id, dims=dims)
    logger.info(
        "Ingestion embedding cache: %s (model=%s, dims=%d, %d vector(s) stored) "
        "— query embeddings stay uncached",
        path,
        model_id,
        dims,
        cache.count(),
    )
    return CachedEmbeddings(inner, cache, model_id=model_id, dims=dims)

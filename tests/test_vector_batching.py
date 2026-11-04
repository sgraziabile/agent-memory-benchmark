"""Batched chunk writes in ``VectorMemoryBackend`` (no cache, no network).

``InMemoryStore.batch()`` embeds every text of a batch in **one** provider
call, so handing it a window of chunks instead of one chunk at a time is the
difference between one HTTP round-trip per chunk and one per window. These
tests pin that batching:

* does not change chunk ids, vectors or retrieval scores;
* keeps the append-only collision guard *before* any embedding;
* degrades to per-chunk ``put()`` when ``PutOp`` is unavailable;
* survives duplicate chunk texts inside one window (a langgraph limitation);
* reports the same ``embedding_tokens`` telemetry as the unbatched path.
"""

import hashlib

import pytest
from langchain_core.embeddings import Embeddings

from agents.level_2_episodic_rag.backends.base import EpisodeRecord, estimate_tokens
from agents.level_2_episodic_rag.backends.vector import VectorMemoryBackend


class FakeEmbeddings(Embeddings):
    """Deterministic vectors + a call log, so batching is observable."""

    def __init__(self, dims: int = 4) -> None:
        self.dims = dims
        self.doc_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [byte / 255.0 for byte in digest[: self.dims]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.doc_calls.append(list(texts))
        return [self._vec(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self._vec(text)


def make_records(count: int, prefix: str = "doc") -> list[EpisodeRecord]:
    return [
        EpisodeRecord.from_document(
            chunk_id=f"{prefix}_{index:03d}",
            content=f"record {index} covers the cooling loop maintenance window",
        )
        for index in range(count)
    ]


def make_backend(embeddings: FakeEmbeddings, *, batch_size: int = 16) -> VectorMemoryBackend:
    return VectorMemoryBackend(
        embeddings,
        dims=embeddings.dims,
        embedding_id="fake-embed",
        chunk_size=1200,
        chunk_overlap=200,
        embed_batch_size=batch_size,
    )


class TestBatchedChunkWrites:
    def test_chunks_are_embedded_in_windows(self):
        inner = FakeEmbeddings()
        backend = make_backend(inner, batch_size=2)

        telemetry = backend.append(make_records(5))

        assert [len(call) for call in inner.doc_calls] == [2, 2, 1]
        assert len(telemetry.chunk_ids) == 5
        assert backend.corpus_stats()["total_chunks"] == 5

    def test_batch_size_one_reproduces_per_chunk_puts(self):
        inner = FakeEmbeddings()
        backend = make_backend(inner, batch_size=1)
        backend.append(make_records(4))
        assert [len(call) for call in inner.doc_calls] == [1, 1, 1, 1]

    def test_batching_does_not_change_retrieval(self):
        backend_one = make_backend(FakeEmbeddings(), batch_size=1)
        backend_many = make_backend(FakeEmbeddings(), batch_size=4)
        records = make_records(5)

        backend_one.append(records)
        backend_many.append(records)

        left = backend_one.retrieve("cooling loop maintenance", top_k=3)
        right = backend_many.retrieve("cooling loop maintenance", top_k=3)
        assert [(e.chunk_id, round(e.score, 6)) for e in left.episodes] == [
            (e.chunk_id, round(e.score, 6)) for e in right.episodes
        ]

    def test_duplicate_texts_in_one_window_do_not_crash(self):
        """langgraph dedupes batch texts but expects one vector per PutOp."""
        inner = FakeEmbeddings()
        backend = make_backend(inner, batch_size=4)
        records = [
            EpisodeRecord.from_document(chunk_id="dup_a", content="same text"),
            EpisodeRecord.from_document(chunk_id="dup_b", content="same text"),
        ]

        telemetry = backend.append(records)

        assert telemetry.chunk_ids == ["dup_a", "dup_b"]
        assert backend.corpus_stats()["total_chunks"] == 2
        hits = backend.retrieve("same text", top_k=2)
        assert {episode.chunk_id for episode in hits.episodes} == {"dup_a", "dup_b"}

    def test_rejects_invalid_batch_size(self):
        with pytest.raises(ValueError):
            make_backend(FakeEmbeddings(), batch_size=0)

    def test_collision_still_raises_before_embedding(self):
        inner = FakeEmbeddings()
        backend = make_backend(inner, batch_size=2)
        backend.append(make_records(1))
        calls_before = len(inner.doc_calls)

        with pytest.raises(ValueError):
            backend.append(
                [
                    EpisodeRecord.from_document(
                        chunk_id="doc_000", content="different content entirely"
                    )
                ]
            )
        assert len(inner.doc_calls) == calls_before  # guard runs first

    def test_falls_back_when_put_op_is_unavailable(self, monkeypatch):
        inner = FakeEmbeddings()
        backend = make_backend(inner, batch_size=4)
        monkeypatch.setattr(
            "agents.level_2_episodic_rag.backends.vector.PutOp", None
        )

        telemetry = backend.append(make_records(3))

        assert [len(call) for call in inner.doc_calls] == [1, 1, 1]
        assert len(telemetry.chunk_ids) == 3
        assert backend.corpus_stats()["total_chunks"] == 3

    def test_telemetry_reports_processed_tokens(self):
        inner = FakeEmbeddings()
        backend = make_backend(inner, batch_size=3)
        records = make_records(3)

        telemetry = backend.append(records)

        expected = sum(estimate_tokens(record.content) for record in records)
        assert telemetry.embedding_tokens == expected


class TimeoutEmbeddings(FakeEmbeddings):
    """Fails with a timeout whenever a request carries more than ``limit`` texts."""

    def __init__(self, limit: int, dims: int = 4) -> None:
        super().__init__(dims)
        self.limit = limit

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if len(texts) > self.limit:
            self.doc_calls.append(list(texts))
            raise TimeoutError("timed out")
        return super().embed_documents(texts)


class TestTimeoutRetry:
    def test_timed_out_window_is_halved_and_retried(self, caplog):
        inner = TimeoutEmbeddings(limit=4)
        backend = make_backend(inner, batch_size=8)

        with caplog.at_level("INFO", logger="agents.level_2_episodic_rag.backends.vector"):
            telemetry = backend.append(make_records(8))

        # 8 → timeout, then 4 + 4 succeed.
        assert [len(call) for call in inner.doc_calls] == [8, 4, 4]
        assert len(telemetry.chunk_ids) == 8
        assert backend.corpus_stats()["total_chunks"] == 8
        assert any("timed out" in record.message.lower() for record in caplog.records)
        assert any("retrying" in record.message.lower() for record in caplog.records)

    def test_retried_chunks_are_retrievable(self):
        inner = TimeoutEmbeddings(limit=2)
        backend = make_backend(inner, batch_size=8)

        backend.append(make_records(6))

        hits = backend.retrieve("cooling loop", top_k=6)
        assert len(hits.episodes) == 6

    def test_single_chunk_timeout_is_re_raised(self):
        inner = TimeoutEmbeddings(limit=0)
        backend = make_backend(inner, batch_size=1)

        with pytest.raises(TimeoutError):
            backend.append(make_records(1))

    def test_non_timeout_errors_still_propagate(self):
        class BrokenEmbeddings(FakeEmbeddings):
            def embed_documents(self, texts):
                raise RuntimeError("provider exploded")

        backend = make_backend(BrokenEmbeddings(), batch_size=4)
        with pytest.raises(RuntimeError):
            backend.append(make_records(4))

    def test_progress_is_logged_per_window(self, caplog):
        backend = make_backend(FakeEmbeddings(), batch_size=2)

        with caplog.at_level("INFO", logger="agents.level_2_episodic_rag.backends.vector"):
            backend.append(make_records(6))

        messages = [record.message for record in caplog.records]
        assert sum(1 for message in messages if "embedding window" in message) == 3
        assert any("ETA" in message for message in messages)


class TestCliBatchFlag:
    def test_default_window_is_8(self):
        from run import build_parser

        assert build_parser().parse_args([]).embed_batch_size == 8

    def test_window_is_forwarded_to_the_backend(self, monkeypatch):
        import run as run_module

        fake = FakeEmbeddings(dims=4)
        monkeypatch.setattr(run_module, "create_embeddings", lambda *_a, **_k: fake)
        monkeypatch.setattr(run_module, "get_embedding_dims", lambda *_a, **_k: 4)

        backend = run_module._build_memory_backend(
            "vector", "qwen3-embedding-0.6b", embed_batch_size=7
        )

        assert backend._embed_batch_size == 7

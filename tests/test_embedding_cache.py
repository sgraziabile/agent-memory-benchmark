"""On-disk ingestion-embedding cache (``core/embedding_cache.py``).

Pins the contract of TODOs/embedding-cache-todo.md §5 — no network, no Ollama:

* keys are content-addressed and scoped by model + dims;
* float32 storage round-trips approximately (~1e-7), never exactly;
* a cache file survives process/instance boundaries;
* duplicates are collapsed *within* a call (one provider vector per text);
* ``embed_query`` is never served from disk (metric honesty for P3);
* wrong-``dims`` rows and corrupt blobs are misses, not crashes;
* a second backend over the same file makes **zero** provider calls and bills
  **zero** ingestion tokens, while reporting hits;
* ``cache_dir`` never reaches the provider constructor.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3

from array import array
from pathlib import Path

import pytest

from agents.level_2_episodic_rag.backends.base import EpisodeRecord, estimate_tokens
from agents.level_2_episodic_rag.backends.vector import VectorMemoryBackend
from core import embedding_cache
from core.embedding_cache import (
    CachedEmbeddings,
    EmbeddingCache,
    cache_path_for,
    make_key,
)
from tests.test_vector_batching import FakeEmbeddings, make_backend, make_records

DIMS = 4
MODEL_ID = "fake/model"


# ── Helpers ──────────────────────────────────────────────────────────────────


def make_cache(
    tmp_path: Path, *, dims: int = DIMS, model_id: str = MODEL_ID
) -> EmbeddingCache:
    return EmbeddingCache(tmp_path / "cache.sqlite", model_id=model_id, dims=dims)


def make_client(
    inner: FakeEmbeddings, tmp_path: Path, *, dims: int = DIMS
) -> CachedEmbeddings:
    cache = make_cache(tmp_path, dims=dims)
    return CachedEmbeddings(inner, cache, model_id=MODEL_ID, dims=dims)


def make_cached_backend(
    inner: FakeEmbeddings,
    tmp_path: Path,
    *,
    batch_size: int = 16,
    dims: int = DIMS,
) -> tuple[VectorMemoryBackend, CachedEmbeddings]:
    """A ``VectorMemoryBackend`` whose embedder writes into ``tmp_path``."""
    client = make_client(inner, tmp_path, dims=dims)
    return make_backend(client, batch_size=batch_size), client


# ── Keys ─────────────────────────────────────────────────────────────────────


class TestMakeKey:
    def test_key_is_stable_across_calls_and_processes(self):
        first = EmbeddingCache.make_key(MODEL_ID, DIMS, "chunk text")
        second = embedding_cache.make_key(MODEL_ID, DIMS, "chunk text")
        expected = hashlib.sha256(
            f"{MODEL_ID}\x00{DIMS}\x00chunk text".encode("utf-8")
        ).hexdigest()
        assert first == second == expected

    def test_key_is_scoped_by_model(self):
        assert make_key("model-a", 1024, "t") != make_key("model-b", 1024, "t")

    def test_key_is_scoped_by_dims(self):
        assert make_key("model-a", 1024, "t") != make_key("model-a", 4096, "t")

    def test_key_is_scoped_by_text(self):
        assert make_key("model-a", 1024, "t1") != make_key("model-a", 1024, "t2")

    def test_key_fields_cannot_bleed_into_each_other(self):
        # NUL separators make ("a", "bc") and ("ab", "c") distinct.
        assert make_key("a", 4, "bc") != make_key("ab", 4, "c")

    def test_token_estimate_matches_the_backend_heuristic(self):
        # ``core`` duplicates the 4-chars-per-token heuristic to avoid importing
        # the agents package; parity is what keeps telemetry consistent.
        for text in ["", "a", "abcd" * 10, "x" * 1001]:
            assert embedding_cache.estimate_tokens(text) == estimate_tokens(text)


# ── Storage ──────────────────────────────────────────────────────────────────


class TestEmbeddingCacheStorage:
    def test_float32_roundtrip_is_approximate_not_exact(self, tmp_path):
        cache = make_cache(tmp_path)
        key = cache.make_key(MODEL_ID, DIMS, "roundtrip")
        values = [0.1, 1 / 3, -2.718281828, 1e-8]

        cache.put_many({key: values})
        (stored,) = cache.get_many([key]).values()

        assert stored == pytest.approx(values, abs=1e-6)
        # float32, not float64: the values are NOT bit-identical to the input.
        assert stored != values
        cache.close()

    def test_vectors_are_four_kilobytes_per_1024_dims(self, tmp_path):
        cache = make_cache(tmp_path, dims=1024)
        key = cache.make_key(MODEL_ID, 1024, "sizing")
        cache.put_many({key: [0.5] * 1024})

        with sqlite3.connect(cache.path) as conn:
            (blob,) = conn.execute(
                "SELECT vector FROM embeddings WHERE key = ?", (key,)
            ).fetchone()
        assert len(blob) == 4096
        cache.close()

    def test_entries_persist_across_instances(self, tmp_path):
        key = make_key(MODEL_ID, DIMS, "persisted")
        first = make_cache(tmp_path)
        first.put_many({key: [1.0, 2.0, 3.0, 4.0]})
        first.close()

        second = make_cache(tmp_path)
        assert second.count() == 1
        assert second.get_many([key])[key] == pytest.approx(
            [1.0, 2.0, 3.0, 4.0], abs=1e-6
        )
        second.close()

    def test_wal_mode_and_parent_directories_are_created(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "cache.sqlite"
        cache = EmbeddingCache(path, model_id=MODEL_ID, dims=DIMS)
        with sqlite3.connect(path) as conn:
            (mode,) = conn.execute("PRAGMA journal_mode").fetchone()
        assert mode.casefold() == "wal"
        assert cache.count() == 0
        cache.close()

    def test_context_manager_closes_the_connection(self, tmp_path):
        with EmbeddingCache(
            tmp_path / "cache.sqlite", model_id=MODEL_ID, dims=DIMS
        ) as cache:
            cache.put_many({make_key(MODEL_ID, DIMS, "t"): [0.0] * DIMS})
            assert cache.count() == 1

        with pytest.raises(RuntimeError, match="closed"):
            cache.count()

    def test_put_many_rejects_wrong_vector_length(self, tmp_path):
        cache = make_cache(tmp_path)
        key = cache.make_key(MODEL_ID, DIMS, "bad width")

        with pytest.raises(ValueError, match="dimensions"):
            cache.put_many({key: [0.0, 1.0]})

        # Validated before writing: nothing was persisted.
        assert cache.count() == 0
        cache.close()

    def test_put_many_writes_nothing_when_any_vector_is_malformed(self, tmp_path):
        cache = make_cache(tmp_path)
        good = cache.make_key(MODEL_ID, DIMS, "good")
        bad = cache.make_key(MODEL_ID, DIMS, "bad")

        with pytest.raises(ValueError, match="dimensions"):
            cache.put_many({good: [0.0] * DIMS, bad: [0.0] * (DIMS + 1)})

        assert cache.count() == 0
        cache.close()

    def test_wrong_dims_rows_are_misses(self, tmp_path):
        cache = make_cache(tmp_path)
        key = cache.make_key(MODEL_ID, DIMS, "foreign vector")
        with sqlite3.connect(cache.path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?, ?)",
                (
                    key,
                    MODEL_ID,
                    DIMS * 2,
                    array("f", [0.0] * (DIMS * 2)).tobytes(),
                    "now",
                ),
            )
            conn.commit()

        assert cache.get_many([key]) == {}
        cache.close()

    def test_undecodable_blobs_are_misses(self, tmp_path, caplog):
        cache = make_cache(tmp_path)
        truncated = cache.make_key(MODEL_ID, DIMS, "truncated blob")
        wrong_width = cache.make_key(MODEL_ID, DIMS, "too few floats")
        with sqlite3.connect(cache.path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?, ?)",
                (truncated, MODEL_ID, DIMS, b"\x01\x02\x03", "now"),  # 3 bytes
            )
            conn.execute(
                "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?, ?)",
                (wrong_width, MODEL_ID, DIMS, array("f", [0.0, 1.0]).tobytes(), "now"),
            )
            conn.commit()

        with caplog.at_level("WARNING", logger=embedding_cache.__name__):
            hits = cache.get_many([truncated, wrong_width])

        assert hits == {}
        assert any("undecodable" in record.message.lower() for record in caplog.records)
        assert any("float(s)" in record.message for record in caplog.records)
        cache.close()

    def test_get_many_looks_keys_up_in_windows(self, tmp_path):
        cache = make_cache(tmp_path)
        keys = [cache.make_key(MODEL_ID, DIMS, f"chunk {i}") for i in range(450)]
        cache.put_many({key: [1.0] * DIMS for key in keys})

        assert len(cache.get_many(keys)) == 450
        cache.close()

    def test_cache_path_is_one_file_per_embedding_id(self, tmp_path):
        path = cache_path_for(tmp_path, "ollama:qwen3-embedding:0.6b")
        assert path.parent == tmp_path
        assert path.suffix == ".sqlite"
        # Colons are illegal on Windows, so the file name is sanitised...
        assert ":" not in path.name
        # ...while two different registry ids never share a file.
        assert path != cache_path_for(tmp_path, "qwen3-embedding-0.6b")


# ── CachedEmbeddings: documents (cached) ─────────────────────────────────────


class TestCachedDocuments:
    def test_cold_call_embeds_everything_and_reports_misses(self, tmp_path):
        inner = FakeEmbeddings()
        client = make_client(inner, tmp_path)
        texts = ["alpha chunk", "beta chunk"]

        vectors = client.embed_documents(texts)

        assert inner.doc_calls == [texts]  # exactly one provider call
        assert len(vectors) == 2
        assert client.last_call_stats["hits"] == 0
        assert client.last_call_stats["misses"] == 2
        assert client.last_call_stats["tokens_sent"] == sum(
            estimate_tokens(t) for t in texts
        )
        assert client.cache.count() == 2

    def test_duplicate_texts_are_embedded_once_and_order_is_preserved(self, tmp_path):
        inner = FakeEmbeddings()
        client = make_client(inner, tmp_path)

        vectors = client.embed_documents(["alpha", "beta", "alpha"])

        # One provider vector per DISTINCT text, in first-occurrence order.
        assert inner.doc_calls == [["alpha", "beta"]]
        assert vectors[0] == vectors[2]
        assert vectors[0] != vectors[1]
        assert client.last_call_stats == {
            "hits": 0,
            "misses": 2,
            "tokens_sent": estimate_tokens("alpha") + estimate_tokens("beta"),
            "latency_ms": client.last_call_stats["latency_ms"],
        }

    def test_warm_call_embeds_nothing_and_reports_hits(self, tmp_path):
        inner = FakeEmbeddings()
        client = make_client(inner, tmp_path)
        client.embed_documents(["alpha", "beta"])

        inner.doc_calls.clear()
        vectors = client.embed_documents(["alpha", "beta"])

        assert inner.doc_calls == []  # zero provider calls
        assert client.last_call_stats["hits"] == 2
        assert client.last_call_stats["misses"] == 0
        assert client.last_call_stats["tokens_sent"] == 0
        assert len(vectors) == 2

    def test_partial_hits_only_send_the_misses(self, tmp_path):
        inner = FakeEmbeddings()
        client = make_client(inner, tmp_path)
        client.embed_documents(["alpha"])

        inner.doc_calls.clear()
        client.embed_documents(["alpha", "gamma"])

        assert inner.doc_calls == [["gamma"]]
        assert client.last_call_stats["hits"] == 1
        assert client.last_call_stats["misses"] == 1

    def test_lifetime_counters_accumulate(self, tmp_path):
        inner = FakeEmbeddings()
        client = make_client(inner, tmp_path)
        client.embed_documents(["alpha"])
        client.embed_documents(["alpha", "beta"])

        assert client.lifetime_stats() == {
            "hits": 1,
            "misses": 2,
            "tokens_sent": estimate_tokens("alpha") + estimate_tokens("beta"),
            "calls": 2,
        }

    def test_empty_input_is_a_no_op(self, tmp_path):
        inner = FakeEmbeddings()
        client = make_client(inner, tmp_path)

        assert client.embed_documents([]) == []
        assert inner.doc_calls == []
        assert client.last_call_stats["hits"] == 0

    def test_async_documents_mirror_the_sync_cache(self, tmp_path):
        inner = FakeEmbeddings()
        client = make_client(inner, tmp_path)

        first = asyncio.run(client.aembed_documents(["alpha", "alpha", "beta"]))
        second = asyncio.run(client.aembed_documents(["alpha", "beta"]))

        assert inner.doc_calls == [["alpha", "beta"]]
        # The first call returns the provider's exact floats (3 rows, the
        # duplicate re-expanded); the second returns the float32 vectors read
        # back from disk → equal within 1e-6, but not bit-equal.
        assert len(second) == 2
        assert first[0] == first[1]  # the in-call duplicate reuses one vector
        for left, right in ((first[0], second[0]), (first[2], second[1])):
            assert left == pytest.approx(right, abs=1e-6)
        assert client.last_call_stats["hits"] == 2



    def test_mismatched_dims_are_rejected_at_construction(self, tmp_path):
        cache = make_cache(tmp_path, dims=DIMS)
        with pytest.raises(ValueError, match="must match"):
            CachedEmbeddings(FakeEmbeddings(), cache, model_id=MODEL_ID, dims=DIMS * 2)


# ── CachedEmbeddings: queries (never cached) ────────────────────────────────


class TestQueriesAreNeverCached:
    def test_embed_query_always_calls_the_provider(self, tmp_path):
        inner = FakeEmbeddings()
        client = make_client(inner, tmp_path)

        client.embed_query("cooling loop")
        client.embed_query("cooling loop")

        assert inner.query_calls == ["cooling loop", "cooling loop"]
        assert client.lifetime_stats()["calls"] == 0
        assert client.cache.count() == 0

    def test_aembed_query_always_calls_the_provider(self, tmp_path):
        inner = FakeEmbeddings()
        client = make_client(inner, tmp_path)

        asyncio.run(client.aembed_query("cooling loop"))
        asyncio.run(client.aembed_query("cooling loop"))

        assert inner.query_calls == ["cooling loop", "cooling loop"]
        assert client.cache.count() == 0


# ── Backend integration: cold run then warm run over the same file ──────────


class TestWarmBackend:
    def test_second_backend_embeds_nothing_and_reports_hits(self, tmp_path):
        records = make_records(5)

        cold_inner = FakeEmbeddings()
        cold_backend, _ = make_cached_backend(cold_inner, tmp_path)
        cold = cold_backend.append(records)

        assert len(cold_inner.doc_calls) == 1  # one batched window
        assert cold.embedding_cache_hits == 0
        assert cold.embedding_cache_misses == 5
        assert cold.embedding_tokens == sum(
            estimate_tokens(record.content) for record in records
        )

        warm_inner = FakeEmbeddings()
        warm_backend, _ = make_cached_backend(warm_inner, tmp_path)
        warm = warm_backend.append(records)

        assert warm_inner.doc_calls == []  # ZERO provider calls
        assert warm.embedding_cache_hits == 5
        assert warm.embedding_cache_misses == 0
        assert warm.embedding_tokens == 0  # nothing was billed
        assert warm_backend.embedding_cache_stats() == {"hits": 5, "misses": 0}

    def test_warm_run_retrieves_identically(self, tmp_path):
        records = make_records(6)

        cold_backend, _ = make_cached_backend(FakeEmbeddings(), tmp_path)
        cold_backend.append(records)
        warm_backend, _ = make_cached_backend(FakeEmbeddings(), tmp_path)
        warm_backend.append(records)

        cold_hits = cold_backend.retrieve("cooling loop maintenance", top_k=4)
        warm_hits = warm_backend.retrieve("cooling loop maintenance", top_k=4)

        assert [(e.chunk_id, e.rank) for e in cold_hits.episodes] == [
            (e.chunk_id, e.rank) for e in warm_hits.episodes
        ]
        for left, right in zip(cold_hits.episodes, warm_hits.episodes):
            assert left.score == pytest.approx(right.score, abs=1e-6)

    def test_queries_are_embedded_live_even_when_warm(self, tmp_path):
        records = make_records(3)
        cold_backend, _ = make_cached_backend(FakeEmbeddings(), tmp_path)
        cold_backend.append(records)

        warm_inner = FakeEmbeddings()
        warm_backend, _ = make_cached_backend(warm_inner, tmp_path)
        warm_backend.append(records)
        warm_backend.retrieve("cooling loop", top_k=3)
        warm_backend.retrieve("cooling loop", top_k=3)

        # Two searches → two live provider query calls (never served from disk).
        assert warm_inner.query_calls == ["cooling loop", "cooling loop"]

    def test_duplicate_text_window_is_split_and_served_from_cache(self, tmp_path):
        # InMemoryStore dedupes batch texts but needs one vector per PutOp, so
        # the backend splits duplicates out and writes them individually — with a
        # cache attached the duplicate is a HIT instead of a second embed.
        records = [
            EpisodeRecord.from_document(chunk_id="doc_a", content="shared text"),
            EpisodeRecord.from_document(chunk_id="doc_b", content="shared text"),
            EpisodeRecord.from_document(chunk_id="doc_c", content="distinct text"),
        ]
        inner = FakeEmbeddings()
        backend, _ = make_cached_backend(inner, tmp_path, batch_size=8)

        telemetry = backend.append(records)

        assert sorted(telemetry.chunk_ids) == ["doc_a", "doc_b", "doc_c"]
        assert backend.corpus_stats()["total_chunks"] == 3
        embedded = [text for call in inner.doc_calls for text in call]
        assert sorted(embedded) == ["distinct text", "shared text"]
        assert telemetry.embedding_cache_misses == 2
        assert telemetry.embedding_cache_hits == 1

    def test_manifest_labels_a_warm_run(self, tmp_path):
        records = make_records(4)
        cold_backend, _ = make_cached_backend(FakeEmbeddings(), tmp_path)
        cold_backend.append(records)

        warm_backend, _ = make_cached_backend(FakeEmbeddings(), tmp_path)
        warm_backend.append(records)
        dest = warm_backend.export_snapshot(
            tmp_path / "memory" / "run_02", {"run_id": "run_02"}
        )
        manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))

        assert manifest["embedding"]["cache"] == {"hits": 4, "misses": 0}

    def test_reset_clears_cache_counters_but_keeps_the_file(self, tmp_path):
        records = make_records(3)
        backend, client = make_cached_backend(FakeEmbeddings(), tmp_path)
        backend.append(records)
        assert backend.embedding_cache_stats() == {"hits": 0, "misses": 3}

        backend.reset()

        assert backend.embedding_cache_stats() == {"hits": 0, "misses": 0}
        assert client.cache.count() == 3  # the cache file survives the reset

    def test_uncached_backend_reports_no_cache(self, tmp_path):
        backend = make_backend(FakeEmbeddings())
        telemetry = backend.append(make_records(3))

        assert telemetry.embedding_cache_hits == 0
        assert telemetry.embedding_cache_misses == 0
        assert backend.embedding_cache_stats() is None


# ── Factory wiring (cache_dir must never reach the provider) ────────────────


class ConstructedEmbeddings(FakeEmbeddings):
    """Fake provider client that records the kwargs it was constructed with."""

    captured: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dims=DIMS)
        ConstructedEmbeddings.captured = dict(kwargs)


@pytest.fixture
def factory(monkeypatch):
    """``core.model_factory`` with the provider class replaced by a recorder."""
    from core import model_factory

    monkeypatch.setattr(
        model_factory,
        "_resolve_embeddings_class",
        lambda provider: ConstructedEmbeddings,
    )
    model_factory.clear_embeddings_cache()
    yield model_factory
    model_factory.clear_embeddings_cache()


class TestFactoryWiring:
    def test_cache_dir_is_never_forwarded_to_the_provider(self, factory, tmp_path):
        client = factory.create_embeddings(
            "qwen3-embedding-0.6b", cache_dir=tmp_path
        )

        assert isinstance(client, CachedEmbeddings)
        # The reserved kwarg was popped before the constructor ran.
        assert "cache_dir" not in ConstructedEmbeddings.captured
        assert ConstructedEmbeddings.captured["model"] == "qwen3-embedding:0.6b"
        assert client.cache.path == cache_path_for(tmp_path, "qwen3-embedding-0.6b")
        assert client.model_id == "ollama/qwen3-embedding:0.6b"
        assert client.dims == 1024

    def test_without_cache_dir_the_client_is_returned_unwrapped(self, factory):
        client = factory.create_embeddings("qwen3-embedding-0.6b")

        assert not isinstance(client, CachedEmbeddings)
        assert "cache_dir" not in ConstructedEmbeddings.captured

    def test_cache_dir_is_part_of_the_client_cache_key(self, factory, tmp_path):
        cached = factory.create_embeddings("qwen3-embedding-0.6b", cache_dir=tmp_path)
        again = factory.create_embeddings("qwen3-embedding-0.6b", cache_dir=tmp_path)
        fresh = factory.create_embeddings("qwen3-embedding-0.6b")

        assert again is cached  # identical configuration → shared client
        assert fresh is not cached  # opted-in and uncached runs never share


# ── CLI wiring ──────────────────────────────────────────────────────────────


class TestCliFlag:
    def test_flag_defaults_to_off(self):
        from run import build_parser

        assert build_parser().parse_args([]).embed_cache is None

    def test_flag_is_forwarded_to_the_backend(self, monkeypatch, tmp_path):
        import run as run_module

        seen: dict[str, Any] = {}
        client = CachedEmbeddings(
            FakeEmbeddings(),
            EmbeddingCache(
                cache_path_for(tmp_path, "qwen3-embedding-0.6b"),
                model_id=MODEL_ID,
                dims=DIMS,
            ),
            model_id=MODEL_ID,
            dims=DIMS,
        )

        def fake_create_embeddings(embedding_id, *, cache_dir=None):
            seen["embedding_id"] = embedding_id
            seen["cache_dir"] = cache_dir
            return client

        monkeypatch.setattr(run_module, "create_embeddings", fake_create_embeddings)
        monkeypatch.setattr(run_module, "get_embedding_dims", lambda *_a, **_k: DIMS)

        backend = run_module._build_memory_backend(
            "vector", "qwen3-embedding-0.6b", embed_cache_dir=tmp_path
        )

        assert seen == {"embedding_id": "qwen3-embedding-0.6b", "cache_dir": tmp_path}
        assert backend.embedding_cache_stats() == {"hits": 0, "misses": 0}




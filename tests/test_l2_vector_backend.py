"""Offline tests for L2 Milestone 2: VectorMemoryBackend.

Covers TODOs/phase-2-l2-vector-implementation-plan.md Milestone 2:
- append + dense retrieval ranking via a deterministic fake embedding
- ``exclude_session_id`` chat-only exclusion (documents never excluded)
- long-content chunking with RecursiveCharacterTextSplitter
- reset isolation and snapshot export (manifest + episodes + L2 norms)

All tests are fully offline — ``DeterministicFakeEmbedding`` maps each
word to a stable hash bucket (zlib.crc32) and L2-normalizes, so texts
sharing words score high cosine similarity, giving realistic ranking
without any API call.
"""

import json
import zlib

import pytest
from langchain_core.embeddings import Embeddings

from agents.level_2_episodic_rag.backends.base import (
    EpisodeRecord,
    estimate_tokens,
)
from agents.level_2_episodic_rag.backends.vector import VectorMemoryBackend

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


def make_record(
    chunk_id: str,
    session_id: str,
    turn: int,
    content: str,
    *,
    source: str = "chat",
) -> EpisodeRecord:
    return EpisodeRecord.from_content(
        chunk_id=chunk_id,
        session_id=session_id,
        turn_number=turn,
        source=source,  # type: ignore[arg-type]
        content=content,
    )


def make_backend(**kwargs) -> VectorMemoryBackend:
    return VectorMemoryBackend(
        DeterministicFakeEmbedding(), DIMS, embedding_id="fake-test", **kwargs
    )


# ── Constructor validation ────────────────────────────────────────────────────


class TestConstructor:
    def test_rejects_nonpositive_dims(self):
        with pytest.raises(ValueError, match="dims"):
            VectorMemoryBackend(DeterministicFakeEmbedding(), 0)

    def test_rejects_overlap_ge_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            VectorMemoryBackend(
                DeterministicFakeEmbedding(), DIMS, chunk_size=100, chunk_overlap=100
            )


# ── Append + retrieve ────────────────────────────────────────────────────────


class TestAppendRetrieve:
    def test_relevant_episode_ranks_first(self):
        backend = make_backend()
        backend.append(
            [
                make_record("ep_s1_t1", "s1", 1, "the reactor uses helium cooling"),
                make_record("ep_s2_t1", "s2", 1, "my favorite color is blue"),
            ]
        )
        result = backend.retrieve("helium reactor cooling", top_k=2)
        assert len(result.episodes) == 2
        assert result.episodes[0].chunk_id == "ep_s1_t1"
        assert result.episodes[0].score > result.episodes[1].score
        assert [e.rank for e in result.episodes] == [1, 2]

    def test_scores_are_ranked_descending(self):
        backend = make_backend()
        backend.append(
            [
                make_record("ep_s1_t1", "s1", 1, "alpha beta gamma delta"),
                make_record("ep_s2_t1", "s2", 1, "alpha beta gamma"),
                make_record("ep_s3_t1", "s3", 1, "unrelated words entirely here"),
            ]
        )
        result = backend.retrieve("alpha beta gamma", top_k=3)
        scores = [e.score for e in result.episodes]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_truncates(self):
        backend = make_backend()
        backend.append(
            [
                make_record(f"ep_s{i}_t1", f"s{i}", 1, f"topic number {i} words")
                for i in range(1, 5)
            ]
        )
        result = backend.retrieve("topic number", top_k=2)
        assert len(result.episodes) == 2

    def test_append_telemetry_populated(self):
        backend = make_backend()
        tel = backend.append(
            [make_record("ep_s1_t1", "s1", 1, "some content about reactors")]
        )
        assert tel.chunk_ids == ["ep_s1_t1"]
        assert tel.append_latency_ms >= 0.0
        assert tel.embedding_latency_ms >= 0.0
        assert tel.embedding_tokens == estimate_tokens("some content about reactors")

    def test_retrieve_telemetry_populated(self):
        backend = make_backend()
        backend.append([make_record("ep_s1_t1", "s1", 1, "reactor helium")])
        result = backend.retrieve("helium", top_k=1)
        assert result.embedding_latency_ms >= 0.0
        assert result.embedding_tokens == estimate_tokens("helium")
        assert result.retrieval_latency_ms >= result.embedding_latency_ms

    def test_retrieve_on_empty_backend(self):
        result = make_backend().retrieve("anything", top_k=5)
        assert result.episodes == []
        assert result.embedding_tokens == 0

    def test_re_append_identical_record_is_idempotent(self):
        backend = make_backend()
        record = make_record("ep_s1_t1", "s1", 1, "same content")
        backend.append([record])
        backend.append([record])
        assert len(backend._chunks) == 1

    def test_conflicting_chunk_id_raises(self):
        backend = make_backend()
        backend.append([make_record("ep_s1_t1", "s1", 1, "original")])
        with pytest.raises(ValueError, match="append-only"):
            backend.append([make_record("ep_s1_t1", "s1", 1, "different content")])


# ── exclude_session_id semantics ────────────────────────────────────────────


class TestSessionExclusion:
    """Chat episodes of the excluded session are hidden; documents never are."""

    def test_chat_of_excluded_session_hidden(self):
        backend = make_backend()
        backend.append(
            [
                make_record("ep_s1_t1", "s1", 1, "reactor helium cooling design"),
                make_record("ep_s2_t1", "s2", 1, "reactor helium cooling backup"),
            ]
        )
        result = backend.retrieve("reactor helium", top_k=5, exclude_session_id="s1")
        chunk_ids = [e.chunk_id for e in result.episodes]
        assert "ep_s1_t1" not in chunk_ids
        assert "ep_s2_t1" in chunk_ids

    def test_document_chunks_never_excluded(self):
        backend = make_backend()
        backend.append(
            [
                make_record(
                    "doc_spec01_c0",
                    "s1",
                    1,
                    "reactor helium cooling specifications document",
                    source="document",
                ),
                make_record("ep_s1_t1", "s1", 1, "reactor helium chat aside"),
            ]
        )
        result = backend.retrieve("reactor helium", top_k=5, exclude_session_id="s1")
        sources = [e.source for e in result.episodes]
        assert "chat" not in sources
        assert "document" in sources

    def test_no_exclusion_returns_everything(self):
        backend = make_backend()
        backend.append(
            [
                make_record("ep_s1_t1", "s1", 1, "reactor helium alpha"),
                make_record("ep_s2_t1", "s2", 1, "reactor helium beta"),
            ]
        )
        result = backend.retrieve("reactor helium", top_k=5)
        assert {e.session_id for e in result.episodes} == {"s1", "s2"}


# ── Long-content chunking ────────────────────────────────────────────────────


class TestChunking:
    def test_long_content_split_into_subchunks(self):
        backend = make_backend(chunk_size=100, chunk_overlap=20)
        long_text = " ".join(f"reactor detail {i}" for i in range(60))
        tel = backend.append([make_record("ep_s1_t1", "s1", 1, long_text)])
        assert len(tel.chunk_ids) > 1
        assert tel.chunk_ids[0] == "ep_s1_t1_c0"
        assert set(backend._chunks) == set(tel.chunk_ids)
        # All sub-chunks are retrievable and carry chunk metadata.
        result = backend.retrieve("reactor detail", top_k=50)
        assert len(result.episodes) == len(tel.chunk_ids)
        for episode in result.episodes:
            assert episode.metadata["parent_chunk_id"] == "ep_s1_t1"
            assert episode.metadata["chunk_count"] == len(tel.chunk_ids)

    def test_short_content_keeps_original_chunk_id(self):
        backend = make_backend()
        tel = backend.append(
            [make_record("ep_s1_t2", "s1", 2, "short turn about tea")]
        )
        assert tel.chunk_ids == ["ep_s1_t2"]
        assert "ep_s1_t2" in backend._chunks

    def test_subchunk_sizes_consistent(self):
        backend = make_backend(chunk_size=100, chunk_overlap=20)
        long_text = "x" * 50 + " " + "y" * 50 + " " + "z" * 50
        backend.append([make_record("ep_s1_t1", "s1", 1, long_text)])
        for value in backend._chunks.values():
            assert value["char_count"] == len(value["content"])
            assert value["estimated_tokens"] == estimate_tokens(value["content"])


# ── Reset isolation ──────────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_all_episodes(self):
        backend = make_backend()
        backend.append(
            [
                make_record("ep_s1_t1", "s1", 1, "reactor helium one"),
                make_record("ep_s2_t1", "s2", 1, "reactor helium two"),
            ]
        )
        backend.reset()
        assert backend._chunks == {}
        result = backend.retrieve("reactor helium", top_k=5)
        assert result.episodes == []

    def test_backend_reusable_after_reset(self):
        backend = make_backend()
        backend.append([make_record("ep_s1_t1", "s1", 1, "first episode")])
        backend.reset()
        backend.append([make_record("ep_s9_t1", "s9", 1, "second episode")])
        result = backend.retrieve("second episode", top_k=5)
        assert [e.chunk_id for e in result.episodes] == ["ep_s9_t1"]


# ── Snapshot export ───────────────────────────────────────────────────────────


class TestExportSnapshot:
    def test_snapshot_writes_manifest_and_episodes(self, tmp_path):
        backend = make_backend()
        backend.append(
            [
                make_record("ep_s2_t1", "s2", 1, "helium cooling fact one"),
                make_record("ep_s1_t1", "s1", 1, "helium cooling fact two"),
            ]
        )
        dest = backend.export_snapshot(
            tmp_path / "memory" / "run_01",
            {"run_id": "run_01", "scenario_id": "cross_thread", "model_id": "m1"},
        )
        assert (dest / "manifest.json").exists()
        assert (dest / "episodes.json").exists()

        manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["backend"] == "vector"
        assert manifest["embedding"] == {"id": "fake-test", "dims": DIMS}
        assert manifest["total_chunks"] == 2
        assert manifest["run_metadata"]["run_id"] == "run_01"
        assert manifest["total_chars"] == (
            len("helium cooling fact one") + len("helium cooling fact two")
        )

        episodes = json.loads((dest / "episodes.json").read_text(encoding="utf-8"))
        # Sorted by chunk_id for deterministic diffs.
        assert [e["chunk_id"] for e in episodes] == ["ep_s1_t1", "ep_s2_t1"]
        # Fake embeddings are L2-normalized → stored vector norms == 1.0.
        for episode in episodes:
            assert episode["vector_l2_norm"] == pytest.approx(1.0)

    def test_snapshot_after_reset_is_empty_but_valid(self, tmp_path):
        backend = make_backend()
        backend.append([make_record("ep_s1_t1", "s1", 1, "temporary episode")])
        backend.reset()
        dest = backend.export_snapshot(tmp_path / "mem", {"run_id": "r"})
        manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["total_chunks"] == 0
        assert json.loads((dest / "episodes.json").read_text(encoding="utf-8")) == []



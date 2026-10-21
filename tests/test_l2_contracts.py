"""Offline tests for L2 Milestone 1: backend contracts + embeddings factory.

Covers TODOs/phase-2-l2-vector-implementation-plan.md Milestone 1:
- EpisodeRecord / RetrievedEpisode / telemetry schema validation
- EpisodicMemoryBackend abstract interface
- load_embedding_registry / get_embedding_dims / create_embeddings

All tests are fully offline: no provider API is ever contacted
(``OpenAIEmbeddings`` client construction is lazy), and a dummy
``NANOGPT_API_KEY`` satisfies the fail-fast gate.
"""

from pathlib import Path

import pytest
import yaml
from langchain_core.embeddings import Embeddings

from agents.level_2_episodic_rag.backends.base import (
    AppendTelemetry,
    EpisodeRecord,
    EpisodicMemoryBackend,
    RetrievedEpisode,
    RetrievalTelemetry,
    estimate_tokens,
    utc_now_iso,
)
from core.model_factory import (
    clear_embeddings_cache,
    create_embeddings,
    get_embedding_dims,
    load_embedding_registry,
    EMBEDDING_PROVIDER_MAP,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_YAML = REPO_ROOT / "configs" / "models.yaml"


# ── EpisodeRecord contract ───────────────────────────────────────────────────


class TestEpisodeRecord:
    """Verify the persisted-chunk schema and its derived fields."""

    def test_from_content_derives_size_fields(self):
        rec = EpisodeRecord.from_content(
            chunk_id="ep_s1_t2_0",
            session_id="s1",
            turn_number=2,
            source="chat",
            content="The reactor uses helium cooling.",
        )
        assert rec.char_count == len(rec.content)
        assert rec.estimated_tokens == estimate_tokens(rec.content)
        assert rec.timestamp  # auto-derived UTC ISO string
        assert rec.metadata == {}

    def test_from_content_accepts_explicit_timestamp_and_metadata(self):
        rec = EpisodeRecord.from_content(
            chunk_id="doc_spec01_c00",
            session_id="s0",
            turn_number=1,
            source="document",
            content="## Section 1",
            timestamp="2026-10-21T00:00:00+00:00",
            metadata={"doc_id": "spec01", "section_title": "Section 1"},
        )
        assert rec.timestamp == "2026-10-21T00:00:00+00:00"
        assert rec.metadata["doc_id"] == "spec01"

    def test_blank_content_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            EpisodeRecord.from_content(
                chunk_id="ep_s1_t1_0",
                session_id="s1",
                turn_number=1,
                source="chat",
                content="   ",
            )

    def test_zero_turn_number_rejected(self):
        with pytest.raises(ValueError, match="1-based"):
            EpisodeRecord.from_content(
                chunk_id="ep_s1_t0_0",
                session_id="s1",
                turn_number=0,
                source="chat",
                content="text",
            )


# ── Retrieved episodes + telemetry defaults ─────────────────────────────────


class TestRetrievalContracts:
    """Verify retrieval-result and telemetry schema defaults."""

    def test_retrieved_episode_defaults(self):
        ep = RetrievedEpisode(
            chunk_id="ep_s1_t2_0",
            session_id="s1",
            turn_number=2,
            source="chat",
            content="text",
            score=0.87,
            rank=1,
            char_count=4,
            estimated_tokens=1,
        )
        assert ep.metadata == {}

    def test_append_telemetry_defaults(self):
        tel = AppendTelemetry()
        assert tel.chunk_ids == []
        assert tel.append_latency_ms == 0.0
        assert tel.embedding_latency_ms == 0.0
        assert tel.embedding_tokens == 0

    def test_retrieval_telemetry_defaults(self):
        tel = RetrievalTelemetry()
        assert tel.episodes == []
        assert tel.retrieval_latency_ms == 0.0

    def test_backend_interface_is_abstract(self):
        with pytest.raises(TypeError):
            EpisodicMemoryBackend()  # type: ignore[abstract]

    def test_utc_now_iso_is_parseable_utc(self):
        ts = utc_now_iso()
        assert ts.endswith("+00:00") or ts.endswith("Z")


# ── Embedding registry (configs/models.yaml) ──────────────────────────────────


class TestEmbeddingRegistry:
    """Verify the ``embeddings:`` section of models.yaml loads correctly."""

    def test_registry_loads_from_real_config(self):
        registry = load_embedding_registry(MODELS_YAML)
        assert "qwen3-embedding-8b" in registry
        assert "bge-large-en-v1.5" in registry

    def test_primary_dims_declared_explicitly(self):
        # Qwen3-Embedding-8B emits 4096-dim vectors — InMemoryStore must be
        # built with exactly this dimensionality (plan §3.2).
        assert get_embedding_dims("qwen3-embedding-8b", MODELS_YAML) == 4096

    def test_backup_dims_declared_explicitly(self):
        assert get_embedding_dims("bge-large-en-v1.5", MODELS_YAML) == 1024

    def test_unknown_embedding_id_raises_with_available_listed(self):
        with pytest.raises(ValueError, match="Available embedding models"):
            get_embedding_dims("nope-embedding", MODELS_YAML)

    def test_missing_embeddings_key_raises(self, tmp_path):
        cfg = tmp_path / "models.yaml"
        cfg.write_text("models:\n  foo: {}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="top-level 'embeddings'"):
            load_embedding_registry(cfg)

    def test_missing_config_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_embedding_registry(tmp_path / "absent.yaml")

    @pytest.mark.parametrize("bad_dims", [None, 0, -1024, True, "1024"])
    def test_invalid_dims_rejected(self, tmp_path, bad_dims):
        cfg = tmp_path / "models.yaml"
        cfg.write_text(
            yaml.safe_dump({"embeddings": {"bad": {"provider": "nanogpt",
                                                  "model_name": "m", "dims": bad_dims}}}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="'dims'"):
            get_embedding_dims("bad", cfg)


# ── Embedding factory (offline client construction) ──────────────────────────


class TestCreateEmbeddings:
    """Verify create_embeddings without contacting any provider API."""

    @pytest.fixture(autouse=True)
    def _dummy_api_key(self, monkeypatch):
        # Satisfy the NanoGPT fail-fast gate; OpenAIEmbeddings construction
        # is lazy so no network call is made.
        monkeypatch.setenv("NANOGPT_API_KEY", "sk-test-dummy-key")
        clear_embeddings_cache()
        yield
        clear_embeddings_cache()

    def test_constructs_openai_compatible_client(self):
        emb = create_embeddings("qwen3-embedding-8b", MODELS_YAML)
        assert isinstance(emb, Embeddings)

    def test_yaml_params_forwarded_to_client(self):
        # timeout / max_retries from models.yaml must reach the constructor.
        emb = create_embeddings("qwen3-embedding-8b", MODELS_YAML)
        assert getattr(emb, "request_timeout", None) == 60.0 or getattr(
            emb, "timeout", None
        ) in (60.0, 60)

    def test_identical_config_returns_cached_instance(self):
        a = create_embeddings("qwen3-embedding-8b", MODELS_YAML)
        b = create_embeddings("qwen3-embedding-8b", MODELS_YAML)
        assert a is b

    def test_clear_cache_forces_new_instance(self):
        a = create_embeddings("bge-large-en-v1.5", MODELS_YAML)
        clear_embeddings_cache()
        b = create_embeddings("bge-large-en-v1.5", MODELS_YAML)
        assert a is not b

    def test_kwargs_override_changes_cache_slot(self):
        a = create_embeddings("qwen3-embedding-8b", MODELS_YAML, timeout=99.0)
        b = create_embeddings("qwen3-embedding-8b", MODELS_YAML)
        assert a is not b
        assert getattr(a, "request_timeout", getattr(a, "timeout", None)) in (
            99.0,
            99,
        )

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("NANOGPT_API_KEY", raising=False)
        clear_embeddings_cache()
        with pytest.raises(ValueError, match="NANOGPT_API_KEY is not set"):
            create_embeddings("qwen3-embedding-8b", MODELS_YAML)

    def test_unknown_provider_raises(self, monkeypatch, tmp_path):
        # setitem mutates the module-level dict in place, which
        # _resolve_embeddings_class reads at call time.
        monkeypatch.setitem(
            EMBEDDING_PROVIDER_MAP, "bogus", ("no_such_pkg_xyz", "Nope")
        )
        cfg = tmp_path / "models.yaml"
        cfg.write_text(
            yaml.safe_dump({"embeddings": {"x": {"provider": "bogus",
                                                  "model_name": "m", "dims": 8}}}),
            encoding="utf-8",
        )
        with pytest.raises(ImportError, match="not installed"):
            create_embeddings("x", cfg)

    def test_unknown_provider_name_rejected_pre_import(self):
        assert "bogus" not in EMBEDDING_PROVIDER_MAP
        from core.model_factory import _resolve_embeddings_class

        with pytest.raises(ValueError, match="Unknown embedding provider"):
            _resolve_embeddings_class("bogus")

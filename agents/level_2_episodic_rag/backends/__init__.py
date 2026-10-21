# agents.level_2_episodic_rag.backends — Swappable L2 storage backends.
#
# Both backends implement the unified EpisodicMemoryBackend interface from
# agents/level_2_episodic_rag/backends/base.py:
#   - VectorMemoryBackend  — dense retrieval over a LangGraph InMemoryStore
#                           vector index (Milestone 2)
#   - MarkdownMemoryBackend — BM25 lexical retrieval over append-only .md
#                           sections (deferred — see implementation plan)
#
# The L2 graph and BenchmarkRunner depend only on the interface, never on a
# concrete backend, so the two storage engines stay swappable per run.

from agents.level_2_episodic_rag.backends.base import (
    AppendTelemetry,
    EpisodeRecord,
    EpisodeSource,
    EpisodicMemoryBackend,
    RetrievedEpisode,
    RetrievalTelemetry,
    estimate_tokens,
    utc_now_iso,
)
from agents.level_2_episodic_rag.backends.vector import VectorMemoryBackend

__all__ = [
    "AppendTelemetry",
    "EpisodeRecord",
    "EpisodeSource",
    "EpisodicMemoryBackend",
    "RetrievedEpisode",
    "RetrievalTelemetry",
    "VectorMemoryBackend",
    "estimate_tokens",
    "utc_now_iso",
]

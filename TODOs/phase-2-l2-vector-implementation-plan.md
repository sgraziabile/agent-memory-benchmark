# 🚧 Implementation Plan: Phase 2 — L2 Episodic RAG (Vector Backend First)

> **Status:** AGREED (2026-10-21) — derived from `TODOs/phase-2-l2-plan.md`.
> **Scope:** Implement **L2 with the VectorDB backend only**, split into **6 progressive milestones**. Each milestone is independently verifiable and leaves the harness green.
> **Explicitly deferred:** `MarkdownMemoryBackend` + BM25, `system_event` document ingestion, doc-needle / addendum scenarios, multi-session `belief_revision` / `explicit_forget` splits, full `reorganize_outputs.py` / `token_report.py` overhaul, thesis supervision note.

---

## Current State → Required Changes (Vector-Only Scope)

| Area | Current state | Needed for L2-Vector |
|---|---|---|
| `agents/level_2_episodic_rag/` | Does not exist | Backend protocol + `VectorMemoryBackend` + 3-node graph |
| `core/model_factory.py` | Chat models only | `create_embeddings()` + `@lru_cache`, `embeddings:` registry |
| `configs/models.yaml` | `models:` + `judge:` keys | Top-level `embeddings:` key with explicit `dims` |
| `core/schemas.py` | No session / telemetry fields | `session_id`/`attachments` on turns, ~15 telemetry fields, `L2AgentState` |
| `core/runner.py` | Single fixed `thread_id`; `system_event` is a no-op `continue`; token slicing via last `AIMessage` | Per-session `thread_id`, `HumanMessage.id` slicing, telemetry capture, snapshot export before `reset()` |
| `run.py` / `agents/__init__.py` | Levels 0–1 only | Registry entry, `--backend` flag, memory config injection |
| `pyproject.toml` | No `langchain-text-splitters` | Must be added (`RecursiveCharacterTextSplitter` lives there in the 0.3 ecosystem) |

**Existing test that must be updated:** `tests/test_scaffolding.py::TestResolveLevelIds::test_unimplemented_level_raises_with_available_listed` asserts `resolve_level_ids("2")` raises "not implemented" — once L2 is registered, this test must switch to an unimplemented number (e.g. `9`).

---

## Milestone 1 — Data Contracts + Embeddings Plumbing *(pure additions, offline)*

1. **`agents/level_2_episodic_rag/backends/base.py`:**
   - `EpisodeRecord` (`chunk_id`, `session_id`, `turn_number`, `source: "chat" | "document"`, `content`, `char_count`, `estimated_tokens`, `timestamp`, `metadata`)
   - `RetrievedEpisode` (`chunk_id`, `session_id`, `turn_number`, `source`, `content`, `score`, `rank`, `char_count`, `estimated_tokens`, `metadata`)
   - `AppendTelemetry` (`append_latency_ms`, `embedding_latency_ms`, `embedding_tokens`, `chunk_ids`)
   - `RetrievalTelemetry` (`episodes`, `retrieval_latency_ms`, `embedding_latency_ms`, `embedding_tokens`)
   - `EpisodicMemoryBackend` protocol: `append(records)`, `retrieve(query, top_k, *, exclude_session_id=None)`, `reset()`, `export_snapshot(dest_dir, run_metadata)`
2. **`configs/models.yaml`:** top-level `embeddings:` key — `qwen3-embedding-8b` (nanogpt, `dims: 4096`) and `bge-large-en-v1.5` (`dims: 1024`), with base URL + explicit dims so `InMemoryStore` never hardcodes dimensions.
3. **`core/model_factory.py`:** `load_embedding_registry()` + `create_embeddings()` (`@lru_cache`, mirroring `create_model`).
4. **`pyproject.toml`:** add `langchain-text-splitters>=0.3.0`.

**Verification:** offline pytest (registry parsing, schema defaults) — no live API needed.

---

## Milestone 2 — `VectorMemoryBackend` *(offline testable)*

- **`agents/level_2_episodic_rag/backends/vector.py`:**
  - Storage: LangGraph `InMemoryStore(index={"embed": embeddings, "dims": dims})`
  - Chunking: `RecursiveCharacterTextSplitter` (default `chunk_size` ~2000 chars / `chunk_overlap` ~200 chars)
  - `retrieve`: cosine top-`k` with `exclude_session_id` filter for `source="chat"` episodes only (`source="document"` always allowed)
  - `reset()` for strict run isolation between benchmark tuples
  - `export_snapshot(dest_dir, run_metadata)`: `manifest.json` + `episodes.json` (includes `vector_l2_norm`, never raw vector arrays)

**Verification:** unit tests with `DeterministicFakeEmbedding(size=1024)` — append→retrieve roundtrip, exclusion filter, reset isolation, snapshot contents, oversized-turn chunking. 100% offline / API-key-free.

---

## Milestone 3 — Harness Extensions *(backward compatible with L0/L1)*

- **`core/schemas.py`:**
  - `ConversationTurn`: add `session_id: str = "default"` + `attachments: list[str] = []`
  - `TurnResult`: full per-turn telemetry (§4.2) — `session_id`, `role`, `retrieved_context_chars`, `retrieved_context_tokens`, `retrieval_latency_ms`, `append_latency_ms`, `embedding_latency_ms`, `embedding_tokens`, `retrieved_memories`, `appended_chunk_ids`
  - `BenchmarkRunSummary`: memory config fields (`memory_backend`, `embedding_model_id`, `top_k`, `chunk_size`, `chunk_overlap`, `memory_snapshot_path`) + aggregated token/latency metrics (§4.3)
  - `L2AgentState(AgentState)`: per-turn overwrite telemetry channels (no `add_messages` reducer)
- **`core/runner.py`:**
  - Per-session `thread_id = f"{run_id}_{scenario.id}_{turn.session_id}"` (defaults to `"default"` → 100% backward compatible)
  - `HumanMessage.id = uuid4()` + multi-message slicing rule (§4.1): sum `usage_metadata` over all `AIMessage`s after that `HumanMessage.id`
  - Read telemetry channels from `agent_graph.invoke()` output state (guarded with `.get()` so L0/L1 untouched)
  - Snapshot export to `memory/<run_id>/` immediately before `backend.reset()`

**Verification:** full pytest suite green + short **live L0/L1 smoke run** to prove zero regression on existing scenarios.

---

## Milestone 4 — L2 Graph (`retrieve_node → model_node → append_node`)

- **`agents/level_2_episodic_rag/graph.py`:**
  - `retrieve_node`: query backend with current user turn text, `exclude_session_id=current` for chat episodes; format `[Memory #1 | id=... | session=... | turn=... | source=... | score=...]` provenance blocks
  - `model_node`: prefix-cache-preserving injection — `[SystemMessage(system_prompt), *prior_thread_messages, SystemMessage(retrieved_memories_block), latest_human_message]`; ephemeral SystemMessages never persisted into `MemorySaver` state
  - `append_node`: persist formatted turn `User: {input}\nAssistant: {response}` (chunked if oversized) with `session_id`, `turn_number`, `source="chat"`, `timestamp`
  - `MemorySaver` checkpointer for intra-session short-term state
- **`agents/__init__.py`:** register `level_2_episodic_rag: Level2EpisodicRAGAgent`; fix `resolve_level_ids("2")` test to use an unimplemented level number.
- **Verification:** offline unit test with a fake chat model + fake embeddings simulating a 2-session conversation — cross-session recall works, current-session chat excluded, telemetry populated, MemorySaver not polluted by RAG SystemMessages.

---

## Milestone 5 — Runner + CLI Integration

- **`run.py`:** add `--backend` / `-b` flag (default `vector` for L2+; comma-split reserved for future multi-backend runs); inject `memory_backend`, `embedding_model_id`, `top_k`, `chunk_size`, `chunk_overlap` via `config["configurable"]`.
- **Runner lifecycle:** construct backend per run, export snapshot before `reset()` (Milestone 3 hook), pass backend params through the graph.
- **Exports:** add `memory_backend` (and embedding identity) to `summary.csv` / `results.json` merge identity. *(Full script overhaul deferred.)*

**Verification:** `python run.py --level 2 --list-*` sanity checks + dry-run paths without API keys where possible.

---

## Milestone 6 — End-to-End Verification *(Week-1 Day-5 milestone)*

1. **Pre-check:** verify `Qwen/Qwen3-Embedding-8B` is subscription-included on the live NanoGPT catalog (`GET /api/v1/models?detailed=true`) before burning tokens.
2. Author `test_cross_thread_01.yaml` — facts seeded in `session_id: "s1"`, probed in `session_id: "s2"`.
3. Live sweep: `python run.py --level 0,1,2 --model <model> --scenario cross_thread_01`.
   - **Expected:** L0 / L1 score **0%** cross-thread (amnesia); L2-Vector scores **100%**, with full `retrieved_memories` telemetry and `outputs/runs/<ts>/memory/<run_id>/` snapshots.

**Success criterion:** the Week-1 milestone of `phase-2-l2-plan.md` — cross-thread recall works for L2-vector with telemetry + snapshots, no regression for L0/L1.

---

## Deferred Until Vector Pass Is Verified

| Deferred item | Blocking dependency |
|---|---|
| `MarkdownMemoryBackend` (BM25okapi, `.md` sections, `index.md`) | Vector baseline must exist first for the comparative hypothesis (§3.3) |
| `system_event` document ingestion + `attachments` resolution | Agreed: defer until cross-thread chat recall works end-to-end |
| `test_doc_needle_01.yaml`, `test_document_addendum_revision_01.yaml`, `datasets/documents/` fixtures | Requires `system_event` ingestion |
| Multi-session `belief_revision` / `explicit_forget` splits (`s1`/`s2`/`s3`) | Append-only collision analysis (Week 2, Day 4) |
| Full `scripts/reorganize_outputs.py` + `scripts/token_report.py` overhaul | Only the minimal `memory_backend` merge-key slice is done in Milestone 5 |
| Thesis supervision experiment note | After comparative sweeps |

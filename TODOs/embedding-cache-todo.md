# TODO — Content-addressed cache for **ingestion** embeddings (Option A)

> **Status:** PROPOSED (2026-11-04) — not implemented.
> **Scope:** D16 and every document-heavy L2 sweep. Reduce repeated ingestion of
> the *same* chunks to ~0 s without changing retrieval semantics.
> **Explicitly out of scope:** caching query embeddings, persisting an
> `InMemoryStore` (store rehydration), batch-API changes.

---

## 1. Why

D16 re-ingests the corpus for every `(top_k, chat-model)` tuple, and the local
`qwen3-embedding:0.6b` embedder recomputes every chunk from scratch.

Measured 2026-11-04 on the 8-document D16 cell (`document_corpus_size_008_01`,
`--chunk-size 600`, 115 chunks, 15,975 embedding tokens, CPU Ollama):

| configuration | ingestion | billed embedding tokens |
|---|---:|---:|
| per-chunk puts (pre-batching) | 154 s | 15,975 |
| batched puts, 8–16 chunks/window (**shipped**) | 110–131 s | 15,975 |
| batched + **warm embedding cache** (this TODO) | **≈0.004 s** | **0** (queries excluded) |

Batching already removed the per-call overhead; what is left is token-bound
(~10 ms/token here), so extrapolating to the D16 grid:

| cell | chunks | cold | warm (this TODO) |
|---|---:|---:|---:|
| 8 docs | ~118 | ~2 min | ~0 s |
| 60 docs | ~880 | ~15–20 min | ~0 s |
| 200 docs | ~2,940 | ~50 min | ~0 s |
| 15-run grid (5 k × 3 sizes) | — | **≈7 h** | **≈20 min** (one cold pass) |

Every re-run, every `top_k` arm and every scenario sharing `policy_v1.md` /
`ops_handbook.md` pays for the same vectors again.

## 2. Design (Option A — content-addressed vector cache)

Two small pieces; no change to `InMemoryStore`, chunking or ranking.

### 2.1 `EmbeddingCache` (new `core/embedding_cache.py`)

```python
class EmbeddingCache:
    def __init__(self, path, *, model_id: str, dims: int): ...
    @staticmethod
    def make_key(model_id: str, dims: int, text: str) -> str:
        return sha256(f"{model_id}\x00{dims}\x00{text}".encode()).hexdigest()
    def get_many(self, keys) -> dict[str, list[float]]: ...
    def put_many(self, vectors: dict[str, list[float]]) -> None: ...
    def count(self) -> int: ...
```

* Storage: **SQLite** (`stdlib`) + `array("f").tobytes()` → **4 KB per
  1024-dim vector** (~12 MB for the 200-doc corpus). Table:
  `embeddings(key PK, model, dims, vector BLOB, created_at)`.
* Rows whose `dims` differ from the configured value, or whose blob cannot be
  decoded, are **misses**, never exceptions.
* `get_many` looks keys up in windows of ~400 (SQLite parameter cap).
* WAL mode + `check_same_thread=False`; one writer at a time (the runner is
  single-threaded today).

### 2.2 `CachedEmbeddings` wrapper (`langchain_core.embeddings.Embeddings`)

* `embed_documents(texts)` — cached:
  * key every text; collapse duplicates **inside the call** (one provider
    vector per distinct text, original order restored on return);
  * look up hits, call `inner.embed_documents(misses)` once, persist, reassemble;
  * expose `last_call_stats = {hits, misses, tokens_sent, latency_ms}`.
* `embed_query(text)` — **never cached**, always the real provider call. Query
  embedding sits inside the measured `retrieval_latency_ms` window; caching it
  would turn the P3 latency metric into a measurement of the cache.
* Async variants mirror the sync behaviour.

### 2.3 Wiring

1. `core/model_factory.create_embeddings(..., cache_dir: Path | None = None)`:
   wrap the registry client when `cache_dir` is set. Carry the directory
   through the existing `lru_cache` key (a reserved kwargs item that is popped
   before the provider constructor, so it can never leak into a request body).
2. `run.py`: `--embed-cache DIR` (default **off**), forwarded to
   `_build_memory_backend`, and recorded in `memory_backend_config`
   (`embed_cache`) next to `embed_batch_size`.
3. `VectorMemoryBackend`: nothing to change *if* the wrapper is transparent —
   confirm `_TimingEmbeddings` totals still match (it counts
   `sum(estimate_tokens(text))`, i.e. tokens **processed**).

## 3. Metric honesty (do not skip)

* Keep `embedding_tokens` = tokens **processed**, so cached and uncached runs of
  the same workload stay comparable; add a separate `embedding_tokens_sent`
  (miss volume) and `embedding_cache_hits` / `_misses` on `AppendTelemetry`.
* Record `{hits, misses}` in `manifest.json` (`embedding.cache`) and in the run
  summary, so a warm run is *labelled* warm. A 30× faster ingestion must never
  be reportable as a faster embedder.
* Report the cache path + hit rate in the D16 results table.

## 4. Decisions required

| # | Decision | Suggested default |
|---|---|---|
| C1 | Cache on by default? | **No** — opt-in flag, recorded per run |
| C2 | Default location | `outputs/embed_cache/<embedding_id>.sqlite` (git-ignored) |
| C3 | Invalidation | content+model hash only; bump `model_id` on embedder upgrades (record the Ollama digest in the manifest) |
| C4 | float32 round-trip | accepted; document that cosine differs by ~1e-7 from an uncached run |
| C5 | Concurrency | single writer today; WAL is enough for future multi-process sweeps |
| C6 | Cache size cap / eviction | none for now (12 MB worst case); revisit for 1M-chunk corpora |

## 5. Acceptance criteria

- [ ] Second run of `document_corpus_size_008_01` (`k=5`, `--chunk-size 600`)
      reports `ingestion_latency_ms < 100 ms` and `embedding_cache_hits == 115`.
- [ ] Retrieval is unchanged: identical `chunk_id`/`rank` list and scores within
      `1e-6` of the cold run; `lr_report.py` Δcos moves < 1e-4.
- [ ] `embed_query` is never served from disk (counting client: 2 queries →
      2 provider calls).
- [ ] Corrupt blob and wrong-`dims` rows are misses, not crashes.
- [ ] A cache directory is never forwarded to the provider constructor.
- [ ] Unit tests: key stability/scope, dedupe-in-call, persistence across
      processes, hit/miss counters, `batch_size=1` parity, duplicate-text
      windows (langgraph dedupes batch texts — see `test_vector_batching.py`).

## 6. Reference implementation sketch

```python
def embed_documents(self, texts: list[str]) -> list[list[float]]:
    keys = [self.cache.make_key(self.model_id, self.dims, t) for t in texts]
    first = {}
    for i, k in enumerate(keys):
        first.setdefault(k, i)
    cached = self.cache.get_many(list(first))
    missing = [k for k in first if k not in cached]
    sent = 0
    if missing:
        vectors = self.inner.embed_documents([texts[first[k]] for k in missing])
        self.cache.put_many(dict(zip(missing, vectors)))
        cached.update(zip(missing, vectors))
        sent = sum(estimate_tokens(texts[first[k]]) for k in missing)
    self.stats.record(hits=len(first) - len(missing), misses=len(missing), tokens_sent=sent)
    return [cached[k] for k in keys]
```


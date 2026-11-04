# `datasets/documents/` — Document Corpus Fixtures

Documents are the harness's **second memory-ingestion channel** (plan WS-D):
they are chunked and appended to the active `EpisodicMemoryBackend` with
`source="document"` **without ever invoking the chat model**. Ingestion costs
embedding tokens only.

## Committed fixtures

| File | Contents | Used by |
|------|----------|---------|
| `policy_v1.md` | Northwind access policy: roles → authorization levels table, change management (§3.2 is the **only** place that states the reboot level = Level-1), break-glass account `iris-breakglass-77` | `document_needle_01`, `document_addendum_revision_01` |
| `addendum_1.md` | Addendum superseding §3.2 (`Level-2` reboots) and setting the 2026-10-01 runbook compliance window | `document_addendum_revision_01` (via `system_event`) |
| `ops_handbook.md` | Service ports (telemetry ingestion gateway = `7443`), escalation leads (`Rostova`), failover cluster `dr-atlas-04` | `document_cross_thread_01`; `document_needle_01` (topically adjacent distractor that makes the corpus exceed `top_k`) |

## How a scenario references them

Paths resolve **relative to the scenario YAML first**, then here:

```yaml
# Scenario-level corpus: ingested once per tuple, BEFORE turn 1.
documents:
  - doc_id: policy_v1
    path: policy_v1.md
    metadata: {title: "Access Policy (Revision 1)"}

turns:
  # Mid-conversation ingestion: appended when the turn is reached.
  - turn_number: 2
    session_id: s1
    role: system_event
    content: "Compliance published Addendum 1, superseding section 3.2."
    attachments:
      - addendum_1.md
```

Chunk IDs are deterministic:

| Channel | Chunk ID |
|---------|----------|
| Preload (`documents:`) | `doc_<doc_id>` (backend adds `_c<i>` when chunking) |
| `system_event` file | `doc_<file stem>_t<turn>` |
| `system_event` inline content | `doc_evt_<session>_t<turn>` |

Provenance (`doc_id`, `source_path`, `sha256`, char counts) lands in
`outputs/runs/<ts>/memory/<run_id>/manifest.json` under
`run_metadata.documents`, and document chunks are also exported to
`documents.json`.

## Generating a larger corpus

`scripts/generate_corpus.py` produces a seeded, deterministic corpus so
"context volume" is a parameter rather than hand-written text:

```bash
python scripts/generate_corpus.py --docs 20 --chars-per-doc 5000 --seed 7
python scripts/generate_corpus.py --docs 40 --chars-per-doc 8000 --seed 7 --dry-run
```

It writes `generated/corpus_000.md … corpus_00N.md` plus
`generated/corpus_manifest.json` (seed, per-file `sha256`, and which document
holds the needle). Reference them from a scenario as
`path: generated/corpus_003.md`.

## D16 — corpus-size curve (`--needles`)

D16 (stress plan §5.3.5) measures `Recall@5` against corpus size. Instead of a
single hand-written needle it plants **N opaque (entity, answer) pairs**:

```bash
python scripts/generate_corpus.py --docs 8   --chars-per-doc 6000 --seed 7 \
    --needles 8 --out datasets/documents/d16_corpus_008
python scripts/generate_corpus.py --docs 60  --chars-per-doc 6000 --seed 7 \
    --needles 8 --out datasets/documents/d16_corpus_060
python scripts/generate_corpus.py --docs 200 --chars-per-doc 6000 --seed 7 \
    --needles 8 --out datasets/documents/d16_corpus_200
python scripts/build_d16_scenarios.py        # emits the three scenario YAMLs
```

Each needle renders as one sentence

> `The calibration code for the cooling loop LOOP-412 is ARC-9F3K-2214.`

and the generated probe asks only for the entity:

> `What is the calibration code for the cooling loop LOOP-412?`

**Non-inferability contract** (enforced by the generator, which exits non-zero
otherwise): the answer token occurs **exactly once** in the whole corpus, never
in its own probe, and its entity also occurs exactly once — so a correct
response cannot be guessed, inferred from the prompt, or reconstructed from a
sibling document. The gold chunk is necessary. Every corpus owns its own
(answer, entity) set (`seed + corpus size`), so probes are meaningful only
inside their corpus.

`corpus_manifest.json` gains a `needles` list (`doc_id`, `entity`, `answer`,
`probe`, `sentence`, `section`); the legacy `needle` key and single-needle
behaviour are unchanged (the no-needle corpus is byte-identical to the
pre-`--needles` output for the same seed).

Report the resulting curve with:

```bash
python scripts/lr_report.py --scenario-prefix document_corpus_size \
    --csv outputs/lr_recall.csv
```

It recomputes **chunk-level** Recall@k, MRR, Δcos and coverage from
`retrieved_memories`, because the scenario assertion only pins the document
prefix (`must_retrieve_chunk: corpus_0NN` also accepts sibling sub-chunks).

### Ingestion cost: batched embedding requests

Chunks are written through `InMemoryStore.batch()` windows, so one provider call
embeds up to `--embed-batch-size` chunks instead of one call per chunk.

| switch | effect |
|---|---|
| (always on) | one embedding request per window instead of one per chunk |
| `--embed-batch-size N` | chunks per request. Default **8** (≈7–9 s per window on CPU Ollama); 16 windows take ≈35–44 s at `chunk_size=1200` |
| `configs/models.yaml` → embedding `timeout: 240` | headroom for the 16-chunk case; 60 s aborted a 14-minute ingestion on a single slow window (2026-11-04) |
| (automatic) | a **timed-out window is halved and retried**, so a slow request costs one retry instead of the whole ingestion |

Every window logs its progress, so a long ingestion is observable:

```
embedding window 7/47: 8 chunk(s) embedded, 0 written individually,
17.4s this window, 122s elapsed, ETA 1219s
```

Measured on the 8-document D16 cell (115 chunks, 15,975 embedding tokens,
`qwen3-embedding:0.6b` on CPU): **~154 s → 110–131 s** (~1.2–1.4×; the spread is
CPU load variance). What remains is token-bound (~7–10 ms/token on this
machine), so a 200-document cold ingestion is still ≈50 min and the 15-run grid
≈7 h.

### Ingestion cost: on-disk embedding cache (`--embed-cache`)

Batching removed the per-call overhead; the cache removes the *tokens*. Pass
`--embed-cache DIR` and every chunk vector is stored in a content-addressed
SQLite file (`DIR/<embedding-id>.sqlite`, ~4 KB per 1024-dim vector), so a
second run of the same corpus re-embeds nothing.

```bash
# cold: ~160 s ingestion, 15,975 embedding tokens, 0 cache hits
python run.py --level 2 --model minimax-m3 --scenario document_corpus_size_008_01 \
    --backend vector --embedding-model qwen3-embedding-0.6b \
    --chunk-size 600 --top-k 5 --embed-cache outputs/embed_cache

# warm: SAME command → ~10 ms ingestion, ~120 query tokens, 123/123 hits
```

| switch | effect |
|---|---|
| `--embed-cache DIR` | opt-in (default **off**); one SQLite file per embedding model, recorded in `memory_backend_config` and in the run summary |
| (automatic) | **ingestion** vectors only — `embed_query` always calls the provider, because the query embedding sits inside the measured `retrieval_latency_ms` window (a cached query would make P3 a measurement of the cache) |
| (automatic) | vectors are keyed by `sha256(model \x00 dims \x00 text)`; a wrong-`dims` row or a corrupt blob is a **miss**, never a crash |
| (automatic) | `embedding_tokens` means tokens **actually sent**: a warm run reports ~0 instead of the cold 15,975 |

Warm-run honesty: `embedding_cache_hits` / `embedding_cache_misses` are reported
in `results.json` / `summary.csv` (`AppendTelemetry` → `TurnResult` →
`BenchmarkRunSummary`) and in `manifest.json`:

```json
"embedding": {"id": "qwen3-embedding-0.6b", "dims": 1024,
              "cache": {"hits": 120, "misses": 3}}
```

Measured 2026-11-04 (CPU Ollama, 8-doc cell, `--chunk-size 600`, 123 chunks):

| run | ingestion | tokens billed | cache hits/misses | pass rate |
|---|---:|---:|---|---:|
| cold (`rm -rf outputs/embed_cache`) | 146–160 s | 15,975 | 0 / 123 | 8/8 |
| warm (identical command) | **6–12 ms** | **117–236** (8 live query embeddings + any chat chunk whose text changed) | 120–123 / 0–3 | 8/8 |

Retrieved `chunk_id`/`rank` lists and scores are identical to the cold run
(Δscore = 0.0: the corpus chunks round-trip through float32). Only the *chat*
chunk text varies between runs (the model's wording), which is why a warm run can
show a handful of misses and a few hundred tokens rather than exactly 0.

`outputs/embed_cache/` is git-ignored — delete it to force a cold measurement.

Budget: ingestion is charged per tuple. Measured 2026-11-04 with
`qwen3-embedding-0.6b` (Ollama, CPU): 8 docs ≈ 12k tokens ≈ **154 s**; at
≈12 ms/token the 60-doc cell is ≈20 min and the 200-doc cell ≈65 min per run —
each of which collapses to a few milliseconds on the second run with
`--embed-cache`.




| Level | Preload | `system_event` | Result |
|-------|---------|----------------|--------|
| L0 | skipped (+warning) | skipped (no-op) | document probes fail → ablation floor |
| L1 | skipped (+warning) | skipped (no-op) | document probes fail (no long-term store) |
| L2+ | appended with `source="document"`, `session_id="corpus"`, `turn_number=0` | appended at that turn | retrievable from any later session |

Disable ingestion for control runs with `python run.py ... --no-documents`.

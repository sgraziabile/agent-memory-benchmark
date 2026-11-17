# Level 2 (Episodic Naive RAG) — D16 corpus-size curve: 8 vs 60 vs 200 documents

> **Date:** 2026-11-05
> **Scope:** every `document_corpus_size_*` run in `outputs/runs/` — 13 executed runs covering the
> 8- and 60-document cells of the D16 curve (`k ∈ {1, 2, 5}` at 60 docs), plus the first
> 200-document attempt (§8).
> **Agent:** `level_2_episodic_rag` — append-only store, `START → retrieve → model → append → END`;
> retrieval excludes the current session, so the candidate pool is `source="document"` chunks only
> (chat episodes are appended but never retrieved inside `s1`).
> **Backend:** `VectorMemoryBackend` + `qwen3-embedding-0.6b` (local Ollama, 1 024 dims),
> `chunk_overlap=200`, `--embed-batch-size 8`.
> **Models:** `minimax-m3` (8-doc cell, `chunk_size=600`) and `glm-5.3-flash` (8/60/200 cells,
> `chunk_size=1200`), prompt `baseline_react` in all runs.
> **Comparability:** all runs were produced on the post-`9de61ca` memory-block format (content only,
> no `id`/`session`/`turn`/`source`/`score` header), i.e. the same code path. The grid is **not**
> fully factorial — see §5 for exactly which cells are matched and which are confounded.
> **Related:** [`TODOs/phase-2-l2-document-stress-plan.md`](../TODOs/phase-2-l2-document-stress-plan.md)
> §5.3.5 (D16), §5.3.3 (metrics), §6 (reporting rules);
> [`notes/2026-11-03_l2_document_topk_ablation.md`](2026-11-03_l2_document_topk_ablation.md) (the
> fixture-level `top_k` ablation these curves extend);
> [`notes/2026-10-30_l2_episodic_rag_summary.md`](2026-10-30_l2_episodic_rag_summary.md) findings
> 1.13 (L-R never probed) and 1.14 (no preference signal);
> [`datasets/documents/README.md`](../datasets/documents/README.md) (generator contract);
> reporting tool: [`scripts/lr_report.py`](../scripts/lr_report.py).

---

## 0. Executive summary (30 seconds)

- **The curve is flat in accuracy and steep in cost.** At `top_k=5` the retrieval layer answers
  **8/8 probes with 8 documents and 8/8 with 60 documents** (chunk-level Recall@5 = 100 % in both
  corpora; 72/72 probes aggregated in the 8-doc cell, Wilson CI **[94.9 %, 100 %]**). What grows is
  the *ingestion bill*: per-document ingestion is flat in corpus size at fixed chunking
  (**12.4 → 12.8 s/doc**), while moving from 8 docs@600 to 60 docs@1200 multiplies the store
  **115 → 376 chunks**, the embedding work **16.0 k → 103.8 k tokens** and the wall clock
  **154 s → 769 s** for a 7.5× corpus.
- **Every error is a retrieval error.** `P(correct given the gold chunk is in the window) = 100 %` in
  all five cells, so **end-to-end accuracy equals `Recall@k`**: the agent never failed to use a
  retrieved gold chunk. The L2 score reported by this suite is a *pure retrieval score* (D16-12).
- **`k` is the variable that carries the signal, not corpus size.** At fixed model/embedder/
  chunking (`glm-5.3-flash`, `chunk_size=1200`), dropping to `k=1` costs
  **−12.5 pp at 8 docs (7/8 probes, 87.5 %)** and **−37.5 pp at 60 docs (5/8 probes, 62.5 %)**;
  restoring `k=2` at 60 docs recovers **100 %** — at half the cost of `k=5`. Coverage at the two
  extremes is **0.26 % → 1.30 %** of the store: the whole accuracy difference lives inside a
  **1 %-wide retrieval window**.
- **Retrieval is deterministic; only the generator is stochastic.** The nine `minimax-m3` runs over
  the same corpus reproduce **identical ranks `[1,1,1,1,1,1,2,1]` and identical gold scores to 4
  decimals** (0.5704). At 60 docs the `k=1` failure set is **exactly** the set of probes whose gold
  chunk sits at **rank 2** at `k=5` (T1, T3, T8): the `k=1 → k=5` delta is a pure cut-off effect
  with **zero sampling noise**.
- **The minimum viable `k` is 2 in both corpora — measured, not inferred** (`k=2` at 60 docs:
  16/16). The worst observed rank of the answer-bearing chunk is **2** (8 docs: 1/8 probes at rank 2;
  60 docs: 3/8 probes at rank 2). No probe ever needed `k>2`, and no probe was reachable at `k=1`
  when its gold sat at rank 2.
- **The gold-vs-competitor margin does not degrade with corpus size.** Mean
  `cos(q, gold) − max cos(q, non-gold)` is **+0.0314** at 8 docs and **+0.0313** at 60 docs
  (≈5 % of a 0.57 absolute score). Corpus growth changes *how often the gold chunk lands at rank 2*,
  not *how strongly it wins when it is in the window*.
- **All 3 retrieval misses are clean abstentions.** `glm-5.3-flash` answered *"I don't have the …
  code"* in 3/3 misses and **fabricated no code**. The competitor was another **needle document**
  in 2/3 misses (probe-template collision) and a same-entity filler document in 1/3.
- **`k=2` is the efficient point, and it is now measured where it matters.** At 60 documents
  `k=1 → 2 → 5` gives **62.5 % → 100 % → 100 %** (`20261105_173916`, 16/16 assertions). `k=2`
  matches `k=5` exactly (same ranks, same MRR 0.812) while cutting the per-turn context from
  **5 581 → 2 278 chars** and the run cost from **14 954 → 7 208 tokens** (−52 %), **12 530 →
  7 983 ms/turn** and **$0.00184 → $0.00090** (−51 %). Anything above `k=2` buys nothing at this
  corpus size.
- **Cost of a perfect answer: 450–935 tokens and $5.6e-5–$1.2e-4 per passed assertion.** The
  retrieved block is **64–84 % of the prompt tokens** at `k=5`, **71.9 %** at `k=2` and **46–50 %**
  at `k=1`; the gold chunk itself is only **24–31 %** of the injected characters at `k=5` (62.8 % at
  `k=2`). `k=5` buys no extra accuracy over `k=2` here — it only adds distractor mass.
- **The 200-document point is not measured yet.** No `document_corpus_size_200_01` run existed in
  `outputs/runs/` when this note was written; a first attempt on 2026-11-05 died at embedding window
  23/158 from an embedding-server connection reset (§8.1) and wrote no artifact. Its 184 embedded
  chunks are cached, so a re-run resumes. The outcome must be read against the **pre-registered
  prediction of §8.2**; every number in §3–§7 rests on the 8- and 60-document corpora only.

---

## 1. What is actually being measured

D16 is the *retrieval-limit* (L-R) experiment: eight opaque `(entity, answer)` needles are planted
across a seeded synthetic corpus, one probe per needle, and the agent is asked for the entity's
code. The generator guarantees that **the answer token occurs exactly once in the whole corpus and
never in the probe**, so a correct answer is impossible without the gold chunk in the prompt
(non-inferability contract, `datasets/documents/README.md`).

Each graded turn carries exactly two assertions:

| assertion | what it grades |
|---|---|
| `must_contain "<code>"` | the answer token reached the response — *generation* |
| `must_retrieve_chunk corpus_0NN` | a chunk of the gold document reached the window — *retrieval* (substring match on `chunk_id`, so sibling sub-chunks satisfy it) |

8 turns × 2 assertions = **16 assertions per run**, and **8 independent probes** per run. The
distinction matters: at `k=1` the two assertions co-fail by construction (no chunk → no token), so
`16` is not the evidence count — `8` is (§9, threat T-2).

### 1.1 Corpus geometry (offline census, no embeddings)

Counted with the production splitter (`RecursiveCharacterTextSplitter`, `chunk_overlap=200`) over
the exact fixture files the scenarios reference. Embedding tokens are the backend's own
`estimate_tokens` (§`agents/level_2_episodic_rag/backends/base.py:36`).

| corpus | docs | chars | `chunk_size=1200` | `chunk_size=600` |
|---|---:|---:|---:|---:|
| `d16_corpus_008` | 8 | 62 401 | **49 chunks** · ~13.7 k emb-tok | **115 chunks** · ~15.6 k emb-tok |
| `d16_corpus_060` | 60 | 473 236 | **376 chunks** · ~103.2 k emb-tok | **880 chunks** · ~118.0 k emb-tok |
| `d16_corpus_200` | 200 | 1 576 158 | **1 257 chunks** · ~343.8 k emb-tok | **2 933 chunks** · ~392.9 k emb-tok |

Where each needle lands (sub-chunk index inside its document), which is what makes a probe
retrievable at all:

| corpus | `chunk_size=1200` | `chunk_size=600` |
|---|---|---|
| 8 docs | `c0` ×7, `c1` ×1 | `c2` ×8 |
| 60 docs | `c0` ×5, `c1` ×3 | `c1` ×1, `c2` ×6, `c3` ×1 |
| 200 docs | `c0` ×4, `c1` ×4 | `c2` ×8 |

At `chunk_size=600` every needle sits in the third sub-chunk (`c2`) of a 14–15-chunk document, and
the document prefix assertion is satisfied by any of its siblings — this is why the report keeps two
levels of "hit" (`doc_hit` vs `chunk_hit`, `scripts/lr_report.py`).

### 1.2 Probes

| corpus | probes (`doc_id` per turn T1…T8) |
|---|---|
| 8 docs | `corpus_000 … corpus_007` (one per document) |
| 60 docs | `corpus_000, 007, 015, 022, 030, 037, 045, 052` (stride ≈ n/8) |
| 200 docs | `corpus_000, 025, 050, 075, 100, 125, 150, 175` (stride 25) |

All eight probes live in **one session** (`s1`), so the current session's own chat chunks are
excluded from retrieval and every window is documents only. The needle sentence is always in
**section 2** of its document and shares a single template
(`The <role> code for the <object> <ENTITY> is <CODE>.`) — the consequence of that template is
finding **D16-7** (§7).

---

## 2. The executed grid

| corpus | `chunk_size` | model | `k` | runs | result |
|---|---:|---|---:|---:|---|
| 8 docs | 600 | `minimax-m3` | 5 | **9** | 16/16 in all 9 |
| 8 docs | 1200 | `glm-5.3-flash` | 1 | 1 | 14/16 |
| 60 docs | 1200 | `glm-5.3-flash` | 5 | 1 | 16/16 |
| 60 docs | 1200 | `glm-5.3-flash` | 1 | 1 | 10/16 |
| 200 docs | 1200 | `glm-5.3-flash` | 5 | 1 | aborted at ingest (§8.1) |
| 200 docs | 1200 | `glm-5.3-flash` | 1 | 0 | not run |
| 60 docs | 1200 | `glm-5.3-flash` | 2 | 1 | **16/16 — the efficient point (§3.1)** |

The 8-doc cell is the only one with replication (9 runs, by design: it was the ingestion/perf
canary). Every other cell has **n = 1 run**, which is why §8 and §9 are explicit about what a single
8-probe run can and cannot conclude.

**Missing cells that the D16 plan asks for but the grid does not yet contain**

| missing | why it matters |
|---|---|
| 8 docs, `glm-5.3-flash`, `chunk_size=1200`, `k=5` | the only cell that would make the `k=1 → k=5` recovery comparable across corpora at fixed model+chunking |
| 60 docs, `chunk_size=600` | isolates the chunk-size effect from the corpus-size effect (currently 8@600 and 60@1200 differ in two factors) |
| any cell with `--no-documents` | the absence control: proves the answers are unreachable without the corpus (stress plan §6, defect class B3) |
| L0/L1 on the same scenarios | the expected 0 % baselines are declared in the YAML but never measured |

---

## 3. Run inventory and headline metrics

All 13 executed runs. `ingest` is `ingestion_latency_ms`; `cache` is `embedding.cache`
(`hits/misses`) from the memory manifest — three 8-doc runs ingested at **0 s** because the
content-addressed cache (`--embed-cache`) was warm, which is the whole point of that switch.

| run | corpus | model | chunk | `k` | assertions | pass rate | ingest | cache | corpus chunks | tokens | cost |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| `20261104_113129` | 8 | minimax-m3 | 600 | 5 | 16/16 | **100 %** | 154 s | off | 123 | 8 164 | $0.002022 |
| `20261104_133434` | 8 | minimax-m3 | 600 | 5 | 16/16 | **100 %** | 110 s | 0/123 | 123 | 8 157 | $0.002018 |
| `20261104_133734` | 8 | minimax-m3 | 600 | 5 | 16/16 | **100 %** | **0 s** | 123/0 | 123 | 8 157 | $0.002018 |
| `20261104_140845` | 8 | minimax-m3 | 600 | 5 | 16/16 | **100 %** | 131 s | off | 123 | 8 157 | $0.002018 |
| `20261104_144151` | 8 | minimax-m3 | 600 | 5 | 16/16 | **100 %** | 119 s | off | 123 | 8 157 | $0.002018 |
| `20261104_154935` | 8 | minimax-m3 | 600 | 5 | 16/16 | **100 %** | 147 s | 0/123 | 123 | 8 157 | $0.002018 |
| `20261104_155555` | 8 | minimax-m3 | 600 | 5 | 16/16 | **100 %** | 160 s | 0/123 | 123 | 8 157 | $0.002018 |
| `20261104_155914` | 8 | minimax-m3 | 600 | 5 | 16/16 | **100 %** | **0 s** | 120/3 | 123 | 8 170 | $0.002024 |
| `20261104_160026` | 8 | minimax-m3 | 600 | 5 | 16/16 | **100 %** | **0 s** | 123/0 | 123 | 8 157 | $0.002018 |
| `20261104_152012` | 8 | glm-5.3-flash | 1200 | 1 | 14/16 | 87.5 % | 100 s | off | 57 | 5 627 | $0.000756 |
| `20261104_145027` | 60 | glm-5.3-flash | 1200 | 5 | 16/16 | **100 %** | 769 s | off | 384 | 14 954 | $0.001840 |
| `20261104_160957` | 60 | glm-5.3-flash | 1200 | 1 | 10/16 | 62.5 % | 780 s | off | 384 | 6 112 | $0.000867 |
| `20261105_173916` | 60 | glm-5.3-flash | 1200 | **2** | 16/16 | **100 %** | **527 s** | 166/218 | 384 | 7 208 | $0.000896 |

The last row is the `k=2` cell and shows the cache in action inside one run family: the **same
corpus** that took **769–780 s** cold (`chunk 1200`, rows 11–12) took **527 s** with **166 cache
hits / 218 misses** (57 711 embedding tokens sent instead of 103 823). Cache hits are partial
because the 60-doc chunks share their filler text with the first 60 documents of the 200-doc corpus
(same generator + seed), while the needle-bearing chunks differ per corpus.

`error_turns = 0` in all 13 runs. `corpus_chunks` is the store size **at snapshot time**: document
chunks plus the 8 chat episodes the run appends (49 + 8 = 57, 376 + 8 = 384, 115 + 8 = 123).

### 3.1 Retrieval metrics (`scripts/lr_report.py`, chunk-level)

`Recall@k` here is the **honest** recall: the chunk whose text contains the answer token reached the
top-k (`doc_hit` is the softer document-prefix version the scenario assertion uses; they coincide in
these corpora because the needle is unique per document).

| corpus | chunk | model | `k` | probes | **Recall@k** | 95 % CI (Wilson) | doc-hit | MRR | Δcos (gold − best other) | coverage | ingest |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 8 | 600 | minimax-m3 | 5 | 72 (9 runs) | **100 %** | [94.9, 100] | 100 % | 0.938 | **+0.0314** | 4.07 % | 91 s* |
| 8 | 1200 | glm-5.3-flash | 1 | 8 | 87.5 % | [52.9, 97.8] | 87.5 % | 0.875 | +0.5899† | 1.75 % | 100 s |
| 60 | 1200 | glm-5.3-flash | 1 | 8 | 62.5 % | [30.6, 86.3] | 62.5 % | 0.625 | +0.5942† | 0.26 % | 780 s |
| 60 | 1200 | glm-5.3-flash | **2** | 8 | **100 %** | [67.6, 100] | 100 % | **0.812** | +0.1558‡ | 0.52 % | **527 s** |
| 60 | 1200 | glm-5.3-flash | 5 | 8 | **100 %** | [67.6, 100] | 100 % | 0.812 | **+0.0313** | 1.30 % | 769 s |

\* mean over the 9 runs, including the three cache-warm ones (0 s).
† **degenerate margin**: at `k=1` the window contains no competitor, so
`delta_cos()` (`scripts/lr_report.py:88`) returns `gold − 0` — i.e. the raw absolute cosine, *not* a
preference margin. Only the `k=5` rows carry a genuine gold-vs-competitor margin. The honest
`k=1` comparison is the pair `gold_cos 0.5899 vs best_other 0.5383` (8 docs) and
`gold_cos 0.5942 vs best_other 0.5932` (60 docs): at 60 documents the best non-gold chunk in the
single-slot window is **0.0010 away** from the gold — the same coin-flip margin reported as finding
1.14 of the 2026-10-30 summary, now at corpus scale.
‡ **partially degenerate margin**: at `k=2`, 2 of the 8 probes retrieve **both chunks of the gold
document** (`doc_corpus_022_c0+c1`, `doc_corpus_030_c0+c1`), so those turns have no competitor and
`delta_cos` returns the raw gold score (0.667, 0.518). The genuine per-probe margins are the ones
measurable in the `k=5` window (identical retrieved set): **T1 −0.044, T2 +0.030, T3 −0.006,
T4 +0.155, T5 +0.034, T6 +0.011, T7 +0.090, T8 −0.020**, mean **+0.031**. In **3 of 8 probes the
gold chunk scores *worse* than its best competitor** yet still sits inside the window — which is
exactly why `k≥2` is required and why a positive/negative Δcos sign alone is not a retrieval verdict.

**`P(correct | gold chunk reached the window) = 100 %` in all five cells**
(`p_correct_given_chunk_hit = 1.0000` in `outputs/lr_recall.csv`, the machine-readable companion of
this note). Whenever the answer-bearing chunk made it into the prompt, the model produced the code.
Consequently **end-to-end accuracy equals `Recall@k`** in this suite: every failed assertion measured
here is a *retrieval* failure, never a generation failure, and the L2 memory layer's answer quality
is numerically identical to its recall. This is the same conditional the top-`k` ablation observed on
the hand-written L-R fixtures, now reproduced on two generated corpora (8 and 60 documents) and two
chat models (`minimax-m3`, `glm-5.3-flash`).

### 3.2 Rank evidence (the core of the curve)

| corpus | chunk | gold rank per probe T1…T8 | probes at rank ≥2 | minimum viable `k` |
|---|---:|---|---:|---:|
| 8 | 600 | `[1,1,1,1,1,1,2,1]` | 1/8 (12.5 %) | **2** |
| 8 | 1200 | `[1,–,1,1,1,1,1,1]` (– = not retrieved) | unknown (only `k=1` observed) | ≤ 2 |
| 60 | 1200 | `[2,1,2,1,1,1,1,2]` | **3/8 (37.5 %)** | **2** |
| 60 | 1200, `k=1` | `[–,1,–,1,1,1,1,–]` | — | — |
| 60 | 1200, `k=2` | `[2,1,2,1,1,1,1,2]` | 3/8 (identical to `k=5`) | **2** |

The `k=1` row is a **strict subset** of the `k=5` row: the three probes it loses (T1, T3, T8) are
*exactly* the three probes whose gold chunk sits at rank 2 when five slots are available. Retrieval
scores are identical across the two runs (same corpus, same embedder, same queries), so
`Recall@5 − Recall@1 = 37.5 pp` is a **pure cut-off effect with no sampling noise** — the cleanest
single measurement in this note.

The `k=2` row **reproduces the `k=5` rank vector exactly** and scores the same MRR (0.812), with the
same per-probe cosines (`T1 0.722/0.678`, …). Three consequences:

1. **The rank distribution is a property of the retriever, not of `k`**: measuring it once at `k=5`
   predicts every `k ≥ 2` for free, and a `k=1` run can be *derived* from a `k=5` run
   (`{rank=1}` set) without spending another ingestion.
2. **`k=2` is sufficient at 60 documents** — the drop from 62.5 % to 100 % happens entirely between
   `k=1` and `k=2` (D16-3 now measured, not inferred).
3. **`k` counts chunks, not documents**: at `k=2`, 2 of 8 probes spend their whole window on one
   document (`022_c0+c1`, `030_c0+c1`). A 2-slot window is sometimes a *1-document* window, which is
   the sibling-chunk caveat (T-9) showing up in the data.

---

## 4. Cost structure: what corpus size actually buys and costs

### 4.1 Ingestion (the steep axis)

| cell | chunks | emb tokens | ingest | s / doc | s / chunk | **ms / token** |
|---|---:|---:|---:|---:|---:|---:|
| 8 docs @600 | 115 | 15 977 | 154 s | 19.2 | 1.34 | **9.64** |
| 8 docs @1200 | 49 | 14 138 | 100 s | 12.4 | 2.03 | **7.04** |
| 60 docs @1200 (`k=5`) | 376 | 103 823 | 769 s | 12.8 | 2.05 | **7.40** |
| 60 docs @1200 (`k=1`) | 376 | 103 822 | 780 s | 13.0 | 2.07 | **7.51** |
| 60 docs @1200 (`k=2`, 44 % cache hits) | 376 | **57 711 sent** (166 hits / 218 misses) | **527 s** | 8.8 | 1.40 | — (cache) |

Two regularities:

1. **Per document, ingestion is flat in corpus size and linear in characters** (12.4 → 12.8 → 13.0
   s/doc for 8 → 60 docs at identical chunking). Extrapolation to 200 docs is therefore
   **200 × 13 s ≈ 43 min** for one cold ingestion — the single largest cost in the whole L2 suite.
2. **Smaller chunks are ~30 % more expensive per token** (9.64 vs 7.0–7.5 ms/token) for the same
   corpus: `chunk_size=600` doubles the window count and every window pays its own request
   overhead. Chunk size trades **retrieval precision** (§3.2) against **ingestion time** — and
   against 1.5× more vectors to store.
3. **The embedding cache removes the entire cost from repeat runs**: three of the nine 8-doc runs
   report `ingest = 0 s` with `123/0` cache hits, bit-identical retrieval (ranks and scores
   unchanged). Cache-warm runs are the only way a multi-`k` grid over a large corpus is affordable.
   The `k=2` run quantifies the partial case in the *same* corpus and the *same* chunking:
   **166 hits / 218 misses → 527 s against 769–780 s cold (−32 %)** with 44 % of the vectors reused.

### 4.2 Per-query cost

| cell | ctx chars / turn | ctx tokens / turn | ctx share of prompt | gold chunk chars | **payload share** | retrieval ms / turn |
|---|---:|---:|---:|---:|---:|---:|
| 8 @600, `k=5` | 2 551 | 637 | 64.0 % | 542 | 24.2 % | 200–270 (1 221 cold) |
| 8 @1200, `k=1` | 1 159 | 289 | 49.7 % | 1 145 | 98.8 % | 183 |
| 60 @1200, `k=1` | 1 123 | 281 | 46.4 % | 1 085 | 96.6 % | 233 |
| 60 @1200, `k=2` | **2 278** | **569** | **71.9 %** | 1 140 | **62.8 %** | 338 |
| 60 @1200, `k=5` | 5 581 | 1 395 | 84.3 % | 1 138 | 30.8 % | 224 |

- **Retrieval latency is corpus-invariant** (183–338 ms for a 57- vs 384-chunk store): the query
  embedding (~180 ms on local Ollama) dominates a brute-force cosine scan of 1 000 chunks. L-R is
  therefore a *token/ingestion* problem, not a latency problem. (The `k=2` run's 338 ms is
  Ollama-load noise, not a `k` effect — it was measured while another ingestion was running.)
- **Cost of the window scales with `k` and with chunk size, not with corpus size**: the payload share
  (useful chars ÷ injected chars) is **~1/k** — 24.2 % at `k=5`@600, 30.8 % at `k=5`@1200, 62.8 % at
  `k=2`, 96.6–98.8 % at `k=1`. That is the formal statement of "dilution": it is a function of how
  many slots you open, not of how big the corpus is.
- **Tokens per passed assertion** (16 assertions per run): **450** (60@1200 `k=2`) → **510**
  (8@600 `k=5`) → **935** (60@1200 `k=5`) → **382** (60@1200 `k=1`) → **352** (8@1200 `k=1`). Cost per
  passed assertion: **$5.6e-5, $1.26e-4, $1.15e-4, $5.4e-5, $4.7e-5**. `k=2` at 60 docs is therefore
  **2.1× cheaper per assertion than `k=5` at the same accuracy and the same corpus**.

### 4.3 The three curves, side by side

```
Recall@5   8 docs ############################ 100 %  (coverage 4.07 %, 510 tok/assert)
          60 docs ############################ 100 %  (coverage 1.30 %, 935 tok/assert)
Recall@2  60 docs ############################ 100 %  (coverage 0.52 %, 450 tok/assert)
Recall@1   8 docs ##########################--  87.5 % (coverage 1.75 %, 352 tok/assert)
          60 docs ##################--------  62.5 % (coverage 0.26 %, 382 tok/assert)
ingestion  8 docs ###                            154 s   /   16 k emb-tokens
          60 docs ###########################    769 s   /  104 k emb-tokens
          60 docs (44 % cache hits) ############  527 s   /   58 k sent
```

---

## 5. What is actually comparable (and what is not)

The grid mixes models and chunk sizes, so every claim below is tagged with the comparison it
actually supports.

| comparison | factors held fixed | factors varied | verdict |
|---|---|---|---|
| **A. corpus 8 → 60, `k=1`** | model (`glm-5.3-flash`), chunk (1200), embedder, prompt | corpus size | **clean** — 87.5 % → 62.5 % (−25 pp) |
| **B. `k=1 → 2 → 5` at 60 docs** | everything | `k` only | **clean** — 62.5 % → **100 %** → 100 % with retrieval identical (§3.2) |
| **C. `k=1` → `k=5` at 8 docs** | — | model **and** `k` | **confounded** — no 8-doc `glm@1200 k=5` cell exists (§2) |
| **D. 8 docs@600 → 60 docs@1200** | embedder, model family? | chunk **and** corpus, and minimax vs glm | **confounded** — not usable as a curve point |
| **E. 9 replication runs of 8@600** | everything | time, cache state | **clean** repeatability check |

The two clean comparisons (A, B) are exactly the two that make the curve interpretable:

- **Corpus size hurts `Recall@1` and leaves `Recall@2`/`Recall@5` intact** over the 7.5× range
  tested. The mechanism is visible in §3.2: bigger corpora push more probes out of rank 1 into rank
  2, but nothing out of rank 2.
- Because B is noise-free (identical scores across the three `k` values), the **`k` effect is
  measured, not estimated**: accuracy saturates at `k=2` and the remaining 3 slots are pure cost.
  The `k=1` misses are not "the model got unlucky"; they are the deterministic consequence of a
  one-slot window.

### 5.1 Repeatability (the 9-run 8-doc cell)

| quantity | across 9 runs |
|---|---|
| pass rate | **16/16 in 9/9** |
| gold chunk rank, T1…T8 | identical: `[1,1,1,1,1,1,2,1]` |
| gold / best-competitor cosine | identical: 0.5704 / 0.5390 (Δ +0.0314) |
| retrieved context chars / turn | identical: 2 551 |
| total tokens | 8 157 in 7 runs, 8 164 / 8 170 in 2 (generation-length noise only) |
| latency / turn | **2 658 – 7 183 ms (2.7×)** — the only quantity with real variance |
| ingestion | 119–160 s cold, **0 s warm** (cache hit) |

**Retrieval and grading are deterministic; latency is not.** Any L2 accuracy number from this suite
should be reported with the caveat that only generation is stochastic, and that latency variance is
environmental (CPU load, model warm-up), not a property of the agent.

---

## 6. Failure anatomy: the three `k=1` misses at 60 documents

All three misses have the same shape — the single retrieved chunk is a **different document**, the
model says so explicitly, and no assertion records the near-miss.

| turn | probe | gold doc | retrieved (rank 1) | competitor score vs gold | model behaviour |
|---|---|---|---:|---:|---|
| T1 | `LOOP-420` | `corpus_000` | `doc_corpus_045_c0` | **0.722 vs 0.678** (Δ −0.044) | *"I don't have the calibration code for LOOP-420. The documentation provided only contains the calibration code for LOOP-969, which is BQR-485R-6500."* |
| T3 | `BSTORE-440` | `corpus_015` | `doc_corpus_039_c0` | 0.539 vs ? (gold absent) | *"I don't have the decommissioning code … the documentation covers object store compaction … it doesn't contain bucket BSTORE-440."* |
| T8 | `RACK-628` | `corpus_052` | `doc_corpus_007_c1` | 0.518 vs ? (gold absent) | *"I don't have the provisioning code for rack row RACK-628 … only RACK-923, which is UVA-046I-7504."* |

Three observations:

1. **0/3 hallucinated codes.** Every miss is a correct *abstention* — the model refuses rather than
   inventing an answer, and it names the entity it could not find. On this fixture the cost of a
   retrieval miss is a refusal, not a fabricated fact.
2. **The strongest competitor is the probe set itself, not the filler.** T1 and T8 lost to another
   **needle document** (same sentence template, different entity); only T3 lost to a filler document
   that merely shared the entity type. This is finding **D16-7** below: the generator's template
   makes the eight needles each other's hardest distractors.
3. **Over-disclosure is ungraded.** Each miss volunteers a neighbouring needle's code
   (`BQR-485R-6500`, `UVA-046I-7504`) — and so does the *passing* T2 at 8 docs
   (`RACK-857 = CXC-DW7D-3327`, an adjacent entity). No assertion in D16 tests "did the answer leak
   values that were not asked for", so this behaviour costs nothing in the pass rate, exactly as in
   the 2026-11-03 top-`k` note (T1 pollution).

---

## 7. Findings

| # | finding | evidence |
|---|---|---|
| **D16-1** | **`Recall@5` is 100 % at 8 and at 60 documents.** Within the 7.5× corpus range tested, `k=5` is sufficient at every probe. | §3.1: 72/72 probes [94.9, 100] and 8/8 [67.6, 100] |
| **D16-2** | **`Recall@1` degrades with corpus size: 87.5 % → 62.5 %** at fixed model, chunk size and embedder. | §5 comparison A |
| **D16-3** | **The minimum viable `k` is 2 in both corpora, and it is now measured**: at 60 documents `k=2` scores **16/16 (100 %)** while `k=1` scores 10/16 (62.5 %). No probe needed `k>2`, and no probe survived `k=1` when its gold sat at rank 2. | §3.2 rank vectors, `20261105_173916` |
| **D16-4** | **The `k=1 → k=5` loss is a pure cut-off effect.** The failing set at `k=1` equals the rank-2 set at `k=5`, and scores are identical to 4 decimals across runs. | §3.2, §5.1 |
| **D16-5** | **Corpus size does not degrade the gold/competitor margin** (+0.0314 at 8 docs vs +0.0313 at 60 docs); it only changes how often the gold lands at rank 2. | §3.1 |
| **D16-6** | **`k=5` is not free: 64–84 % of the prompt is retrieved context and only 24–31 % of those characters are the gold chunk.** Cost per passed assertion rises from 450 (`k=2`) to 935 (`k=5`) tokens at 60 documents. | §4.2 |
| **D16-7** | **The probe template is the dominant distractor.** Two of the three `k=1` misses at 60 docs lost to another *needle* document, because all eight needles share one sentence shape. The fixture therefore measures "needle vs sibling needle" discrimination, not only needle-vs-filler recall. | §6 |
| **D16-8** | **Ingestion, not inference, is the scaling wall**: 12.4 → 13.0 s/doc (linear), 16 k → 104 k embedding tokens, 154 s → 769 s wall-clock for 7.5× corpus, while retrieval latency stays at ~200 ms. | §4.1, §4.2 |
| **D16-9** | **The embedding cache changes the experiment, not just the speed**: `ingest = 0 s` with `123/0` hits and *bit-identical* retrieval. Any multi-`k` grid at 200 docs is only feasible warm. | §4.1, §5.1 |
| **D16-10** | **Every miss is an abstention, never a fabrication**, and every miss also over-discloses a neighbouring needle's code — a behaviour no assertion in D16 grades. | §6 |
| **D16-11** | **The 16 assertions are 8 independent events**: at `k=1` the `must_contain` and `must_retrieve_chunk` rules co-fail by construction, so pass rates must be reported per probe, never as n = 16. | §1, §9 T-2 |
| **D16-12** | **Generation is not the bottleneck: `P(correct given chunk hit) = 100 %` in all five cells.** End-to-end accuracy equals `Recall@k`, so the L2 score is a retrieval score. | §3.1, `outputs/lr_recall.csv` |
| **D16-13** | **`k=2` dominates `k=5` at 60 documents**: identical recall (100 %), identical MRR (0.812), identical ranks — at **−52 % tokens, −36 % latency/turn, −51 % cost** and a 59 % smaller retrieval block (2 278 vs 5 581 chars/turn). No probe needed the 3rd–5th slot. | §3.1, §3.2, §4.2 |
| **D16-14** | **The rank vector is `k`-invariant, so retrieval measurements compose**: the `k=2` run reproduces the `k=5` scores to 3 decimals, which means a single `k=5` run yields the `k=1`, `k=2` and `k=5` points *without* re-ingesting, and a `k=1` result can be derived analytically from it. | §3.2 |

---

## 8. The 200-document point

### 8.1 Status: aborted (embedding-server connection reset), resumable from cache

The 200-document cell had **no run** in `outputs/runs/` when this note was written. A first attempt was
launched on 2026-11-05 with the exact configuration of the 60-document cells (so the resulting row is
directly comparable) and **died at embedding window 23/158**:

```
17:29:07 | ERROR | Run 1/1 FAILED (scenario=document_corpus_size_200_01, model=glm-5.3-flash):
                   [WinError 10054] An existing connection was forcibly closed by the remote host
17:29:07 | ERROR | 1 of 1 tuples FAILED and were excluded from exports
```

The failure is **infrastructural, not experimental**: the local Ollama server reset the HTTP connection
mid-ingestion (the server was restarted; two `ollama serve` processes came up minutes later). Two
consequences worth recording:

- **A failed tuple writes no artifact.** `core/runner.py` excludes failed tuples from the export, so
  the run directory is created and left **empty** — a killed ingestion is indistinguishable from a run
  that never happened unless you read the log. Always capture stderr when launching a long ingestion.
- **The ingestion cache makes the abort cheap.** 23 windows × 8 chunks = **184 vectors survived** in
  `outputs/embed_cache/qwen3-embedding-0.6b.sqlite` (310 rows in total, including the 8-doc corpus), so
  a re-run re-embeds only the remaining ~1 073 chunks instead of all 1 257.

The command to (re)launch it:

```bash
python run.py --model glm-5.3-flash \
    --scenario document_corpus_size_200_01 --level 2 --backend vector \
    --embedding-model qwen3-embedding-0.6b \
    --top-k 5 --chunk-size 1200 --chunk-overlap 200 \
    --embed-cache outputs/embed_cache
```

| quantity | value |
|---|---|
| documents / chunks / chars | 200 / **1 257** / 1 377 086 |
| embedding tokens (census) | **≈ 343 800** (3.3× the 60-doc cell) |
| embedding windows | 158 (`--embed-batch-size 8`) |
| observed throughput | **31–62 s per 8-chunk window** (≈ 19 ms/token on this machine today; windows 1–16 took 724 s, i.e. ≈ 45 s/window) |
| observed ETA | **≈ 110 min** cold (2.6× the 60-doc cell, and **2× the 50 min** estimated in `datasets/documents/README.md` — CPU throughput varies ≥2.5× between sessions, so ingestion ETAs must be measured, never planned from one benchmark) |
| snapshot size | 1 257 doc chunks + 8 chat episodes = **1 265 chunks** |
| coverage at `k=5` / `k=1` | **0.40 %** / **0.08 %** of the store |

The ingestion is **cache-resumable**: the SQLite cache stores every vector as it is produced, so the
follow-up `k=1` run on the same corpus costs ~0 s of ingestion.

To complete the row once the run lands:

```bash
python scripts/lr_report.py --scenario-prefix document_corpus_size --csv outputs/lr_recall.csv
python scripts/lr_report.py --scenario-prefix document_corpus_size --verbose   # per-probe ranks
python run.py --model glm-5.3-flash --scenario document_corpus_size_200_01 --level 2 \
    --backend vector --embedding-model qwen3-embedding-0.6b \
    --top-k 1 --chunk-size 1200 --chunk-overlap 200 --embed-cache outputs/embed_cache
```

### 8.2 Pre-registered prediction (written before the result existed)

Stated so the 200-doc outcome is a test, not a story:

| prediction | basis | falsified if |
|---|---|---|
| **`Recall@5` at 200 docs ∈ [62.5 %, 100 %]**, i.e. **0–3 probes miss** | rank≥2 events grew 0/8 → 3/8 from 8 → 60 docs; only rank>5 events reduce Recall@5 and none has been observed yet | ≥4 probes miss (Recall@5 ≤ 50 %) ⇒ the curve has a knee between 60 and 200 docs and `k=5` is corpus-size-bound |
| **`Recall@1` at 200 docs ≤ 62.5 %** (≈ 50 % central estimate) | log-linear fit of miss rate on `ln(docs)` through (8 docs, 1/8) and (60 docs, 3/8) gives 0.52 at 200 docs | ≥6/8 probes hit at rank 1 ⇒ the 8→60 degradation was noise, not a trend |
| **gold/competitor Δcos stays ≈ +0.03** | D16-5: margin was corpus-invariant from 8 → 60 docs | mean margin collapses (< +0.01) or turns negative |
| **retrieval latency stays ≈ 200 ms** | brute-force scan of 1 265 chunks is ≪ query embedding | latency > 500 ms |
| **ingestion ≈ 344 k embedding tokens, ≥ 100 min cold on CPU** | census + measured 19 ms/token | tokens ≪ 344 k (would mean chunks were skipped) |

Note the arithmetic humility: with **8 probes**, the Wilson interval for a perfect 8/8 is
**[67.6 %, 100 %]** and for 7/8 is **[52.9 %, 97.8 %]** — the 200-doc cell can distinguish "clearly
broken" from "clearly fine", but **cannot** resolve "100 %" from "88 %".

**Re-running the same scenario does *not* fix that.** Retrieval is deterministic (§5.1: identical
scores to 4 decimals across 9 runs), so a warm re-run reproduces the *same* 8 ranks: it is a
**pseudo-replica** (threat T-11), not a new sample. The only lever that narrows a `Recall@k`
interval is **more independent probes** — more needles per corpus
(`generate_corpus.py --needles 24` + `build_d16_scenarios.py`, which emits one turn per needle) or
more corpus seeds. The intervals are:

| independent probes | all-pass 95 % CI |
|---:|---|
| 8 (today) | [67.6, 100] |
| 16 | [80.6, 100] |
| **24** | **[86.2, 100]** |
| 32 | [89.3, 100] |
| 64 | [94.3, 100] |

---

## 9. Threats to validity

| id | threat | effect | mitigation |
|---|---|---|---|
| **T-1** | **8 probes per run.** | CI width: 8/8 → [67.6, 100]; 5/8 → [30.6, 86.3]. Small differences (e.g. 87.5 % vs 100 %) are **not** separable. | report CIs, never point estimates alone; **add independent probes** (regenerate with `--needles 24`) — NOT more runs of the same probes (T-11) |
| **T-2** | **Assertion-pair correlation.** `must_contain` + `must_retrieve_chunk` co-fail by construction when the chunk is missing. | 16 assertions look like n = 16 but are n = 8 (D16-11); pass rate = `1 − misses/8`. | aggregate by probe (`scripts/lr_report.py`), not by assertion |
| **T-3** | **Confounded grid.** 8-doc cell = `minimax-m3` + `chunk 600`; 60-doc cells = `glm-5.3-flash` + `chunk 1200`. | the 8→60 comparison is clean **only** at `k=1` (§5 A); any 8@600 vs 60@1200 difference mixes model, chunk size and corpus. | add the missing cells in §2 |
| **T-4** | **Single run per `glm` cell.** | generation non-determinism is unmeasured there (it was nil in the 9-run cell: 16/16 × 9). | warm-cache re-runs (2 min each) — these measure **generation** stability only, not retrieval (T-11) |
| **T-5** | **No absence control** (`--no-documents`). | nothing proves the 8 answers require the corpus; non-inferability is a *fixture* property, not a measured one. | one `--no-documents` run per cell; expected 0/16 |
| **T-6** | **No L0/L1 baseline.** | the YAML declares "fails 0 %" for L0/L1 and it is never measured on these scenarios. | 2 runs |
| **T-7** | **Weak embedder.** Only `qwen3-embedding-0.6b` is executable here (`8b` → HTTP 429 `subscription_only`, summary 1.15). | absolute recall is a **lower bound** for a stronger embedder, and the *shape* of the curve may change (a better embedder pushes more probes to rank 1). | re-run one `k=1` cell with a hosted embedder when the subscription allows |
| **T-8** | **Template collision (D16-7).** | the eight needles are each other's hardest distractors, so Recall@k measures sibling-vs-sibling discrimination as much as needle-vs-filler retrieval. | accept and name it, or randomise the needle template per document |
| **T-9** | **Sibling-chunk assertion.** `must_retrieve_chunk corpus_0NN` is a substring match, so any sub-chunk of the gold document satisfies it. | at `chunk_size=600` a document-prefix hit can be a chunk that does *not* contain the needle → the assertion is softer than Recall@k. | two-level report (`doc_hit` vs `chunk_hit`); they coincided in every cell measured here |
| **T-10** | **Ingestion wall-clock is machine-dependent** (7.4 ms/token on 2026-11-04 vs 19 ms/token on 2026-11-05). | `ingestion_latency_ms` is **not** comparable across sessions. | report embedding *tokens* as the cost metric, and latency only within one session |
| **T-11** | **Pseudo-replication.** Re-running the same scenario (same corpus, same embedder, same probes) with a warm cache does **not** add independent samples: retrieval is deterministic, so the 8 probes return the *same* ranks every time. | "3 runs × 8 probes = 24 samples" is **false**; it would leave the `Recall@k` CI at [67.6, 100] while *looking* like more evidence. | replicas are legitimate for *generation* stability, latency and cost variance, and cache reproducibility; for `Recall@k` add **probes** (`--needles 24`) or **seeds** |

---

## 10. Conclusions

**For the thesis (the three claims the data supports today):**

1. **Naïve RAG with `k≥2` is robust to corpus size over the range 8 → 60 documents** (7.5×), with
   **100 % chunk-level Recall@k** in both corpora at `k=5`, a **corpus-invariant gold/competitor
   margin** (+0.031 both) and — measured at 60 documents — **100 % at `k=2`** as well. What corpus
   growth destroys is *rank-1 precision* (87.5 % → 62.5 %), not a 2-slot window. Because
   `P(correct given chunk hit) = 100 %` in every cell, this is also a complete account of the
   agent's accuracy: **the L2 system never failed to answer a question whose evidence it had
   retrieved**.
2. **`k` — not corpus size — is the accuracy and cost dial, and `k=2` is where it should be set.**
   At 60 documents `k=1 → 2 → 5` reads 62.5 % → **100 %** → 100 % while the context block goes
   1 123 → **2 278** → 5 581 chars/turn: the last three slots buy **zero** accuracy and cost **+145 %
   context, +57 % latency and +105 % dollars**. The honest economic statement of an L2 memory layer
   is therefore *"accuracy ⇒ minimum viable `k` × chunk size"*, and the tuning question is
   *"what is the smallest `k` that still covers rank 2?"* — not *"how much context can we afford?"*.
3. **The failure mode is safe but blind.** All retrieval misses produce **abstentions, not
   fabrications** (0/3 hallucinated codes), and every miss is accompanied by an **ungraded
   over-disclosure** of a neighbouring needle's code. A benchmark that only grades "did the asked
   code appear" cannot see this behaviour.
4. **The measurement itself is cheap once the ingestion is cached**: because the rank vector is
   `k`-invariant (D16-14), one warm run at `k=5` yields the whole `k` curve for a corpus, and a
   `k=2` re-run costs minutes (527 s of which 44 % of the vectors were cache hits). The expensive
   part of an L-R experiment is the *first* ingestion, never the sweeps.

**What must be reported alongside every number above:** the coverage ratio (`k / corpus_chunks`:
0.26 %–4.07 % here), `chunk_size`, the embedding model, and the probe count (8) with its Wilson
interval. Without them a `k=5` number from an 8-document corpus is not comparable to a `k=5` number
from a 60-document corpus — exactly the reporting rule the stress plan (§6) already demands.

**Immediate next steps, in order of information per minute of compute:**

| # | action | cost | what it settles |
|---|---|---|---|
| 1 | finish the 200-doc `k=5` run + the warm `k=1` companion | ~15–40 min with a warm partial cache, then ~2 min | the third curve point + the pre-registered predictions of §8.2 |
| 2 | `k=2` at 8 docs (`glm-5.3-flash`, `chunk 1200`) | ~2 min (cache warm) | closes confound C from the other end: `k=2` at 8 docs should also be 100 % (D16-3) |
| 3 | 8 docs, `glm-5.3-flash`, `chunk 1200`, `k=5` | ~2 min (cache warm) | the 8-doc `k=5` cell at the same model/chunking as the 60-doc cells |
| 4 | `--no-documents` control on one corpus | ~2 min | T-5: proves the answers need the corpus |
| 5 | **more probes**: regenerate `d16_corpus_060` with `--needles 24` + `build_d16_scenarios.py` (24 turns) and run one cell | partial ingestion (~60 % cache hits: only the needle-bearing documents change) | moves `Recall@5` from [67.6, 100] to **[86.2, 100]** — the only lever that narrows the recall CI (T-11) |
| 6 | 3 warm replicas of one `glm` cell | ~6 min | measures **generation** stability (same context ⇒ same answer?), latency and cost spread; does **not** narrow the recall CI (T-11) |
| 7 | 60 docs with `chunk_size=600` | ~20 min cold | separates the chunk-size effect from the corpus-size effect (T-3) |

---

## 10.1 What adding needles does — and does not — buy

The cheapest way to narrow the recall CI is more probes (§8.2, T-11). Before spending that compute,
note exactly what the fixture does with them (`scripts/generate_corpus.py:137-219`,
`scripts/build_d16_scenarios.py:140-145`):

| property of the fixture | consequence for more needles |
|---|---|
| **one needle per document**, documents chosen by even spacing (`1 + k*docs//count`) | 60 docs → up to 60 probes; **8 docs → still 8 probes** (the 8-doc cell can only grow by adding *corpus seeds*, not needles) |
| **all needles in `section: 2`** (0-based index, constant) | more needles add **no positional diversity**: "lost in the middle" stays unmeasured, and every recall number is conditional on the needle sitting in the 3rd section |
| **one sentence template**, with artifact/subject/entity-prefix cycling on `k % len(...)` | 24 needles ≈ 4 near-twin sentences per (artifact, subject) pair → **more same-template decoys per probe**, i.e. D16-7 gets worse by construction |
| **single session `s1`** for all turns | the chat history grows with the probe index, and **over-disclosure leaks neighbouring codes** (measured: *"only LOOP-969 = BQR-485R-6500"*, *"RACK-923 = UVA-046I-7504"*) |

Four consequences:

1. **It is a new difficulty point, not more evidence for the same point.** `Recall@k` at 24 needles
   is measured against 23 same-template competitors instead of 7, so it is **not** comparable with
   the 8-needle cells in §3. Run it as its own cell (own corpus dir, own scenario id) and keep the
   8-needle runs as the baseline.
2. **The non-inferability contract is corpus-level, not session-level.** With enough over-disclosure,
   the answer to probe *j* can already be in the chat when probe *j* is asked, so `must_contain`
   could pass **without any retrieval** while `must_retrieve_chunk` fails. With 8 needles that pair
   never dissociated (0/104 probes); with 24 the channel is ~3× wider. Mitigation: one session (or
   `thread_id`) per probe, or grade the **dissociation** of the two assertions explicitly.
3. **Probes stop being independent (cluster effect).** Probes sharing an (artifact, subject) pair
   compete for the same decoys, so their failures correlate; Wilson assumes i.i.d. Bernoulli and
   will report an **optimistic** interval. Report the effective sample size or bootstrap by cluster
   (entity type / artifact).
4. **Turn position becomes a cost confound**: the prompt grows with history (715 → 1 068 tokens
   across 8 turns), so at 24 turns the last probes pay far more context. Report cost per turn index.

**What a good 24-probe cell should vary on purpose** (otherwise the extra probes only buy precision):
needle section/depth (0..N-1 instead of always 2), one session per probe, and either a second
sentence template or a deliberate "near-duplicate decoys" arm. With those three knobs the same
compute answers *"does recall depend on where the fact sits in the document?"* and *"how much of the
failure is near-duplicate competition vs volume?"* — two questions the current 8-probe design cannot
answer at all.

---

## 11. Reproduction

```bash
# 1. corpora + scenarios (deterministic, already committed)
python scripts/generate_corpus.py --docs 8   --chars-per-doc 6000 --seed 7 --needles 8 --out datasets/documents/d16_corpus_008
python scripts/generate_corpus.py --docs 60  --chars-per-doc 6000 --seed 7 --needles 8 --out datasets/documents/d16_corpus_060
python scripts/generate_corpus.py --docs 200 --chars-per-doc 6000 --seed 7 --needles 8 --out datasets/documents/d16_corpus_200
python scripts/build_d16_scenarios.py

# 2. the runs in this note (L2 + vector backend + local embedder)
python run.py --model minimax-m3    --scenario document_corpus_size_008_01 --level 2 \
    --backend vector --embedding-model qwen3-embedding-0.6b \
    --top-k 5 --chunk-size 600  --chunk-overlap 200 --embed-cache outputs/embed_cache
python run.py --model glm-5.3-flash --scenario document_corpus_size_060_01 --level 2 \
    --backend vector --embedding-model qwen3-embedding-0.6b \
    --top-k 5 --chunk-size 1200 --chunk-overlap 200 --embed-cache outputs/embed_cache
python run.py --model glm-5.3-flash --scenario document_corpus_size_060_01 --level 2 \
    --backend vector --embedding-model qwen3-embedding-0.6b \
    --top-k 1 --chunk-size 1200 --chunk-overlap 200 --embed-cache outputs/embed_cache

# 3. the metrics in §3 (chunk-level recall, MRR, Δcos, coverage)
python scripts/lr_report.py --scenario-prefix document_corpus_size --csv outputs/lr_recall.csv
python scripts/lr_report.py --scenario-prefix document_corpus_size --verbose
```

The offline corpus census in §1.1 needs no network: it loads each fixture, splits it with
`langchain_text_splitters.RecursiveCharacterTextSplitter(chunk_size, chunk_overlap=200)` — the same
splitter `VectorMemoryBackend` uses — and sums `estimate_tokens` per chunk.

> **Shell note.** The `\` continuations above are bash-style. In **PowerShell** they are literal
> arguments (argparse then aborts before the banner), so either write each command on one line or use
> PowerShell's backtick. This matters for `--embed-cache`: a command that silently loses the flag
> produces `embed_cache=off` in the banner (`run.py:370`, `run.py:429-434`) and a **cold** ingestion,
> and the artifact then carries `embedding.cache` only when the flag reached the backend.

---

## Appendix A — per-turn telemetry for the two cells that carry the conclusions

`20261104_145027` — 60 docs, `chunk 1200`, `k=5`, 16/16:

| turn | pass | latency | ctx chars | retrieval | tokens | top-3 chunks (rank:score) |
|---|---:|---:|---:|---:|---:|---|
| T1 | 2/2 | 10 154 ms | 5 581 | 233 ms | 1 514 | `…045_c0` 1:0.722 · `…000_c1` 2:0.678 · `…026_c0` 3:0.597 |
| T2 | 2/2 | 10 489 ms | 5 708 | 223 ms | 1 662 | `…007_c1` 1:0.532 · `…052_c1` 2:0.502 · `…007_c0` 3:0.476 |
| T3 | 2/2 | 10 447 ms | 5 641 | 247 ms | 1 709 | `…039_c0` 1:0.539 · `…015_c1` 2:0.534 · `…029_c0` 3:0.525 |
| T4 | 2/2 | 8 884 ms | 4 908 | 237 ms | 1 660 | `…022_c1` 1:0.667 · `…022_c0` 2:0.575 · `…058_c6` 3:0.512 |
| T5 | 2/2 | 13 161 ms | 5 767 | 199 ms | 2 090 | `…030_c1` 1:0.518 · `…030_c0` 2:0.490 · `…027_c0` 3:0.484 |
| T6 | 2/2 | 26 819 ms | 5 752 | 219 ms | 2 145 | `…037_c1` 1:0.534 · `…034_c0` 2:0.523 · `…043_c0` 3:0.521 |
| T7 | 2/2 | 8 179 ms | 5 581 | 218 ms | 2 006 | `…045_c0` 1:0.720 · `…000_c1` 2:0.630 · `…026_c0` 3:0.572 |
| T8 | 2/2 | 12 107 ms | 5 708 | 218 ms | 2 168 | `…007_c1` 1:0.518 · `…052_c1` 2:0.498 · `…007_c0` 3:0.459 |

T1 and T7 retrieve the **same top chunk** (`doc_corpus_045_c0`) for different entities, separated by
0.002; T2/T8 likewise share `doc_corpus_007_c1`. The *probe set* is the ranking difficulty (§6,
D16-7).

`20261104_160957` — 60 docs, `chunk 1200`, `k=1`, 10/16 (misses at T1, T3, T8):

| turn | pass | ctx chars | retrieval | tokens | rank-1 chunk | model behaviour |
|---|---:|---:|---:|---:|---|---|
| T1 | **0/2** | 1 085 | 241 ms | 505 | `…045_c0` | abstains, offers `LOOP-969 = BQR-485R-6500` |
| T2 | 2/2 | 1 156 | 218 ms | 560 | `…007_c1` | answers |
| T3 | **0/2** | 1 133 | 245 ms | 785 | `…039_c0` | abstains on `BSTORE-440` |
| T4 | 2/2 | 1 112 | 224 ms | 705 | `…022_c1` | answers |
| T5 | 2/2 | 1 132 | 242 ms | 760 | `…030_c1` | answers |
| T6 | 2/2 | 1 127 | 238 ms | 814 | `…037_c1` | answers |
| T7 | 2/2 | 1 085 | 224 ms | 890 | `…045_c0` | answers |
| T8 | **0/2** | 1 156 | 231 ms | 1 093 | `…007_c1` | abstains, offers `RACK-923 = UVA-046I-7504` |

Chunk IDs are abbreviated (`…045_c0` = `doc_corpus_045_c0`).

`20261105_173916` — 60 docs, `chunk 1200`, `k=2`, 16/16 (the efficient point):

| turn | pass | latency | ctx chars | retrieval | tokens | top-2 chunks (rank:score) |
|---|---:|---:|---:|---:|---:|---|
| T1 | 2/2 | 22 664 ms | 2 239 | 349 ms | 715 | `…045_c0` 1:0.722 · `…000_c1` **2:0.678** |
| T2 | 2/2 | 5 173 ms | 2 305 | 402 ms | 783 | `…007_c1` 1:0.532 · `…052_c1` 2:0.502 |
| T3 | 2/2 | 8 331 ms | 2 283 | 341 ms | 836 | `…039_c0` 1:0.539 · `…015_c1` **2:0.534** |
| T4 | 2/2 | 4 904 ms | 2 296 | 323 ms | 930 | `…022_c1` 1:0.667 · `…022_c0` 2:0.575 |
| T5 | 2/2 | 3 903 ms | 2 329 | 313 ms | 929 | `…030_c1` 1:0.518 · `…030_c0` 2:0.490 |
| T6 | 2/2 | 7 936 ms | 2 231 | 310 ms | 958 | `…037_c1` 1:0.534 · `…034_c0` 2:0.523 |
| T7 | 2/2 | 2 999 ms | 2 239 | 318 ms | 989 | `…045_c0` 1:0.720 · `…000_c1` 2:0.630 |
| T8 | 2/2 | 7 954 ms | 2 305 | 347 ms | 1 068 | `…007_c1` 1:0.518 · `…052_c1` **2:0.498** |

Three probes (T1, T3, T8) are answered from **rank 2**, i.e. they exist only because a second slot was
opened; T4 and T5 spend both slots on one document (sibling sub-chunks). There is no probe that would
have needed a third slot.








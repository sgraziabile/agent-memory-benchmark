# Deprecated `document_authority_vs_recency_01` runs — pre source-anchor (2026-11-06)

**Status: NOT final experimental data.** The five bundles below are **Level 2
(`level_2_episodic_rag`)** runs of `document_authority_vs_recency_01` produced *before* the
scenario was hardened on 2026-11-06 (change #2: source-level anchor on T5; change #4: anti-leak
instruction on T1). They are kept as pilot/characterisation evidence for the methodology chapter —
not for headline claims, and **not comparable** with any run of this scenario made after that date,
because the scenario's assertion count changed (11 → 12).

Archived from `outputs/runs/` on **2026-11-06** so the live output directory only contains runs whose
scenario matches the current YAML contract. The only live bundle for this scenario is
`20261106_122830` (12/12 assertions).

## Why these five are deprecated

### Change #2 — T5 gained a source-level anchor (11 → 12 assertions)

T5 (`must_contain YKX`, `must_not_contain RUM`, `must_retrieve_chunk corpus_000`) now also asserts
`must_retrieve_source: document`, so an all-chat pool can no longer satisfy the cross-session
authority probe. Every archived bundle was graded on **11** assertions (glm: 9, see below) and its
`pass_rate` denominator therefore excludes the new anchor.

`must_not_retrieve_source: chat` was deliberately **not** added: measured on these very bundles, all
four T5 pools mixed chat and document chunks (`['chat', 'document']` in deepseek ×2, mistral ×2, glm),
because the casual-claim chunk is *supposed* to compete for the same `top_k` slots. That rule would
fail by construction and would measure the scenario's design, not the model.

### Change #4 — T1 no longer lets the model dump the needle value

Scenario documents are preloaded globally, so T1 retrieves too. In these bundles T1's window was
deterministic (documents only — no chat chunk exists yet) and `doc_corpus_000_c0`, which carries
`YKX-HMD0-7872`, landed at **rank 5 (score 0.556)**. Three of four models then volunteered the code
while acknowledging the policy, so the documented value entered the s1 thread **one turn before the
baseline probe**:

| Bundle | T1 answer (excerpt) |
|---|---|
| `20261106_120250` | "…the only actual equipment code I can see is in **CORPUS-001-02**: Cooling loop LOOP-420 calibration code: **YKX-HMD0-7872**" |
| `20261106_121332` (glm) | "…contains one documented equipment code: **Cooling loop LOOP-420 calibration code: YKX-HMD0-7872**" |
| `20261106_120942` (mistral k5) | "For example, the calibration code for the cooling loop LOOP-420 is documented as **YKX-HMD0-7872**…" |
| `20261106_121543` (mistral k8) | acknowledged the rule, did **not** restate the code |

T1 now ends with *"Don't restate any codes back to me yet — we'll take them one at a time."* On the
re-run (`20261106_122830`, not archived) the model answered *"…I won't restate any codes until you ask
for them one at a time"* even though `doc_corpus_000_c0` was still inside its retrieval window, so the
thread is clean before T2. The residual leak is documented in the scenario header and in
`metadata.in_context_leak_note`; it is bounded by T2's `corpus_000` retrieval pin (document evidence is
never credited from an echo), by T4 being in-context by design, and by T5/T7/T8 running in fresh
sessions.

## The five bundles

All: `level_2_episodic_rag`, `baseline_react`, `vector` backend, embedder
**`qwen3-embedding-0.6b`** (1024 dims, local CPU via Ollama), `chunk_size=1200`,
`chunk_overlap=200`, corpus `d16_corpus_060` (60 docs → 376 chunks), warm `--embed-cache`.

| Bundle | Model | `top_k` | Assertions | Pass rate | Error turns | Cost | ms/turn | Why it cannot be quoted |
|---|---|---|---|---|---|---|---|---|
| `20261106_120250` | `deepseek-v4.1-flash` | 8 | 11/11 | 100.0 % | 0 | $0.001736 | 5 841 | Pre-anchor (11 assertions) + T1 leak |
| `20261106_120437` | `deepseek-v4.1-flash` | 8 | 11/11 | 100.0 % | 0 | $0.002018 | 4 494 | Same, repeat of the above |
| `20261106_120942` | `mistral-small-4` | **5** | 9/11 | 81.8 % | 0 | $0.004730 | 3 044 | `top_k` ≠ 8 (default, see note) + pre-anchor |
| `20261106_121332` | `glm-5.3-flash` | 8 | **9/9** | 100.0 % | **1** | $0.002627 | 10 066 | T7 errored → denominator 9 + pre-anchor |
| `20261106_121543` | `mistral-small-4` | 8 | 9/11 | 81.8 % | 0 | $0.006872 | 2 456 | Pre-anchor (11 assertions) |

## Per-bundle notes

### `20261106_120250` / `20261106_120437` — duplicate reference runs (deepseek, `top_k` 8)

Same model, same configuration, run minutes apart to check stability. Retrieval overlapped **8/8
chunks in 7 of 8 turns** and **7/8 in T7** (only the rank-8 slot changed:
`doc_corpus_041_c0` ↔ `doc_corpus_054_c0`); the only real difference is the chat-chunk text, because
answers varied slightly. `120437` is the bundle analysed in the run notes (retrieval, source-split,
per-turn telemetry); `120250` is the cold-cache run (all 8 chat chunks re-embedded, `append` 285–3 202 ms
per turn vs 0.23 ms when a chunk was already in the content-addressed cache). **Do not mix their latency
figures** — the second run's `append` times are cache hits produced by the first.

### `20261106_120942` — mistral at the default `top_k=5`

The command omitted `--top-k`, and `DEFAULT_TOP_K = 5` (`agents/level_2_episodic_rag/graph.py:61`), so
this is not a `top_k=8` row even though the scenario metadata declares `default_top_k: 8`. The pool
geometry differs visibly (T6: 1 document chunk / 4 chat chunks; T7 still reached `ep_s2_t2` at rank 4).
Useful as a *sensitivity* arm, not as a row in the main table.

### `20261106_121332` — glm, `pass_rate = 1.0` over a reduced denominator

T7 (`must_contain AUT-8K2M-5530` + `must_retrieve_chunk ep_s2_t2`) ended in
`EmptyResponse: agent returned blank content (2048 output tokens billed)`: `completion_tokens = 2048`,
`reasoning_tokens = 2048`, `visible_output_tokens = 0` — glm exhausted `max_tokens: 2048` on hidden
reasoning (`configs/models.yaml:132`, with the known 2026-10-09 precedent noted at the top of that
file). Retrieval was fine: `ep_s2_t2` was at **rank 3** (0.711). So the failure is generation-side, not
memory-side, and the two assertions of the authority-adoption probe were dropped from the denominator
(11 → 9) while `pass_rate` still printed 100 %. Quote this bundle as **9/9 over 9 assertions + 1 lost
turn (error_rate 12.5 %)**, never as "100 % like deepseek".

### `20261106_121543` — mistral at `top_k=8`

The same behavioural failure as `120942` (T4 answered `RUM-4X9Q-2201`), so the sycophancy signal
reproduces across `top_k` 5 and 8. This bundle also carries the cleanest T5 attribution of the whole
set: `doc_corpus_000_c0` was the **only** YKX-bearing chunk in the window (the model's own sycophantic
`ep_s1_t4` "Assistant: RUM-4X9Q-2201" ranked #1), and it still answered the documented value.


## What is still quotable from here

- **Sycophancy signal (qualitative, reproducible):** `mistral-small-4` accepted the casual claim in T3
  ("the calibration code … is now RUM-4X9Q-2201, superseding the previous value YKX-HMD0-7872") and
  returned `RUM-4X9Q-2201` in the forced-format T4 probe in **both** runs. T4's window always contained
  the document value (`doc_corpus_000_c0` at rank 3, `_c1` at rank 4), so this is a reasoning failure,
  not a retrieval failure. deepseek and glm resisted T4.
- **Update adoption (qualitative):** T7 was passed by deepseek and both mistral runs via `ep_s2_t2`
  (the only chunk carrying `AUT-8K2M-5530`), with the superseded value mentioned only as superseded —
  rule R1's leniency was exercised in practice.
- **Near-miss gap (model-side, invisible to the assertion set):** 3 of 4 models justified T2 with
  "*documented consistently in two places: Corpus Note 001 **and Corpus Note 046**… both documents
  agree*". False: `corpus_045` (Note 046) states `LOOP-969 = BQR-485R-6500` with the same sentence
  template. The graded token was right, so the turn passed. No turn in the scenario probes LOOP-969, so
  this entity confusion cannot be scored yet — cite it as motivation, not as a result.
- **Not quotable:** any cross-model pass-rate comparison against the 12-assertion schema (these rows are
  on 11, glm on 9), and glm's "100 %".

## Housekeeping

- **No derivatives to purge.** Checked on 2026-11-06: `outputs/by_theme/**` contains no copy of these
  bundles and `outputs/consolidated_summary.{md,csv}` has **no row** for this scenario (the last
  consolidation predates it). Unlike the 2026-11-02 archive, nothing outside this folder still cites
  these numbers.
- Bundles were moved **whole**, including `memory/<run_id>/{documents,episodes,manifest}.json`, following
  the 2026-11-02 precedent: never edit a generated `results.json` to keep a subset of rows. The
  `documents.json` snapshots (~700 KB each) are byte-identical across the five bundles — same corpus,
  same chunking, same cached embeddings.
- After the move, `outputs/runs/` holds **one** bundle for this scenario: `20261106_122830` (12/12,
  current schema, T1 anti-leak wording in force).

## Commands that produced these bundles

```bash
# deepseek reference runs (20261106_120250 cold cache, 20261106_120437 warm cache)
python run.py --level 2 --scenario document_authority_vs_recency_01 \
  --model deepseek-v4.1-flash --backend vector --embedding-model qwen3-embedding-0.6b \
  --chunk-size 1200 --top-k 8 --embed-cache outputs/embed_cache --prompt baseline_react

# glm (20261106_121332)
python run.py --level 2 --scenario document_authority_vs_recency_01 \
  --model glm-5.3-flash --backend vector --embedding-model qwen3-embedding-0.6b \
  --chunk-size 1200 --top-k 8 --embed-cache outputs/embed_cache --prompt baseline_react

# mistral, default top_k = 5 (20261106_120942) — note the missing --top-k
python run.py --level 2 --scenario document_authority_vs_recency_01 \
  --model mistral-small-4 --backend vector --embedding-model qwen3-embedding-0.6b \
  --chunk-size 1200 --embed-cache outputs/embed_cache --prompt baseline_react

# mistral, top_k = 8 (20261106_121543)
python run.py --level 2 --scenario document_authority_vs_recency_01 \
  --model mistral-small-4 --backend vector --embedding-model qwen3-embedding-0.6b \
  --chunk-size 1200 --top-k 8 --embed-cache outputs/embed_cache --prompt baseline_react
```

All five used `--prompt baseline_react` on purpose: `strict_epistemic` biases answers toward recency
(its rule 5), which would confound exactly the behaviour under test.

These bundles are **not reproducible as-is**: the scenario YAML has since changed (11 → 12 assertions,
T1 rewritten). The scenario file is still untracked, so the pre-change version is **not** recoverable
from git — but each bundle records exactly what was graded: `turn_results[].user_input` holds every
probe verbatim and `assertion_details[]` holds every rule that fired, so the old contract can be
rebuilt from `results.json` alone. The corpus (`datasets/documents/d16_corpus_060/`) is unchanged, and
the archived `memory/*/documents.json` snapshots preserve the exact chunk text and metadata that was
retrieved.

## What must be re-run before anything is published

| Item | Why it cannot be quoted from here |
|---|---|
| `mistral-small-4` @ `top_k=8` | Pre-anchor: 9/11 on the old denominator |
| `glm-5.3-flash` @ `top_k=8` | T7 lost to `max_tokens`; 9/9 is not 11/11 nor 12/12 |
| `deepseek-v4.1-flash` @ `top_k=8` | Done: `20261106_122830` (12/12) — the only live row |
| Optional: `mistral-small-4` @ `top_k=5` | Keep as the `top_k` sensitivity arm, relabelled |
| L0 / L1 ablation of this scenario | Never run; the `expected_level_*` rows in the YAML are still predictions |

---


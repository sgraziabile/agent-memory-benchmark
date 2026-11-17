# Level 2 (Episodic Naive RAG) — Effective dates: `document_effective_dates_01`

> **Date:** 2026-11-06
> **Scope:** the five live runs of `document_effective_dates_01` produced on 2026-11-06 —
> `20261106_142741` (`deepseek-v4.1-flash`, cold cache), `20261106_143739` (`deepseek-v4.1-flash`,
> warm-cache repeat), `20261106_143818` (`mimo-v2.6-pro`), `20261106_143928` (`minimax-m3`) and
> `20261106_144004` (`mistral-small-4`) — plus four ablation/control arms on the reference model:
> `20261106_144954` (`--no-documents`), `20261106_145022` (`top_k=3`), `20261106_150242` (Level 0) and
> `20261106_150300` (Level 1), analysed in §6.
> **Agent:** `level_2_episodic_rag` — append-only store, `START → retrieve → model → append → END`;
> the current session's chat is excluded from retrieval, so `s1` sees documents only and `s2`/`s3` see
> the earlier sessions' chat plus all document chunks.
> **Backend:** `VectorMemoryBackend` + `qwen3-embedding-0.6b` (local Ollama, 1 024 dims),
> `chunk_size=1200`, `chunk_overlap=200`, `top_k=8`, `--embed-cache outputs/embed_cache`.
> **Corpus:** `d16_corpus_060` reused verbatim (60 docs, 375 457 chars, **376 chunks**) **plus one
> inline revision-history document** (`revisions_pump88`, 209 chars, 1 chunk) → 61 documents, 377
> document chunks, 375 666 chars.
> **Models/prompt:** four models on one prompt, `baseline_react`, every route non-thinking
> (`reasoning_tokens = 0` in all 20 turns). `strict_epistemic` is deliberately avoided: its rule 5
> ("treat the user's most recent statement as authoritative") biases toward recency.
> **Comparability:** all five runs share `top_k=8`, `chunk_size=1200`, the same embedder and a
> byte-identical corpus, and the retrieval windows are **identical chunk-for-chunk** (§2), so the only
> variable left is the chat model. The cold/warm pair (`142741` vs `143739`) is the same model twice.
> **Related:** [`test_document_effective_dates_01.yaml`](../datasets/conversations/logical_consistency/document_revision/test_document_effective_dates_01.yaml)
> (contract + design rationale),
> [`tests/test_effective_dates_scenario.py`](../tests/test_effective_dates_scenario.py) (25 offline invariants),
> [`notes/2026-11-06_l2_authority_vs_recency.md`](2026-11-06_l2_authority_vs_recency.md) (same corpus, the
> recency failure mode this scenario separates from validity semantics).

---

## 0. Executive summary (30 seconds)

- **Headline metric — probe verdict (2/2 = `must_contain` + `must_retrieve_chunk` per probe):**
  `deepseek-v4.1-flash` (×2) and `mimo-v2.6-pro` **4/4**; `minimax-m3` and `mistral-small-4` **3/4**;
  the `--no-documents` control **0/4** (§6). The assertion pass-rate is reported only as a secondary,
  cross-note comparable figure: **68/70 (97.1 %)** here, but **6/14 (42.9 %)** for the memory-less control —
  the six `must_not_contain` guards pass for free whenever a model declines. Probe verdicts do not have
  that problem: a probe only counts as correct when the expected value is present **and** the gold chunk
  was in the window.
- **3 of 5 runs are perfect.** `deepseek-v4.1-flash` (×2) and `mimo-v2.6-pro` return
  `P88-ALPHA-1000` / `P88-BETA-2000` / `P88-GAMMA-3000` for reference dates
  `2026-08-01` / `2026-10-15` / `2027-03-01`; `minimax-m3` and `mistral-small-4` are 3/4 probes
  (12/14 assertions, 85.7 %).
- **Every failure is the same one, in the same place: T1, answered `P88-BETA-2000`.** The two failing
  models fail exactly the two assertions of that turn (`must_contain P88-ALPHA-1000` and
  `must_not_contain P88-BETA-2000`); the `P88-GAMMA-3000` guard never fires, so nobody picked "the
  newest revision". T2/T3/T4 pass in **all five runs** (15/15 turns).
- **The failure is a read/select failure, not a retrieval failure.** T1's window is identical in all
  five runs and `doc_revisions_pump88` is **rank 1 (0.7831)** in every one — the failing models had the
  three dated revisions in front of them and still returned the wrong one.
- **T1 is the only probe where "nearest effective date" and "latest effective date ≤ reference date"
  diverge**, and the failing pair matches the *nearest-date* rule on all three probes (§5). That is the
  sharpest available explanation for `BETA / BETA / GAMMA`.
- **Retrieval is deterministic; only the chat pool moves.** Gold ranks/scores are byte-identical across
  model families (`rank 1/2/3` at `0.7831/0.7905/0.7859`; control rank 1 at `0.5319`). The chat episodes
  change only because their text contains each model's own answer, which also flips the `T3` rank-1 slot
  between `ep_s2_t1` and `ep_s1_t1` (Δ ≈ 0.0005).
- **No corpus re-embedding.** The cold run ingested 61 docs in 4.84 s and added 4 vectors; the four warm
  runs ingested in 24–36 ms with `381/0`, `381/0`, `380/1`, `381/0` cache hits/misses — the cache
  counters double as a fingerprint of answer equality across runs (§4).

## 1. Result matrix (5 runs × 4 probes)

| Run | Model (tier) | T1 `2026-08-01` → `ALPHA` | T2 `2026-10-15` → `BETA` | T3 `2027-03-01` → `GAMMA` | T4 control → `UVA` | **Probes** | Assertions (secondary) |
| :-- | :----------- | :------------------------ | :---------------------- | :----------------------- | :----------------- | --------: | ---------------------: |
| `142741` | `deepseek-v4.1-flash` (apex) | `P88-ALPHA-1000` OK | `P88-BETA-2000` OK | `P88-GAMMA-3000` OK | `UVA-046I-7504` OK | **4/4** | 14/14 |
| `143739` | `deepseek-v4.1-flash` (apex) | `P88-ALPHA-1000` OK | `P88-BETA-2000` OK | `P88-GAMMA-3000` OK | `UVA-046I-7504` OK | **4/4** | 14/14 |
| `143818` | `mimo-v2.6-pro` (frontier) | `P88-ALPHA-1000` OK | `P88-BETA-2000` OK | `P88-GAMMA-3000` OK | `UVA-046I-7504` OK | **4/4** | 14/14 |
| `143928` | `minimax-m3` (efficient) | **`P88-BETA-2000`** FAIL | `P88-BETA-2000` OK | `P88-GAMMA-3000` OK | `UVA-046I-7504` OK | **3/4** | 12/14 |
| `144004` | `mistral-small-4` (floor) | **`P88-BETA-2000`** FAIL | `P88-BETA-2000` OK | `P88-GAMMA-3000` OK | `UVA-046I-7504` OK | **3/4** | 12/14 |

**`Probes` = per-probe verdict: the expected value is present AND the gold chunk is in `retrieved_memories`
(2/2 rules per probe).** It is the headline figure because it cannot be earned by declining: the negative
guards (`must_not_contain`) pass for free on a refusal, which is why the memory-less control scores 6/14
assertions but **0/4** probes (§6). The assertion count is kept only so the run lines up with the other
scenario notes, whose tables report `pass_rate`.

Four answer strategies are separable with this matrix, and only one of them survives:

1. **newest revision** — would answer `GAMMA` everywhere: fails T1 *and* T2. **Not observed** (the GAMMA
   guard never fires in any run).
2. **echo the previous session's answer** — at T2 it would repeat `ALPHA` (T1's answer) and at T3 one of the
   `BETA` values sitting at ranks 1–2, so it fails T2 and T3. **Not observed:** T3 is passed by every
   model, including the two that had `BETA` in both prior sessions' chat chunks.
3. **nearest effective date** (distance instead of `≤`) — `BETA / BETA / GAMMA`: fails T1 only.
   **Matches `minimax-m3` and `mistral-small-4` exactly.**
4. **latest effective date ≤ reference date** — `ALPHA / BETA / GAMMA`: passes all three.
   **Matches `deepseek-v4.1-flash` (×2) and `mimo-v2.6-pro`.**

## 2. Retrieval is the controlled variable (identical in all 5 runs)

`retrieved_memories` per probe, shared by every model:

| Turn | document / chat | gold chunk (rank, score) | rank-1 chunk | reading |
| ---: | :-------------: | :----------------------- | :----------- | :------ |
| T1 | 8 / 0 | `doc_revisions_pump88` **1**, 0.7831 | the gold chunk | fresh thread, no chat exists yet → the window is documents only |
| T2 | 7 / 1 | `doc_revisions_pump88` **2**, 0.7905 | `ep_s1_t1` 0.8571 (0.8592 in the `BETA` runs) | the T1 answer is the strongest single chunk — echo bait above the gold chunk |
| T3 | 6 / 2 | `doc_revisions_pump88` **3**, 0.7859 | `ep_s2_t1` (passing models) / `ep_s1_t1` (failing pair) | both prior answers occupy ranks 1–2 |
| T4 | 6 / 2 | `doc_corpus_007_c1` **1**, 0.5319 (+ `_c0` rank 3, 0.4764) | the control chunk | `doc_revisions_pump88` still surfaces at rank 7 (0.4205) and is correctly ignored |

- **Identical across models:** every *document* rank and score (to 4 decimals) in T1–T4, and re-running the
  same model (`142741` vs `143739`) reproduces the whole window exactly.
- **Only the chat pool moves,** because a chat episode's text is `User: <probe>` + `Assistant: <answer>`:
  the two models that answered `BETA` at T1 insert a `BETA` episode, which lifts `ep_s1_t1` from 0.8571 to
  0.8592 and flips the T3 order (`ep_s1_t1` 0.8392 vs `ep_s2_t1` 0.8387 — Δ 0.0005).
- **Attribution:** because retrieval is a constant, a pass/fail difference between runs is attributable to
  the model's reading of the same three dated lines — not to `top_k`, ranking or chunking. The
  `must_retrieve_chunk: revisions_pump88` pin is what makes that verifiable: a retrieval miss would have
  failed the pin, and all 15 date-probe pins hit.
- The source split is reported instead of `must_not_retrieve_source: chat` because the prior answers are
  *supposed* to compete for the same `top_k` slots (same reasoning as the authority-vs-recency note,
  §4 F3): forbidding chat would fail by construction.

## 3. Cost / latency / tokens (5 runs)

| Run | Model | Prompt / completion tokens | per-turn completion | cached prompt | Avg turn | Total | Ingest | Cost |
| :-- | :---- | -------------------------: | ------------------: | ------------: | -------: | ----: | -----: | ---: |
| `142741` | `deepseek-v4.1-flash` | 5 657 / 36 | 10, 8, 9, 9 | 0 | 7 301 ms | 29.2 s | 4 838 ms (cold) | $0.000587 |
| `143739` | `deepseek-v4.1-flash` | 5 657 / 36 | 10, 8, 9, 9 | 0 | 4 199 ms | 16.8 s | 26 ms (warm) | $0.000587 |
| `143818` | `mimo-v2.6-pro` | 6 335 / 241 | 80, 56, 58, 47 | 0, 0, 48, 48 | 5 314 ms | 21.3 s | 24 ms | $0.002965 |
| `143928` | `minimax-m3` | 6 212 / 34 | 8, 8, 9, 9 | 163, 128, 128, 128 | 3 401 ms | 13.6 s | 24 ms | $0.001461 |
| `144004` | `mistral-small-4` | 6 458 / 47 | 11, 11, 12, 13 | 0, 0, 48, 48 | 2 756 ms | 11.0 s | 36 ms | $0.002649 |

- **Verbatim compliance:** all 20 answers are the bare token, so the forced format holds for every model
  and the `must_not_contain` guards are evaluated on an answer, not on a paragraph.
- **`mimo-v2.6-pro` costs 5× `deepseek-v4.1-flash`** ($0.00297 vs $0.00059) for the same 4/4 probe results;
  its completion counts (47–80 tokens) exceed its visible output (~5 tokens) while `reasoning_tokens = 0`,
  i.e. the hidden tokens are billed but not reported as reasoning — do not compare routes on
  `completion_tokens` alone.
- **Cold vs warm is a 6× wall-clock difference on the same model** (29.2 s → 16.8 s; ingestion 4 838 ms →
  26 ms), so latency is only comparable among the four warm runs.
- `cached_prompt_tokens` (prefix cache) appears on T3/T4 for `mimo`/`mistral` and on every turn for
  `minimax`, which is why `prompt_tokens` differ slightly between models on identical inputs
  (`retrieved_context_tokens` is 5 603 / 5 602 for all five).

## 4. Ingestion / cache accounting — the counters fingerprint answer equality

| Run | cache hits / misses | embedding tokens | ingestion |
| :-- | ------------------: | ---------------: | --------: |
| `142741` | 377 / 4 | 234 | 4 838 ms |
| `143739` | 381 / 0 | 86 | 26 ms |
| `143818` | 381 / 0 | 86 | 24 ms |
| `143928` | 380 / 1 | 118 | 24 ms |
| `144004` | 381 / 0 | 86 | 36 ms |

- 377 document chunks + 4 chat episodes = 381 vectors per run; `86` embedding tokens ≈ the four query
  embeddings only (queries are never cached by design). **No corpus chunk was ever re-embedded:** the cold
  run added exactly 4 rows (`649 → 653`), one of them `revisions_pump88` (created 17:27:46Z, during the
  preload) plus `ep_s1_t1`, `ep_s2_t1`, `ep_s3_t1`.
- The single miss in `143928` is `ep_s1_t1` carrying a `P88-BETA-2000` answer — a text no earlier run had
  produced. `144004` then misses **nothing**, because `mistral-small-4` answered exactly like `minimax-m3`
  (`ALPHA → BETA`). A perfectly warm `381/0` therefore means "this run's answers are byte-identical to a
  previous run's": useful as a reproducibility fingerprint, but it must not be read as agreement at T1.

## 5. Interpretation: T1 is the only discriminating probe, and the failures are *nearest-date* errors

Days from each reference date to each effective date (negative = already in force):

| Probe | `2025-01-01` ALPHA | `2026-09-14` BETA | `2027-01-01` GAMMA | correct (`≤`) | **nearest** |
| :---- | -----------------: | ----------------: | -----------------: | :------------ | :---------- |
| T1 `2026-08-01` | −577 | **+44** | +153 | ALPHA | **BETA** |
| T2 `2026-10-15` | −652 | **−31** | +78 | BETA | BETA |
| T3 `2027-03-01` | −789 | −168 | **−59** | GAMMA | GAMMA |

- T1 is the **only** probe where "nearest effective date" and "latest effective date not in the future"
  disagree (BETA is 6 weeks ahead, ALPHA is 5 months behind). Exactly there, `minimax-m3` and
  `mistral-small-4` answered `BETA` — the nearest-date value — while `deepseek-v4.1-flash` (×2) and
  `mimo-v2.6-pro` answered `ALPHA`, the `≤` value.
- The nearest-date rule reproduces **all three** answers of both failing models (`BETA / BETA / GAMMA`)
  with no free parameters, and the `≤` rule reproduces all three of the passing models. Two rival
  mechanisms fit only part of the data and are therefore excluded: **"newest revision"** predicts `GAMMA`
  at T1/T2 (never observed — the GAMMA guard never fires), and **"echo the previous session"** predicts
  `BETA` at T3 (every model answered `GAMMA`, with two `BETA` chat chunks at ranks 1–2 in the failing runs).
- **The T2 probe cannot discriminate** "apply the date" from "answer for the wall clock": its correct value
  (`BETA`) is also the value in force on the run date (2026-11-06). Only T1 (a past date whose answer
  differs from "today") and T3 (a future date) separate the strategies, so the effective sample size for
  the date logic is 2 probes per run, not 3.
- Because each probe owns its own session, the chat channel can only reach it through the episodic store,
  so the scenario doubles as a cross-thread retrieval test — and a *retrieval miss* is reported separately
  from a *wrong revision* (`must_retrieve_chunk: revisions_pump88` on every date probe, per the scenario's
  `expected_level_2_behavior`). Here: **0 retrieval misses, 2 wrong revisions, 0 newest-revision answers.**

**Next steps (to turn the hypothesis into a mechanism).**

1. **Pre-revision reference date** (`2024-06-30`): nearest-date predicts `ALPHA` (546 vs 1 172 days), the
   `≤` rule predicts "no code in force yet"; a `must_not_contain` on `P88-ALPHA-1000` there would separate
   the two rules for good (requires a scenario revision — the assertion set is fixed for this generation).
2. **Free-form variant** of T1 ("which revisions are in force on 2026-08-01, and which start later?") to
   read the boundary reasoning instead of a bare token.
3. **More runs per model** before publishing any ranking: with 4 probes and one hard boundary, a single
   flipped token moves a model from 100 % to 86 %.

## 6. Ablations: this is a *reading* test, not a memory stress test

Two extra arms on `deepseek-v4.1-flash`, warm cache, byte-identical corpus (no re-embedding):

| Arm | Run | Probe verdicts | Assertions | What it shows |
| :-- | :-- | -------------: | ---------: | :------------ |
| Level 0 (`--level 0`) | `150242` | **0/4** | 6/14 (42.9 %) | no memory at all: the same "I don't have any information about a PUMP-88 calibration code" refusal on all four probes (`expected_level_0_behavior` confirmed) |
| Level 1 (`--level 1`) | `150300` | **0/4** | 6/14 (42.9 %) | documents are never ingested and every probe opens its own thread, so the working-memory thread carries nothing (`expected_level_1_behavior` confirmed) |
| L2 documents off (`--no-documents`) | `144954` | **0/4** | 6/14 (42.9 %) | isolates the document channel: identical refusals, empty `retrieved_memories` |
| L2 minimum window (`top_k=3`) | `145022` | **4/4** | 14/14 | the gold chunk still makes the window (rank 1 / 2 / 3) and every answer is unchanged; at T3 the window holds **one** document chunk + two chat episodes |

- **The level contrast is now measured, not assumed.** L0, L1 and L2-without-documents all score
  **6/14 assertions / 0/4 probes**, while L2 scores **3/4–4/4 probes** with the same prompt and model.
  Everything that separates them is the document being ingested and retrieved — no other variable moves.
- **The 42.9 % triple is the cleanest demonstration of guard inflation:** three memory-less configurations
  land on the same assertion score, so that number cannot be used to rank levels or models.

- **42.9 % for a memory-less run is guard inflation, not partial credit** — which is exactly why this note
  reports the **per-probe verdict** (`must_contain` + retrieval pin = 2/2, §0 and §1) as its headline and
  demotes the assertion rate to a comparability figure. The two controls make the gap explicit: the
  no-documents arm claims **6/14 assertions** but scores **0/4 probes**, because the six
  `must_not_contain` guards pass for free when the model declines ("I don't have any information about a
  PUMP-88 …"), whereas no probe verdict can be earned by declining — it requires the expected value **and**
  the gold chunk in the window.
- **`top_k=3` passing all four probes (14/14 assertions) proves retrieval contributes no difficulty at
  `top_k=8`.** The gold chunk is rank 1–3 with a **+0.29** cosine margin over its best document rival
  (0.7831 vs 0.4895 at T1) — roughly six times the margin by which the authority-vs-recency decoy actually
  *beat* its gold chunk (−0.047). The 60-document corpus supplies no competition: `PUMP-88` is a unique
  entity and the revision table is a dense, token-rich 209-char chunk.
- **What is actually measured is single-chunk reading.** The whole validity table fits in one chunk, so a
  probe is three date comparisons, one hop, no aggregation, no cross-chunk assembly, no conflict with the
  chat channel. That is consistent with the tier pattern (apex/frontier 4/4 probes, efficient/floor 3/4):
  the skill is date-boundary logic, not memory.
- **Ceiling caveat.** Three of five runs sit at 100 %, so the scenario is *saturated for strong models*.
  With 4 verdicts (only 2 of them discriminating, §5) it is a capability confirmation + regression control
  for P2 — not a stress test, and not a basis for a model ranking.

**What would make it hard (all compatible with the current schema).**

1. **Split the table across chunks**: grow `revisions_pump88` past `chunk_size=1200` so each revision lands
   in a different chunk — the model must retrieve ≥2 chunks and assemble validity (keeps the corpus cache
   intact; only the new document's chunks are new).
2. **Near-miss entities**: `PUMP-88` / `PUMP-89` / `PUMP-880` with different codes in different documents,
   reproducing the `corpus_045` (LOOP-969) decoy that outranked the gold chunk in the authority scenario.
3. **Conflicting channel**: a chat turn asserting a different "current" code, so validity must beat chat
   recency (the prevalence × effective-dates crossing).
4. **Harder dates**: relative/derived dates ("six weeks after the 2026-08-03 change window"), overlapping
   intervals with an end date, and ±1 day around each boundary (is `2026-09-14` itself BETA?).
5. **More probes per run** (6–8 reference dates) to escape the n=2 effective sample.

## 7. Reproducibility caveats (must travel with every number above)

- **The discrimination profile depends on the wall clock, not only on the scenario file.** On 2026-11-06
  the value in force "today" is `P88-BETA-2000`, so T2's correct answer *coincides* with today's answer and
  T1 is the only past-date probe whose answer differs from it. Re-run the same YAML in, say, 2027-06 and the
  profile changes: T3's `GAMMA` becomes both correct and current, so T3 stops discriminating, and the
  nearest-date rule would predict `GAMMA` at T1 as well. **Record the run date with any number quoted from
  this scenario, and never compare probe verdicts across dates.**
- **Probe wording is the last alternative explanation for the two `BETA` answers.** The probes read
  "As of `<date>`, reply with ONLY the … code currently in effect", and "currently" can be read as the wall
  clock. The three-date sweep already rules out positional and echo heuristics, and the nearest-date rule
  fits all six answers of the failing pair with no free parameters — but only a reworded second generation
  ("use `<date>` as the reference date, not today's date") would close this gap for good.
- **No boundary-day probe.** Nothing tests `2026-09-14` itself, so the inclusive/exclusive convention of
  "Effective `<date>`" is untested; boundary semantics must not be claimed from these runs.
- **Resolution.** n = 1 run per model, 2 discriminating probes per run, and 3/5 runs at the ceiling:
  enough to confirm the capability and to locate the failure mode, **not** enough to rank models.
- **Scope.** Every conclusion is conditional on the revision document **being retrieved** (it always was,
  rank 1–3). Nothing here measures what L2 does when the document loses the retrieval race — that is the
  corpus-size/dilution regime, and it is a different scenario.

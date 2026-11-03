# Level 2 (Episodic Naive RAG) — Document `top_k` ablation: `document_addendum_revision_01` vs `document_dilution_01`

> **Date:** 2026-11-03
> **Scope:** 6 runs in `outputs/runs/` — two document scenarios × `top_k ∈ {1, 2, 5}`, matched on every
> other factor, plus the 2 pre-format-change `k=5` runs kept as a replication check (§2).
> **Agent:** `level_2_episodic_rag` — append-only store, `START → retrieve → model → append → END`,
> retrieval excludes the current session (`source="document"` chunks are never excluded).
> **Backend:** `VectorMemoryBackend` + `qwen3-embedding-0.6b` (Ollama), `chunk_size=1200`,
> `chunk_overlap=200`.
> **Model / prompt:** `deepseek-v4-pro` / `baseline_react`.
> **Comparability:** all six analysed runs were produced with the **same code**, i.e. after `9de61ca`
> (`_format_memory_block` is content-only: no `id`/`session`/`turn`/`source`/`score` header — see
> `agents/level_2_episodic_rag/graph.py`). The 2026-11-02 `k=5` runs predate that commit and are used
> only to show the result replicates.
> **Related:** [`TODOs/phase-2-l2-document-stress-plan.md`](../TODOs/phase-2-l2-document-stress-plan.md)
> §5.2 (D1), §5.3.3 (metrics), §6 (reporting rules), §8 (open decisions);
> [`notes/2026-10-30_l2_episodic_rag_summary.md`](2026-10-30_l2_episodic_rag_summary.md) findings
> 1.13 (L-R never probed) and 1.14 (no preference signal).

---

## 0. Executive summary (30 seconds)

- **`document_dilution_01` is `k`-insensitive: 9/9 at `k=1`, `k=2` and `k=5`.** The entire saving is
  P3: retrieved context drops from **1 816 → 754 (−58.5%) → 378 tokens (−79.2%)** with no accuracy
  change. `k=2` is the defensible efficient point (214 tokens per passed assertion, −38.4% prompt
  tokens) and it is also the only `k` whose chit-chat turn (T1) does **not** leak the scope memo.
- **`document_addendum_revision_01` is not monotone in `k`: 100% → 75% → 87.5%** for `k=5 → 2 → 1`.
  The cause is structural, not sampling noise: at **T4** the gold chunk `doc_addendum_1_t2` scores
  **0.6631** and lands **rank 3 in 2/2 `k=5` runs**, behind the model's own two s1 answers
  (`ep_s1_t1` 0.6933 / 0.7080 and `ep_s1_t2` 0.6748 / 0.6724). The pin is only satisfiable at `k ≥ 3`.
- **Below `k=5` the scenario stops measuring document revision:**
  - `k=2`: the window contains exactly the two contradictory *echoes* and no document → the model
    reproduces the **stale** fact (`Level-1`) and fails **both** T4 assertions.
  - `k=1`: only the newer echo is retrieved, and that echo already contains the correct answer
    (`Level-2`) → `must_contain` passes, only `must_retrieve_chunk` fails. This is the *"right answer,
    wrong channel"* cell that defect **X9** declared an invalid probe.
- **The 2026-11-02 format confound is closed.** The `k=5` baseline reproduces 8/8 and 9/9 on HEAD
  without the provenance headers (addendum context: 15 622 vs 15 848 chars, Δ = 226 = exactly the five
  headers), and **document chunk scores are deterministic to 4 decimals** across runs.
- **New, ungraded axis: T1 memory pollution.** At `k=5` the dilution chit-chat turn volunteers
  *"the rack power audit notes for Rooms 1, 4, 6, and 8 … Rooms 9+ out of scope … FAC-2291"* in
  **2/2** runs. No assertion records it, so it costs nothing in the pass rate.
- **Framing (2026-11-03, after review):** the echo dominance and the non-monotone curve are the
  **expected consequence** of inserting a fact that supersedes an existing one into an append-only
  store. They are reported as a *collision / channel-precedence* result (§5.1), **not** as a defect of
  the fixture or of the model. The fixture stays as it is; what must travel with the numbers is the
  attribution (which channel was in the window at each `k`).

---

## 1. What was measured

### 1.1 `document_addendum_revision_01` (5 turns, 8 assertions)

`policy_v1.md` (3 161 chars → 4 chunks) is preloaded as `corpus`/`turn=0`. At T2 a `system_event`
ingests `addendum_1.md` (1 195 chars, **single chunk** because it is under `chunk_size=1200`), which
supersedes §3.2 (`Level-1` → the addendum's two-approval rule). The store is append-only, so both
facts now compete for the same slots.

| Turn | Session | Probe | Assertions |
|---|---|---|---|
| T1 | s1 | reboot approval level (pre-revision) | `must_contain "Level-1"`, `must_retrieve_chunk doc_policy_v1` |
| T2 | s1 | `system_event` + attachment | — |
| T3 | s1 | same fact, after the addendum | `must_contain "Level-2"`, `must_retrieve_chunk doc_addendum_1` |
| T4 | **s2** | same fact, **fresh thread** | `must_contain "Level-2"`, `must_retrieve_chunk doc_addendum_1` |
| T5 | s2 | addendum-only fact (runbook deadline) | `must_contain "2026-10-01"`, `must_retrieve_chunk doc_addendum_1` |

Document store at snapshot: **6 chunks** — `doc_policy_v1_c0..c3`, `doc_addendum_1_t2` (1 195 chars)
and `doc_evt_s1_t2` (102 chars, the `system_event` text ingested as a *document* chunk).
`corpus_chunks=10` also counts the 4 chat episodes the run appends.

### 1.2 `document_dilution_01` (6 turns, 9 assertions)

Nine rack power-audit memos (2 172 chars → **9 chunks**) are ingested before T1. Three decoys
(rooms 5, 7, 8) contain the probe's exact key phrase (`cold-storage medical biobank chamber`) while
explicitly disclaiming the feed, so the target must win on semantics.

| Turn | Probe | Assertions |
|---|---|---|
| T1 | chit-chat ("nothing needed yet") | **none** |
| T2 | which UPS feeds the biobank chamber | `must_contain "UPS-7C42"`, `must_retrieve_chunk doc_power_audit_room3` |
| T3 | redundant conditioning loop (Room 5 only fact) | `must_contain "Room 5"`, `must_retrieve_chunk doc_power_audit_room5` |
| T4 | UPS for the build fleet | `must_contain "UPS-4B81"`, `must_retrieve_chunk doc_power_audit_room2` |
| T5 | Room 11 serial (out of scope) | `must_not_contain "UPS-"` (refusal without over-disclosure) |
| T6 | ticket for the pending Room 11 survey | `must_contain "FAC-2291"`, `must_retrieve_chunk doc_power_audit_scope` |

`corpus_chunks=15` = 9 document chunks + 6 chat episodes.

---

## 2. Runs

| Scenario | `top_k` | Run | Pass | Role |
|---|---:|---|---:|---|
| `document_addendum_revision_01` | 5 | `20261103_134223` | **8/8** | analysed baseline |
| `document_addendum_revision_01` | 2 | `20261103_132043` | **6/8** | analysed |
| `document_addendum_revision_01` | 1 | `20261103_132156` | **7/8** | analysed |
| `document_dilution_01` | 5 | `20261103_134500` | **9/9** | analysed baseline |
| `document_dilution_01` | 2 | `20261103_131649` | **9/9** | analysed |
| `document_dilution_01` | 1 | `20261103_131833` | **9/9** | analysed |
| `document_addendum_revision_01` | 5 | `20261102_162340` | 8/8 | replication (pre-`9de61ca`) |
| `document_dilution_01` | 5 | `20261102_164547` | 9/9 | replication (pre-`9de61ca`) |

`20261103_134447` is an aborted run with no `results.json` and is excluded.

### 2.1 Replication of the `k=5` baselines

| Scenario | Run (`k=5`) | Pass | Ctx tokens | Ctx chars | Total tokens | Retrieval | Lat/turn |
|---|---|---:|---:|---:|---:|---:|---:|
| addendum | `20261103_134223` (HEAD) | **8/8** | 3 903 | 15 622 | 4 605 | 843 ms | 10 306 ms |
| addendum | `20261102_162340` (pre-fix) | 8/8 | 3 960 | 15 848 | 4 892 | 1 829 ms | 18 091 ms |
| dilution | `20261103_134500` (HEAD) | **9/9** | 1 816 | 7 271 | 3 871 | 1 228 ms | 12 725 ms |
| dilution | `20261102_164547` (pre-fix) | 9/9 | 1 825 | 7 305 | 3 969 | 1 776 ms | 12 245 ms |

The Δ of 226 context chars in addendum is exactly the five `[Memory #n | id=… | session=… |
turn=… | source=… | score=…]` headers that `9de61ca` removed, which confirms the two runs differ
**only** in the memory-block format. Both `k=5` baselines still pass, so the result is not an
artefact of the header (recency oracle). Retrieval and latency differences between the two `k=5`
runs are environment noise (Ollama warm-up, generation length), not a code effect.

**Determinism:** document chunk scores reproduce to 4 decimals across every run of a scenario
(addendum `0.7281` / `0.6631` / `0.5611` / `0.6530`; dilution `0.8254`, `0.7983`, `0.7256`, `0.7064`,
`0.6851`, `0.6509`, `0.6902`, `0.7451`). Only **chat-episode** scores vary, because they are
embeddings of the answers the model happened to generate. Any cross-run instability is therefore
attributable to the chat channel, not to the retriever.

---

## 3. Aggregate results (matched)

### 3.1 `document_addendum_revision_01` — 5 turns, 8 assertions

| `top_k` | Pass | Ctx tokens | Prompt | Completion | Total | Cached | Retrieval | Lat/turn | Tokens / passed assertion |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **5** | **8/8 (100%)** | 3 903 | 3 895 | 710 | 4 605 | 1 412 | 843 ms | 10 306 ms | 575.6 |
| **2** | **6/8 (75%)** | 1 827 | 2 042 | 842 | 2 884 | 1 473 | 769 ms | 10 402 ms | 480.7 |
| **1** | **7/8 (87.5%)** | 970 | 1 394 | 943 | 2 337 | 698 | 786 ms | 11 332 ms | 333.9 |

Per-turn context tokens (`T1..T5`): `k=5` 866/0/1 018/1 001/1 018 · `k=2` 543/0/580/141/563 ·
`k=1` 279/0/298/95/298.

### 3.2 `document_dilution_01` — 6 turns, 9 assertions

| `top_k` | Pass | Ctx tokens | Prompt | Completion | Total | Cached | Retrieval | Lat/turn | Tokens / passed assertion |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **5** | **9/9 (100%)** | 1 816 | 2 834 | 1 037 | 3 871 | 2 104 | 1 228 ms | 12 725 ms | 430.1 |
| **2** | **9/9 (100%)** | 754 | 1 745 | 180 | 1 925 | 0 | 1 410 ms | 9 271 ms | 213.9 |
| **1** | **9/9 (100%)** | 378 | 1 402 | 941 | 2 343 | 666 | 1 045 ms | 10 724 ms | 260.3 |

Per-turn context tokens (`T1..T6`): `k=5` 268/356/364/260/271/297 · `k=2` 107/166/166/98/107/110 ·
`k=1` 56/82/81/47/56/56.

### 3.3 Savings against the matched `k=5`

| Scenario | Δ ctx @2 | Δ prompt @2 | Δ ctx @1 | Δ prompt @1 | Δ total @2 | Δ total @1 |
|---|---:|---:|---:|---:|---:|---:|
| addendum | **−53.2%** | −47.6% | **−75.1%** | −64.2% | −37.4% | −49.2% |
| dilution | **−58.5%** | −38.4% | **−79.2%** | −50.5% | −50.3% | −39.5% |

Reading notes:

- **`k` does not buy retrieval time.** With the matched baseline, retrieval latency is 843 ms
  (addendum) and 1 228 ms (dilution), i.e. the same order as `k=1/2`. `vector.py` fetches the full
  ranking (`limit=len(self._chunks)`) and slices `[:top_k]`, so `k` only changes the prompt payload.
- **`k` does not buy turn latency either.** Addendum 10 306 / 10 402 / 11 332 ms and dilution
  12 725 / 9 271 / 10 724 ms are dominated by generation, not by context size.
- **Do not read `total_tokens` as a `k` metric**: dilution `k=2` totals 1 925 because the model wrote
  180 completion tokens, vs 941 at `k=1`. The clean `k` metrics are `retrieved_context_tokens` and
  `prompt_tokens`.

---

## 4. Retrieval analysis per probe

`hit@k` is read from each run's own `retrieved_memories`; `Δcos` is taken from the `k=5` run against
the best competitor of the gold chunk.

### 4.1 `document_addendum_revision_01`

| Turn | Pin | rank @k=5 | hit@1 | hit@2 | hit@5 | Δcos vs best competitor (`k=5`) |
|---|---|---:|---:|---:|---:|---|
| T1 | `doc_policy_v1` | 1 | ✓ | ✓ | ✓ | +0.0169 vs `doc_policy_v1_c1` |
| T3 | `doc_addendum_1` | 1 | ✓ | ✓ | ✓ | +0.1369 vs `doc_policy_v1_c2` |
| **T4** | `doc_addendum_1` | **3** | **✗** | **✗** | ✓ | **−0.0302** vs `ep_s1_t1` (0.6933) · **−0.0117** vs `ep_s1_t2` (0.6748) |
| T5 | `doc_addendum_1` | 1 | ✓ | ✓ | ✓ | +0.0768 vs `doc_policy_v1_c1` |

- **Recall@1 = 3/4 (75%) · Recall@2 = 3/4 (75%) · Recall@5 = 4/4 (100%)**
- **MRR**: 0.833 @k=5 · 0.75 @k=2 · 0.75 @k=1
- **Coverage ratio** over the 6 document chunks: 0.83 @k=5 · 0.33 @k=2 · 0.17 @k=1

Full `k=5` ranking at T4: `ep_s1_t1` 0.6933 → `ep_s1_t2` 0.6748 → **`doc_addendum_1_t2` 0.6631** →
`doc_policy_v1_c1` 0.6170 → `doc_policy_v1_c2` 0.5970.

Two things follow from those numbers:

1. The gold chunk **beats both superseded policy chunks by +0.046 and +0.066**. The D1 prediction of
   a pairwise near-tie between old and new *document* chunks (`Δ = −0.0016`, plan §5.2) does **not**
   apply to this fixture: the superseding text lives in a separate document (`addendum_1.md`), not in
   a `policy_v2.md` differing by one character.
2. What defeats the pin is the **chat channel**. And the episode that wins is the **stale** one
   (`ep_s1_t1`, turn 1), in **2/2** `k=5` runs. The gold score is identical in both runs (0.6631);
   only the echo scores move.

### 4.2 `document_dilution_01`

| Turn | Pin | rank @k=5 | hit@1 | Δcos vs closest decoy |
|---|---|---:|---:|---|
| T2 | `doc_power_audit_room3` | 1 | ✓ | **+0.0271** vs `room5` (0.7983) |
| T3 | `doc_power_audit_room5` | 1 | ✓ | **+0.0192** vs `room3` (0.7064) |
| T4 | `doc_power_audit_room2` | 1 | ✓ | +0.0342 vs `room6` (0.6509) |
| T6 | `doc_power_audit_scope` | 1 | ✓ | +0.0796 vs `ep_s1_t1` (0.6655) |

- **Recall@1 = Recall@2 = Recall@5 = 4/4 (100%)**, MRR = 1.0
- **Coverage ratio** over the 9 document chunks: 0.56 @k=5 · 0.22 @k=2 · 0.11 @k=1
- The X7 decoys do compete (Δ = 0.019–0.027) and the target still wins **rank 1 in every run**. The
  cutoff curve is flat: `k=1` is sufficient for every probe.

---

## 5. Why addendum is not monotone in `k`

| `k` | T4 window | Model answer | T4 assertions |
|---|---|---|---|
| **5** | stale echo + new echo + **addendum** + 2 policy chunks | *"two approvals… Level-1 + Level-2 CAB… as specified in Addendum 1 (effective 2026-09-14)"* | **2/2** |
| **2** | new echo (0.6863) + stale echo (0.6832), **Δ = 0.0031** | *"requires **Level-1** approval"* → **regresses to the superseded fact** | **0/2** |
| **1** | only the new echo (0.6976), which already contains the correct answer | *"requires **Level-2**… two approvals"* | 1/2 (pin only) |

1. **`k=5` is not "more context is better"; it is the only `k` where the document is inside the
   window.** With the addendum present the model resolves the contradiction correctly in **2/2** runs,
   with and without provenance headers.
2. **`k=2` is the worst of the three.** The window holds exactly the two contradictory echoes and no
   document, and the model picks the obsolete one. This is the failure mode the plan attributes to the
   absence of a preference/recency signal (finding 1.14, §5.2 D1), observed here at the cutoff.
3. **`k=1` is not a conceptual improvement.** The correct answer comes from `ep_s1_t2` (`source="chat"`),
   i.e. the model recalls its own previous answer; the document pin fails. This is the
   "correct answer + gold absent" cell that X9 classified as an invalid probe.
   The 100% → 75% → 87.5% curve is therefore an artefact of **which channel fits in the window**.

**Consequence for the scenario:** the T4 pin is satisfiable only at `k ≥ 3`. Below that,
`document_addendum_revision_01` stops measuring document revision and starts measuring reconciliation
of conflicting self-echoes.

### 5.1 How to report this without misattributing it

The `k`-dependence is a **result, not a defect**. "Insert a document that supersedes an existing fact"
is precisely the operation an append-only store cannot express, and the echo/document competition at T4
is the observable consequence of that limitation. What changes with `k` is *which channel fits in the
window*, so every number has to travel with its attribution.

| Claim allowed by these runs | Claim **not** allowed |
|---|---|
| In an append-only store without provenance, a superseding document **does not displace the stale chat memory**: gold lands at rank 3 behind two self-echoes (Δ = −0.0117 / −0.0302). | "L2 fails document revision at `k=2`." (it fails *because* the document is not in the window) |
| At `k=2` the model reproduces the superseded fact when only the two contradictory echoes are in context (0/2 assertions). | "Accuracy is non-monotone in `k`" as a property of the model (it is a property of the window) |
| At `k=1` the correct answer comes from `source="chat"`; `must_contain` passes while `must_retrieve_chunk` fails — answer right, document channel unused. | "L2 answers correctly at `k=1`." (it answers from its own previous answer) |
| The store has **no primitive for obsolescence**: `turn_number` and `source` are stored but never score, so recency and authority are unavailable to retrieval. | "The retriever is broken." (it returns the most similar chunk, exactly as specified) |

Required in any table that reports these runs: `top_k`, the gold `rank` / `Δcos` for each pinned probe,
and the `source` split of the retrieved chunks. Without those three columns the same numbers can be
read as a reasoning failure instead of an architectural one.

This is also why the scenario is worth keeping **as a collision test**: it is the cleanest evidence in
the repo that supersession needs an `UPDATE`/`DELETE` primitive (or an authority/recency score), which
is the L3 argument.

---

## 6. Findings

| # | Finding | Evidence |
|---|---|---|
| **F1** | **Dilution is `k`-insensitive; its cost is pure P3.** 9/9 at `k=1/2/5`; context −58.5% (−k2) and −79.2% (−k1); 214 tokens per passed assertion at `k=2` vs 430 at `k=5`. | §3.2, §4.2 |
| **F2** | **Addendum T4 is `k`-coupled by construction**: gold rank 3 in 2/2 `k=5` runs, gold score deterministic at 0.6631, defeated by two chat echoes (Δ = −0.0117 and −0.0302). | §4.1 |
| **F3** | **The pass-rate curve is non-monotone because the failing channel changes**: at `k=2` the model reproduces the stale fact (0/2 on T4); at `k=1` it answers correctly from its own prior answer (1/2). | §5 |
| **F4** | **`must_contain` alone cannot attribute a fact to a channel.** At `k=1`, T4's `Level-2` is satisfied by `ep_s1_t2` (`source="chat"`). Any claim about document memory needs the `must_retrieve_chunk`/`source` pair. | §5 item 3 |
| **F5** | **The D1 "one-character supersession" cosine prediction does not transfer to this fixture.** Gold beats the superseded policy chunks by +0.046/+0.066; the competition is chat-vs-document, not doc-vs-doc. | §4.1 |
| **F6** | **`k` does not reduce retrieval or turn latency.** The store returns the full ranking and slices; only the prompt payload shrinks. Reporting a "latency win" from a smaller `k` would be wrong. | §3.3 |
| **F7** | **T1 memory pollution is real, reproducible and ungraded.** At `k=5` the dilution chit-chat turn enumerates rooms and the `FAC-2291` memo (2/2 runs); at `k=1` it leaks the scope memo; only `k=2` stays neutral. | §4.2, run `20261103_134500` T1 |
| **F8** | **The 2026-11-02 format confound is closed.** Both `k=5` baselines pass on HEAD without provenance headers; Δ context chars = 226 = the five removed headers. | §2.1 |

---

## 7. Threats to validity / caveats

1. **N = 1 per cell at `k=1` and `k=2`; N = 2 at `k=5`.** The *retrieval* side is deterministic
   (document scores identical to 4 decimals), so the structural claim (gold rank 3 ⇒ `hit@2 = 0`) is
   not sampling noise. The *behavioural* side is stochastic: T4 at `k=2` is decided by a 0.0031
   difference between two echoes whose embeddings depend on the answers the model itself generated.
2. **Four retrieval pins per scenario is a very small probe set.** 4/4 gives a Wilson 95% CI of
   roughly [0.51, 1.00]: "Recall@1 = 100%" in dilution is compatible with a true recall near 0.5.
   Do not report point estimates without the interval, and do not present a 0.019 margin as robustness.
3. **`coverage ratio` must use document chunks (6 and 9), not `corpus_chunks` (10 and 15),** because
   the latter includes the chat episodes appended during the run.
4. **`corpus_chars` is not comparable across runs** for the same reason (it is a snapshot taken at the
   end and grows with the verbosity of the model's answers: 6 295 / 5 818 / 6 081 in addendum).
5. **`Δcos` for chat chunks is not comparable across runs**; for document chunks it is.
6. **`completion_tokens` varies ±40% between runs of the same cell**, so total-token and latency
   comparisons across `k` are noisy by construction.
7. **The addendum is a single chunk** (1 195 chars < `chunk_size=1200`), so this scenario exercises
   *single-chunk revision*, not chunk-boundary behaviour (R-B).
8. **The `system_event` text is ingested as a document chunk** (`doc_evt_s1_t2`, 102 chars), so the
   document channel holds three sources (policy, addendum, event echo). It ranked 4th at T3 (0.5025)
   and 3rd at T5 (0.4721), i.e. it competes but never wins.

---

## 8. Proposed changes to the tests (from the weaknesses observed here)

Priority reflects "does this invalidate a number we would publish?".

### P0 — the measured variable changes with `k` (document it, do not "fix" it)

| ID | Weakness | Change | Where | Cost |
|---|---|---|---|---|
| **A1** | `document_addendum_revision_01` T4's gold chunk is **rank 3** (2/2 runs), so the `must_retrieve_chunk` pin is only satisfiable at `k ≥ 3`. At `k ≤ 2` the turn measures chat-echo reconciliation, not document revision. | Redesign the T4 probe so the chat echo cannot answer it: (a) a **zero-lexical-echo paraphrase** of the s1 question (same pattern already enforced by `tests/test_stage4_scenarios.py::test_probes_use_zero_lexical_echo_paraphrase`), and/or (b) assert an **addendum-only token** that the s1 answers never contain (a clause id, a CAB quorum, the effective date) — the trick T5 already uses with `2026-10-01`. | `datasets/conversations/logical_consistency/document_revision/test_document_addendum_revision_01.yaml` T4 + a guard test in `tests/test_document_ingestion.py` | 1 YAML edit + 1 guard test |
| **A2** | Nothing in the fixture or the runner records that a probe needs a minimum `k`. A run at `k=1` silently produces a "failed retrieval" that is a cutoff artefact. | Declare `metadata.min_top_k: 3` on the scenario (`metadata` is already `dict[str, Any]` — `core/schemas.py:228`, so no schema change), and have the runner log a warning when `--top-k` is below it. Report it in the summary table. | scenario YAML + `core/runner.py` (near the `--top-k` resolution in `run.py`) | small |
| **A3** | Dilution's T5 (`must_not_contain: "UPS-"`) is a leak canary whose teeth **decrease with `k`**: with `k=1` the model never sees the other serials, so passing it says nothing. | Report it as a separate `over_disclosure` metric (per-`k`), never as evidence of low-`k` robustness. It is also a token-*class* negation (X10a) — the deterministic "no match outside an allow-list" rule of §8 decision 10 is still the right long-term fix. | reporting / `TODOs` §8 decision 10 | small |

> **Decision recorded (2026-11-03, after review):** A1 is **not** a fixture defect and the T4 probe
> stays as it is. Inserting a document that supersedes an existing fact into an append-only store, and
> the resulting competition between the document and the self-echoes, is the object of measurement
> (see §5.1). The scenario is reported as a *collision / channel-precedence* test. Consequences:
> **A2** (`min_top_k` + attribution in the summary) and **A3** (per-`k` over-disclosure metric) are the
> only mandatory items in this block; the probe redesign below survives as an **optional companion
> scenario (C1)**, not as a correction of this fixture.

### P1 — measurement quality (cheap, high value)

| ID | Weakness | Change | Where | Cost |
|---|---|---|---|---|
| **B1** | `document_dilution_01`'s corpus is 9 docs / 2 172 chars, **smaller than one `chunk_size`**: `k=5` already sees 56% of the document chunks. "Dilution" is nominal; `k=1` passes because the store is tiny. | Scale to 20–60 docs (plan cards D5/D16) before claiming low-`k` robustness. | `datasets/documents/` + `scripts/generate_corpus.py` | 1 fixture + re-baseline |
| **B2** | Four pins per scenario and margins of **0.019–0.027** at the hard decoys. 4/4 is compatible with a true recall near 0.5. | Report `Δcos` (sign **and** magnitude) next to every `hit@k` (plan §5.3.3) and run ≥3 replicates per cell before quoting a percentage. | reporting + run matrix | 2 runs/cell |
| **B3** | The addendum scenario has **no absence control**: nothing proves T3–T5 are unreachable without the documents (dilution does document its `--no-documents` contract). | Add the same contract and one `--no-documents` run: T1, T3, T4, T5 must all fail. | YAML header + 1 run | 1 run |
| **B4** | `summary.md` omits `top_k`, `embedding_model_id`, `chunk_size` and `corpus_chunks`, although `summary.csv` already carries them (`core/runner.py:1350-1370` vs `:1397`). This is exactly the column whose absence lets a `k=1` run be compared with a `k=5` run by accident. | Add those columns to the Markdown table. | `core/runner.py:1397` (`headers` / `table_rows`) | ~10 lines |
| **B5** | `addendum_1.md` is 1 195 chars, i.e. **one chunk**. If the claim is "chunk collision", the superseding clause should be able to straddle a 1 200-char boundary (R-B). | Either rename the claim to "single-chunk revision" or grow the addendum past `chunk_size` so it splits into 2+ chunks. | `datasets/documents/addendum_1.md` | fixture |

### P2 — what the experiment shows is missing

| ID | Gap | Change |
|---|---|---|
| **C1** | There is **no scenario that grades document-vs-echo precedence**. | `document_vs_echo_01`: s1 establishes the old fact, a document supersedes it, s3 probes with a low-overlap paraphrase; assert `must_contain` the new fact + `must_retrieve_chunk doc_*` + `must_not_retrieve_chunk ep_s1_*` (rule already supported: `core/runner.py:163`). Note: this **cannot** be bolted onto the current T4 — at `k=5` the stale echo is retrieved at rank 1, so the assertion would fail today. |
| **C2** | **T1 pollution (D12) is ungraded** in both document scenarios. | `memory_pollution_01`: chit-chat turn graded with `must_not_contain: "FAC-2291"` (R10-safe: the token never appears in the user turn). Keep it as its own scenario so dilution's meaning is preserved — bolting it onto dilution T1 would immediately turn 9/9 into 8/9 at `k=5` and 8/9 at `k=1`. |
| **C3** | The curve jumps `k=2 → 5` and the transition of addendum T4 is at `k=3`. | Add `k=3` to the sweep (1 run per scenario) to bracket the boundary instead of interpolating. |

### What **not** to change

- **Do not lower the suite default `top_k` to 1 or 2 on the strength of this note.** Only dilution
  supports it. Addendum loses its document pin and its headline number becomes uninterpretable.
- **Do not add `must_not_retrieve_chunk: ep_s1_t1` to the current T4.** At `k=5` the stale echo is
  legitimately retrieved at rank 1, so the assertion would fail today and convert a *diagnostic*
  signal into a red test without measuring anything new (C1 is the place for it).
- **Do not add `must_not_contain: "Level-1"`** to T3/T4/T5: a correct answer may legitimately say
  "Level-1 is no longer sufficient" (rule R1). The current negative probe (T5's addendum-only
  deadline) is the correct pattern.

---

## 9. Reproduction

```powershell
# k = 5 (baseline; also the only setting where addendum T4's pin is satisfiable)
python run.py --level 2 --scenario document_addendum_revision_01 --model deepseek-v4-pro `
  --prompt baseline_react --backend vector --embedding-model qwen3-embedding-0.6b --top-k 5
python run.py --level 2 --scenario document_dilution_01 --model deepseek-v4-pro `
  --prompt baseline_react --backend vector --embedding-model qwen3-embedding-0.6b --top-k 5

# k = 2 (efficient point for dilution; worst case for addendum)
python run.py --level 2 --scenario document_addendum_revision_01 --model deepseek-v4-pro `
  --prompt baseline_react --backend vector --embedding-model qwen3-embedding-0.6b --top-k 2
python run.py --level 2 --scenario document_dilution_01 --model deepseek-v4-pro `
  --prompt baseline_react --backend vector --embedding-model qwen3-embedding-0.6b --top-k 2

# k = 1 (dilution: 100%; addendum: answer right, document pin red)
python run.py --level 2 --scenario document_addendum_revision_01 --model deepseek-v4-pro `
  --prompt baseline_react --backend vector --embedding-model qwen3-embedding-0.6b --top-k 1
python run.py --level 2 --scenario document_dilution_01 --model deepseek-v4-pro `
  --prompt baseline_react --backend vector --embedding-model qwen3-embedding-0.6b --top-k 1
```

Per-run telemetry used for every number above: `results.json` →
`turn_results[].retrieved_memories[]` (`chunk_id`, `rank`, `score`, `source`, `char_count`),
`retrieved_context_tokens`, `prompt_tokens`, `retrieval_latency_ms`; and
`memory/<run_id>/manifest.json` → `document_chunks`, `total_chunks`, `by_source`.

---

## 10. One-line conclusion

`document_dilution_01` can be quoted at `k=2` (same accuracy, half the context) — but the corpus must
grow first. `document_addendum_revision_01` is a valid **collision / channel-precedence** result at
every `k`, provided the `k`-dependence is reported with its attribution: the gold chunk sits at rank 3
behind the model's own answers, so at `k=5` the document acts as the tie-breaker (8/8), at `k=2` only
the two contradictory echoes are in context and the model reproduces the superseded fact (6/8), and at
`k=1` the answer is right but the document channel is unused (7/8). The non-monotone
100% → 75% → 87.5% curve is the signature of that substitution — expected for an append-only store —
and it is the strongest evidence in the repo that supersession needs an `UPDATE`/`DELETE` primitive.


# Extraction-Difficulty Calibration Audit (D1–D3) vs Memory-Level Ablation (L0–L4)

**Repository:** `agent-memory-benchmark` (Thesis harness, DCIC/UNS — KRR group)
**Audit date:** 2026-11-10
**Auditor role:** Principal AI Test Engineer / Benchmark Architect
**Scope:** `datasets/conversations/**/*.yaml` (33 scenarios), `datasets/documents/**` (272 corpus files), `core/schemas.py`, `core/runner.py`, `core/metrics.py`, `core/dataset_layout.py`, `configs/models.yaml`, `configs/prompts.yaml`, `tests/**` (19 files), `outputs/runs/**` (empirical run records in `results.json`).
**Out of scope:** `archive/**` (frozen snapshots), agent graphs beyond the retrieval-context contract they impose on scenarios.

---

## 0. Method & Evidence Base

Every number in this report was computed from the repository, not estimated. Reproduce with
`python run.py --list-scenarios` (the 33 ids the loader sees) and `pytest -q` (19 files, offline).

| Evidence artefact | How it was derived | Used in |
|---|---|---|
| Turn/conversation word counts | `\S+` tokenisation of every `turns[].content` + every resolved `documents[]` (inline and `path`) | §1, §2, §3 |
| Distractor-claim counts | manual coding per scenario, cross-checked against `metadata.distractor_*` | §2, §3 |
| Assertion histogram | 232 rules parsed by `AssertionRule.type` | §1, §3C |
| Probe-session classification | for each asserting turn: `session_id` == first chat turn of that session ⇒ intra-session, else cross-session | §3B |
| Lexical uniqueness ratio | unique(long sentences) / total(long sentences) across a scenario's whole context | §3A |
| Declared vs evaluated retrieval pins | `core.metrics.declared_retrieval_assertions` | §3C |
| Empirical level behaviour | every `outputs/runs/*/results.json`, grouped by `(scenario_id, agent_level)` | §3C |
| Schema behaviour of unknown keys | `BenchmarkScenario.model_validate({… "difficulty_tier": "D3"})` | §3B, §4 |

**Nature of the evidence.** This is a static + telemetry-replay audit; no new model runs were
executed. Model-dependent statements are labelled *empirical* (backed by `results.json`) or
*structural* (backed by code and therefore model-independent).

---

## 1. Executive Summary & Calibration Score

### 1.1 Headline

**Calibration score: 3.5 / 10.**

| Dimension | Score | One-line verdict |
|---|---:|---|
| **A. Signal-to-noise & surface form (D1/D2/D3)** | **3 / 10** | D1 exists only as a *de-facto* micro-context regime (median turn = 16 words); D2 exists in 3 families; D3 exists **only** as retrieval scale over machine-generated near-duplicate prose. The word bands that define the tiers (200–400 / 400–800 / 600–1000+) are almost entirely unoccupied. |
| **B. Orthogonality (difficulty vs memory ablation)** | **4 / 10** | The *reasoning* for separation is excellent and written per scenario (`expected_level_N_behavior`, three paired-control families), but it is **prose, not schema**: nothing enforces it, nothing reports it, and 76 % of probes never exercise retrieval at all. |
| **C. Assertion rigor & metric desensitization** | **4 / 10** | Retrieval-metric design is genuinely strong (declared denominator, Wilson CI, `must_not_retrieve_*` supported). The **response** layer is one pooled substring `pass_rate` in which an amnesic agent empirically scores **1.000** (`adversarial_forget_01`, L0, 3 models). No store-level assertion exists, so *false updates* and *context blending* are detectable only when an author remembered a negative text guard. |
| **D. Trajectory & temporal progression** | **6 / 10** | Strongest axis: the runner sends only the turn delta (`HumanMessage(content=turn.content)`), so no prompt re-states prior facts and turns are genuinely incremental. Missing: a declarative **cumulative** gold (`expected_state`) — cumulative memory state is unverifiable, only per-turn responses are. |

---
### 1.2 Does the suite discriminate extraction difficulty (D1–D3) from memory capability (L0–L4)?

**Not as a suite. Yes in three isolated pockets — and the metric layer throws that separation away.**

1. **There is no D-tier concept in the codebase.** The tokens `D1`/`D2`/`D3` appear in `SPEC.md §5.8`
   and `core/metrics.py` as the *retrieval-recall policy decisions* (declared denominator, rule-level
   unit, `must_retrieve_chunk`-only) — a real **nomenclature collision** with FEB-30 tiers. The only
   difficulty field that exists is `metadata.difficulty` with three values (`basic` 5, `moderate` 10,
   `hard` 18), and it encodes **reasoning** difficulty, not extraction difficulty:
   `in_context_derivation_01` is labelled `hard` although it is a 143-word prompt with 8 explicit
   bullet premises and zero noise (extraction tier **D1**), while `document_dilution_01` is `hard` on
   354 words of near-duplicate memos (extraction tier **D2**).

2. **The scenarios do not span the tier bands.** 203 of 215 turns are under 50 words; **zero** turns
   fall in 200–399 (D1's band) and **zero** in 400–799 (D2's band). Volume enters only through
   documents, and then it jumps straight to 6.9 k–173 k words. The suite measures *"how many turns /
   how many documents"*, not *"how hard is it to extract this claim from its context"*.

3. **The discriminating signal is real but unrecorded.** Where the authors built a ladder on purpose —
   `document_dilution_01` (9 same-schema memos, 3 sharing the probe's exact key phrase),
   `entity_binding_01` (4 entities × 2 attribute surfaces, 8 retrieval pins), `top_k_saturation_01`
   (10 near-identical inventory entries) — the design meets or exceeds the D2 bar. But the reporting
   layer collapses every assertion into one `pass_rate`, so a D-tier failure and an L-tier failure are
   numerically identical in `summary.csv`.

**Consequence for the thesis.** A claim of the form *"L3 outperforms L2 on revision"* is currently
supportable (the L2-append-only vs L3-CRUD cells are well instrumented). A claim of the form *"our
scenarios are D1/D2/D3 calibrated"* is **not** supportable from the current artefacts: there is no
field to cite, no word-band test and no per-tier aggregate — and the empirical counter-example
(`L0 = 1.000` on a `hard` scenario) shows that at least two scenarios reward amnesia.

### 1.3 What is genuinely strong (do not refactor away)

1. **Retrieval pins make failure causal.** `must_retrieve_chunk` is evaluated against
   `retrieved_memories` telemetry, so "the gold chunk never reached the model" is provable rather than
   inferred (`core/runner.py::_evaluate_retrieval_assertion`). Only 10/33 scenarios use it — a
   coverage problem, not a design problem.
2. **The recall denominator cannot be gamed by infrastructure.** `declared_retrieval_assertions`
   counts pins from the YAML, so an errored probe stays in the denominator and counts as a miss
   (`core/metrics.py`, SPEC §5.8 policy D1) — a mature decision documented against a real artefact
   (notes F7).
3. **Three paired in-context controls** establish the reasoning ceiling a memory failure must be
   measured against: `cross_thread_derived_01` ↔ `in_context_derivation_01`,
   `derived_belief_revision_01` ↔ `in_context_derived_revision_01`,
   `backward_modus_tollens_01` ↔ `in_context_modus_tollens_01`. The interpretation rule is stated in
   each file (`pass control + fail scenario ⇒ memory failure; fail both ⇒ reasoning failure`).
4. **Delta-only turn dispatch.** The runner never repeats earlier context, so a cross-session probe
   cannot be answered by prompt duplication — memory is genuinely required.
5. **Rich per-scenario metadata** (`expected_level_0..4_behavior`, `failure_taxonomy`,
   `headline_metric`, `observational_probe_note`) with unusually honest self-criticism
   (e.g. `memory_injection_01` states outright that amnesic declines inflate `pass_rate` to 60 %).

The gap is that items 3–5 live in YAML prose and `metadata: dict[str, Any]`; nothing in
`core/schemas.py`, `core/runner.py` or `summary.csv` knows they exist.

---

## 2. Inventory & Classification of Current Scenarios

Column meanings: **Current Assigned Tier** = the only difficulty field that exists today
(`metadata.difficulty`, three values, no D-tier equivalent). **Actual Effective Tier** = this audit's
coding of the extraction difficulty the scenario actually measures, with evidence (context words,
distractor count, channel) in parentheses. **Target Memory Level Tested** = the lowest level that can
in principle pass (derived from `session_id` count, document ingestion and revision semantics, *not*
from `metadata.expected_level_*_behavior`). **Risk** = the dominant way the cell can produce a
false positive or a mis-attributed failure.

### 2.1 Chat-only scenarios (23)

| Scenario ID | Current Assigned Tier | Actual Effective Tier (D1/D2/D3) | Target Memory Level Tested | Risk / Confounding Factor |
| :--- | :--- | :--- | :--- | :--- |
| `belief_revision_01` | basic | **D1** (77 w, 5 turns, 0 distractor claims) | L1 (one session) | Probe is intra-session ⇒ L2's `exclude_session_id` never fires; `must_contain "3"` is a 1-char-class assertion (passes on "2023", "3 cats", …). |
| `explicit_forget_01` | moderate | **D1** (74 w) | L1 | 2 of 5 rules are `must_not_contain` → an amnesic refusal already banks 40 % (empirical L0 = 0.400). |
| `negative_hallucination_01` | moderate | **D1** (73 w) | L1 (nominally) — **inverted in practice** | Empirical: L0 = **1.000**, L1 = 0.719 (11 runs). More memory lowers the score; the scenario penalises the capability it names. |
| `adversarial_forget_01` | hard | **D1** (90 w) | L1 | **Only negative rules** (2 of 2). Empirical L0 = **1.000** on 3 models. The file *declares* this ("Passes purely by amnesia"), so the design is intentional — the defect is that the declaration is prose only, so the cell's 1.000 is pooled with quality scores in `summary.csv`. |
| `adversarial_forget_02` | hard | **D1** (187 w) | L1 | Same: 2 negative rules, no positive "preserved facts survive" probe. L0 = 1.000 (3 runs), L1 mean 0.929 (7). Intent also declared in prose ("Passes purely by amnesia"). |
| `attrition_01` | moderate | **D1** (119 w; "distractors" = 6 trivia Q&A turns, **0 distractor claims**) | L1 | Noise lives in conversation *length*, not in the *context of the claim*: no dilution pressure on extraction. Discriminates L0 (0.667) vs L1 (1.000) only. |
| `needle_haystack_01` | moderate | **D1/D2** (116-w haystack, 4 lexically distinct distractor projects) | **L1 — empirically, not "attention"** | File claims "tests attention, not persistence", but L0 = **0.000** and L1 = 1.000 (8 runs): every probe needs T1 across turns. No pins ⇒ no attribution. |
| `temporal_multi_hop_01` | moderate | **D1** (59 w, 2 hops) | L1 | 2 assertions for 4 turns; no per-hop partial credit. |
| `3hop_temporal_01` | hard | **D1** (163 w, 3 hops, single session) | L1 | **1 assertion for 9 turns** — 8 turns carry no signal, so one numeric slip erases the whole run. |
| `in_context_control_01` | basic | **D1** (59 w, all premises in-turn) | L0 ceiling control | Correctly built as a control; values echo the prompt by design (unusable as a *test*). |
| `in_context_needle_01` | basic | **D1** (78 w, 4 in-turn distractors, distinct lexicon) | L0 ceiling control | Single turn; measures attention only; value echo. |
| `in_context_multihop_01` | basic | **D1** (47 w) | L0 ceiling control | 1 of 3 L0 runs failed a 47-word single-turn prompt ⇒ noise floor unquantified at n=3. |
| `in_context_derivation_01` | hard | **D1** (143 w, 8 explicit bullets, 0 noise) | Reasoning control for `cross_thread_derived_01` | Its `hard` label is *reasoning* difficulty. Any aggregate keyed on `difficulty` mixes it with genuinely hard-extraction cells. |
| `in_context_modus_tollens_01` | hard | **D1** (167 w) | Reasoning control for `backward_modus_tollens_01` | Menu echo: the three legal labels are printed in the probe, so `must_contain` is satisfiable by restating the menu. |
| `in_context_derived_revision_01` | moderate | **D1** (147 w) | Reasoning control for `derived_belief_revision_01` | Premise echo is unavoidable (that is the point of a control) — ceiling-only artefact. |
| `overhead_baseline_01` | basic | **D1** (204 w, **zero memory demand**) | All levels (P3 control) | 7 of 10 values are ≤2 chars (`O`, `0`, `6`, `7`, `9`, `24`): substring matching is saturated, so it cannot detect a wrong answer. Correct for P3, but must be excluded from every accuracy aggregate. |
| `cross_thread_01` | moderate | **D1** (198 w, 12 turns, 2 sessions) | L2 | **No retrieval pins** ⇒ a failure cannot be attributed to retrieval vs read-time vs generation. Noise = trivia turns only. |
| `cross_thread_derived_01` | hard | **D1 + reasoning** (256 w, 11 turns, 3 sessions) | L2 (L1 inside sessions) | 2 of 3 rules are negatives on the *final answer*; 8 of 11 turns unasserted. Good paired control (`in_context_derivation_01`). |
| `backward_modus_tollens_01` | hard | **D1→D2** (308 w, rule stated once in T3, 13 turns) | L2 | T12/T13 assert one of three enumerated labels **without `must_not_contain` guards for the other two** ⇒ listing the menu passes. |
| `derived_belief_revision_01` | hard | **D2** (284 w: rule + inventory + revision over 5 turns) | L2 | Best revision cell in the suite (positive + stale-value guard), but 11 of 13 turns unasserted and no pins. |
| `entity_binding_01` | hard | **D2** (323 w, 8 coreference-split attribute chunks, uniq 0.727) | L2 | Strongest cell overall (8 pins + 12 negatives + paired commitment probes). Residual: retrieval ambiguity is *intentional* but undocumented as a calibration parameter. |
| `top_k_saturation_01` | hard | **D2/D3** (376 w, 10 near-identical entries, 3 sessions) | L2 | **9 negatives vs 1 positive** ⇒ refusal scores 0.90. No pins, so dilution is inferred rather than proven (`must_not_retrieve_chunk` is supported but unused). |
| `memory_injection_01` | hard | **D1/D2** (299 w; payload inside one 127-w turn) | L1 (T6) + L2 (T8) | 6 of 10 rules are negative; the file itself documents the amnesia inflation ("L0 scores 60 % on assertion count") with no metric to correct it. |



### 2.2 Document-memory scenarios (10)

| Scenario ID | Current Assigned Tier | Actual Effective Tier (D1/D2/D3) | Target Memory Level Tested | Risk / Confounding Factor |
| :--- | :--- | :--- | :--- | :--- |
| `document_addendum_revision_01` | hard | **D2** (450 doc w, mid-run `system_event` supersession) | L2 vs L3 (append-only vs mutation) | 4 pins, but the *same* gold chunk (`doc_addendum_1`) is pinned on 3 turns: document-level recall is dominated by one document. |
| `document_cross_thread_01` | moderate | **D2** (512 doc w, 1 document) | L2 | **No pins at all** ⇒ retrieval failure invisible; 2 of 5 rules negative. |
| `document_needle_01` | moderate | **D2** (962 doc w, 2 documents) | L2 | 2 pins on the same document; the precision turn negates a token *class* (`iris-breakglass-77`), a known-fragile pattern. |
| `document_dilution_01` | hard | **D2** (354 doc w, 9 same-schema memos, 3 sharing the probe phrase) | L2 | Schema-level competition is the real difficulty, but it is asserted only in a comment; the *scenario* records no similarity/overlap number that a test could enforce. Volume (354 w) is below D1's own band, so tier-by-word-count would misclassify it. |
| `document_authority_vs_recency_01` | hard | **D3-retrieval / D1-extraction** (52 k w templated corpus, 60 docs) | L2 vs L3 (authority vs recency) | 97 % of corpus sentences are literal duplicates; the target is one verbatim sentence. Measures near-duplicate retrieval, not realistic extraction. Extraction tier is D1 once the chunk is in context. |
| `document_effective_dates_01` | hard | **D3-retrieval / D2-extraction** (52 k w, 61 docs) | L2 vs L3 | 6 of 14 rules negative; effective-date arithmetic adds a *reasoning* confound on top of extraction, and the two are not separable in `pass_rate`. |
| `document_prevalence_update_vs_distractor_01` | hard | **D3-retrieval / D2-extraction** (52 k w, 60 docs) | L2 vs L3 | Update-vs-distractor design (QZX/YKX) is solid, but the distractor lives in the same synthetic corpus ⇒ same realism caveat. |
| `document_corpus_size_008_01` | moderate | **D3** (6.9 k w, 8 docs) | L2 scaling | Needle is a verbatim single sentence: extraction is D1, only retrieval scales. 8/16 rules are pins; no precision pins. |
| `document_corpus_size_060_01` | hard | **D3** (52 k w, 60 docs) | L2 scaling | 6.5 k words of context per asserted claim; 97 % duplicate sentences. |
| `document_corpus_size_200_01` | hard | **D3** (173 k w, 200 docs) | L2 scaling | 21.6 k words per asserted claim; 98 % duplicate sentences; no `must_not_retrieve_chunk` precision pin, so "retrieved the right chunk among 200" is not balanced by "did not retrieve the wrong one". |

### 2.3 Aggregate picture

| Measure | Value | Reading |
|---|---:|---|
| Scenarios | 33 (18 single-thread / 15 multi-thread) | — |
| Turns | 215 (214 `user`, 1 `system_event`) | — |
| Turns per extraction tier (this audit) | **D1 = 16**, mixed D1/D2 = 4, **D2 = 7**, **D3 = 6** | D1 is 2.4× the size of D2 alone, and no scenario sits in D1's *and* D2's word bands |
| Turns with 50–199 words | 12 / 215 (5.6 %) | the only "long-ish" chat turns |
| Turns with 200–399 words | **0 / 215 (0 %)** | D1's declared band is unoccupied |
| Turns with 400–799 words | **0 / 215 (0 %)** | D2's declared band is unoccupied |
| Turns with ≥800 words | **0 / 215** | D3 volume exists only as documents |
| Median / p75 / max turn length | 16 / 27 / 167 words | micro-context regime |
| Assertion rules | 232 = 121 `must_contain` + 55 `must_not_contain` + 55 `must_retrieve_chunk` + 1 `must_retrieve_source` + **0 `must_not_retrieve_*`** | zero precision pins anywhere |
| Scenarios declaring retrieval pins | 10 / 33 (30 %) | `retrieval_recall` is `null` for the other 23 |
| Asserting turns | 113, of which **86 intra-session (76.1 %)** and 27 cross-session (23.9 %) | 3 in 4 probes never touch retrieval |
| Scenarios with a paired in-context control | 3 / 33 | the only enforced D-vs-L separation |
| Assertion values ≤2 chars | 10 | substring matching degenerates to "contains a digit/letter" |

---


## 3. Diagnostic Gaps & Flaws Identified

### 3.A Signal-to-noise ratio & surface form (D1 vs D2 vs D3)

**A1 — The tier bands are empty; the suite is a "D0" regime.** Median turn = 16 words, p75 = 27.
`belief_revision_01` states its target claim in 23 words; `explicit_forget_01` in 24. FEB-30 D1 asks for
a 200–400 word context with zero distractors. The chat cells are *an order of magnitude* below that
floor, so the extraction step is trivially easy and the only thing measured is whether the text
survived into the prompt at all. **Fixed by** the word-band calibration in §4 plus a
`test_difficulty_calibration.py` ratchet (§5, H3).

**A2 — "Distractor claims" are frequently not claims at all.** `attrition_01` calls its six trivia
questions "distractor turns" (`metadata.distractor_turns: 6`), but they are *separate turns* about
Lisbon, stacks vs queues, RGB, García Márquez and photosynthesis. They add conversation length, not
competition inside the context of the target claim: nothing in them is a candidate answer to "when
does Falcon launch?". `cross_thread_01` has the same shape (8 seeding turns + trivia). These scenarios
measure context-window attenuation — a working-memory property — never extraction difficulty.

**A3 — Where distractors are real, they are lexically distinct and trivially filtered.**
`needle_haystack_01` embeds `DELTA-7749` among four projects whose names, leads, budgets and focus
areas are all different words. Any embedding model separates those clusters (cosine ≈ 0.1–0.2). The
repo's own record shows the consequence: **L0 = 0.000, L1 = 1.000** — the cell is saturated at both
ends, and the intermediate region where extraction difficulty lives is never sampled (see C6).

**A4 — The one genuine D2 family works for the right reason, but is undocumented.**
`document_dilution_01` (9 memos, identical schema, three decoys carrying the probe's exact key phrase,
pairwise cosine 0.99968 per the file comment) and `entity_binding_01` (4 servers × port + weekday,
split across coreference-thin chunks) are best-in-class: the difficulty is *lexical competition*, not
volume. Yet neither records the competition numerically in a form a test could enforce, and
`document_dilution_01` is only 354 words — *below* the D1 band — which proves word count alone is the
wrong calibration variable. **Recommendation:** add `near_duplicate_competition` (max pairwise
similarity + shared-phrase count) as a first-class field beside `context_words`.

**A5 — D3 is synthetic, not "real multi-author noise".** The 60-document corpus has **3,159 sentences
of which 212 are unique (6.7 %)**; the 200-document corpus 10,513 / 232 unique (2.2 %). Most sentences
are one of ~14 templates ("The operator records the change ticket identifier on every command they
execute." ×162) each stamped with a synthetic `Reference CORPUS-001-01` label. Unique-sentence ratio
per scenario:

| Scenario | Sentences | Unique | Ratio |
|---|---:|---:|---:|
| `document_corpus_size_200_01` | 10 513 | 232 | **0.022** |
| `document_effective_dates_01` | 3 164 | 89 | 0.028 |
| `document_corpus_size_060_01` | 3 167 | 92 | 0.029 |
| `document_authority_vs_recency_01` | 3 170 | 95 | 0.030 |
| `document_prevalence_update_vs_distractor_01` | 3 171 | 94 | 0.030 |
| `document_corpus_size_008_01` | 427 | 40 | 0.094 |
| `entity_binding_01` (chat) | 22 | 16 | 0.727 |
| all other 26 scenarios | 2–75 | = sentences | **1.000** |

The distribution is **bimodal**: either zero lexical redundancy (retrieval trivial) or ~97 %
redundancy (a near-duplicate stress test no realistic document set produces). No scenario occupies the
realistic 0.6–0.9 band. A claim about "extraction from dense real-world documents" cannot rest on a
corpus with ~14 distinct sentences.

**Chunk-level correction (measured).** The redundancy is *combinatorial*, not literal: with the harness
defaults (`chunk_size=600`, `overlap=200`) the 60-doc corpus splits into **880 chunks (14.7/doc)**, all
**880 lexically distinct** — each chunk is a different *permutation* of the ~14 template sentences plus a
`Corpus Note NNN` header. So no duplicate-detection effect is being tested either. What matters is
**claim density**, and there the corpus is effectively empty:

| Corpus | Docs | Chunks | Chunks carrying an opaque code (a claim) | Chunks in which the probe entity occurs |
|---|---:|---:|---:|---:|
| `d16_corpus_008` | 8 | 115 | 9 (7.8 %) | 1 |
| `d16_corpus_060` | 60 | 880 | 10 (**1.1 %**) | 1 |
| `d16_corpus_200` | 200 | 2 933 | 10 (**0.34 %**) | 1 |

99 % of the haystack states no fact at all, and no chunk anywhere holds a *competing* value for the
probed predicate. The corpus is therefore a **keyword-unique needle in a fact-free haystack**.


**A6 — The D3 *extraction* is D1.** Every needle in the D16 corpora is a single verbatim sentence
("The calibration code for the cooling loop LOOP-530 is EOW-WTEL-5216."). Once the chunk reaches the
prompt the answer needs no inference. The D16 cells are **D3-retrieval / D1-extraction**: they
calibrate `top_k` pressure, not extraction. Conflating the two is the most consequential mislabel in
the suite, because it lets "D3 hardness" be claimed for a test whose ground truth is a copy-paste
string.

**A7 — Only one scenario family requires true cross-statement inference.**
`document_authority_vs_recency_01` (documented value vs later official update) and
`document_prevalence_update_vs_distractor_01` are the closest to legitimate D3: the answer needs a
stale statement combined with a superseding one plus an authority rule. Both live inside the templated
corpus, so the inference is real while the surrounding noise is artificial.

**A8 — A pure lexical baseline already solves the D16 corpus (measured).** BM25 (k1 = 1.5, b = 0.75,
same 600/200 chunking, zero semantics, zero embeddings) ranks the gold chunk inside the top-5 for
**24 of 24** probes across the three corpus sizes and at rank 1 for **20 of 24**:

| Corpus | Chunks | Gold chunk BM25 rank 1 | Gold chunk BM25 in top-5 |
|---|---:|---:|---:|
| `d16_corpus_008` | 115 | 7/8 | **8/8** |
| `d16_corpus_060` | 880 | 6/8 | **8/8** |
| `d16_corpus_200` | 2 933 | 7/8 | **8/8** |

Because every probe names the opaque entity token (`LOOP-420`) and that token occurs in exactly one
chunk, the task reduces to a keyword lookup: the size axis measures **ingestion and top-k plumbing**, not
semantic retrieval, and any flat curve across 8 → 200 documents is a property of the probe rather than of
the memory architecture. This does not invalidate the cells — it *re-types* them: they are a
**harness-validation / retrieval-attribution control**, not a D3 extraction test. The discipline this
suggests (a lexical baseline is part of the corpus definition) is step **M8** in §5.




### 3.B Orthogonality (difficulty vs memory ablation)

**B1 — 76 % of probes are intra-session, where L1 wins by construction.** 86 of 113 asserting turns sit
in the *same* `session_id` as their evidence. `agents/level_2_episodic_rag/graph.py` calls
`backend.retrieve(query, top_k, exclude_session_id=session_id)`, so L2 *cannot* retrieve same-session
episodes: for those 86 probes L2 behaves like L1 plus overhead. A failure on an intra-session probe can
therefore never be an extraction/retrieval finding — the evidence is verbatim in the checkpointed
history. Only 27 probes (all in `multi_thread/` plus the document families) actually exercise the memory
mechanism.

**B2 — Scenarios advertised as extraction tests are memory tests (empirical).** `needle_haystack_01`
declares `expected_level_0_behavior: "Should succeed within single context window; tests attention, not
persistence"`. Measured: **L0 mean 0.000 (n=3), L1 mean 1.000 (n=5)**. Every probe depends on T1, so the
cell measures persistence. The metadata statement is falsified by the repository's own telemetry, and no
test guards against that class of drift.

**B3 — Document scenarios are structurally L2-only, silently.** In `run_single`, a `system_event` turn
with no backend attached logs *"no memory backend"* and `continue`s: ingestion never happens, no
assertion is evaluated, and the reported pass rate is computed over chat turns only. On
`document_effective_dates_01`, L0 and L1 both land on **0.429** — the same number for two very different
architectures, because both fail the four retrieval pins identically. The cell carries no information
about the L axis, yet it sits in the same `summary.csv` as informative cells.

**B4 — Intrinsic (store) state is never asserted; only its shadow is.** `AssertionRule` has exactly six
types, all reading either the response text or `retrieved_memories`. `memory/<run_id>/` snapshots are
exported after every run (`export_snapshot`) but **no assertion ever reads them**. Consequences:
(i) a *false update* (store duplicating a superseded value) is undetectable unless an author remembered
a `must_not_contain` on the *response*; (ii) *context blending* (correct value attributed to the wrong
document) is detectable only through the 12 hand-written negatives in `entity_binding_01`;
(iii) L3/L4's core claim — *the store is consistent, not merely the answer* — is tested nowhere.

**B5 — Extrinsic generation and intrinsic state are pooled in one number.** `pass_rate =
passed_assertions / total_assertions` over *all* rule types (`core/runner.py`). A scenario with 8 text
rules and 8 retrieval pins reports one figure; when it drops the reader cannot tell whether retriever or
reader failed. `retrieval_recall` is well built but secondary: present for 10/33 scenarios and never
combined with the text score in a reported decomposition.

**B6 — The difficulty/level matrix is not a matrix.** Per-level expectations exist as prose —
**all 33** scenarios declare `expected_level_0_behavior`, decaying to 21 (L1), 18 (L2), 5 (L3) and
2 (L4) — but there is no execution constraint tying a scenario to a level, no machine-readable
expected-outcome schema (`{"L0": "fail", "L1": "pass"}`), and no test that would catch a scenario an
amnesic agent passes. So the expectations cannot be *checked*: a run that contradicts them is
indistinguishable from a broken expectation.


**B7 — The schema silently discards tier annotations.** Verified:
`BenchmarkScenario.model_validate({… "difficulty_tier": "D3"})` succeeds and the key is dropped;
`ConversationTurn(..., doc_claims=[…], expected_state={…})` keeps only
`turn_number, role, content, assertions, session_id, attachments`. Pydantic's default `extra="ignore"`
means a future D-tier or gold annotation would be silently lost — the same "contract you don't honor"
failure the authors already fixed for `attachments` with an explicit validator.



### 3.C Assertion rigor & metric desensitization

**C1 — The substring engine has no notion of a claim.** `must_contain` is
`value.casefold() in response.casefold()` after NFKC folding. There is no regex, no numeric equality, no
negation scope, and no "the answer must be the *only* value" operator. Ten assertion values are ≤ 2
characters and therefore saturated:

| Scenario / turn | Value | Why it is degenerate |
|---|---|---|
| `overhead_baseline_01` T2/T4/T6/T7/T9/T10 | `6`, `0`, `7`, `O`, `24`, `9` | any digit, or the letter "o" anywhere, passes ("2026", "no", "to") |
| `belief_revision_01` T2/T5 | `3` | matches "3 cats" (correct) but also "2023", "13", "30" |
| `attrition_01` T6 | `Au` | matches "because", "author", "launch" |

`overhead_baseline_01` is a deliberate P3 control, but its values are also the *evidence* behind "all
levels score 100 %", so its accuracy measurement is vacuous.

**C2 — Menu-echo loophole.** `backward_modus_tollens_01` T12 asks for "exactly one of YES_EU_CENTRAL,
NO_EU_CENTRAL, CANNOT_DETERMINE" and asserts only `must_contain: NO_EU_CENTRAL`. A model that prints the
whole menu — or that answers "it is not YES_EU_CENTRAL … so NO_EU_CENTRAL" — passes. Nothing validates
the forced format (contrast `memory_injection_01`, which at least adds `must_not_contain` guards on the
attack tokens). Same pattern in T13 and in `in_context_modus_tollens_01`.

**C3 — Negative-heavy cells invert discrimination (empirical; the single worst artefact).**
`adversarial_forget_01/02` carry **only** `must_not_contain` rules. Measured across every run in
`outputs/runs/`:

| Scenario | Level | Runs | Mean pass_rate | Max |
|---|---|---:|---:|---:|
| `adversarial_forget_01` | L0 | 3 | **1.000** | 1.000 |
| `adversarial_forget_01` | L1 | 6 | 0.583 | 1.000 |
| `adversarial_forget_02` | L0 | 3 | **1.000** | 1.000 |
| `adversarial_forget_02` | L1 | 7 | 0.929 | 1.000 |
| `negative_hallucination_01` | L0 | 3 | **1.000** | 1.000 |
| `negative_hallucination_01` | L1 | 8 | 0.719 | 1.000 |

An amnesic agent scores a *perfect* pass on two `hard` scenarios and beats L1 on a third. The mechanism
is arithmetic: 55 of 232 rules (24 %) are negatives and a refusal passes all of them.
`top_k_saturation_01` is the clearest structural case — 9 negatives + 1 positive ⇒ **any** "I don't
know" answer scores 0.90.

**Nuance (checked in the files).** For `adversarial_forget_01/02` this is *intended*: both declare
`expected_level_0_behavior: "Passes purely by amnesia"`, because a privacy test where an agent that never
stored the secret does not leak it is legitimately passing. The defect is therefore **not the design but
its representation**: the intent lives in prose, so `pass_rate = 1.000` from an amnesic agent is pooled with
quality scores in every aggregate and looks like a memory win. The fix is a declared, machine-readable
`kind: privacy` (+ `amnesia_passes: true`) that excludes the cell from quality aggregates — not a rewrite of
the scenario.


**C4 — Success and safety share one denominator.** Because `must_contain` and `must_not_contain` are
summed, "did it remember?" and "did it avoid leaking?" are averaged. A scenario can improve its score by
adding guards without adding memory pressure, and `pass_rate` therefore drifts whenever an author
hardens a cell (documented in several file comments as "raw assertion-level pass_rate stops flattering
them"). **Recommendation:** report `recall_score`, `precision_score` (guards) and `retrieval_recall` as
three separate columns and never aggregate them.

**C5 — Retrieval metrics are independent — but almost nowhere.** `retrieval_recall`, the Wilson CI, the
declared denominator and `retrieval_assertions_evaluated` are genuinely well designed (SPEC §5.8). Two
gaps: (i) only 10/33 scenarios declare pins, so 23 cells report `null` and drop out of aggregates;
(ii) **zero** scenarios use `must_not_retrieve_chunk` / `must_not_retrieve_source` although the engine
supports them — so "the retriever fetched the distractor too", the actual failure mode of a D2/D3 cell,
is never a metric, only a hand-written text guard. There is also no `Hit@1` / `MRR` over the gold rank
even though `retrieved_memories` carries `rank` for every episode: the data is exported, the metric is
not computed in-run (`scripts/lr_report.py` computes chunk-level Recall@k offline).

**C6 — No mechanism detects false updates or context blending at the store level.** The *scenario*
layer compensates in four places (`entity_binding_01` 12 negatives, `top_k_saturation_01` 9,
`document_effective_dates_01` 6, the QZX/YKX pair in `document_prevalence_…`), but nothing distinguishes
*correct answer with a duplicated stale binding* (inconsistent store) from *correct answer with a clean
store*. Since L3's value proposition is mutation instead of append, this is the largest measurement hole
relative to the thesis claims.

**C7 — LLM compensation loopholes (where a stronger model masks a broken memory).** Five concrete
paths, all visible in the artefacts:

1. **Prompt-level revision instruction.** `configs/prompts.yaml::strict_epistemic` says *"Treat the
   user's most recent statement as the authoritative source when conflicts arise"* and *"you MUST
   explicitly acknowledge the contradiction and UPDATE your understanding"*. On `belief_revision_01`
   (D1, 77 words) the prompt alone can produce a correct answer with a broken store: the model just
   repeats the last value it saw. The harness never records which prompt was used in `pass_rate`
   comparisons across scenarios (`prompt_id` is exported but never used as a control).
2. **Recency heuristics on append-only stores.** `entity_binding_01`/`document_effective_dates_01` ask
   for "today's" value; a model that defaults to the most recent *text* it read can be right for the
   wrong reason, and the run cannot tell that from a correct store mutation (see C6).
3. **In-context carry-over via the model's own previous answer.** `entity_binding_01` T13 and
   `document_dilution_01` both note that a correct free-form answer may be *copied from the model's own
   stored commitment answer* (`ep_s3_t1..t4`). The observation is documented in prose; no assertion
   distinguishes "bound from evidence" from "paraphrased from my own earlier output".
4. **Refusal as a passing strategy** (C3): 24 % of rules are satisfiable by abstaining.
5. **Menu echo** (C2) and **short-token saturation** (C1): cheap wins that do not require the claim.



### 3.D Trajectory & temporal progression

**D1 — Turn-level increments are genuinely new information (strong).** Every seed turn introduces a
fact once and never restates it: `cross_thread_01` T1/T3/T5/T7 each add one entity-attribute pair;
`cross_thread_derived_01` T1/T3/T4/T7/T8 each add one contract clause; `derived_belief_revision_01`
splits rule (T3), inventory (T4), and revision (T9). This is exactly the "distinct new information per
turn" requirement, and it is verified by the runner contract: only `HumanMessage(content=turn.content)`
is sent, with history retained by the agent's checkpointer.

**D2 — Probes avoid restating the premise (strong, with named exceptions).** The hardened document
scenarios enforce this in tests (`test_no_positive_assertion_echoes_its_own_probe`, `R10` in the
comments). Exceptions found:
`backward_modus_tollens_01` T12/T13 (the labels are printed in the probe), and the in-context controls,
where echo is intended.

**D3 — There is no cumulative gold.** The YAML can express *what the user said* (`turns[].content`) and
*what the answer must contain* (`assertions`), but not *what the memory should contain after turn N*.
Consequences: no test can check "the store now holds 4 bindings and 0 stale ones"; no test can check
that a later turn did not silently drop an early fact; drift across a multi-turn run is only visible
through the final probe. `metadata.intrinsic_expectations` does not exist as a concept — the closest is
prose in `metadata.preserved_document_facts` (1 scenario).

**D4 — Turn-specific input is not decomposed from cumulative state.** The `turn_number`-addressed
assertions give turn-level scores, but a scenario cannot declare *which* claims a turn is allowed to
change. `derived_belief_revision_01` should assert "T9 changes the tier of srv-alpha and nothing else";
today that is implied by T13 but never asserted per-turn.

**D5 — `system_event` turns carry no `docClaims`.** The addendum scenario's `system_event` content is a
free-text notice plus an attachment; nothing declares *which* claim it supersedes. The revision is
therefore inferred by the reader from prose ("addendum 1 supersedes §3.2") instead of being a
machine-checkable assertion pair.

---


## 4. Concrete YAML Scenario Template (Calibrated D1–D3)

One topic — *a deploy-region revision plus two constraints that must survive it* — implemented three
times. The three files below were **validated against the live schema**: each parses with
`BenchmarkScenario.model_validate`, each lies inside its declared word band, and the three share
byte-identical probe turns (verified — see §4.7).

### 4.1 The seven design rules the ladder encodes

| # | Rule | Why it matters for calibration |
|---|---|---|
| R1 | **Hold the L-demand fixed; vary only the D-tier.** Same probe text, same session layout (`s1` seed → `s2` probe) in all three rungs. | Otherwise a D1-vs-D3 delta mixes extraction difficulty with memory difficulty and can never be attributed. |
| R2 | **Differentiate by *extraction surface*, not by *volume alone*.** D1 states the claim verbatim in 225 words; D2 embeds it in 434 words of narrative with 3 same-lexicon distractors; D3 requires a two-document inference across 1 104 words with 9 distractors. | Volume alone is the metric the suite currently mis-uses (`document_dilution_01` is D2-hard at 354 words). |
| R3 | **Distractors must be *lexically adjacent*, not semantically distant.** Apollo keeps `eu-central-1` (the trap value), Kepler uses `us-east-2` (a trap value for the wrong entity), Orion keeps a sibling freeze window. | A distant distractor is filtered by any embedding model and measures nothing (see §3.A3). |
| R4 | **Ground truth must be provably unique.** Every distractor is scoped by service name and the precedence rule (bulletin beats handbook) is printed in the handbook itself; the in-context twin proves solvability. | Prevents "D3 = impossible" from being read as "D3 = hard". |
| R5 | **Each turn declares only *new* information; no probe restates a premise.** `target_claims[].tier_of_claim` marks which claim each tier is supposed to stress, and `assertion_turn` binds it to the probe that scores it. | Keeps the D-axis measured per claim and keeps trajectory clean (§3.D). |
| R6 | **Intrinsic and extrinsic assertions are separated by construction.** `assertions:` scores the chat output (extrinsic, supported today); `metadata.intrinsic_expectations.bindings` declares the store-level gold (`current` / `absent` / `preserved`) that refactor step H2 turns into `must_store` / `must_not_store` rules. | Fixes §3.B4 and §3.C6 without breaking the current schema. |
| R7 | **Every rung has an in-context twin** (`nimbus_region_in_context_01`) whose single prompt contains the rung's whole context, so `pass twin + fail rung` reads as *memory/retrieval failure* and `fail both` reads as *reasoning or miscalibration*. | Generalises the three existing paired controls to every rung of the ladder. |

**Field reference — the two new blocks.** `metadata.calibration` is free-form today (so it validates), and
each key is intended to become a typed field in step H1:

| Key | Type | Meaning |
|---|---|---|
| `tier` | `D1｜D2｜D3` | the extraction tier this file is calibrated to |
| `twin_ladder` / `in_context_control` | list / str | the sibling rungs and the ceiling twin |
| `context_word_target` / `context_words_actual` | `[lo, hi]` / int | band and measurement (seed turns + documents, probe turns excluded) |
| `probe_words_actual` | int | probe-side words, constant across the ladder |
| `distractor_claim_target` / `_actual` | `[lo, hi]` / int | count of *competing claims* (not turns) |
| `inference_required` / `inference_hop` | bool / str | whether the answer must be derived, and from which passages |
| `per_claim_tier` | map | per-claim tier, so one file can host a D3 claim beside D1-easy controls |
| `noise_ratio_estimate` | str | share of context words that are not the target claim |

### 4.2 D1 — Explicit / Verbatim (rung 1)

Proposed location: `datasets/conversations/multi_thread/difficulty_calibration/nimbus_region/test_nimbus_region_d1_01.yaml`
(not yet added to the tree — adding it changes the loader inventory from 33 to 36; see step L4)

```yaml
# ─────────────────────────────────────────────────────────────────
# test_nimbus_region_d1_01.yaml — CALIBRATED D1 (Explicit / Verbatim)
# -----------------------------------------------------------------
# Extraction-calibration ladder, rung 1 of 3. The SAME four decisions,
# the SAME four probe turns and the SAME session layout (s1 seed ->
# s2 probe) as the D2 and D3 twins. ONLY the surface form of the
# context changes, so the L0..L4 delta is held constant across the
# ladder and any score difference is attributable to extraction.
#
# Twin set:  d1 <-> d2 <-> d3   (this file is the ladder floor)
# ─────────────────────────────────────────────────────────────────

id: nimbus_region_d1_01
name: "Nimbus Deploy Region — D1 Explicit (Ladder Floor)"
description: >
  Four atomic Nimbus release-governance decisions are stated verbatim in a
  single 200-400 word context turn with zero distractor claims. The probe
  session (s2) asks for the revised region, a preserved constraint and a
  preserved owner. D1 measures whether the agent can hold and return an
  explicitly stated claim; any failure here is a memory-mechanism failure,
  never an extraction failure.
category: difficulty_calibration
metadata:
  difficulty: basic
  stage: "6.1"
  sessions: ["s1", "s2"]
  # ── Extraction calibration block (D1/D2/D3 axis, not the L0-L4 axis) ──
  calibration:
    tier: D1
    twin_ladder: ["nimbus_region_d1_01", "nimbus_region_d2_01", "nimbus_region_d3_01"]
    in_context_control: nimbus_region_in_context_01
    context_word_target: [200, 400]
    context_words_actual: 225
    context_words_definition: "seed-turn words + document words; excludes probe turns (40 words, identical across the ladder)"
    probe_words_actual: 40
    distractor_claim_target: 0
    distractor_claims_actual: 0
    documents: 0
    inference_required: false
    probes_identical_across_ladder: true
    expected_ceiling: "L0 may fail (cross-session); extraction itself is at ceiling for any instruction-following model."
  target_claims:
    - id: C1
      subject: nimbus
      predicate: deploy_region
      value: us-east-2
      supersedes: eu-central-1
      tier_of_claim: D1
    - id: C2
      subject: nimbus
      predicate: freeze_window
      value: "Friday 17:00-21:00 UTC"
      revised: false
      tier_of_claim: D1
    - id: C3
      subject: nimbus
      predicate: rollback_owner
      value: "Priya Raman"
      revised: false
      tier_of_claim: D1
  intrinsic_expectations:
    note: >
      Store-level gold. Enforced once the `must_store` rule family lands
      (refactor H2); today it is documented here and partially observable
      through the `must_retrieve_chunk` pin on ep_s1_t1.
    bindings:
      - {key: "nimbus.deploy_region", value: "us-east-2", state: current}
      - {key: "nimbus.deploy_region", value: "eu-central-1", state: absent}
      - {key: "nimbus.freeze_window", value: "Friday 17:00-21:00 UTC", state: preserved}
      - {key: "nimbus.rollback_owner", value: "Priya Raman", state: preserved}

turns:
  # ── s1: seed. New information only, stated once, never re-stated ──
  - turn_number: 1
    session_id: s1
    role: user
    content: |
      Release-governance review for the Nimbus service, recorded 2026-03-04.
      I am logging four decisions with you now so that we can work from them
      later without re-reading this note. Nothing outside these four items is
      in scope for this review. The review was chaired by the platform group
      and attended by the three service owners named in the minutes; this is
      the approved extract of those minutes, and the four decisions below are
      quoted from it without paraphrase. No item in this note is provisional,
      and no item is waiting on a later sign-off.

      Decision 1, deploy region. Nimbus deploys to us-east-2 as of this
      review. The previous region, eu-central-1, is retired for Nimbus and
      must not be used again.

      Decision 2, freeze window. No Nimbus deployment may start between
      Friday 17:00 UTC and Friday 21:00 UTC. This window is unchanged by
      this review.

      Decision 3, rollback ownership. Priya Raman owns all Nimbus rollbacks.
      This is unchanged by this review.

      Decision 4, approval path. A production Nimbus deploy needs one
      approval from the platform on-call engineer before it starts. This is
      recorded for completeness and is not revised.

      Please confirm that you have recorded all four decisions. Do not
      summarise them, do not add recommendations, and do not ask me for the
      missing context — there is none: the four items above are the whole
      record.
    assertions: []

  # ── s2: probes. Wording identical across the ladder; no premise echo ──
  - turn_number: 2
    session_id: s2
    role: user
    content: "New session. I am preparing the change ticket for the Nimbus redeploy. Which region does Nimbus deploy to today?"
    assertions:
      - type: must_contain
        value: "us-east-2"
      - type: must_not_contain
        value: "eu-central-1"
      - type: must_retrieve_chunk
        value: "ep_s1_t1"

  # Constraint inference: only asked here to keep the probe set identical
  # to the D2/D3 twins (where it is the discriminating turn).
  - turn_number: 3
    session_id: s2
    role: user
    content: "Can the Nimbus redeploy start on Friday at 19:00 UTC?"
    assertions:
      - type: must_contain
        value: "no"
      - type: must_not_contain
        value: "yes"

  # Preservation: a revision on C1 must not disturb C2/C3.
  - turn_number: 4
    session_id: s2
    role: user
    content: "And who owns Nimbus rollbacks, and what is the freeze window?"
    assertions:
      - type: must_contain
        value: "Priya Raman"
      - type: must_contain
        value: "17:00"
      - type: must_contain
        value: "21:00"
```



### 4.3 D2 — Contextual / Embedded (rung 2)

Proposed location: `datasets/conversations/multi_thread/difficulty_calibration/nimbus_region/test_nimbus_region_d2_01.yaml`

```yaml
# ─────────────────────────────────────────────────────────────────
# test_nimbus_region_d2_01.yaml — CALIBRATED D2 (Contextual / Embedded)
# -----------------------------------------------------------------
# Rung 2 of the extraction ladder. Identical probes, identical session
# layout, identical target claims as nimbus_region_d1_01 — the four
# decisions are now EMBEDDED in a 400-800 word release-notes narrative
# that also carries 3 unrelated distractor claims sharing the target's
# lexicon (region names, freeze windows, rollback owners) but bound to
# two sibling services (Apollo, Orion).
#
# Why the distractors are same-lexicon: a distractor that is semantically
# distant (a cooking recipe) is filtered by any embedding model and
# measures nothing. These force entity binding, which is the property D2
# is supposed to stress.
# ─────────────────────────────────────────────────────────────────

id: nimbus_region_d2_01
name: "Nimbus Deploy Region — D2 Embedded (Ladder Mid)"
description: >
  The same four Nimbus decisions, embedded in a release-notes narrative with
  three distractor claims bound to Apollo and Orion. D2 measures whether the
  agent separates the target claim from lexically adjacent claims in the same
  context; the probe wording is byte-identical to the D1 and D3 twins.
category: difficulty_calibration
metadata:
  difficulty: moderate
  stage: "6.2"
  sessions: ["s1", "s2"]
  calibration:
    tier: D2
    twin_ladder: ["nimbus_region_d1_01", "nimbus_region_d2_01", "nimbus_region_d3_01"]
    in_context_control: nimbus_region_in_context_01
    context_word_target: [400, 800]
    context_words_actual: 434
    context_words_definition: "seed-turn words + document words; excludes probe turns (40 words, identical across the ladder)"
    probe_words_actual: 40
    distractor_claim_target: [2, 3]
    distractor_claims_actual: 3
    documents: 0
    inference_required: false
    probes_identical_across_ladder: true
    expected_ceiling: "L1+ should reach it; a failure here with a D1 pass isolates entity binding, not retrieval."
  distractor_claims:
    - {id: X1, subject: apollo, predicate: deploy_region, value: eu-central-1, conflicts_with_target: false}
    - {id: X2, subject: apollo, predicate: rollback_owner, value: "Tomas Iversen", conflicts_with_target: false}
    - {id: X3, subject: orion, predicate: freeze_window, value: "Saturday 02:00-04:00 UTC", conflicts_with_target: false}
  target_claims:
    - {id: C1, subject: nimbus, predicate: deploy_region, value: us-east-2, supersedes: eu-central-1, tier_of_claim: D2}
    - {id: C2, subject: nimbus, predicate: freeze_window, value: "Friday 17:00-21:00 UTC", revised: false, tier_of_claim: D2}
    - {id: C3, subject: nimbus, predicate: rollback_owner, value: "Priya Raman", revised: false, tier_of_claim: D2}
  intrinsic_expectations:
    note: "Same union as D1 plus one separation check: Apollo's eu-central-1 must not be bound to nimbus."
    bindings:
      - {key: "nimbus.deploy_region", value: "us-east-2", state: current}
      - {key: "nimbus.deploy_region", value: "eu-central-1", state: absent}
      - {key: "apollo.deploy_region", value: "eu-central-1", state: current}
      - {key: "nimbus.rollback_owner", value: "Priya Raman", state: preserved}
      - {key: "apollo.rollback_owner", value: "Tomas Iversen", state: preserved}

turns:
  - turn_number: 1
    session_id: s1
    role: user
    content: |
      Release-governance review notes, 2026-03-04, folded into the quarterly
      platform report. I am pasting the review section verbatim so you have
      the same picture the reviewers had; the decisions it records are the
      ones we will work from later, without re-reading this note.

      The quarter opened with the Frankfurt tenancy under review. The
      platform group had already flagged it in January, when the cost model
      for the eu-central-1 footprint came back 40 percent above the Ohio
      equivalent and the latency numbers for our traffic mix turned out to be
      a wash. After two rounds of discussion the group signed off on moving
      Nimbus, the customer-facing release service, off Frankfurt and onto
      us-east-2 as of this review. The previous region, eu-central-1, is
      retired for Nimbus and must not be used again. The migration itself ran
      during the March 3 maintenance window and produced no customer-visible
      downtime, which the report calls out as the reason the decision was
      taken as final rather than provisional.

      The same section records three items that did not change, and the review
      is explicit that they are unchanged so nobody re-litigates them in
      April. The freeze window for Nimbus remains Friday 17:00 UTC to Friday
      21:00 UTC: no Nimbus deployment may start inside it. Rollback ownership
      for Nimbus remains with Priya Raman, who has held it since the service
      launched. A production Nimbus deploy still needs one approval from the
      platform on-call engineer before it starts; the review only tightened
      the wording, not the rule.

      Two sibling services came up in the same session and are recorded here
      for completeness, because their entries share the vocabulary of the
      Nimbus entries and the report is easy to misread if it is skimmed.
      Apollo, the billing pipeline, continues to deploy to eu-central-1 and
      has no migration planned this half; its Frankfurt latency profile is
      what the cost discussion was actually comparing against. Apollo
      rollbacks stay with Tomas Iversen, who took them over when the billing
      team was split. Orion, the batch reconciliation job, keeps its own
      freeze window, Saturday 02:00 UTC to 04:00 UTC, and its region was not
      discussed at all in this review.

      The reviewers closed the section with a note that the distinction
      between the three services matters more than the individual values,
      since the shared vocabulary is the main source of confusion in the
      ticket history. Please confirm that you have recorded the Nimbus
      decisions. Do not summarise them and do not ask me for the missing
      context: the four Nimbus items above are the whole record for Nimbus,
      and the Apollo and Orion lines are context, not Nimbus state.
    assertions: []

  - turn_number: 2
    session_id: s2
    role: user
    content: "New session. I am preparing the change ticket for the Nimbus redeploy. Which region does Nimbus deploy to today?"
    assertions:
      - type: must_contain
        value: "us-east-2"
      - type: must_not_contain
        value: "eu-central-1"
      - type: must_retrieve_chunk
        value: "ep_s1_t1"

  - turn_number: 3
    session_id: s2
    role: user
    content: "Can the Nimbus redeploy start on Friday at 19:00 UTC?"
    assertions:
      - type: must_contain
        value: "no"
      - type: must_not_contain
        value: "yes"

  - turn_number: 4
    session_id: s2
    role: user
    content: "And who owns Nimbus rollbacks, and what is the freeze window?"
    assertions:
      - type: must_contain
        value: "Priya Raman"
      - type: must_contain
        value: "17:00"
      - type: must_contain
        value: "21:00"
      # Context-blending guard: the sibling owner must not surface here.
      - type: must_not_contain
        value: "Tomas Iversen"
```



### 4.4 D3 — Inferential / Injected (rung 3)

Proposed location: `datasets/conversations/multi_thread/difficulty_calibration/nimbus_region/test_nimbus_region_d3_01.yaml`
(D3 also needs the two documents below under `datasets/documents/calibration/`, and `documents[].path`
resolved relative to the YAML)

```yaml
# ─────────────────────────────────────────────────────────────────
# test_nimbus_region_d3_01.yaml — CALIBRATED D3 (Inferential / Injected)
# -----------------------------------------------------------------
# Rung 3 of the extraction ladder. Identical probes, identical target
# claims as the D1 and D2 twins, but the context is now
#   (a) two real-looking background documents (handbook + vendor
#       bulletin, ~90% noise by word count),
#   (b) 9 distractor claims sharing the target's lexicon, including
#       the trap that eu-central-1 is the *correct* answer for Apollo,
#       so a lexical match is actively wrong, and
#   (c) one claim (C1) that is stated NOWHERE verbatim: the handbook
#       says Frankfurt, the later bulletin says the Frankfurt tenancy
#       was decommissioned and Nimbus now runs in Ohio. Deriving
#       "Nimbus deploys to us-east-2 today" needs the two statements
#       combined AND the authority/recency rule in handbook §9.
#
# Solvability contract: every claim is scoped by service name and the
# precedence rule is stated in the handbook, so the ground truth is
# unique. If the in-context twin (nimbus_region_in_context_01) is at
# ceiling and this is not, the gap is retrieval; if both fail, the
# ground truth is miscalibrated and the scenario must be rewritten.
# ─────────────────────────────────────────────────────────────────

id: nimbus_region_d3_01
name: "Nimbus Deploy Region — D3 Inferential (Ladder Ceiling)"
description: >
  The same four Nimbus decisions, now recoverable only from a 750-word
  operations handbook plus a 330-word vendor bulletin. The deploy region is a
  two-document inference: the handbook records Frankfurt, the later bulletin
  records that the Frankfurt tenancy was decommissioned and the Nimbus
  workloads migrated to Ohio. The freeze window and rollback owner are stated
  verbatim in the handbook and are annotated per-claim as D1-easy, so a run can
  separate "failed the inference" from "failed the whole document".
category: difficulty_calibration
metadata:
  difficulty: hard
  stage: "6.3"
  sessions: ["s1", "s2"]
  default_top_k: 5
  calibration:
    tier: D3
    twin_ladder: ["nimbus_region_d1_01", "nimbus_region_d2_01", "nimbus_region_d3_01"]
    in_context_control: nimbus_region_in_context_01
    context_word_target: [600, 1200]
    context_words_actual: 1104
    context_words_definition: "seed-turn words + document words; excludes probe turns (40 words, identical across the ladder)"
    probe_words_actual: 40
    distractor_claim_target: [6, 20]
    distractor_claims_actual: 9
    documents: 2
    inference_required: true
    inference_hop: "handbook.§2 (nimbus -> eu-central-1) + bulletin.Summary (Frankfurt decommissioned, Nimbus -> us-east-2) + handbook.§9 (bulletin wins)"
    probes_identical_across_ladder: true
    noise_ratio_estimate: "~0.93 of document words are not the target claim"
    expected_ceiling: "D1-easy claims (C2, C3) should survive at L2+; C1 is the tier discriminator."
  per_claim_tier:
    C1: D3
    C2: D1
    C3: D1
  distractor_claims:
    - {id: X1, subject: apollo, predicate: deploy_region, value: eu-central-1, note: "lexical trap: correct for Apollo, wrong for Nimbus"}
    - {id: X2, subject: apollo, predicate: rollback_owner, value: "Tomas Iversen"}
    - {id: X3, subject: apollo, predicate: freeze_window, value: "Sunday 00:00-06:00 UTC"}
    - {id: X4, subject: orion, predicate: deploy_region, value: "us-west-1"}
    - {id: X5, subject: orion, predicate: freeze_window, value: "Saturday 02:00-04:00 UTC"}
    - {id: X6, subject: orion, predicate: rollback_owner, value: "Daniel Okafor"}
    - {id: X7, subject: kepler, predicate: deploy_region, value: "us-east-2", note: "second trap: correct value for the wrong entity"}
    - {id: X8, subject: kepler, predicate: rollback_owner, value: "Ingrid Sollberg"}
    - {id: X9, subject: nimbus, predicate: capacity_ratio, value: "flagged twice", note: "same entity, different predicate"}
  target_claims:
    - {id: C1, subject: nimbus, predicate: deploy_region, value: us-east-2, supersedes: eu-central-1, tier_of_claim: D3, assertion_turn: 2}
    - {id: C2, subject: nimbus, predicate: freeze_window, value: "Friday 17:00-21:00 UTC", revised: false, tier_of_claim: D1, assertion_turn: 4}
    - {id: C3, subject: nimbus, predicate: rollback_owner, value: "Priya Raman", revised: false, tier_of_claim: D1, assertion_turn: 4}
  intrinsic_expectations:
    note: >
      Store-level gold. The stale binding must be superseded, not duplicated
      (false-update check), and no distractor binding may be written under the
      nimbus key (context-blending check).
    bindings:
      - {key: "nimbus.deploy_region", value: "us-east-2", state: current}
      - {key: "nimbus.deploy_region", value: "eu-central-1", state: absent}
      - {key: "apollo.deploy_region", value: "eu-central-1", state: current}
      - {key: "kepler.deploy_region", value: "us-east-2", state: current}
      - {key: "nimbus.rollback_owner", value: "Priya Raman", state: preserved}
      - {key: "nimbus.freeze_window", value: "Friday 17:00-21:00 UTC", state: preserved}

documents:
  - doc_id: ops_handbook_2025_11
    path: ops_handbook_2025_11.md
    metadata:
      title: "Platform Operations Handbook — Deployment Section"
      version: "2025.11"
      authority: handbook
      published: "2025-11-14"
  - doc_id: vendor_bulletin_2026_03_03
    path: vendor_bulletin_2026_03_03.md
    metadata:
      title: "Vendor Bulletin — Frankfurt tenancy decommission"
      version: "2026-03-03"
      authority: vendor_bulletin
      published: "2026-03-03"

turns:
  # A single ingest acknowledgement keeps the chat shape comparable to the
  # D1/D2 twins without restating any target claim in the chat channel.
  - turn_number: 1
    session_id: s1
    role: user
    content: "I have loaded the operations handbook and the latest vendor bulletin into your document memory. I will ask about them in a later session. Just confirm you have them."
    assertions: []

  - turn_number: 2
    session_id: s2
    role: user
    content: "New session. I am preparing the change ticket for the Nimbus redeploy. Which region does Nimbus deploy to today?"
    assertions:
      - type: must_contain
        value: "us-east-2"
      # Stale-value guard: the handbook value must not survive the revision.
      - type: must_not_contain
        value: "eu-central-1"
      # Intrinsic-adjacent: both documents had to be consulted.
      - type: must_retrieve_chunk
        value: "ops_handbook_2025_11"
      - type: must_retrieve_chunk
        value: "vendor_bulletin_2026_03_03"

  - turn_number: 3
    session_id: s2
    role: user
    content: "Can the Nimbus redeploy start on Friday at 19:00 UTC?"
    assertions:
      - type: must_contain
        value: "no"
      - type: must_not_contain
        value: "yes"

  - turn_number: 4
    session_id: s2
    role: user
    content: "And who owns Nimbus rollbacks, and what is the freeze window?"
    assertions:
      - type: must_contain
        value: "Priya Raman"
      - type: must_contain
        value: "17:00"
      - type: must_contain
        value: "21:00"
      # Context-blending guards: sibling values must not be attributed to Nimbus.
      - type: must_not_contain
        value: "Tomas Iversen"
      - type: must_not_contain
        value: "Daniel Okafor"
```

### 4.5 Supporting documents for D3 (a) — operations handbook

`datasets/conversations/multi_thread/difficulty_calibration/nimbus_region/ops_handbook_2025_11.md`

```markdown
# Platform Operations Handbook — Deployment Section (excerpt)

Version 2025.11 (superseded in part — see vendor bulletin 2026-03-03).

## 1. Scope

This section governs deployment of the services listed in the service registry
below. It is maintained by the Platform Operations group and reviewed twice a
year. Changes to a service's region, freeze window or rollback owner must be
recorded here before they take effect; an entry that is not in this handbook is
not a change, and a change that has not yet been folded into a handbook release
is tracked in the change log.

## 2. Service registry

The registry is the authoritative list of deployable services. Each entry
records the tenancy the service deploys to, the owning group and any standing
exceptions. The registry is deliberately terse: it is copied into tickets, so
it lists values without rationale.

- nimbus — customer-facing release service. Tenancy: eu-central-1 (Frankfurt).
  Owning group: Platform Operations. Standing exceptions: none.
- apollo — billing pipeline. Tenancy: eu-central-1 (Frankfurt). Owning group:
  Billing Engineering. Standing exceptions: regional data residency review
  pending since 2025-09, no action required from platform.
- orion — batch reconciliation job. Tenancy: us-west-1 (Oregon). Owning group:
  Data Platform. Standing exceptions: batch-only, exempt from interactive
  latency SLOs.
- kepler — internal metrics relabeler. Tenancy: us-east-2 (Ohio). Owning group:
  Observability. Standing exceptions: none.

## 3. Change windows and freeze windows

Every service has a freeze window inside which no deployment may start. The
window is expressed in UTC and applies to the start of the deploy only; a
deploy that began before the window opened is allowed to finish. Emergency
rollbacks are exempt from freeze windows but are recorded in the incident log
regardless.

- nimbus — Friday 17:00 UTC to Friday 21:00 UTC. Introduced after the 2025
  holiday incident, when a Friday evening deploy was followed by a two-hour
  recovery. The window has not moved since.
- apollo — Sunday 00:00 UTC to Sunday 06:00 UTC, aligned with the billing
  reconciliation run.
- orion — Saturday 02:00 UTC to Saturday 04:00 UTC, aligned with the weekly
  batch.
- kepler — no freeze window; the service carries no customer traffic.

## 4. Rollback ownership

A rollback owner is a single named engineer accountable for reverting a bad
deployment. Ownership follows the service, not the tenancy, so a service that
changes region keeps its owner. Ownership is recorded per service and is the
only accepted authority during a rollback: an approval from anyone else is
recorded but does not authorise the action.

- nimbus — Priya Raman (since service launch).
- apollo — Tomas Iversen (transferred when the billing team was split).
- orion — Daniel Okafor.
- kepler — Ingrid Sollberg.

## 5. Approval paths

A production deploy requires one approval from the platform on-call engineer
before it starts. Approval is recorded in the change ticket and expires after
four hours. Non-production deploys require no approval. This rule is uniform
across services and has no exceptions; the previous per-service override was
retired with the 2025.06 release of this handbook.

## 6. Escalation

Escalation follows the on-call rotation, which is published weekly. The first
escalation contact is the platform on-call engineer, the second is the duty
manager, the third is the director of engineering. Persistent customer-visible
failures trigger the incident communications template rather than the generic
escalation path. Escalation contacts are not recorded in this handbook because
the rotation changes weekly and a stale contact is worse than no contact.

## 7. Secrets and identifiers

Service credentials rotate every 90 days. Rotation is scheduled by the
security group and announced two weeks in advance. Identifiers printed in this
handbook are internal and must not be copied into customer-facing material.
The secrets rotation runbook lives in the security group's space and is not
reproduced here.

## 8. Capacity review

Capacity is reviewed quarterly against the peak-to-average ratio of the
preceding quarter. A service that exceeds a ratio of four is flagged for a
dedicated review. Nimbus has been flagged twice in the last three quarters,
which is one of the reasons its tenancy has been contested; the capacity
review itself does not change a tenancy, it only opens the discussion.

## 9. Known gaps

This handbook does not record tenancy migrations that are still in flight. When
a migration is agreed but not yet completed, the registry keeps the source
tenancy and the change log records the destination. The tenancy in the registry
is therefore correct as of this release and may be wrong by the time it is
read. Where a vendor bulletin contradicts this section, the bulletin wins:
bulletins are issued after the fact and carry the vendor's confirmation that
the migration completed.
```

### 4.5 Supporting documents for D3 (b) — vendor bulletin

`datasets/conversations/multi_thread/difficulty_calibration/nimbus_region/vendor_bulletin_2026_03_03.md`

```markdown
# Vendor Bulletin — Frankfurt tenancy decommission (2026-03-03)

To all tenants of the Frankfurt (eu-central-1) managed tenancy.

## Summary

The Frankfurt managed tenancy reached end of life on 2026-02-28. All Nimbus
workloads hosted in that tenancy were migrated by the vendor during the
maintenance window of 2026-03-03 and are now running in the Ohio
(us-east-2) tenancy. No further deployments to the Frankfurt tenancy will be
accepted: the API returns 410 Gone for any Nimbus pipeline targeting it.

## What changed for you

- Workbook exports: unchanged, the export endpoint is region-agnostic.
- Pipeline IDs: preserved across the migration, so existing change tickets
  referencing a Nimbus pipeline remain valid.
- DNS: the CNAME target changed; internal resolvers were updated during the
  same window and no action is required.
- Latency: median latency for the traffic mix observed during the pilot was
  not materially different from Frankfurt, so no configuration change is
  recommended.

## What did not change

The migration was tenancy-only. Service-level controls — approval paths,
freeze windows and rollback ownership — are governed by your own handbook and
were not touched by this work. If your handbook records a freeze window or a
rollback owner for a service, that record remains correct.

Apollo is out of scope for this bulletin. It runs on a dedicated Frankfurt
tenancy that is not part of the managed offering and is not affected by the
decommission, so any Apollo entry in your registry continues to apply as
written.

Orion was not hosted in Frankfurt and is not mentioned in the migration
manifest.

## Effective dates

The migration completed on 2026-03-03 at 04:12 UTC. This bulletin supersedes
the tenancy statements in handbook releases published before that date, and
only the tenancy statements: every other section of the handbook remains in
force.
```



### 4.6 The in-context twin (mandatory companion, generated — do not hand-write)

`nimbus_region_in_context_01` is a **single user turn** built by concatenating the rung's whole context
plus the three probes:

```yaml
id: nimbus_region_in_context_01
name: "Nimbus Deploy Region — In-Context Ceiling Control (all rungs)"
category: in_context_control
metadata:
  difficulty: basic
  stage: "6.4"
  calibration:
    tier: CONTROL
    twins: [nimbus_region_d1_01, nimbus_region_d2_01, nimbus_region_d3_01]
    construction: >
      turn 1 content = join(scenario.turns[0].content for the rung) + the three probe
      strings, verbatim. Regenerate with:
        python scripts/build_calibration_twin.py --ladder nimbus_region --rung d3
    interpretation: >
      pass twin + fail rung  -> retrieval/extraction failure (memory mechanisms are implicated)
      fail twin + fail rung  -> reasoning failure or miscalibrated ground truth (rewrite the rung)
      pass twin + pass rung  -> the rung is below the model's ceiling; raise the tier
turns:
  - turn_number: 1
    role: user
    content: |
      <CONCATENATION OF THE RUNG'S CONTEXT, VERBATIM>

      Based strictly on the material above, answer these three questions:
      (1) Which region does Nimbus deploy to today?
      (2) Can the Nimbus redeploy start on Friday at 19:00 UTC?
      (3) Who owns Nimbus rollbacks, and what is the freeze window?
    assertions:
      - {type: must_contain,      value: "us-east-2"}
      - {type: must_not_contain,  value: "eu-central-1"}
      - {type: must_contain,      value: "no"}
      - {type: must_contain,      value: "Priya Raman"}
```

The twin is what makes a D-tier **solvable by construction**: if the strongest model fails it, the
ground truth is wrong and the rung must be rewritten rather than reported as "hard".

### 4.7 Verification of this template (executed)

The three ladder files and both documents were loaded through the real schema and measured:

```
$ python validate_template.py
nimbus_region_d1_01  tier=D1  ctx_words=265  target=200-400   declared=225   in_band=True  probes=3  docs=0  pins=1
nimbus_region_d2_01  tier=D2  ctx_words=474  target=400-800   declared=434   in_band=True  probes=3  docs=0  pins=1
nimbus_region_d3_01  tier=D3  ctx_words=1144 target=600-1200  declared=1104  in_band=True  probes=3  docs=2  pins=2

ladder probe text identical across D1/D2/D3: True

seed-vs-probe split (context_words_actual must equal context)
nimbus_region_d1_01  seed_chat= 225  docs=   0  context= 225  probe_turns= 40  declared_ok=True
nimbus_region_d2_01  seed_chat= 434  docs=   0  context= 434  probe_turns= 40  declared_ok=True
nimbus_region_d3_01  seed_chat=  29  docs=1075  context=1104  probe_turns= 40  declared_ok=True

ALL TEMPLATE CHECKS PASS: True
```

Checks applied: (1) `BenchmarkScenario.model_validate` accepts every file; (2) `context_words_actual`
equals the measured seed+document word count; (3) the context sits inside its declared band;
(4) `probe_words_actual` is constant across the ladder; (5) the three probe turns are byte-identical;
(6) R10 hygiene — no `must_contain` value appears in its own probe turn; (7) `declared_retrieval_assertions`
sees the pins (1 / 1 / 2).

### 4.8 Schema patch required to make the intrinsic half real (step H2)

The template is runnable today, but the store-level expectations stay in `metadata` until
`core/schemas.py` grows four small additions (all backward compatible):

```python
# core/schemas.py — additions, not replacements
Tier = Literal["D1", "D2", "D3"]

class AssertionRule(BaseModel):
    type: Literal[
        "must_contain", "must_not_contain",
        "must_retrieve_chunk", "must_not_retrieve_chunk",
        "must_retrieve_source", "must_not_retrieve_source",
        "must_store", "must_not_store",            # NEW: intrinsic, reads the store snapshot
    ]
    value: str
    case_sensitive: bool = False
    scope: Literal["extrinsic", "intrinsic"] = "extrinsic"   # NEW: keeps the two families apart
    tier: Tier | None = None                                 # NEW: per-rule tier for aggregation

class ConversationTurn(BaseModel):
    doc_claims: list[str] = Field(default_factory=list)      # NEW: claims introduced THIS turn
    model_config = ConfigDict(extra="forbid")                # NEW: stop silently dropping keys

class BenchmarkScenario(BaseModel):
    tier: Tier | None = None                                 # NEW
    expected_state: list[dict[str, Any]] = Field(default_factory=list)  # NEW: cumulative gold
    model_config = ConfigDict(extra="forbid")
```

`must_store` is evaluated against the exported `memory/<run_id>/` snapshot (already written by
`export_snapshot`), with `value` matching a `key=value` pair declared in `expected_state`. That is the
only change in this report that touches the metric layer rather than the data layer, and it is what
turns "false update" and "context blending" from prose into measurable properties.

---


## 5. Actionable Refactoring Steps

Prioritised by impact on the thesis' ability to claim D-tier calibration. Each step names the files it
touches and its acceptance test.

### High impact — do first (≈1.5 days; unblocks every D-tier claim)

| # | Step | Files | Acceptance test |
|---|---|---|---|
| **H1** | **Add typed tier + gold fields, and stop silently dropping keys.** `Tier = Literal["D1","D2","D3"]`; `BenchmarkScenario.tier`, `.expected_state`; `ConversationTurn.doc_claims`; `AssertionRule.scope` + `.tier`; `model_config = ConfigDict(extra="forbid")` on all three. | `core/schemas.py` | `tests/test_schemas_tier.py`: `tier: D4` raises; an unknown top-level key raises; existing scenarios still load unchanged. |
| **H2** | **Implement intrinsic assertions.** `must_store` / `must_not_store` evaluated against the exported snapshot, wired into `evaluate_assertions` behind `scope: intrinsic`, exported as `intrinsic_pass_rate` **separately** from `pass_rate`. | `core/runner.py`, `core/schemas.py`, `core/metrics.py` | `tests/test_store_assertions.py`: a run that answers correctly but duplicates a stale binding fails `must_not_store` while passing `must_contain`. |
| **H3** | **Enforce the word bands with a ratchet.** A test reads every scenario's calibration block (or a `CALIBRATION_BANDS` table keyed by id) and asserts the band bounds, the distractor-count range, and a constant `probe_words` across `twin_ladder`. Report-only first, hard-fail once the 33 files are annotated. | `tests/test_difficulty_calibration.py` (new) | the test fails on a scenario whose seed turn is 15 words but which declares `tier: D2`. |
| **H4** | **Split the score into three columns.** `recall_score` (`must_contain`), `guard_score` (`must_not_contain`), `retrieval_recall` + in-run `hit_at_1` / `mrr` from `retrieved_memories[].rank`. Keep `pass_rate` for compatibility but stop citing it as the headline. | `core/runner.py`, `core/metrics.py`, `core/schemas.py`, `_export_results` | `summary.csv` gains three columns; `summary.md` shows them side by side; SPEC §5.8 updated. |
| **H5** | **Fix the two amnesia-rewarding scenarios.** Add a positive "preserved facts survive" probe to `adversarial_forget_01/02` and a positive recall probe to `negative_hallucination_01`. | the three YAMLs | `test_no_scenario_is_passed_by_an_amnesic_agent`: ≥1 positive rule per scenario and `L0 pass_rate < 0.5` wherever `expected_level_0_behavior` says fail. |

### Medium impact — same week (≈1.5 days; makes the matrix a matrix)

| # | Step | Files | Acceptance test |
|---|---|---|---|
| **M1** | **Add `must_not_retrieve_*` precision pins** to the 10 scenarios that already carry positive pins, so distractor overlap becomes a metric rather than a text guard. | 10 YAMLs | a `retrieval_precision` column exists; `document_dilution_01` proves the room-7/8 decoys are *not* fetched. |
| **M2** | **Rename or remove `metadata.difficulty`.** It conflates reasoning with extraction (§3.A1). Replace with `tier` (H1), `reasoning_depth: 1..3` and `distractor_claims`. | all 33 YAMLs | no file carries `difficulty`; aggregates group by `tier`. |
| **M3** | **Turn the trivia-turn "distractors" into claim distractors** in `attrition_01` and `cross_thread_01`: embed 2–3 same-entity alternative values inside the seed turn instead of appending unrelated Q&A turns. | 2 YAMLs | `distractor_claims_actual >= 2` and context inside 400–800 words. |
| **M4** | **Generate a realistic D3 corpus.** Replace the ~14-template corpus (unique ratio 0.022–0.09) with paraphrased multi-source text (3 handbook versions + 2 vendor bulletins + release notes) targeting unique ratio 0.6–0.9, and make the needle answerable only from ≥2 sentences. | `scripts/generate_corpus.py`, `datasets/documents/d16_*` | `test_corpus_diversity`: unique ratio in [0.6, 0.95]; needle needs ≥2 sentences. |
| **M5** | **Add `near_duplicate_competition`** (max pairwise cosine between gold chunk and hardest distractor + shared-phrase count) to every document scenario. | 10 YAMLs + helper in `core/documents.py` | value present and reproducible from the manifest. |
| **M6** | **Close the menu-echo loophole**: for forced-label probes, validate the format in the runner (`metadata.format_constraint`) or add `must_not_contain` guards for the two rejected labels. | `backward_modus_tollens_01`, `in_context_modus_tollens_01` | an answer listing all three labels fails. |
| **M7** | **Kill the short-token assertions**: replace `3`, `0`, `O`, `6`, `7`, `9`, `24`, `Au` with full-claim assertions (`"3 years old"`, `"0 degrees"`, `"the symbol O"`) and add a lint rejecting `len(value) <= 2` outside an allow-list. | 3 YAMLs + `tests/test_assertion_hygiene.py` | dataset-wide lint modelled on the existing `test_scaffolding.py` hygiene tests. |
| **M8** | **Make the lexical baseline part of every corpus definition.** A dependency-free BM25 over the same chunking must *not* solve a cell that claims D2/D3: `hit@5 <= 0.6`. If BM25 already solves it, the cell is auto-downgraded to D1 (lookup) and reported as a plumbing control. Add `metadata.corpus_difficulty.lexical_baseline_hit_at_5` so the number travels with the data. | `tests/test_corpus_difficulty.py` (new), `scripts/generate_corpus.py`, 10 YAMLs | `document_corpus_size_*` reports `lexical_hit_at_5 = 1.00` and is therefore re-typed as a control; every future D3 cell reports ≤ 0.60. |
| **M9** | **Upgrade the corpus generator's claim grammar** (see the companion note `CORPUS_VALIDITY.md`): `--claims-per-doc`, `--competing-siblings K`, `--superseded-rate`, `--paraphrase-pool`, `--doc-kind`, `--needle-depth-quantile`, `--probe-style {entity_id,semantic}`. Target claim density ≥ 5 % of chunks and ≥ 2 competing values per gold predicate at D2, ≥ 5 at D3. | `scripts/generate_corpus.py` | the claim-density and competition tests in M8 pass on a regenerated D2/D3 corpus. |



### Low impact — polish (≈0.5 day)

| # | Step | Files | Acceptance test |
|---|---|---|---|
| **L1** | **Generalise the R10 no-echo test to the whole dataset** (today it covers only `document_*`). | `tests/test_assertion_hygiene.py` | one test over all 33 scenarios, with an explicit allow-list for in-context controls. |
| **L2** | **Rename the SPEC §5.8 `D1/D2/D3` policy tags** to `P1/P2/P3` so the collision with FEB-30 tiers disappears. | `SPEC.md`, `core/metrics.py` docstrings | no `D`-token left in the metrics contract. |
| **L3** | **Treat `prompt_id` as a control:** `strict_epistemic` *instructs* revision, so it must not be mixed with `baseline_react` when claiming memory effects (§3.C7.1). | `scripts/lr_report.py`, `README.md` | reports group by `(scenario, model, prompt)` or warn. |
| **L4** | **Document the new fields** in `README.md` and `datasets/documents/README.md`, and add the ladder to the scenario tables. | 2 READMEs | `--list-scenarios` count in docs updated (33 → 36 when the ladder lands). |

**Execution order.** H1 → H3 (cheap, high signal: the dataset becomes self-describing) → H4 (the metric
split that makes every later comparison meaningful) → H2 (intrinsic assertions: the real thesis
instrument) → H5 → M1/M6/M7 (hygiene) → M3/M4/M5 (content realism) → L*.

**Estimated effect.** H1+H3+H4 alone raise the calibration score from **3.5 to ≈6/10** on dimensions A
and C with no new scenario content, because existing signal stops being discarded. H2 unlocks the
L3/L4 claims; M4 unlocks any D3 claim.

---

## Appendix A — Per-scenario assertion inventory (evidence for §2/§3)

Positive / negative / retrieval-rule counts and context volume, 232 rules total:

| Scenario | `must_contain` | `must_not_contain` | retrieval pins | context words (chat + docs) |
|---|---:|---:|---:|---:|
| `belief_revision_01` | 5 | 2 | 0 | 77 |
| `explicit_forget_01` | 3 | 2 | 0 | 74 |
| `negative_hallucination_01` | 1 | 1 | 0 | 73 |
| `adversarial_forget_01` | 0 | 2 | 0 | 90 |
| `adversarial_forget_02` | 0 | 2 | 0 | 187 |
| `attrition_01` | 9 | 0 | 0 | 119 |
| `needle_haystack_01` | 10 | 0 | 0 | 162 |
| `in_context_needle_01` | 1 | 0 | 0 | 78 |
| `in_context_control_01` | 2 | 0 | 0 | 59 |
| `in_context_multihop_01` | 1 | 0 | 0 | 47 |
| `in_context_derivation_01` | 1 | 0 | 0 | 143 |
| `in_context_modus_tollens_01` | 1 | 0 | 0 | 167 |
| `in_context_derived_revision_01` | 1 | 1 | 0 | 147 |
| `temporal_multi_hop_01` | 2 | 0 | 0 | 59 |
| `3hop_temporal_01` | 1 | 0 | 0 | 163 |
| `overhead_baseline_01` | 10 | 0 | 0 | 204 |
| `cross_thread_01` | 5 | 0 | 0 | 198 |
| `cross_thread_derived_01` | 1 | 2 | 0 | 256 |
| `backward_modus_tollens_01` | 2 | 0 | 0 | 308 |
| `derived_belief_revision_01` | 2 | 1 | 0 | 284 |
| `entity_binding_01` | 4 | 12 | 8 | 323 |
| `top_k_saturation_01` | 1 | 9 | 0 | 376 |
| `memory_injection_01` | 4 | 6 | 0 | 299 |
| `document_addendum_revision_01` | 4 | 0 | 4 | 517 |
| `document_cross_thread_01` | 3 | 2 | 0 | 579 |
| `document_needle_01` | 3 | 1 | 2 | 1 037 |
| `document_dilution_01` | 4 | 1 | 4 | 420 |
| `document_authority_vs_recency_01` | 5 | 2 | 5 | 52 064 |
| `document_effective_dates_01` | 4 | 6 | 4 | 51 988 |
| `document_prevalence_update_vs_distractor_01` | 7 | 3 | 5 | 52 091 |
| `document_corpus_size_008_01` | 8 | 0 | 8 | 6 971 |
| `document_corpus_size_060_01` | 8 | 0 | 8 | 51 993 |
| `document_corpus_size_200_01` | 8 | 0 | 8 | 172 989 |
| **Total** | **121** | **55** | **56** | — |

## Appendix B — Empirical level behaviour (from `outputs/runs/*/results.json`)

Only cells with L0 or L1 records; `n` = recorded runs.

| Scenario | L0 mean (n) | L1 mean (n) | Reading |
|---|---:|---:|---|
| `adversarial_forget_01` | **1.000 (3)** | 0.583 (6) | amnesia wins |
| `adversarial_forget_02` | **1.000 (3)** | 0.929 (7) | amnesia wins |
| `negative_hallucination_01` | **1.000 (3)** | 0.719 (8) | amnesia wins |
| `needle_haystack_01` | 0.000 (3) | 1.000 (5) | persistence test, not attention |
| `attrition_01` | 0.667 (3) | 1.000 (5) | correct discrimination |
| `belief_revision_01` | 0.286 (4) | 1.000 (7) | correct discrimination |
| `explicit_forget_01` | 0.400 (3) | 0.800 (6) | correct discrimination |
| `temporal_multi_hop_01` | 0.500 (3) | 1.000 (5) | correct discrimination |
| `in_context_control_01` | 1.000 (4) | 1.000 (5) | ceiling control (expected) |
| `in_context_needle_01` | 1.000 (3) | 1.000 (5) | ceiling control (expected) |
| `in_context_multihop_01` | 0.333 (3) | 1.000 (5) | a 47-word single-turn prompt failed 2/3 L0 runs — noise floor |
| `document_effective_dates_01` | 0.429 (1) | 0.429 (1) | identical score ⇒ zero information about L |

## Appendix C — Files inspected

`core/schemas.py` (AssertionRule, DocumentSpec, ConversationTurn, BenchmarkScenario, TurnResult,
BenchmarkRunSummary) · `core/runner.py` (`_normalize_for_matching`, `_evaluate_retrieval_assertion`,
`evaluate_assertions`, `_system_event_specs`, `_ingest_documents`, `run_single` turn loop and
aggregation, snapshot export) · `core/metrics.py` · `core/dataset_layout.py` · `core/documents.py` ·
`agents/level_0_reactive/graph.py` · `agents/level_1_working_memory/graph.py` ·
`agents/level_2_episodic_rag/graph.py` (`retrieve_node`, `exclude_session_id`) · `configs/models.yaml` ·
`configs/prompts.yaml` · all 33 `datasets/conversations/**/test_*.yaml` · `datasets/documents/**`
(manifests, corpus sample, hand-written docs) · `tests/**` (hygiene, stage 3/4/5, document ingestion,
retrieval recall, dataset layout) · `outputs/runs/**` · `SPEC.md` · `README.md` · `AUDIT_REPORT.md`
(prior code audit: its §3.5 and §3.6 overlap with C1/C7 here and remain open).



## Appendix D — Pre-existing test drift found while validating this audit

`python -m pytest -q` on the audited tree: **585 passed, 1 failed** (16.7 s, offline). The failure is
pre-existing and unrelated to this report (no dataset file was modified):

```
tests/test_cost_reporting.py::TestGetModelPricing::test_rebuilt_registry_prices_every_model
    assert len(registry) == 10
    AssertionError: assert 11 == 10        # configs/models.yaml now ships 11 chat models (mimo-v2.6-pro added)
```

It is the same brittleness class the prior `AUDIT_REPORT.md` §3.12 flagged ("brittle tests coupled to
config values"). Two consequences worth carrying into H3/M2:

1. The suite has **cardinality assertions where they are cheap** (model count, document count = 60,
   turns = 8) and **none where they would carry scientific meaning** (context word bands, distractor
   counts, per-tier aggregates). The ratchet in H3 simply moves that existing habit onto the axis that
   matters.
2. `python run.py --list-scenarios` reports **33** scenarios and exits 0 — the loader agrees with this
   audit's inventory.

## Appendix E — Verdict in one paragraph

The harness is a well-engineered *memory* ablation rig (5 levels, per-turn telemetry, causal retrieval
pins, honest metadata, paired controls) that has not yet been calibrated as an *extraction* benchmark:
there is no tier field, the tier word bands are empty for chat turns, D3 exists only as templated-corpus
retrieval, 76 % of probes bypass retrieval entirely, no store-level assertion exists, and a single pooled
substring `pass_rate` lets an amnesic agent score 1.000 on two `hard` scenarios. The fix is mostly
additive — typed tier/gold fields, a band ratchet, a three-way metric split, `must_store` assertions and
one realistic D3 corpus — and the calibrated ladder in §4 is a working, schema-validated starting point.

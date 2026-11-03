# 🚧 Completion Plan: Phase 2 — Closing L2 (Evidence Gaps + Ablations)

> **Status:** PROPOSED (2026-10-30) — derived from the 2026-10-30 supervision review; companion to
> [`notes/2026-10-30_l2_episodic_rag_summary.md`](../notes/2026-10-30_l2_episodic_rag_summary.md).
> **Scope:** close the evidence gaps that (a) prevent declaring L2 complete and (b) leave the
> **L2 → L3** step unjustified. Does **not** implement L3.
> **Explicitly out of scope:** `MarkdownMemoryBackend` + BM25 (§5, decision required), giant
> conversation sweeps (§5), L3 implementation.
> **Coding Agent Instructions:** wait for explicit user instructions before implementing individual
> workstreams. WS-B and WS-E (Hygiene) need **no code** and can be executed first.

---

## 0. Why this plan exists — the framing correction

There are two ways to justify L3, and only one of them survives review:

| Argument | Survives "just use better RAG"? | Survives "a better model"? | Verdict |
|---|:--:|:--:|---|
| Retrieval dilution / chunk collision / `top_k` cutoff | Yes | Yes | ❌ Not a valid L3 justification |
| Entity-binding fragmentation | Yes (more `top_k`) | Partially | ❌ Not a valid L3 justification |
| Backward contraction (derived-belief MT) | Partially | Probably | ❌ Belongs to L3→L4, see 1.7 |
| **Selective forgetting (`DELETE`)** | **No — better retrieval makes it worse** | No | ✅ **L3 justification** |
| **Atomic revision / dedup (`UPDATE`)** | No | No | ✅ **L3 justification** |
| **Belief enumeration (state ≠ transcript)** | No | No | ✅ **L3 justification** |

**Decision recorded:** the L2 → L3 bridge rests on the last three rows, which are exactly L3's
declared target capability in [`phase-2-l2-plan.md`](phase-2-l2-plan.md) §1 — *"Atomic belief
revision & mechanism-backed selective forgetting"*. Retrieval findings are reported as **L2
characterisation**, not as the L3 justification.

**Corollary (why forgetting is the strongest card):** raising `top_k` increases recall of
*everything*, including revoked secrets. Better retrieval therefore makes non-disclosure
**strictly harder**. No retrieval configuration can produce deletion; only `DELETE` can. This
inverts the usual objection.

---

## 1. Findings that motivate the plan (verified from `outputs/runs/*/results.json`)

| # | Finding | Evidence |
|--:|---|---|
| 1.1 | 19 L2 runs; **14 valid**, 5 discarded for infrastructure (2× HTTP 429 `billing_restriction`, 3× Ollama `ConnectionError`). **96.2% (76/79)** over 114 turns. | per-run `results.json`; `outputs/consolidated_summary.md` |
| 1.2 | **5 of 14 valid runs never touch the store** (`retrieved_total = 0`): all turns share `session_id: default` and `retrieve_node` passes `exclude_session_id=session_id`. They measure L1 + overhead. | `graph.py:171-174`; `vector.py:310-312`; `schemas.py:110` |
| 1.3 | **All 14 runs used the fallback embedder** `qwen3-embedding-0.6b` (Ollama, local CPU, 1024 dims). The primary `qwen3-embedding-8b` (4096 dims) was attempted twice and failed with HTTP 429. | `models.yaml:150-172`; runs `20261023_134441`, `20261023_153323` |
| 1.4 | **13 of 14 runs used `deepseek-v4-flash`** (`tier: efficient`). No `frontier` model has ever run L2. | `models.yaml:67-74` |
| 1.5 | **Chat models have no `temperature` pinned**; only `judge` declares `temperature: 0`. Every run is n=1 under provider-default sampling. | `models.yaml:122-124` |
| 1.6 | **`top_k = 5` in all 14 runs, and it covers 38–62% of the entire store.** Retrieval has never been selective. | measured per run (8–13 chunks); `graph.py:61` |
| 1.7 | **Backward contraction failure** (`backward_modus_tollens_01`, 50%): 5 chunks retrieved, the Compliance contradiction present at **rank 4**, model still answered `YES_EU_CENTRAL`. **But:** per the ablation ladder this is L3's *expected failure mode* and **L4's target capability**. | run `20261026_125446`; `phase-2-l2-plan.md:16-17` |
| 1.8 | **Counter-evidence to "L2 cannot reason":** `derived_belief_revision_01` scored 100% (re-derived `Level-1` with the stale `Level-3` cache in context). The LLM *does* re-derive when retrieval is correct. | run `20261026_121129` |
| 1.9 | **`explicit_forget_01` is single-session → 0 retrieval.** The revoked PIN `9173` is stored in 4 of 5 chunks (`ep_default_t1..t4`) with no delete primitive, but the test never retrieves, so the append-only failure is **asserted by design, never measured**. | run `20261023_155026`; its `memory/.../episodes.json` |
| 1.10 | **No L1 baseline for `deepseek-v4-flash`** on `explicit_forget_01`, so the 80% is not comparable to any same-model L1 run. | `outputs/consolidated_summary.md` |
| 1.11 | **Chunking never fired:** `chunk_size=1200`, `chunk_overlap=200`; max observed chunk = **1155 chars**; **0 splits in 114 episodes**; no `_0`/`_1` chunk-id suffixes. `chunk_overlap` was never applied. | `vector.py:138-139`, `:207`; all `episodes.json` |
| 1.12 | **`chunk_size`/`chunk_overlap` are `null` in all 14 `results.json`** — `run.py` omits them from `memory_backend_config`; the real values live only in `manifest.json`. | `run.py` (`memory_backend_config`); `runner.py:766` |
| 1.13 | **Provenance header leaks the binding:** `_format_memory_block` injects `session=` and `turn=`. In `entity_binding_01` T11 the **rank-1 chunk was wrong** (7443) yet the model answered correctly (7080) by using the metadata. | `graph.py:90-91`; run `20261027_131450` |
| 1.14 | **Two-step probe contaminates the graded turn:** T9 (s3, 0 assertions) answered *"srv-alpha … port 7443"*; T10 (s3) then has it in the checkpointer history via `payload.extend(messages[:-1])`. **T10 measures L1, not L2.** | `graph.py` (`model_node`); run `20261027_131450` |
| 1.15 | **The multi-entity probe already exists and already fails, ungraded:** T9 asked for 2 servers; retrieval returned **4 port chunks + 1 hostname chunk**; the model resolved 1 of 2 (*"no information about srv-beta"*). | run `20261027_131450`, T9 |
| 1.16 | **`system_event` document ingestion is a no-op:** the runner logs and `continue`s; `attachments` is never resolved; `datasets/documents/` does not exist; `source="document"` is never produced. | `runner.py:449-451`; `phase-2-l2-vector-implementation-plan.md:5,109-110` |
| 1.17 | **`MarkdownMemoryBackend` does not exist** (`deferred`); `run.py` rejects any backend except `vector`; `rank_bm25` is not installed nor declared in `pyproject.toml`. | `backends/__init__.py:7-8`; `run.py::_resolve_backend_choice`; `tests/test_harness_m5.py:340-342` |
| 1.18 | **`consolidated_summary.md` reports infrastructure-errored runs as `0.0%`**, poisoning any average computed over it. | 5 rows in `outputs/consolidated_summary.md` |
| 1.19 | **Entity binding predicted to fail but passed 8/8** — `expected_level_2_behavior` in the YAML was **falsified**. Mechanism is 1.13 + 1.14 + a rank-5 margin (`ep_s2_t1` at 0.683 in T11 vs. absent in T9). | `test_entity_binding_01.yaml` metadata; run `20261027_131450` |

---

## 2. Acceptance criteria — what "L2 complete" means

L2 may be declared complete only when **all** of the following hold:

- [ ] **C1.** At least one full sweep ran with the **primary** embedder (`qwen3-embedding-8b`) on the discriminating scenarios (1.3).
- [ ] **C2.** At least **two model tiers** (`efficient` + `frontier`) ran the discriminating scenarios (1.4).
- [ ] **C3.** `temperature` is pinned, **or** N≥3 with a reported rate (1.5).
- [ ] **C4.** `top_k` is ablated (`{1,5,50}`) on at least the two retrieval-sensitive scenarios (1.6).
- [ ] **C5.** **Selective forgetting is measured cross-session** with the `top_k` inversion curve (1.9).
- [ ] **C6.** **Atomic revision is measured** (duplication / staleness-by-majority) (WS-B2).
- [ ] **C7.** The **two-step probe contamination** is removed or the graded turn re-attributed to L1 (1.14).
- [ ] **C8.** Every scenario either exercises the store or is explicitly labelled **in-context control** (1.2).
- [ ] **C9.** `consolidated_summary.md` excludes (or annotates) `error_turns > 0` runs (1.18).
- [ ] **C10.** `chunk_size`/`chunk_overlap` are recorded in `results.json`/`summary.csv` (1.12).
- [ ] **C11.** L2's implemented scope (**chat-only, vector-only**) is stated in `README.md` / `SPEC.md` (1.11, 1.16, 1.17).

---

## 3. Workstreams

### WS-A — Settle the confounds ("L2 never ran at its best")

| ID | Action | Deliverable | Verification |
|---|---|---|---|
| **A1** | Re-run the discriminating scenarios with **`qwen3-embedding-8b`** (4096 dims). **Precondition:** verify subscription inclusion via `GET /api/v1/models?detailed=true` — the 2026-10-23 attempts failed with `billing_restriction: subscription_only`. | 3 runs: `backward_modus_tollens_01`, `top_k_saturation_01`, `cross_thread_01` + `overhead_baseline_01` (P3 was CPU-inflated) | `summary.csv` shows `embedding_model_id = qwen3-embedding-8b`; `error_turns = 0` |
| **A2** | **Frontier-tier sweep** on the same three scenarios: `qwen3-235b`, `deepseek-v4-pro`, `kimi-k2.6`. | 9 runs | Per-model pass rates; **explicit statement** of whether the MT failure is model-dependent |
| **A3** | **Pin sampling.** Either add `temperature: 0` to every chat entry in `configs/models.yaml`, or run N≥3 and report a rate. | `models.yaml` change **or** a replication script | Documented decision; **caveat:** `temperature: 0` invalidates the oscillation-style experiment |

> **Note on A3:** pinning `temperature: 0` makes runs reproducible but removes the ability to observe reasoning variance. Recommended: pin `temperature: 0` for all benchmark runs and rely on **cross-phrasing** replication instead of sampling for the variance argument.

---

### WS-B — The three tests that actually justify L3 *(no code required)*

These are the **priority** workstream. All three survive "just use better RAG" and "just use a better model".

#### B1 — `explicit_forget_01` → multi-session + `top_k` inversion
- **Rewrite** `datasets/conversations/logical_consistency/explicit_forget/test_explicit_forget_01.yaml` as
  `s1` (seed `EMP-8042` + PIN `9173`) → `s2` (formal revocation) → `s3` (probe). This makes the PIN enter
  the store **from another session**, so `retrieve_node` actually surfaces it.
- **Add** `must_not_contain: "9173"` on **every** turn after the revocation, **including refusal turns**
  (see 1.9 — the observed leak happens inside the refusal explanation).
- **Run at `top_k ∈ {5, 50}`.** This produces the **inversion curve**: leakage must **increase** with `top_k`.
- **Predictions:** L2 @5 → leaks (fail); L2 @50 → leaks *more* (worse); L3 → `DELETE` → no leak regardless of `top_k`.
- **Also required:** run a same-model L1 baseline (closes 1.10).

#### B2 — `redundant_restatement_01` *(new scenario)* — atomic revision / dedup
- **Variant A (starvation):** five sessions restate *"the failover cluster is `dr-atlas-04`"*, each paired with a
  *different* secondary fact; probe asks for a secondary fact. Expected L2: the 5 near-duplicates fill `top_k`
  and **displace** the needed chunk.
- **Variant B (staleness by majority — the stronger one):** `s1..s3` restate `dr-atlas-04`; `s4` says *"we migrated —
  the cluster is now `dr-atlas-07`"*; `s5` probes. Store = 3 old vs 1 new.
  - Assertions: `must_contain: "dr-atlas-07"` + `must_not_contain: "dr-atlas-04"`.
  - **Predicted L2:** `dr-atlas-04` (chunk majority) → FAIL. **Predicted L3:** `dr-atlas-07` (upsert/revision) → PASS.
- **Why this is the load-bearing test:** no model can de-duplicate a store or determine which of 4 conflicting
  chunks is current. The failure is **architectural**, not capability.
- **Bonus metric (free):** *distinct facts stored* vs *chunks stored* — a growing gap is the dedup argument as a figure.
  Derivable from `manifest.json::total_chunks` + `episodes.json`.

#### B3 — Belief enumeration *(new scenario)* — state ≠ transcript
- **Probe:** *"List everything you currently believe about X: location, audit status, and last change."*
- **Predicted L2:** incomplete or contradictory enumeration — `top_k` limits what it can see and there is **no
  canonical list** to enumerate from. Partial prior evidence: `top_k_saturation_01` could not enumerate 10 serials.
- **Predicted L3:** complete, non-contradictory enumeration from the mutated state.
- **Assertions:** one `must_contain` per fact + `must_not_contain` for wrong/extra facts.

**Verification for WS-B:** each scenario loads via the real pipeline (`pytest tests/test_stage*_scenarios.py`
pattern), assertions 0 on seeding turns, and an `expected_level_N_behavior` written **before** the run.

---

### WS-C — Validity & ablations

| ID | Action | Detail | Why |
|---|---|---|---|
| **C1** | **`top_k` grid** `{1, 5, 50}` | `backward_modus_tollens_01` + `top_k_saturation_01` (+ B1, B2) | Turns "L2 fails" into "L2 cannot improve through its own knobs". `top_k=50` pass on `top_k_saturation_01` proves cutoff; `top_k=50` fail on MT proves it is not a context problem |
| **C2** | **Blind-provenance ablation** | Config flag that formats the injected block **without** `session`, `turn`, or semantic `chunk_id` (opaque IDs). Run `entity_binding_01`, `backward_modus_tollens_01`, `cross_thread_01` under both modes | Determines whether L2's apparent capability comes from the architecture or from the **harness's formatting** (1.13). Affects `graph.py::_format_memory_block` only |
| **C3** | **Decontaminate two-step probes** | Audit every scenario using `open probe → commitment` (`entity_binding_01`, `derived_belief_revision_01`, `backward_modus_tollens_01`, `top_k_saturation_01`). Fix by moving the open probe to a **separate session**, or by grading only the commitment turn and labelling it accordingly | The open probe's answer enters the checkpointer history and contaminates the graded turn (1.14) |
| **C4** | **Migrate single-session scenarios** | `explicit_forget_01` (→ B1), `adversarial_forget_01`, `belief_revision_01` → multi-session. **Keep** `in_context_derived_revision_01` and `in_context_modus_tollens_01` as single-session **by design** (labelled in-context controls) | 5 of 14 runs measure nothing about L2 (1.2) |
| **C5** | **Grade the multi-entity probe** | Promote `entity_binding_01` T9 from observational to graded (assertions on 2 entities) and **add a 4-entity probe**: *"give me the admin console port for each of srv-alpha, srv-beta, srv-gamma, srv-delta, one per line"* | The failure already exists but is ungraded (1.15). At 4 entities, **8 chunks** are needed and `top_k=5` provably cannot fit them |

**Note on C5:** the failure signature is *"the retrieval returns all the values but not the keys to bind them"*
(4 port chunks + 1 hostname chunk). It is the cleanest one-sentence statement of L2's entity-binding limit.

---

### WS-D — `system_event` document ingestion *(code milestone; enables document + addendum revision)*

> **Status: ✅ DONE (2026-11-02).** Implemented with two channels: scenario-level `documents:` preload
> (new) and `system_event` + `attachments` ingestion (the item below). Delivered together with: vector
> backend document support, provenance (`sha256`, `source_path`) in snapshot manifests, `documents.json`,
> `corpus_stats()` dilution reporting, CLI knobs (`--top-k`, `--chunk-size`, `--chunk-overlap`,
> `--no-documents`), fixtures in `datasets/documents/`, a deterministic corpus generator
> (`scripts/generate_corpus.py`), four document scenarios, and `tests/test_document_ingestion.py`.
> Deviations from the sketch below: `TurnResult` gained dedicated `ingested_documents` /
> `ingested_chars` / `ingested_chunk_ids` fields instead of reusing `appended_chunk_ids`, so chat
> persistence and document ingestion stay independently auditable.
>
> Predicted signatures below are kept as the experimental rationale.



Required only if documents are declared **in scope** (see §5). It is the only way to exercise the chunking
path at all (1.11) and the most realistic framing of the deletion/atomic-revision argument.

1. **`core/runner.py`** — replace the no-op at `:449-451`:
   - Load text from `turn.content` **and** every path in `turn.attachments`.
   - For L2+: chunk and append directly into the active `EpisodicMemoryBackend` with `source="document"`,
     **without invoking the chat model**.
   - Emit `TurnResult(role="system_event", status="ok", append_latency_ms=…, embedding_latency_ms=…,
     embedding_tokens=…, appended_chunk_ids=[…])`.
   - Populate `ingestion_latency_ms` (currently hardcoded `0.0` at `runner.py:709`).
   - Exclude `system_event` turns from assertion denominators and `avg_latency_per_turn_ms`.
2. **`attachments` resolution** — relative to the scenario YAML directory, falling back to `datasets/documents/`.
3. **Fixtures** — `datasets/documents/policy_v1.md` (name->auth-level table, ≥ 6 sections) and
   `addendum_1.md` (*"§3.2 is superseded: prod-tier reboots now require Level-2"*).
4. **`test_document_addendum_revision_01.yaml`** — ingest v1 → chat probes → ingest addendum → probe
   *"What authorization does a prod-tier reboot require today?"*
   - Predicted L2: FAIL or unstable — the v1 §3.2 chunk outnumbers and out-scores the single addendum chunk.
   - Predicted L3: PASS — the judge detects supersession and issues `UPDATE`.
   - **This is the 30:1 argument:** a 10-chunk policy plus a 1-chunk addendum leaves the addendum
     structurally outnumbered. No model fixes that; only `UPDATE` does.
5. **Optional:** `test_doc_needle_01.yaml` (single-fact retrieval in a large document).

**Verification:** offline pytest with `DeterministicFakeEmbedding` — ingestion produces **>1 chunk** per document
(proving the splitter fires for the first time), `ingestion_latency_ms > 0`, `manifest.json` shows
`source="document"` chunks, and `system_event` turns never invoke the model.

---

### WS-E — Hygiene *(no code for E1/E2 beyond a few lines)*

| ID | Action | Detail |
|---|---|---|
| **E1** | **Filter infrastructure errors** in `consolidated_summary` | Exclude (or annotate) any run with `error_turns > 0`. Today 5 runs appear as `0.0%` and poison averages (1.18) |
| **E2** | **Record chunk settings** | `run.py`: add `chunk_size` / `chunk_overlap` to `memory_backend_config` (+ CLI flags). Today the real values exist only in `manifest.json` (1.12) |
| **E3** | **Declare L2 scope** | `README.md` §Agent Levels Roadmap and `SPEC.md` §4: state **chat-only, vector-only**, and that chunking / `source="document"` / BM25 are **not exercised** (1.11, 1.16, 1.17). Move L2 from "🚧 Next" to "✅ Implemented (scope limited)" |

---

## 4. Priority order (value / cost)

| Order | ID | Workstream | Code? | Runs | Rationale |
|:--:|---|---|:--:|--:|---|
| **1** | B1 | `explicit_forget` multi-session + `top_k` inversion | ❌ | 2 + L1 baseline | The only **unalienable** L3 argument (deletion). Better retrieval makes it worse |
| **2** | B2 | `redundant_restatement_01` | ❌ | 1–2 | Atomic revision — no model can dedup a store |
| **3** | C3 | Decontaminate two-step probes | ❌ | re-run | Without this, existing pass rates are inflated |
| **4** | C5 | Grade the multi-entity probe | ❌ | 1 | Failure already observed (1.15); just needs assertions |
| **5** | C1 | `top_k ∈ {1,5,50}` grid | ❌ | ~6 | Converts failures into causal claims |
| **6** | A1 + A2 | Primary embedder + frontier sweep | ❌ | ~13 | Kills the "under-provisioned L2" and "cheap model" objections |
| **7** | B3 | Belief enumeration | ❌ | 1 | State ≠ transcript |
| **8** | C2 | Blind-provenance ablation | ✅ (~small) | 6 | Separates architecture from harness formatting |
| **9** | A3 | Pin temperature | ✅ (config) | — | Reproducibility |
| **10** | E1 + E2 + E3 | Hygiene + scope declaration | ✅ (~small) | — | Prevents misreported numbers |
| **11** | C4 | Migrate remaining single-session scenarios | ❌ | 2 | Completes coverage |
| **12** | WS-D | Document ingestion | ✅ (**~1 day**) | 2 | High value, highest cost. Only if documents are in scope |

**Steps 1–7 require no code.** With those seven closed, L2 is defensible and L3 is justified on
arguments that survive both standard objections.

---

## 5. Explicitly out of scope (decisions required)

| Item | Status | Decision needed |
|---|---|---|
| **`MarkdownMemoryBackend` + BM25** | ❌ Not implemented (`backends/__init__.py:7-8` says *deferred*; `run.py` rejects any backend except `vector`; `rank_bm25` not installed / not in `pyproject.toml`) | **Declare out of scope for L2** (recorded as future work and as the enabler of the dense-vs-lexical hypothesis + the P3 counterfactual), or budget it as a full milestone. **Recommendation: declare out of scope** — it does not block L3 |
| **Giant conversation sweeps (100+ turns)** | Not planned | Reason: measures L1's context window (a single long session) or degenerates into "many short sessions"; and retrieval dilution is a plumbing axis, not an L3 justification. **At most one scalability probe** as a secondary figure |
| **Oscillation / re-derivation variance** | Not planned | Would require `temperature > 0`, which conflicts with A3. Alternative: cross-phrasing replication |
| **L3 implementation** | Out of scope | This plan ends when §2 acceptance criteria are green |
| **Recall@k ground truth** | Not planned | Requires declaring `target_chunk_ids` per probe turn in each YAML. Only `top_k_saturation_01` declares it. Worth doing opportunistically inside B1/B2/C5 |

---

## 6. Open questions for supervisors

1. **Ownership of derived-belief contraction.** `phase-2-l2-plan.md:16-17` assigns *"multi-hop derived-belief
   invalidation"* to L3's **expected failure mode** and *"derived-belief retraction (facts + rules)"* to **L4's
   target capability**. Therefore `backward_modus_tollens_01` (Stage 5.4) should justify **L3 → L4**, not **L2 → L3**.
   *Confirm this reading before the defense.*
2. **Scope of L2.** Documents + chunking in, or declared out? (WS-D is ~1 day of code; it is the only way to
   exercise the splitter, 1.11.)
3. **Is `MarkdownMemoryBackend` in scope?** It is the declared hypothesis of `phase-2-l2-plan.md` §3.3 and the
   only source of the P3 embedding-cost counterfactual.
4. **Sampling policy.** Pin `temperature: 0` for reproducibility, or keep provider defaults and report n=1?
5. **Is "belief enumeration" a legitimate L2/L3 discriminator**, or does it belong to L4's structured belief state?

---

## 7. Cross-references

- `notes/2026-10-30_l2_episodic_rag_summary.md` — the results summary this plan closes the gaps of.
- `TODOs/phase-2-l2-plan.md` §1 — ablation ladder (target capability / expected failure mode per level).
- `TODOs/phase-2-l2-vector-implementation-plan.md` §Deferred — the deferral list WS-D closes.
- `TODOs/test-expansion-plan.md` Stage 5.x — the belief-revision scenario family (`derived_belief_revision_01`,
  `backward_modus_tollens_01`).
- `AUDIT_REPORT.md` §3.8 — `system_event` silent no-op, still open (WS-D item 1).





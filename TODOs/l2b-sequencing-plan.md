# L2b (Agentic Naive-RAG) × Document-Test Coverage — Sequencing Plan

**Question answered:** implement the Tier-1 document changes first, or build L2b first and polish the
tests afterwards?
**Answer:** neither. Run a **contract-first vertical slice**: freeze the retrieval telemetry contract
(½–1 day) → author **3–4 discriminating cells** (1.5 days) → build L2b against them → widen the corpus
once the agent's failure modes are known.
**Companions:** `DIFFICULTY_AUDIT.md` (test-suite audit + steps H*/M*), `CORPUS_VALIDITY.md` (§12–14
capability catalogue and build order).

---

## 0. Why not "L2b first"

| Reason | Evidence from the repo |
|---|---|
| **Ceiling effect: the comparison would be uninformative** | On today's document cells a dependency-free BM25 reaches **Hit@5 = 24/24** (`CORPUS_VALIDITY.md` §3.3). L2 and L2b would both score ≈100 %, so "agentic retrieval helps" would be untestable — not because it doesn't help, but because the probes are already solved by lookups. |
| **Three new failure modes that the telemetry cannot record** | Agentic RAG adds (i) *no tool call at all*, (ii) *call with a bad query*, (iii) *wasted calls that crowd the context*. `_read_l2_telemetry` reads a fixed key set (`retrieved_memories` = a flat list from exactly one `retrieve_node` call), so none of the three is visible. You would be debugging by hand-reading `results.json`. |
| **Every run before the contract is stable is wasted or backfilled** | The repo already paid this cost once: the `retrieval_recall` denominator incident (notes F7) forced `scripts/backfill_retrieval_recall.py` and a "denominator drift" column. Runs generated under an unstable telemetry contract repeat it. |
| **The document design depends on an interface choice not yet made** | e.g. does the retrieve tool return *content only* (as today) or *content + provenance*? That single decision determines whether `document_provenance_attribution_01` (T1.6) is writable at all. |

## 0.1 Why not "full Tier-1 first"

| Reason | Detail |
|---|---|
| **Half of Tier 1 is code, not documents** | set-recall metric, numeric assertions, store assertions (`must_store`) belong to the same contract work, not to authoring |
| **~4–6 days before the agent can run** | you would author a corpus for an agent whose tool interface is unsettled |
| **The generator rewrite is a one-way commitment** | M9 (claim grammar) regenerates `d16_*` and invalidates the corpus the current notes are built on. Do it *after* seeing which knobs matter, with 3 hand-authored cells carrying the signal in the meantime |

---


## 1. Phase 0 — freeze the L2b retrieval contract (½–1 day, no model code)

The goal: **the same scenario YAML must be able to explain an L2b failure.** Files touched:

| File | Change |
|---|---|
| `core/schemas.py` | `L2AgentState`: add `retrieval_calls: list[dict]`, `retrieval_call_count: int`, `retrieval_skipped: bool`, `tool_call_count: int`, `tool_call_cap_hit: bool`. `TurnResult` + `BenchmarkRunSummary`: mirror the same fields (per turn and aggregated), plus `total_tool_calls`, `avg_calls_per_retrieval_turn`. |
| `core/runner.py` | `_read_l2_telemetry`: read the new channels with the same guarded pattern (absent ⇒ 0 / `False`, so L0/L1/L2 are untouched). |
| `core/metrics.py` | `retrieval_recall` stays rule-level, but the **match set becomes the union of all calls in the turn**; add `calls_per_probe` and `retrieval_call_efficiency` (pins satisfied per call made). |
| `agents/__init__.py` | register `level_2b_agentic_rag`. Note: `--level 2` still resolves uniquely (regex `^level_2(?:_|$)` does not match `level_2b…`), but `--level 2b` **fails today**, because `token.isdigit()` is False so the prefix becomes `2b`. Add an alias map (`{"2b": "level_2b_agentic_rag"}`) or broaden the regex to `^level_(\d+[a-z]?)_`. |
| `run.py` / `configs/` | nothing structural; L2b sets `requires_memory_backend = True`, so the CLI builds an embedder exactly as for L2. |

### 1.1 Semantics to freeze now (these are what make failures attributable)

1. **Union matching.** `must_retrieve_chunk` is satisfied if **any** call in the turn retrieved a matching
   chunk. Optional per-rule `call_index: int` filter, for probes like "the *second* query must find it".
2. **`exclude_session_id` still applies to the tool.** Otherwise L2b silently wins the 86 intra-session
   probes by reading the current thread (document cells are unaffected — documents have no session).
3. **Hard `max_tool_calls` cap per turn** (start at 3), recorded as `tool_call_cap_hit`; without it
   termination is uncontrolled and the P3 latency/cost comparison is meaningless.
4. **Empty retrieval is recorded, not swallowed.** A call returning 0 chunks appears in `retrieval_calls`
   with `chunk_ids: []`. Today "no similarity floor" guarantees `k` results, so "retrieved nothing" is
   unrepresentable.
5. **Context budget is cumulative.** `retrieved_context_tokens` = tokens of the **union actually injected**;
   add `retrieval_context_tokens_gross` (sum over calls, duplicates included) so re-query cost is visible.
6. **Provenance stays off by default.** The content-only memory block is an established design decision
   (`_MEMORY_SEPARATOR`); changing it silently breaks comparability with every prior L2 run. Make it an
   explicit ablation flag `--memory-provenance {off,on}` instead of a refactor.

### 1.2 The five-cell failure taxonomy L2b makes necessary

Every probe must fall into exactly one class; the runner should emit the distribution per run:

| # | Class | Detection |
|---|---|---|
| **F0** | **No retrieval attempted** — answered from parameters / in-context | `retrieval_call_count == 0` on a turn that declared a pin |
| **F1** | **Retrieval attempted, query malformed** — missing entity/predicate token | call exists, `query` fails a per-probe `query_must_contain` check |
| **F2** | **Query fine, gold absent from the union** | pins fail, calls non-empty |
| **F3** | **Gold present, answer wrong** (read/resolution error) | pins pass, text rules fail |
| **F4** | **Gold present, answer right** | both pass |

Without F0–F3 an L2b regression is indistinguishable from a retriever regression.

### 1.3 Phase-0 tests (offline, fake model + fake backend, same style as `tests/test_harness_m3.py`)

- `test_retrieval_calls_recorded` — two tool calls ⇒ `retrieval_call_count == 2`, both queries captured.
- `test_union_matching` — gold in the 2nd call satisfies `must_retrieve_chunk`; a `call_index: 1` filter fails.
- `test_empty_retrieval_is_recorded` — a call returning `[]` yields one entry with `chunk_ids: []`.
- `test_cap_is_enforced_and_flagged` — a model looping 5× stops at 3 and sets `tool_call_cap_hit`.
- `test_exclude_session_still_applies_to_the_tool`.
- `test_level_2b_resolves` — `resolve_level_ids("2b")` (after the alias fix) and `resolve_level_ids("2")`
  both work, unambiguously.

---


## 2. Phase 1 — the cells worth writing for L2b: 3 before the agent, 3 after (1–1.5 days)

Selection criterion: **does a tool-enabled agent have a structural advantage or disadvantage here?** Cells
where one-shot and agentic retrieval are equivalent would cost effort without discriminating.

| Order | Cell | Why it discriminates L2 vs L2b | New machinery |
|---|---|---|---|
| **1** | `document_multihop_bridge_01` (T1.4) | The canonical agentic win: the first query yields the bridge entity, the **second** query resolves it. One-shot `retrieve(query, top_k)` cannot do this at any `k` unless both chunks happen to rank together. | set-recall metric (2 gold chunks as one unit) |
| **2** | `document_conflict_dilution_01` (T1.1 + T1.2) | Siblings share the predicate; the agent can re-query with the entity qualifier, naive retrieval cannot. Also the L2-vs-L3 revision story at 60-doc scale. | generator params (claims/siblings/supersede) + `must_not_retrieve_chunk` pins |
| **3** | `document_split_evidence_01` (T1.5) | Claim severed by the 600-char boundary: a re-formulated query recovers the other half. Cheapest cell in the set — two hand-written documents, no generator work. | none |
| **4** | `document_prevalence_trap_01` (T1.3) — **negative control** | The case where agentic retrieval can plausibly *hurt*: every extra call re-surfaces the 5 stale copies. A study containing only favourable cases is not credible. | 1 doc set + symmetric control |

**Split: 3 before L2b, 3 after.** The four rows are *ranked*, not a quota. Only cells where agentic and
one-shot retrieval can differ **structurally** must exist before the agent is coded, because otherwise the
L2-vs-L2b comparison has no signal:

| Before L2b | Why it cannot wait |
|---|---|
| `document_multihop_bridge_01` | the only cell where a second query is *required* to succeed; without it L2b can only be measured at ceiling |
| `document_conflict_dilution_01` | the only cell where re-querying with an entity qualifier changes the answer |
| `document_prevalence_trap_01` | the only cell where extra retrieval calls can *hurt* — the negative control that stops the study from being a promotional demo |

`document_split_evidence_01` (T1.5) is cheap enough to write in the same sitting, but it is a chunking
diagnostic that both agents face; it can land with Phase 3. The remaining two of the six build-order cells
(`document_precision_scope_01`, `document_injection_01`) are Phase 3 by construction.

**Add the lexical-baseline test (V1) in the same pass.** If BM25 solves a cell, both agents saturate and the
cell cannot discriminate — that check is what prevents repeating the current corpus' failure on a new budget
line.

**Definition of done per cell** (all six, or it is not a test). Tier labels are *optional bookkeeping* — what
must be true is that the cell discriminates:

1. `metadata.calibration` block: measured context words, distractor count, and
   `lexical_baseline_hit_at_5 ≤ 0.6` (a `tier` label is optional; the number is not).
2. ≥1 positive pin **and** ≥1 negative pin (`must_not_retrieve_chunk` on a decoy).
3. A declared in-context twin, so a failure separates retrieval from reasoning.
4. `per_claim_tier` when a cell hosts claims of different tiers (a D3 inference beside D1 lookups).
5. `intrinsic_expectations` (store gold), even while still prose — it becomes `must_store` at step H2.
6. `metadata.expected` per level including L2b: e.g. `{L0: fail, L1: fail, L2: partial, L2b: pass}`, so a
   regression is a *signal* rather than a surprise.

---

## 3. Phase 2 — implement L2b against those cells (2–3 days)

Build order inside the agent, each step independently testable:

1. **Tool definition** — `retrieve(query: str, k: int = 5) -> str` returning the **content-only** block (same
   `_format_memory_block`) plus a parallel structured return for telemetry. Keeping the prompt payload
   identical to L2 isolates *the agentic decision* as the only changed variable.
2. **Loop with cap** — model → tool → model, `max_tool_calls = 3`, cap recorded; "no tool call" is a valid
   terminal state (that is F0, and it must be observable, not an error).
3. **Union + budget** — dedupe chunks across calls by `chunk_id`, inject the union in first-seen order,
   respect the cumulative token budget.
4. **Telemetry** — `retrieval_calls`, `retrieval_call_count`, `retrieval_skipped`, `tool_call_cap_hit`.
5. **Failure classification** — emit F0–F4 per probe (`failure_class` on `TurnResult` when the turn declares
   pins).

Then run the paired experiment: **L2 (one-shot) vs L2b (agentic), same cells, same model, same prompt, same
`top_k`, same call budget.** Report accuracy, pins, calls/probe, tokens, latency and the F0–F4 distribution.


## 4. Phase 3 — widen, with the lessons in hand (3–5 days)

| Step | What | Why now |
|---|---|---|
| 3.1 | Remaining Tier-1: supersession at corpus scale, `document_provenance_attribution_01` | needs Phase-2's answer to "does the tool return provenance?" |
| 3.2 | Tier-2: precision/abstention scope, document-borne injection, alias/cousin, packaging, time-scoped validity | cheap once the contract exists |
| 3.3 | **Regenerate `d16_*` with the claim grammar (M9)** plus the V1–V6 contract | the corpus rewrite should encode what L2b actually struggled with, not a guess |
| 3.4 | Re-run **L0/L1/L2/L2b** over the full document arm for the paired tables | the comparison the thesis needs |

---

## 5. Do **not** do these now

| Don't | Why |
|---|---|
| Regenerate `d16_corpus_*` before Phase 2 | it invalidates the corpus behind `notes/2026-11-05_l2_document_corpus_size_curve.md` and the README numbers — do it once, informed |
| Run L2b on all 33 scenarios | 23 have no retrieval pins and 86/113 probes never reach retrieval; the run buys nothing |
| Turn on prompt provenance as a side effect of L2b | it is a **treatment**, not a fix: make it `--memory-provenance {off,on}` and report it as an ablation arm |
| Silently "fix" the current D3 labels | relabel (`harness_validation`) and keep the numbers — they are your plumbing control |
| Add store assertions (`must_store`) mid-Phase-2 | contract churn again; land H2 after the L2b comparison is published |

---

## 6. Coverage matrix — the document arm is "done" when every row is ticked

| # | Claim type × retrieval behaviour | Cell | Status today |
|---|---|---|---|
| 1 | Single verbatim claim, unique key | `document_needle_01`, `d16_*` | ✅ (as a lookup control) |
| 2 | Claim among same-schema siblings (D2) | `document_dilution_01` | ⚠️ hand-written, 354 w, no competition metric |
| 3 | Claim requiring cross-document inference (D3) | `document_authority_vs_recency_01` | ⚠️ inference real, noise synthetic |
| 4 | **Compositional (2 docs, bridge entity)** | — | ❌ Phase 1 #1 |
| 5 | **Competing + superseded at scale** | — | ❌ Phase 1 #2 |
| 6 | **Evidence severed by chunk boundary** | — | ❌ Phase 1 #3 |
| 7 | **Prevalence / majority-stale trap** | — | ❌ Phase 1 #4 |
| 8 | **Precision & abstention at scale** (deprecated / out-of-scope / other-tenant) | `document_dilution_01` (1 probe only) | ❌ Phase 3 |
| 9 | **Document-borne injection** | — | ❌ Phase 3 |
| 10 | **Source attribution / citation** | — | ❌ blocked on the provenance decision |
| 11 | Alias / coreference across documents | `entity_binding_01` (chat only) | ❌ Phase 3 |
| 12 | Time-scoped validity | `document_effective_dates_01` | ⚠️ wall-clock dependent |
| 13 | Packaging (length / genre) | — | ❌ Phase 3 (after M9) |

Rows 4–7 are the minimum for an **L2b** claim; rows 8–13 the minimum for a "document memory" claim.

---

## 7. Effort summary

| Phase | Effort | Deliverable |
|---|---|---|
| 0 — contract | ½–1 d | telemetry + union semantics + cap + F0–F4 + 6 offline tests |
| 1 — 4 cells | 1–1.5 d | 4 scenarios (2 need generator params), V1 test, twin controls |
| 2 — L2b | 2–3 d | tool, loop, cap, union, telemetry, classification, paired L2-vs-L2b table |
| 3 — widen | 3–5 d | remaining Tier-1/Tier-2, regenerated corpus, full L0–L2b matrix |

**Total to a defensible L2b result: ~4–5 days** (Phases 0–2). The alternative order (L2b first) reaches a
running agent in ~3 days, but with no cell able to distinguish it from L2 — the comparison, which is the
point of the exercise, arrives later and costs a corpus rewrite plus a re-run.

---

## 8. Why this ordering is cheap, not just correct

The three Phase-0 decisions (union matching, call cap, F0–F4 classification) cost one day and are the same
machinery that later supports `must_store`, packaging and provenance ablations. The four Phase-1 cells are
2 of 4 hand-authorable and reuse the generator flags the corpus rewrite (M9) needs anyway. Conversely,
building L2b first costs the same agent code plus a *second* pass over the telemetry, plus a corpus rewrite
on top — which is why the contract-first slice is also the shortest path.



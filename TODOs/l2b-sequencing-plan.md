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

### 2.1 Corpus economics for these cells (size, reuse, generation)

**Size: 40–80 chunks per cell, not 60 documents.** The binding constraint is that the gold and its
competitors must actually compete inside the `top_k` window, so chunk count should be **8–16 × top_k**
(40–80 at `k=5`):

| Chunks | What happens |
|---|---|
| < 25 (< 5×k) | everything fits in the window → both agents pass, no discrimination |
| **40–80 (8–16×k)** | competition binds; the gold does not always dominate; decoys sometimes surface, so `must_not_retrieve_chunk` becomes meaningful |
| > 200 | the failure mode becomes "gold never retrieved" (a `top_k` provisioning finding) and the competitors vanish from the window → conflict tests stop measuring conflict |

Size for the **largest `k` you intend to sweep**, not just the default: your notes run `k ∈ {1, 2, 5, 8}`,
so a cell that will also be run at `k=8` needs **65–130 chunks** (8–16 × 8) to keep the competition binding.
If you only ever measure at `k=5`, stay at 40–80.

The existing 60-document corpus is **376 chunks at `chunk_size=1200` / 880 at 600** (measured) — 5–20× more
than a Phase-1 cell needs, and actively harmful for conflict/prevalence cells because the siblings get
diluted out of the window.

**Documents: 2–4 kB each, not 6 kB.** At `chunk_size=1200 / overlap=200` a 6 kB document yields 6.3 chunks
(measured), so **8 documents × ~4 kB ≈ 45–60 chunks** — one eighth of the current corpus and roughly one
minute of cold embedding. Keep `chunk_size = 1200` for the new cells (the three hand-authored revision cells
already declare 1200; only the D16 scaling cells use 600, for comparability with their existing notes). The
one exception is `document_split_evidence_01`, where chunk size *is* the treatment and must be declared
explicitly.

**Reuse: yes — one shared background + a per-cell claim cluster.**

```
datasets/documents/calibration/
  shared_bg_00.md … shared_bg_07.md     # byte-identical across all cells → embedded once, cached forever
  conflict_c01.md … conflict_c04.md     # entity namespace LOOP-*
  bridge_c01.md … bridge_c03.md         # entity namespace RACK-*
  prevalence_c01.md … prevalence_c06.md # entity namespace VAULT-* (5 stale copies + 1 current)
```

Three rules make reuse safe:

1. **Disjoint entity namespaces per cell.** Otherwise one cell's stale copies become another cell's
   distractors and the cells stop being independent.
2. **Shared filler mentions no cell entity.** Otherwise the "gold entity occurs once" invariant breaks and a
   pin can be satisfied by background text.
3. **One `chunk_size` for every cell that shares the filler.** The embedding cache is content-addressed on
   `sha256(model_id | dims | chunk_text)` (`core/embedding_cache.py::make_key`), so changing chunk size
   invalidates the whole cache — freeze it per experiment.

**Generation: LLM-drafted at authoring time, committed to disk; the deterministic generator only for volume
and for the scaling curve.**

| Part of the corpus | How | Why |
|---|---|---|
| Surrounding prose of claim-bearing documents | LLM-drafted, then reviewed and **committed as static files** | the template generator's unique-sentence ratio is 0.067 (measured); V5 wants 0.5–0.9, and exact-template prose makes retrieval solvable by token matching |
| The claim sentences themselves | **hand-written / hand-approved** | they carry the ground truth and the assertion tokens |
| Opaque answer tokens + bulk filler + the 8/60/200 scaling corpora | deterministic `generate_corpus.py` | byte-identical regeneration, unguessable answers, cheap volume |
| Anything at *run* time | **never** an LLM | breaks reproducibility, the cache, and offline re-runs |

**Embedding cost — the 10–15 min is paid once, not per cell.** `--embed-cache DIR` stores chunk vectors by
content hash and the backend reuses them across runs (`embedding_cache_hits/misses` are in the telemetry, so
confirm hits ≈ chunk count before trusting a timing). Query embeddings are deliberately *not* cached, so
retrieval latency stays honest. Budget for the four cells:

| Cell | Documents | Chunks @1200 | Cold embed | Warm |
|---|---|---:|---:|---:|
| shared background (once) | 8 | ~30 | ~1 min | ~0 |
| `document_multihop_bridge_01` | +3 | ~40 total | ~1 min | ~15 s |
| `document_conflict_dilution_01` | +4 | ~45 total | ~1 min | ~20 s |
| `document_prevalence_trap_01` | +6 | ~50 total | ~1 min | ~25 s |
| `document_split_evidence_01` | 2 (no shared) | 6–8 | seconds | ~0 |

**Development vs measurement.** While building the L2b loop, use a 2-document smoke corpus (~6 chunks) with a
single probe; only the measurement runs need the full cell corpus. That split — plus the cache — is where the
`10–15 min × N` waste actually comes from. (There is no BM25/markdown backend yet: `MarkdownMemoryBackend`
is declared *deferred* in `agents/level_2_episodic_rag/backends/__init__.py`, so the smoke corpus is the only
cheap iteration path.)

### 2.4 Built: `document_multihop_bridge_01` (as executed, 2026-11-11)

The first of the three pre-L2b cells is in the tree. Measured, not estimated:

| Item | Value |
|---|---|
| Corpus | 14 documents in `datasets/documents/calibration/` (2 012–3 022 chars each) |
| Chunks at 1200/200 | **42** (band 35–70) |
| Gold hop-1 chunk | `doc_asset_registry_01_c0` (rank **1** for the composed probe) |
| Gold hop-2 chunk | `doc_calibration_register_02_c1` (rank **42/42** for the composed probe, rank **1** for `NBL-420`) |
| Unique-sentence ratio | 1.000 (256/256) |
| Scenario | `datasets/conversations/multi_thread/compositional_reasoning/document_bridge/test_document_multihop_bridge_01.yaml` |
| Twin | `datasets/conversations/single_thread/compositional_reasoning/in_context_bridge/test_in_context_bridge_01.yaml` |
| Tests | `tests/test_document_multihop_bridge_01.py` — 28 offline invariants, all green |

The design lock is the register's vocabulary: it contains **none** of the probe's content words, so hop 2 is
unreachable without the identifier at *any* `k` — this is what makes `L2a_k15` fail as well, i.e. breadth
cannot substitute for a second query. `pytest -q` = **642 passed**; `--list-scenarios` = **35**.

One deliberate edit outside the new files: `tests/test_assertion_hygiene.py` gained an H1 exemption for
`in_context_bridge_01`, because a ceiling control necessarily contains its own answer (the ratchet already
requires such exemptions to be justified by a `control_in_context` kind, which the twin declares).

### 2.5 Built: `document_conflict_dilution_01` (as executed, 2026-11-11)

Second of the three pre-L2b cells. Reuse honoured: the 8 claim-free `bg_*` filler documents plus **one**
decoy (`decoy_01`); the claim cluster is new (`PMP-*`, pumps on the third riser).

| Item | Value |
|---|---|
| Corpus | 14 documents → **39 chunks** (band 35–70); 5 new + 8 reused filler + 1 reused decoy |
| New documents | `conflict_gold_01` (PMP-310 → `HLX-2C7V-9043`), `conflict_sibling_01/02/03` (PMP-322/335/348), `conflict_superseded_01` (PMP-310 → `SWG-1F9K-5520`, June 2025) |
| Competition, measured (BM25, T2 probe) | window at `k=5` = gold(rank **1**), superseded(**2**), siblings(**3,4,5**) — the answer is surrounded by four distractors |
| Skeleton | identical `##` sections; only the H1 role and the asset body differ once IDs/values/roles/dates are masked; Jaccard 0.90–0.93 vs gold |
| Probes | T2 current value (conflict + staleness), T3 paired commitment for PMP-335 with a bleed guard, T4 history of the same asset, T5 negative-only guard |
| No-hint rule | no user turn contains a precedence cue ("superseded", "previous", "latest", "current", "revision", …); the acknowledgement turn no longer enumerates the documents, so the model cannot learn the corpus map without retrieving. Enforced by `test_no_user_turn_signals_precedence` |
| Scenario | `datasets/conversations/multi_thread/logical_consistency/document_conflict/test_document_conflict_dilution_01.yaml` |
| Tests | `tests/test_document_conflict_dilution_01.py` — 27 offline invariants, all green |

Two differences from the bridge cell, both deliberate: the **lexical baseline is expected to identify the
gold document** here (each probe names its identifier, so reachability is not the variable — competition is),
and the competition gate is therefore "the window must contain the answer *and* ≥2 siblings *and* the
superseded record", the mirror image of the bridge's "hop 2 must be absent". `pytest -q` = **669 passed**;
`--list-scenarios` = **36**.

### 2.6 Built: `document_split_evidence_01` (as executed, 2026-11-11)

Third pre-L2b cell. Same reuse discipline: the 8 `bg_*` filler documents plus **one** decoy (`decoy_01`);
the claim cluster is new (`VLV-*` valves, two headerless exports).

| Item | Value |
|---|---|
| Corpus | 12 documents → **37 chunks**; 3 new (`split_legend_01`, `split_rows_01`, `split_rows_02`) |
| The severance | both exports carry **no header row** (asserted: no chunk contains a field name); the field order exists only in the reading guide |
| The trap | the two riser families are ordered **differently** — certified value at position 2 for the third riser, position 3 for the north riser — so decoding a record with the wrong family's order parses and reads wrongly |
| Measured (BM25) | T2: record rank **1**, guide **outside the window** (> 5) → the row is reachable but undecidable in one hop. T4 (probe names the format): guide rank **3** → solvable in two hops |
| Probes | T2/T3 value probes on two records of the same export, T4 the format probe (runs **after** them, so the order is not carried in the thread), T5 negative-only guard |
| Scenario | `datasets/conversations/multi_thread/compositional_reasoning/document_split/test_document_split_evidence_01.yaml` |
| Tests | `tests/test_document_split_evidence_01.py` — 28 offline invariants, all green |

**One design deviation, learned from the two earlier cells.** Severance is enforced *structurally* (the
extractor "omits the header"; the order lives in a separate guide) instead of relying on the splitter to cut
a table between its header and its first row. With `chunk_overlap = 200` an element that sits on a boundary
survives in **both** chunks, so a splitter-induced severance is unreliable and would drift whenever the prose
changed — the invariant would be an accident of text length rather than a property of the corpus.

**Ranks are asserted as bounds, not exact values**: every chunk that shares no term with the probe scores
zero, so its position is a tie broken by document order. `pytest -q` = **702 passed** (700 at build time; the
strict probe added one invariant — see 2.7); `--list-scenarios` = **37**.

### 2.7 Follow-up: the value probe was measuring answer form (R2, 2026-11-11)

The first two k=5 runs of this cell disagreed by 2/9 for no reason the cell cared about:

| Run | T2 answer | Old verdict |
|---|---|---|
| `20261111_154640` | `TST-9F2M-6630` (the test pressure) alone | FAIL |
| `20261111_155622` | `KVN-4C8R-1194 (with TST-9F2M-6630, dated 2026-01-22)` | PASS |

Both had the guide **outside** the window, so both were guessing; the only difference was whether the guess
was committed or enumerated. `must_contain` accepts the enumeration, so the pass encoded phrasing, not
interpretation. Fix applied: **T2 now asks for the code only and guards the other field of the row**
(`must_not_contain TST-9F2M-6630`), while **T3 keeps its natural phrasing** so a hedge stays visible in the
transcripts. Rule count 9 → 10. Replay of all three transcripts under the new rules: `154640` **6/10**,
`155622` **8/10**, `155646` **8/10** — T2 now fails in every one, always on the same rule.

**Rejected: naming the corpus's own field in the probe (R1).** It looks like the fairer wording, but it
dissolves the cell. Measured BM25 for this probe:

| Probe wording | rows_01 | guide |
|---|---:|---:|
| `"limit"` + strip note | **1** | 10 |
| `"certified value"` + strip note | 3 | **1** |

Naming the field pulls the guide to rank 1, so the schema arrives with the record and there is nothing left to
sever. The term of art therefore stays, and the fact that a model may object to it (the k=15 run did, then
answered correctly) is recorded rather than designed away.

**Live runs under the strict probe** (deepseek-v4.1-flash, `qwen3-embedding-0.6b`, 12 documents / 37 chunks):

| Arm | Run | Score | T2 | T3 | T4 | T5 |
|---|---|---|---:|---|---|---|
| `top_k=5` | `20261111_161728` | **6/10** | FAIL (both rules) | FAIL | PASS | FAIL (`north-riser` absent) |
| `top_k=15` | `20261111_162001` | **6/10** | FAIL (both rules) | FAIL | PASS | FAIL (over-disclosed `MVB-2K9P-6634`) |

Both arms answer `TST-9F2M-6630` alone at T2 — the format constraint works (the answer is a single token) and
the binding is wrong. At k=15 **both guide chunks were in T2's window** (ranks 10 and 13), so that arm is not a
retrieval failure.

**But it is not a clean mis-binding either**, and this is the open issue: "a valve's limit" also reads as a
pressure limit, and the code it chose (`TST-*`) is the pressure. Every wording that removes that reading
retrieves the guide, because the guide is the only document that distinguishes the two fields:

| T2 wording | rows_01 | guide |
|---|---:|---:|
| `"limit"` (current) | 1 | 10 |
| `"rating"` | 1 | **3** |
| `"certified rating"` | 2 | **1** |
| `"rating the team maintains"` | 4 | **1** |
| `"value the certifying team keeps"` | 9 | **1** |

So the severance depends on the probe being underspecified as much as on the corpus split.

**Decision (2026-11-11): accept it, score the pair.** Rewording was measured to dissolve the cell, so instead T3
became the decisive probe and T2 the diagnostic one:

| Probe | Question | Role |
|---|---|---|
| T2 | "What limit does valve VLV-220 carry? Reply with the code only." | **diagnostic** — names a term of art the corpus never binds to an asset |
| T3 | "And valve VLV-247? Reply with the code only." | **decisive** — same row shape, names nothing, so a wrong field cannot be excused |

T3 also carries the strict treatment now (single token + guard on `TST-3M6Q-9058`); while it was loose, a hedge
passed it and it certified nothing. Rules 10 → 11. Verdict classes are read from the pair: **bound** = T3 pass
and T2 pass; **mis-bound** = T3 answered with the pressure token; **indeterminate** = T3 pass, T2 fail (refusal or
noun reading — one binding on a two-way choice is not evidence).

**Live runs under the pair** (deepseek-v4.1-flash, `qwen3-embedding-0.6b`, 12 documents / 37 chunks):

| Arm | Run | Score | T2 | T3 (decisive) | T4 | T5 | Class |
|---|---|---|---:|---|---|---|---|
| `top_k=5` | `20261111_163559` | **8/11** | FAIL — refused | **PASS** (bare `KVN-8T2D-4417`) | PASS | FAIL (wrong family) | indeterminate |
| `top_k=15` | `20261111_163657` | **9/11** | FAIL — refused | **PASS** | PASS | FAIL (wrong family) | indeterminate |

The T2 refusals are the cleanest evidence the confound is real. At k=15 the guide chunks were at ranks **11 and
14** of T2's window and the model replied:

> "The valve records in the documentation contain only four fields — asset identifier, certified value, test
> pressure, and verification date — and none of them is a limit or rating code."

It listed the field order *correctly* and then declined, because "limit" is not one of the corpus's field names.
So T2 was never measuring field position in this model; it measures whether the model will accept "limit" as a
synonym. T3, which asks the same row for the same value with no term of art, passes in both arms — so the cell's
verdict is **indeterminate**, not "bound": one correct pick out of two candidate codes is not evidence.

T5 now fails on the *family* rather than on absence, and quotes the trap in the transcript: it decoded the
north-riser row with the third-riser order "per the field order you described earlier" (T4's own answer, carried
in the thread) and called the block third-riser. That is the `wrong_family` class firing as designed.

**Correction (same day): the k=5 T3 passes above were contamination, not luck.** The conclusion "indeterminate —
one correct pick out of two codes" was wrong. T2's answer names the row's fields ("asset identifier, certified
value, test pressure, verification date" at k=15; "(identifier, KVN code, TST code, and verification date)" at
k=5), and a shared session puts that answer in T3's prompt verbatim (`graph.py:250`), so T3 read the position out
of its own conversation. Four variants at k=5, same corpus:

| Variant | T3 answer | Verdict |
|---|---|---|
| T2 then T3, one thread (runs `163559`, `164916`) | `KVN-8T2D-4417` | PASS |
| T3 with the T2 turn removed | `TST-3M6Q-9058` | FAIL |
| T3 in its own session | `TST-3M6Q-9058` | FAIL (and T2's episode `ep_s2_t1` was the rank-1 memory, 0.7444) |
| T3 asked before T2 | `TST-3M6Q-9058` | FAIL |

Recorded as `failure_taxonomy.t2_leaks_the_field_order`; artefacts in
`archive/2026-11-11_document_split_t2_leak_experiment/`.

### 2.8 Implemented: one probe per scenario (iii), and the positional default it exposed

The pair is now two cells over the same 12 documents:

| Cell | Rules | Role |
|---|---:|---|
| `document_split_binding_01` | 6 | **decisive** — one value probe, asked first, on a *north-riser* record, then the format control |
| `document_split_evidence_01` | 8 | **diagnostic** — the "limit" probe, the family probe, then the format control last |

`bound = binding pass AND diagnostic pass`, two independent runs, so a coin flip passes both with p = 0.25; the
conjunction moved *across runs*, which is what keeps (iii) from being weaker per arm (it is weaker per run — one
50/50 — and that is the whole reason for the second run).

**The split immediately falsified the cell's claim, and the fix was the probe's target, not the corpus.** The
isolated single-probe cell scored **6/6** at k=5 with the guide out of the window (run `171155`). The model was not
lucky: its default is "report the second field of the line", and in the third-riser export the second field *is*
the certified value. Measured on both families:

| Probe | Row | Answer | Guide in window | Verdict |
|---|---|---|---|---|
| VLV-247 (third riser) | `VLV-247 \| KVN-… \| TST-… \| date` | `KVN-8T2D-4417` | no | right, wrong reason |
| VLV-320 (north riser), k=5 | `VLV-320 \| TRQ-… \| MVB-… \| date` | `TRQ-HJZS-1262` | no | position-2 = test pressure |
| VLV-320 (north riser), k=15 | same | `TRQ-HJZS-1262` | **yes** (ranks 10, 12) | same answer with the schema present |

So the decisive probe now asks about a north-riser record (certified value third), where the positional default is
visibly wrong. Recorded as `failure_taxonomy.positional_default` in the binding cell and pinned by
`test_the_probe_targets_the_family_where_position_two_is_wrong`.

**Numbers under the pair** (deepseek-v4.1-flash, `qwen3-embedding-0.6b`, 12 documents / 37 chunks):

| Arm | `document_split_binding_01` | `document_split_evidence_01` | Class |
|---|---|---|---|
| k=5 | **4/6** (`171723`) — T2 `TRQ-HJZS-1262` | **5/8** (`171805`) — T2 pressure code, family probe position-2 | **mis-bound** |
| k=15 | **4/6** (`172039`) — same answer, guide in window | **6/8** (`171852`) — T2 lists both codes (guard fails) | **mis-bound** |

Read: the agent binds by position, not by schema, and the k=15 arm shows it does so with the guide retrieved. That
is a stable, attributable failure at both cutoffs — the thing the cell was built to measure.





**Add the lexical-baseline test (V1) in the same pass.** If BM25 solves a cell, both agents saturate and the
cell cannot discriminate — that check is what prevents repeating the current corpus' failure on a new budget
line.

### 2.2 These cells are not L2b-only: they are the **baseline arm** of the same experiment

All three are valuable — and two of them are *primarily* L2a cells. A delta claim needs the baseline measured
on the **same corpus, same model, same prompt, same budget**; running the new agent alone yields an
uninterpretable number.

| Cell | What L2a (one-shot) tests in it | What L2b adds |
|---|---|---|
| `document_conflict_dilution_01` | read-time disambiguation with **no recency/authority term** in the store (session/turn/source never score) → the canonical naive-RAG failure ("correct value attributed to the wrong sibling") | a second, entity-qualified query |
| `document_prevalence_trap_01` | **top-k dilution**: five stale copies fill a `k=5` window → failure by dilution, not by reasoning. Needs no agentic arm at all to be a result | re-queries, which re-surface stale copies (may make it *worse*) — the shared negative control |
| `document_multihop_bridge_01` | **capability ceiling**: can one retrieval ever co-locate both halves? | query 1 → bridge entity → query 2 |

**Why the five existing chat multihop cells do not cover this** (measured):

| Scenario | User turns / sessions | Retrievable pool at the last probe (L2, other sessions) | Pins | Verdict |
|---|---|---|---:|---|
| `in_context_multihop_01` | 1 / 1 | **0** | no | both premises in one prompt — no retrieval, no memory |
| `temporal_multi_hop_01` | 4 / 1 | **0** (single session ⇒ `exclude_session_id` empties the pool) | no | pure checkpointer/context test |
| `3hop_temporal_01` | 9 / 1 | **0** | no | same; also never been run (0 results in `outputs/`) |
| `cross_thread_derived_01` | 11 / 3 | 10 | no | never been run; no pins ⇒ failure unattributable |
| `derived_belief_revision_01` | 13 / 3 | 10 | no | **one call already carries the premise set** |
| `backward_modus_tollens_01` | 13 / 3 | 10 | no | same |

The decisive datum is in the recorded L2 runs: at `derived_belief_revision_01` T12 the retriever returned
`['ep_s1_t5', 'ep_s2_t2', 'ep_s1_t6', 'ep_s1_t4', 'ep_s1_t3']` — the rule chunk, the revision chunk and three
neighbours, out of a pool of ~7 usable episodes. `backward_modus_tollens_01` shows the same shape. So the
"two-hop" is performed **in-context** by the model, not by retrieval: a single `retrieve(query, 5)` suffices,
and no amount of agentic cleverness changes the outcome. That is why those cells test *compositional
reasoning* (useful, and they need pins) but not *compositional retrieval*.

The new document cell is the only one where the store is much larger than the window **and** the two premises
share no vocabulary, so the composition can only happen through two queries. Cheapest complement, worth doing
in the same sitting: add `must_retrieve_chunk` pins to `cross_thread_derived_01` (it has no runs at all) and
cite the pair in the thesis as *reasoning vs retrieval* composition.

**One-line framing:** same logic, different bottleneck. `cross_thread_derived_01` = retrieval is trivially
sufficient, so the hop is done in-context (tests reasoning); `document_multihop_bridge_01` = retrieval is
*insufficient by construction*, so the hop can only be closed by a second query (tests retrieval
composition). It is not a "denser" version — density alone makes both arms fail equally; what creates the
requirement is **window-sparsity**: store ≫ k plus vocabulary-disjoint premise chunks joined by one bridge
token.



They also serve the rest of the ladder, which is the strongest argument for building them once: L1 fails all
three trivially ("is the answer guessable?" control), L3 is the interesting arm for conflict/prevalence (does
reflection collapse duplicate bindings?), and L4 (defeasible logic) is the natural arm for the prevalence trap
(majority ≠ authority).

### 2.3 The budget confound — the control everyone forgets

L2b with 3 calls × `k=5` sees **15 chunks**; L2a at `k=5` sees 5. Without a budget-matched arm, any L2b win is
equally explainable as *more context*. So each cell runs **three** arms, not two:

| Arm | Retrieval | Why |
|---|---|---|
| `L2a-5` | one call, `k=5` | the default baseline |
| `L2a-15` | one call, `k=15` (≈ calls × k) | **the null hypothesis for agency**: "just retrieve more" |
| `L2b-3x5` | ≤3 tool calls, `k=5` each | the agentic arm |

Expected pattern, which is also the acceptance criterion for the corpus design:

| Cell | L2a-5 | L2a-15 | L2b-3x5 |
|---|---|---|---|
| multihop bridge | fail | **fail** (gold halves must live in *separate* topical clusters, or breadth solves it) | pass if reformulation works |
| conflict dilution | partial | **worse than k=5** (all siblings enter the window) | partial |
| prevalence trap | fail | **worse** (more stale copies) | partial / fail |

If `L2a-15` matches `L2b-3x5` on the bridge cell, the corpus is not testing agency — separate the two gold
clusters further apart in vocabulary before trusting any L2b result.


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

Then run the paired experiment: **L2a (one-shot) vs L2b (agentic), same cells, same model, same prompt**, with
**three arms — `L2a-5`, `L2a-15` (budget-matched one-shot: calls × k), and `L2b-3x5`** (§2.3), because a
two-arm design cannot separate *better queries* from *more context*. Report accuracy, pins, calls/probe,
tokens, latency and the F0–F4 distribution.

**Naming — keep the ladder honest.** Your planning note proposes `level_3_agentic_rag`. Don't: in SPEC §1 and
P2, **L3 is the semantic store (reflection / CRUD / belief revision)**, so reusing that name would put a
*retrieval strategy* on the *memory-tier* axis and make every slide ambiguous. Register it as
`level_2b_agentic_rag` (and add the `2b` alias so `--level 2b` resolves — `resolve_level_ids` currently only
accepts digits). Then the demo reads "L2a vs L2b", and L3 stays reserved for CRUD.

**Demo vehicle.** Do not use `in_context_multihop_01` (single turn — nothing to retrieve, so L2a and L2b are
identical), `cross_thread_derived_01` (chat channel, no documents, no pins) or `top_k_saturation_01` (no pins,
so "one retrieve does not suffice" is inferred, not proven). `document_multihop_bridge_01` *is* the demo
vehicle, and it can be authored in a few hours with 8 shared + 3 cell documents (~40 chunks).



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

---

## 9. Status 2026-11-12 — Phase 0 shipped, plus the assertion fixes

### 9.1 Phase 0 (the contract) — **DONE**

`TODOs/l2b-telemetry-contract-todo.md` is implemented. What landed:

| File | Change |
|---|---|
| `core/schemas.py` | `L2AgentState` + `TurnResult` gained `retrieval_calls`, `retrieval_call_count`, `retrieval_skipped`, `tool_call_cap_hit`. `BenchmarkRunSummary` gained `total_tool_calls`, `avg_calls_per_retrieval_turn`, `tool_call_cap_hits`, `calls_per_probe`, `retrieval_call_efficiency`. |
| `core/metrics.py` | `union_retrieved_episodes()` (the match-set policy), `pins_per_call()`, `mean_calls_per_probe()`. |
| `core/runner.py` | `_read_l2_telemetry` reads the four channels guarded (absent ⇒ `[]`/`0`/`False`); `evaluate_assertions` now receives the **union** of every call; the summary aggregates calls, cap hits and efficiency. |
| `agents/__init__.py` | `LEVEL_ALIASES = {"2b": "level_2b_agentic_rag"}` resolved *through* `AGENT_REGISTRY`, so `--level 2b` fails cleanly today and works the moment the agent is registered. `_available_levels()` lists it as `2b`. |
| `tests/test_l2b_telemetry.py` | 22 tests: guarded reads, union across first/last call, dedup-by-chunk_id, tolerant `chunk_ids` fallback, no-tool = `None` efficiency, alias resolution. |

`pytest -q` = **783 passed**. The three policy decisions encoded (and tested) are union matching,
counted-not-collapsed calls, and `None`-not-`0.0` efficiency.

### 9.2 The three pre-L2b cells — assertion fixes

Two pins were **rewritten** and one **added**; one cell was deliberately left alone.

| Cell | Change | Why |
|---|---|---|
| `document_multihop_bridge_01` | T2 hops → `asset_registry_01_c0` / `calibration_register_02_c1` | the document-level pin was satisfied by a **145-char header chunk** with no figure (`calibration_register_02` splits to `[145, 1113, 759]`). Count unchanged (4), so no denominator arithmetic moved. |
| `document_split_binding_01` | T2 also pins `split_legend_01` (2 → 3 declared) | the row cannot certify the probe; pinning only the row read **8/8 = 100 % recall while T2 failed 16/24**. |
| `document_split_evidence_01` | **unchanged** | the absent guide *is* the construct (`failure_taxonomy.guide_unreachable`); the ratchet `test_the_guide_probe_runs_after_the_value_probes` forbids it. Locked by a new test so it cannot be re-pinned by accident. |

**Measured effect (run `20261112_132030`, k=5):** pass 4/7 = 57.1 %, recall **2/3 = 66.7 %**, with the T2
retrieval window **byte-identical** to the pre-edit run `20261111_192203`. The agent did not change; the
metric did. `retrieval_recall.py --scenario-prefix document_split` now separates the arms on its own:

```
document_split_binding_01  k=5   3 runs  hit 6  pins 7   85.7%   (2 old 2-pin runs + 1 new 3-pin run)
document_split_binding_01  k=15  2 runs  hit 4  pins 4  100.0%   <- the guide is in-window at k=15
```

### 9.3 Three corrections to this plan's own expectations

1. **`document_split_binding_01` is not a 3-arm discriminator.** Its `failure_taxonomy.field_misbinding`
   already records the guide at **ranks 10 and 12 at k=15**, and §9.2's report confirms 4/4 pins there.
   Breadth substitutes for the second call. The pin buys **attribution** (`recall 1/2` ⇒ schema absent,
   `2/2` + fail ⇒ misread), not a k15-proof bridge.
2. **`document_conflict_dilution_01` is saturated** (18/18 assertions, 6/6 pins, gold at rank 1). It is a
   *precision* cell, where L2b can only lose. Do not use it as the L2b evidence.
3. **Therefore exactly one cell carries the agency claim: `document_multihop_bridge_01`** (hop 2 at rank
   42/42, 0/15 retrieved slots in the pilot matrix). It is also the only one with a ceiling twin
   (`in_context_bridge_01`).

### 9.4 What remains

| Phase | State | Blocker |
|---|---|---|
| 0 — contract | ✅ done | — |
| 1 — 4 cells | ✅ all four in the tree | cells 2–3 are **not** L2b discriminators (§9.3) |
| 1b — L0/L1 baselines | ⬜ not run | 18/19 multihilo cells have no measured amnesia baseline; cheapest fix after the contract |
| 2 — L2b agent | ⬜ next | the contract is frozen, so runs made from here are comparable |
| 3 — widen (D3, M9 generator, numeric/`must_store` assertions) | ⬜ **after** Phase 2 — see §10–11 | needs the agent's failure modes |

---

## 10. Phase 3 (detail) — the D3 real-document corpus arm

`DIFFICULTY_AUDIT.md` §A5 and the multihilo difficulty report land on the same gap: **0 of 19 cells satisfy
FEB-30's D3 requirement of facts injected into *real human* documents** (~90 % authentic foreign text). The
60-document corpus is 3 159 sentences of which 212 are unique (**6.7 %**) — the inverse of what D3 asks. §2.3
proposes LLM-drafted static prose, which fixes the ratio but not the criticism (*"the noise is still
model-generated"*).

### 10.1 What a real document buys for free

| D3 needs | How it is faked today | What an authentic document gives |
|---|---|---|
| lexical near-duplicates | 5 hand-written skeletons (Jaccard 0.90–0.93) | **real tables**: 20–40 authentic rows of identical shape |
| supersession with authority | invented precedence rules | **real structures**: USPTO certificates of correction, CFR "as amended at 55 FR …" |
| human register | ~14 sentence templates | 100 % human by construction |

**Corollary: never invent the noise — choose a document whose noise already works, and inject the fact
where the real document already has competition (inside a table), not in a loose paragraph.**

### 10.2 The four candidate cells

| # | Cell | Real host | Injection | Inference required | Metric |
|--:|---|---|---|---|---|
| D3-1 | **real table** — cheapest, and adds table extraction, a mode untested today | 1 PD document with 20–40 homogeneous rows | **1 row** with the opaque token | the body names the row by *description* → resolve description→row→value among 40 real rows | `must_contain` + 3–4 `must_not_contain` on neighbouring rows + chunk pin |
| D3-4 | **cross-domain bridge** — highest value: a *replication* of `document_multihop_bridge_01` in D3 | 2 PD documents from different domains with a real cross-reference | identifier in A, figure in B; **B shares no probe word** | 2-hop composition | 2 pins on the same probe |
| D3-2 | **certificate of correction** — real supersession, zero fabricated authority | real patent + its real USPTO Certificate of Correction | the correctable constant + its correction line | must know the certificate supersedes | both chunks pinned + `must_contain` corrected + `must_not_contain` original |
| D3-3 | **patent family** — real near-duplicates (parent + continuations share 80–95 %) | 3–4 same-family patents | the paragraph that *differs* | which member is in force / which figure applies | per-document pins + sibling guards |

Build order: **D3-1 → D3-4 → D3-2 → D3-3**.

### 10.3 Source families (all legally modifiable)

| Family | Licence | Native text | Risk |
|---|---|---|---|
| USPTO patents (full-text HTML) | public domain | ✅ | family = long documents → volume `>D3` |
| eCFR / Federal Register | US Government work (17 U.S.C. § 105) | ✅ XML/HTML | heavy cross-jargon between sections |
| NIST SP 800 / NASA TM | US Government work | ⚠️ PDF extraction | **PDF is the main operational risk** |
| Gutenberg | text is PD, the eBook carries terms | ✅ txt | strip the boilerplate |
| RFC (IETF) | reproduce **unmodified** only | ✅ txt | **modifying may breach the licence → discard for injection** |

**Recommendation: patents + eCFR**, both text-native and unambiguously modifiable. No PDFs in the first
iteration: extraction will cost more than the cell.

### 10.4 Injection protocol (verisimilitude is what makes it D3 or a fake)

1. Freeze the source: URL, retrieval date, licence basis and `sha256` in `PROVENANCE.md`. Commit the
   **diff** of the injection — a reviewer must be able to see what is authentic without reading the YAML.
2. ≤10 % of the document's tokens, **one injected unit per document** (a row, a paragraph, a certificate line).
3. Never at the start of a section or next to a heading (models use heading proximity as a cue).
4. Never across a chunk boundary: the token must land in exactly one chunk.
5. Imitate the register: hard-wrap width, quote/dash style, date and citation format of the host.
6. **Tell detector:** every word of ≥6 letters in the injected text must also occur in the host's authentic
   text, except the opaque token. A word that exists only in the injected paragraph gives the injection away.

### 10.5 The D3 admission gate — six measurable criteria

Without these, "real documents" is a vibe. A new cell must fail the build if it does not comply:

| # | Criterion | Gate | How it is tested |
|--:|---|---|---|
| 1 | `foreign_text_ratio` | ≥ **0.85** authentic chars / total | `metadata.calibration` + assert |
| 2 | unique-sentence ratio | ≥ **0.50** (the generator gives 0.067) | assert over the corpus |
| 3 | **BM25 rank of the gold chunk for the composed probe** | > `k_max` used | offline, in the cell's test file |
| 4 | answer token in exactly 1 chunk | — | mirror of the bridge's existing test |
| 5 | **tell detector** (§10.4.6) | 0 exclusive words | assert |
| 6 | sentence length inside the host's p5–p95 | register | assert |

Criterion 3 is the one that saves a wasted run: it is §2.9's falsification applied *before* spending money —
if a lexical baseline finds the gold at rank 1 with the probe, the cell is a lookup, not D3.

### 10.6 The band conflict, and why the construct wins

FEB-30's D3 asks for **4–5 documents / 600–1000 words**. One CFR section alone is 800–3000 words. Options:
(A) excerpts, in-band but lose document integrity; (B) full documents, `>D3` volume but the 90 % claim is
literal; (C) hybrid. **Recommendation: B for the replication cell (D3-4), A for the rest** — and do not
sacrifice the construct for the band, since `document_dilution_01` is 354 words and is one of the best cells
in the repo (§A4: word count alone is the wrong calibration variable).

### 10.7 Anti-memorisation guard

A risk real documents add that synthetic ones do not: the model may know the text. It becomes a free
measurement — choose an injected fact that **contradicts** a real value in the document (which is exactly
what a certificate of correction does), and add `must_not_contain: <real value>`. A model answering from
memory then fails an explicit guard instead of scoring an unearned pass.

---

## 11. Sequencing decision — D3 after L2b (record)

**Decision:** build L2b before the D3 corpus, with three cheap exceptions done up front.

Why after: (i) the telemetry contract is upstream of every run, D3 included; (ii) D3's hop count, `k` and
noise density depend on L2b failure modes you do not have yet — a gold at rank 42 is *unsolvable* if the
agent cannot reformulate; (iii) §0 already sequences contract → discriminating cells → agent → *widen*;
(iv) adding scenarios later invalidates nothing, since `pass_rate` is per-scenario and the contract is frozen.

Why the anti-ceiling argument does not apply here: it does, but only to the D16 corpus cells. Measured today,
the pre-L2b cells are **not** saturated (bridge 8/10, binding 4/7), so two informative cells exist. The
cheapest insurance against a wasted sweep is the `L2a-15` arm, not a new corpus.

The three exceptions taken now (all cost no model budget): **decide the tool's provenance interface**
(content-only vs content+provenance — D3-2 forces it, and it changes the tool signature); **write the
admission gate of §10.5 as a spec**; **source and pin 4–5 PD documents** (`sha256` + URL + licence), which is
the long-lead item and can be done while sweeps are in flight.

**Falsification trigger for moving to D3** (same shape as §2.9):

- If the 3-arm sweep separates `L2a-15` from `L2b-3×5` on the bridge → agency is proven with today's cells,
  and D3 becomes a **robustness claim** ("the finding survives real noise"), defensible but optional.
- If it does **not** separate → do not go to D3 yet: the defect is in the agent or in the telemetry, and a
  harder corpus would only hide it behind a wall of noise.





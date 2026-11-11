# Are the synthetic documents valid for testing? — Validity Note

**Companion to** `DIFFICULTY_AUDIT.md` (§3.A5–A8, steps M4/M8/M9)
**Question answered:** is the generated document corpus a valid instrument, and should the generation
methodology change?
**Evidence:** every figure below was measured on the committed corpus with the harness' own chunking
(`chunk_size=600`, `chunk_overlap=200`) and a dependency-free BM25. Reproduction snippets are in §8.

---

## 1. Short answer

**Synthetic documents are valid — but not for the claim they are currently used to support.**

Synthetic corpora are the standard instrument for exactly this kind of work (NIAH / RULER / BABILong
lineage) because they buy three things real documents cannot: **causal attribution**, **controlled
invariance**, and **bit-level reproducibility**. Nothing about "synthetic" invalidates the current
cells. What invalidates the *D3 claim* is the **claim grammar** of the generator: the haystack contains
no competing claims, the needle entity is lexically unique, and the probe hands the model the exact
lookup token. The result is a keyword search dressed as a memory test.

| Decision | Verdict | Reason |
|---|---|---|
| **Keep the synthetic corpus methodology** | ✅ **Yes** | It is the only way to hold content constant while varying corpus size (8 → 60 → 200) and to prove the gold chunk reached the prompt |
| **Keep `document_corpus_size_*` as a D3 "noise" cell** | ❌ **No** | BM25 solves it 24/24 at k=5 (§4); 99 % of chunks state no fact; no chunk competes for the probed predicate |
| **Re-type `document_corpus_size_*` as a plumbing / attribution control** | ✅ **Yes** | That is what it really measures, and it measures it well (2 933 chunks, ingestion + top-k + provenance) |
| **Change the generator's claim grammar** | ✅ **Yes — the main fix** | Add claim density, competing siblings, superseded versions, paraphrase pools, position sweep, and a probe style that does not leak the entity ID |
| **Replace synthetic with real documents** | ⚠️ **Optional, one small arm** | Worth 20–40 authentic docs as an external-validity anchor; not a replacement, since realistic noise cannot be held constant across corpus sizes |
| **Discard existing runs?** | ❌ **No** | Relabel them; the numbers stay valid for the claims they actually support |

---


## 2. Why synthetic is the right instrument here (three properties worth defending)

1. **Causal attribution.** Every probe asks for an opaque answer token (`EOW-WTEL-5216`) that occurs once
   in the corpus and cannot be inferred. So "the answer is in the response" ⇔ "the gold chunk reached the
   prompt". This is what lets `retrieval_recall` be causal rather than inferred — a genuinely strong
   property and the reason *your* L2 failure analysis holds up.
2. **Controlled invariance.** The same seed produces **byte-identical documents** across sizes: the only
   difference between the 8-, 60- and 200-document cells is the number of competing chunks. There is no
   way to obtain 199 additional authentic documents with identical style, length and topic distribution;
   a real corpus would confound "more documents" with "different documents".
3. **Reproducibility and cost.** Deterministic, committable (~1.2 MB for 200 docs), no licensing, no
   drift. A reviewer can regenerate the exact corpus with one command and a seed.

**Therefore the design intent is sound.** The question is not *synthetic vs real* but *which synthetic
distribution*, and whether the probes extract the property the tier claims.

---

## 3. What the current corpus actually is (measured)

### 3.1 Information content

| Measure | 8 docs | 60 docs | 200 docs |
|---|---:|---:|---:|
| Documents | 8 | 60 | 200 |
| Characters | ~48 k | ~360 k | ~1.2 M |
| Sentences | 427 | 3 159 | 10 513 |
| **Unique sentences** | 40 (9.4 %) | 212 (**6.7 %**) | 232 (**2.2 %**) |
| Chunks (600/200) | 115 | 880 | 2 933 |
| Lexically unique chunks | 115 (100 %) | 880 (100 %) | 2 933 (100 %) |

The two uniqueness rows disagree, and the reason matters: chunks are distinct **permutations** of ~14
template sentences plus a `Corpus Note NNN` header, not distinct *information*. Removing the reference
labels does not collapse chunk uniqueness (measured: still 100 %), because the recombination itself
varies. So the corpus is not a duplicate-detection test either — it is a **low-entropy text generator**
whose surface diversity is combinatorial.

### 3.2 Claim density — the decisive measure

A "claim" here means a chunk that states an `(entity, attribute, value)` fact. Counting chunks that carry
an opaque code token (`XXX-XXXX-NNNN`):

| Corpus | Chunks | Chunks with ≥1 claim | Share | Chunks containing the probe's entity |
|---|---:|---:|---:|---:|
| `d16_corpus_008` | 115 | 9 | 7.8 % | **1** |
| `d16_corpus_060` | 880 | 10 | **1.1 %** | **1** |
| `d16_corpus_200` | 2 933 | 10 | **0.34 %** | **1** |

Two consequences:

- **99 % of the haystack states no fact.** It cannot confuse the model about the target predicate,
  because it never asserts anything about it — or about anything else. Distractor *robustness* and
  *context blending* are therefore untestable on this corpus.
- **The gold entity occurs in exactly one chunk.** Retrieval degenerates to a unique-key lookup.

### 3.3 Lexical-baseline test (the number that settles it)

BM25 over the same chunking, no embeddings, no semantics:

| Corpus | Gold at rank 1 | Gold in top-5 |
|---|---:|---:|
| `d16_corpus_008` | 7/8 | **8/8** |
| `d16_corpus_060` | 6/8 | **8/8** |
| `d16_corpus_200` | 7/8 | **8/8** |
| **Total** | **20/24** | **24/24** |

A pure keyword engine with zero semantic capability retrieves the gold chunk every time. The corpus-size
axis therefore measures **ingestion throughput and top-k plumbing**, not retrieval quality; a flat curve
from 8 → 200 documents is a property of the probe, not of the memory architecture.

---


## 4. The five specific defects (in order of impact)

| # | Defect | Evidence | Consequence |
|---|---|---|---|
| **D-1** | **Haystack is claim-free** | 1.1 % of chunks carry a claim | No dilution, no ambiguity, no context blending possible |
| **D-2** | **Needle entity is lexically unique** | entity occurs in 1 chunk out of 880 | Retrieval = exact-key lookup; BM25 Hit@5 = 100 % |
| **D-3** | **Probe leaks the lookup key** | the probe string contains `LOOP-420`, i.e. the rare token itself | The hardest part of retrieval (mapping a description to an identifier) is removed by construction |
| **D-4** | **No competing/superseded value for the probed predicate** | no second value for `(LOOP-420, calibration_code)` anywhere | Revision, authority, prevalence and recency are untestable in this corpus |
| **D-5** | **No position/depth axis** | needle always in section 2 (fixed) | The most standard IR difficulty knob (needle depth) is missing; only corpus *size* varies |

Defects D-2 and D-3 are *deliberate* and defensible for the attribution property (§2.1) — the error is
not building them, it is labelling the result "D3". D-1 and D-4 are the real methodological gaps: a
corpus with 10 facts among 880 chunks cannot evidence extraction difficulty no matter how large it is.

---

## 5. What the corpus *is* valid for today (keep, relabel, report)

Re-type `document_corpus_size_008/060/200_01` from `retrieval_limit` (D3) to **`harness_validation` /
retrieval-attribution control** and keep publishing the numbers, because they support four legitimate
claims:

1. Document ingestion scales: **200 documents / 1.2 M chars / 2 933 chunks** ingested and indexed, with
   per-turn provenance (`ingested_chunk_ids`, `documents_ingested`).
2. The retrieval + attribution path works end-to-end at scale, so an L2 "miss" elsewhere is not plumbing.
3. The answer-token design is sound: a correct answer *proves* the chunk arrived (no lucky guessing).
4. `retrieval_recall` behaves as designed on a controlled pool — useful as the *positive control* against
   which a real D3 cell is compared.

What it must stop being used for: any claim about D3 extraction difficulty, noise robustness, distractor
resistance, or "semantic retrieval degrades with corpus size".

---

## 6. Redesigned generator — concrete specification

Keep the file layout, the seed discipline and the byte-identical-across-sizes property. Change **what is
generated**.

### 6.1 New parameters (`scripts/generate_corpus.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--claims-per-doc` | `2` | number of `(entity, attribute, value)` claims planted per document (was: 0, except the single needle) |
| `--competing-siblings` | `3` | for each claim, plant K *sibling* claims: same attribute, different entity, different value (real dilution) |
| `--competition-mode` | `sibling` | `sibling` (different entity) · `supersede` (same entity, older value + authority/date) · `negate` (same entity, explicit refutation) |
| `--superseded-rate` | `0.25` | share of claims that are later superseded by a dated document (makes revision testable) |
| `--paraphrase-pool` | `12` | realizations sampled per template sentence (kills the 14-template signature) |
| `--doc-kind` | `mixed` | `memo, email, changelog, runbook, release_notes, audit_table` — each with its own layout and register |
| `--needle-depth-quantile` | `0.5` | where in the corpus the gold lands (0.1 / 0.5 / 0.9 sweep = the position axis) |
| `--probe-style` | `entity_id` | `entity_id` (current, lookup) or `semantic` (describe the entity, never name it) |
| `--lexical-uniqueness` | `strict` | `strict` = gold entity token appears once (attribution) · `realistic` = 2–5 mentions across documents (forces discrimination) |

### 6.2 Claim grammar (what each document asserts)

```yaml
# The unit the generator must produce — a claim, not boilerplate:
- id: C_LOOP420
  entity: "cooling loop LOOP-420"        # or a *description* only, for --probe-style semantic
  attribute: calibration_code
  value: EOW-WTEL-5216
  stated_in: {doc: corpus_017, section: 4, depth_quantile: 0.62}
  siblings:                               # real competition, same attribute, different entity
    - {entity: "cooling loop LOOP-969", value: BQR-485R-6500, stated_in: corpus_042}
    - {entity: "cooling loop LOOP-118", value: "vote pending", stated_in: corpus_088}
  superseded_by:                          # optional: makes revision / authority testable
    - {doc: corpus_150, value: KTX-9033-P771, date: 2026-02-11, authority: vendor_bulletin}
  probe: "What is the calibration code for the cooling loop in the west aisle?"
```

### 6.3 Before / after, same seed budget

| Property | Current corpus | Redesigned corpus (targets) |
|---|---|---|
| Unique-sentence ratio | 0.067 | **0.50–0.90** |
| Claim-carrying chunks | 1.1 % | **≥ 5 %** (≈ 44 chunks at 60 docs) |
| Competing values per gold predicate | 0 | **≥ 2 (D2), ≥ 5 (D3)** |
| Gold-entity occurrences | 1 | **2–5** (`realistic`) |
| Probe leakage of the gold token | yes (by design) | only in the `entity_id` variant (kept as the D1 control) |
| Needle depth | fixed §2 | swept 0.1 / 0.5 / 0.9 as separate reported cells |
| BM25 Hit@5 | **1.00** | **≤ 0.60** for D2/D3 cells |
| Byte-identical across sizes | yes | **yes** (keep — same seed reproduces all sizes) |

---


## 7. The validity contract — tests that replace "it looks realistic"

A corpus is calibrated when these six assertions hold. They are cheap, offline and deterministic, so they
run in CI exactly like the existing hygiene tests.

| Test | Threshold | Current status |
|---|---|---|
| **V1 — lexical non-solvability** (BM25 over the same chunking must not solve a D2/D3 cell) | `hit@5 ≤ 0.60` | ❌ **1.00** ⇒ auto-downgrade to D1 |
| **V2 — claim density** (chunks in the gold's attribute family / all chunks) | ≥ 0.05 | ❌ 0.011 |
| **V3 — competition** (chunks with the same attribute and a different value, per gold claim) | ≥ 2 (D2), ≥ 5 (D3) | ❌ 0 |
| **V4 — separability floor** (hardest distractor must be *close* to the gold: Jaccard ≥ 0.5 or cosine ≥ 0.85) | present | ⚠️ top neighbour 0.63–0.76 — passes, but no artefact records it |
| **V5 — diversity band** (unique-sentence ratio) | 0.50–0.90 | ❌ 0.067 / 0.022 |
| **V6 — attribution invariance** (same seed ⇒ byte-identical docs across sizes; gold token unique in the `strict` variant) | holds | ✅ passes |

**V1 is the highest-value addition** because it is a *calibration* test rather than a description: it turns
"is this corpus hard?" into a pass/fail property that cannot be argued with. Sketch:

```python
# tests/test_corpus_difficulty.py  (no new dependency)
def test_lexical_baseline_cannot_solve_d3():
    for scenario in _scenarios_with_documents():
        cal = scenario.metadata.get("corpus_difficulty", {})
        if cal.get("tier") not in {"D2", "D3"}:
            continue
        chunks = _chunks_from(scenario)              # same splitter + sizes as the backend
        golds = _gold_chunk_indices(scenario, chunks)
        hits = sum(_bm25_rank(chunks, probe, k=5) in gold for probe, gold in golds.items())
        assert hits / len(golds) <= 0.60, (
            f"{scenario.id}: BM25 solves {hits}/{len(golds)} probes at k=5 — the cell is a "
            "lookup, not an extraction test. Re-tier it to D1 or regenerate the corpus."
        )
```

Recommended trigger: **run V1 at generation time** and write the result into `corpus_manifest.json`
(`lexical_baseline: {hit_at_1, hit_at_5, n_probes}`), so a corpus' difficulty travels with the data
instead of living in a comment.

---

## 8. Reproduction (exact commands used for §3)

```python
# §3.1 / §3.2 — information content and claim density
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re, pathlib
S = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=200)
files = sorted(pathlib.Path("datasets/documents/d16_corpus_060").glob("*.md"))
chunks = [c for f in files for c in S.split_text(f.read_text(encoding="utf-8"))]
claim = re.compile(r"\b[A-Z]{3}-[A-Z0-9]{4}-\d{4}\b")
len(chunks), sum(1 for c in chunks if claim.search(c))     # -> 880, 10

# §3.3 — lexical baseline (BM25, k1=1.5, b=0.75, dependency-free):
# gold = chunk containing the needle answer token; rank it per manifest probe
# -> Hit@1 20/24, Hit@5 24/24
```

---


## 9. Optional real-document arm (external-validity anchor)

Not a replacement — an **anchor**, because it is immune to the "your generator encoded its own
assumptions" objection.

| Choice | Cost | Notes |
|---|---|---|
| 20–40 public-domain documents (RFCs, Kubernetes/Postgres docs, SEC 10-Ks, arXiv abstracts) | ~200–400 KB committed, one-off curation | authentic multi-author noise, real boilerplate, real vocabulary drift |
| Hand-planted needles (paraphrased, not verbatim) | ~1 h per 10 docs | keeps determinism: files are committed, so runs stay reproducible |
| One D3 cell per theme (revision, needle, dilution) | 3 scenarios | reported separately as the "real-corpus arm", never mixed into synthetic aggregates |

If time is short, the two cheapest realism wins that need **no** real documents are `--paraphrase-pool`
and `--doc-kind` (§6.1): they remove the "14-template signature" a reviewer will notice first.

---

## 10. Migration plan for the existing artefacts

| Step | Action | Effort |
|---|---|---|
| 1 | Add `metadata.corpus_difficulty` with the V1–V6 numbers to the 10 document scenarios (measure once, commit the numbers) | 2 h |
| 2 | Re-type `document_corpus_size_*` to `harness_validation`; exclude them from D-tier aggregates and from any "noise robustness" claim; keep their results as the plumbing control | 1 h |
| 3 | Keep `document_authority_vs_recency_01`, `document_effective_dates_01`, `document_prevalence_update_vs_distractor_01` as the **transitional** D3 family with an explicit caveat in `notes/`: the inference is real, the noise is synthetic and 99 % claim-free | 1 h |
| 4 | Implement `--claims-per-doc`, `--competing-siblings`, `--probe-style semantic`, `--needle-depth-quantile`; regenerate `d16_*` | 1–2 d |
| 5 | Regenerate D2/D3 cells until V1 ≤ 0.60, V2 ≥ 0.05, V3 ≥ 2/5, V5 in band | 0.5 d |
| 6 | Re-run L0–L2 on the regenerated cells; keep the old curves labelled "lexical-lookup control (corpus v1)" | 0.5 d + API cost |
| 7 | (Optional) add the 20–40-document real arm | 1 d |

**No existing run is wasted**: everything measured on corpus v1 stays valid as a plumbing/attribution
result — exactly the sanity control a thesis needs before reporting memory-quality curves.

---

## 11. Bottom line

Synthetic documents are **valid** and are the right tool for attribution and controlled scaling. The current
corpus is not *too synthetic* — it is **too empty**: 10 facts among 880 chunks, a lexically unique needle,
and a probe that hands over the lookup key, which is why a keyword engine reaches Hit@5 = 100 %. Two changes
fix it without giving up reproducibility: **relabel** the current cells as what they are (plumbing /
attribution control), and **upgrade the generator's claim grammar** so the haystack contains competing and
superseded claims, paraphrased surface forms, and a needle depth that is swept rather than fixed. Add V1
(BM25 must fail) as a CI test and the D2/D3 labels become defensible numbers instead of assertions.


## 12. What new documents unlock (capability catalogue)

The current corpus can only ask **"did the gold chunk arrive?"**. Every question of the form **"did the
agent pick the right claim among plausible claims?"** is unreachable, because the haystack asserts nothing
(1.1 % claim-carrying chunks), no two chunks compete for the same predicate, and the probe hands over the
unique key.

Each entry below states **what becomes testable**, **why it is impossible today** (measured reason),
**what the document must contain**, and **which assertion/metric reports it**.

### 12.0 Three prerequisites that are *not* corpus problems

These block whole categories regardless of how good the documents are — fix them alongside the corpus:

| Prerequisite | Current state | Blocks |
|---|---|---|
| **In-text provenance** | the injected memory block is **content only**: no `chunk_id`, `source`, `score`, rank or `[Memory #N]` (`_MEMORY_SEPARATOR`; `test_no_provenance_fields_leak_into_the_prompt`) | any test of *source attribution / citation*: the model cannot attribute a value to a document whose name it never sees. Fix: put provenance in the **body** (letterhead, doc-id line, signature block, a "Vendor Bulletin — …" heading), as real documents do |
| **Set-recall metric** | `must_retrieve_chunk` is scored per rule; `retrieval_recall` is rule-level | *compositional* retrieval ("did **both** gold chunks arrive?" as one unit) |
| **Numeric / structured assertions** | `must_contain` substring only | numeric equivalence, units, rounding, table-row reconstruction, negation scope |

### 12.1 Tier 1 — unlocks the L2-vs-L3 claims the thesis actually needs

**T1.1 Conflict resolution under dilution (competing values).**
*Today:* zero chunks hold a competing value for a probed predicate; the closest cells
(`entity_binding_01`, `document_dilution_01`) run on chat turns or a 354-word memo set.
*Documents:* for each gold claim, K siblings — same attribute, different entity (optionally the same
entity with an older value).
*Reports:* accuracy **and** retrieval precision (`must_not_retrieve_chunk` on the siblings), decomposed
into *wrong sibling retrieved* / *right chunk + wrong value read* / *right chunk + right value + wrong
entity attributed*.
*Discriminates:* read-time disambiguation — exactly L2's weak point, since its store has no
recency/authority term (session/turn/source never score).

**T1.2 Supersession at corpus scale (revision under dilution).**
*Today:* one hand-written 450-word addendum cell; supersession exists nowhere in the corpus, so "does the
architecture revise?" is only testable at 7 chunks with zero competitors.
*Documents:* the same `(entity, attribute)` in two dated documents, the second authority-tagged.
*Reports:* current-value accuracy L2 vs L3 (append-only must resolve at read time; L3 may mutate), plus
`must_store` "stale binding absent" once H2 lands.
*Discriminates:* whether revision survives 50 competing chunks, and whether architecture or model is doing
the revising.

**T1.3 Prevalence / staleness trap (majority bias).**
*Today:* one document says the value, so majority vote cannot mislead — the failure mode is unobservable.
*Documents:* the stale value restated in 5 documents, the current one once (dated + authoritative), plus a
symmetric 1-vs-1 control.
*Reports:* stale-answer rate against the control.
*Discriminates:* majority/frequency bias, and whether an L3 store that collapses duplicates resists better
than L2, which re-injects all five copies.

**T1.4 Multi-hop cross-document composition (bridge entity).**
*Today:* every needle is self-contained in one chunk; no answer requires two documents, so retrieval
completeness is never measured.
*Documents:* claim A in doc X, claim B in doc Y, answer = f(A, B), linked by a shared bridge entity.
*Reports:* set-recall over the gold pair (prerequisite 12.0), then accuracy conditioned on
*both / one / neither retrieved*.
*Discriminates:* top_k sufficiency and the read-time join — the regime where RAG fails silently while
single-chunk recall looks perfect.

**T1.5 Evidence split across a chunk boundary (structural failure).**
*Today:* needles are short whole sentences planted intact; the 600/200 splitter never separates a subject
from its value, so the failure mode is untested.
*Documents:* a claim whose subject, qualifier and value land in different chunks (long sentence, or a table
row detached from its header), plus the same claim re-packaged to fit one chunk as the control.
*Reports:* "value retrieved but subject missing" rate, then accuracy.
*Discriminates:* chunking strategy — the commonest RAG defect, invisible under `chunk_size=600` today.

**T1.6 Source attribution / citation.**
*Today:* structurally impossible (12.0) — the model never sees document identity.
*Documents:* in-text provenance (letterhead, doc id, signature) with two sources disagreeing on one value.
*Reports:* does the answer name the correct source (`must_contain: "vendor bulletin"` plus
`must_not_contain` of the wrong source)?
*Discriminates:* whether the agent can *justify* a value — what an L4 defeasible KB requires.


### 12.2 Tier 2 — properties that become measurable but are not on the critical path

| # | Property | Why it cannot be tested today | Document requirement | Assertion / metric |
|---|---|---|---|---|
| **T2.1** | **Retrieval precision & abstention at scale** | 0 `must_not_retrieve_*` rules exist and the corpus holds no *tempting* false positive (nothing deprecated, out-of-scope or other-tenant) | deprecated versions, an out-of-scope note, a sibling-tenant document that is topically identical | `must_not_retrieve_chunk` / `_source`, `retrieval_precision`, abstention accuracy |
| **T2.2** | **Document-borne prompt injection** | injections exist only in chat (`memory_injection_01`); no document is adversarial, so the ingest → retrieve → re-inject channel is untested | one memo carrying an embedded instruction block | `must_not_contain` on the hijack token; same guard at L1, which never stores it |
| **T2.3** | **Duplicate crowding / de-duplication** | claim-free filler repeats *sentences*, not *claims*; no claim is stated twice, so top-k can never be wasted on repeats | the same claim paraphrased in 3–5 documents | accuracy vs k, plus `must_not_retrieve_chunk` over redundant copies |
| **T2.4** | **Alias / coreference across documents** | entities are unique opaque IDs (`LOOP-420` appears once); no alias, no description-only reference | the same entity as `LOOP-420` / "the north cooling loop" / "the west-aisle loop" in three documents | `must_contain` the value, `must_not_contain` the cousin entity's value |
| **T2.5** | **Packaging axis** | every document is 6 000 chars, one genre, one structure — length/form is held constant, so its effect cannot be isolated | the same claim as a 200-w memo, a 1 000-w report, a changelog line and a table row | recall/accuracy per packaging; `context_words` becomes a controlled variable |
| **T2.6** | **Time-scoped validity** | values are timeless; "true on date D" vs "true today" is unexpressible, and `document_effective_dates_01` flags wall-clock dependence as a known risk | `valid_from` / `valid_to` in text, probes anchored to explicit dates | accuracy per as-of date, with no wall-clock dependence |
| **T2.7** | **Negation scope & numeric normalisation** | codes only; no negated statements, no unit/rounding variants, and the engine cannot compare numbers | "LOOP-420 no longer uses X", "100 calendar days" vs "about three months" | needs a numeric rule type (`must_equal_numeric ± tol`) *before* the documents are worth writing |

### 12.3 Tier 3 — optional

Cross-tenant / region data leakage (privacy at the store, not in the answer) · homonym disambiguation (two
entities sharing a surface name across domains) · abbreviation and multi-language variants · OCR-style
noise.

---

## 13. What still will *not* be testable, even with perfect documents

Say this explicitly in the thesis so the new corpus is not oversold:

| Not testable from documents alone | Why | Where it must be fixed |
|---|---|---|
| **Store-level consistency** (false updates, duplicated stale bindings) | no assertion reads the snapshot | `must_store` / `must_not_store` (audit step H2) |
| **Whether the *model* or the *architecture* revised a belief** | a correct answer is compatible with three different internal states | store assertions + a store-inspection column |
| **"Real-world" distribution** (multi-author prose, jargon, OCR noise) | synthetic generators encode their own assumptions | the 20–40-document real arm (§9) |
| **Cost / latency trade-offs** | already measured | existing telemetry |
| **Intent, helpfulness, hallucinations outside the claim set** | out of scope for a claim-extraction benchmark | separate rubric |

---

## 14. Recommended build order (highest yield per day)

| # | New cell | Tier | Why first | Cost |
|---|---|---|---|---|
| 1 | `document_conflict_dilution_01` — siblings + one supersession at 60 docs | D2/D3 (T1.1, T1.2) | the only cell that makes L2-vs-L3 measurable on documents | generator params + 1 YAML |
| 2 | `document_prevalence_trap_01` — 5 stale vs 1 current, plus symmetric control | D3 (T1.3) | cheap, clean, publishes a strong failure-mode result | 1 doc set + 2 YAMLs |
| 3 | `document_split_evidence_01` — claim severed by the 600-char boundary, plus re-packaged control | D2 (T1.5) | isolates chunking, needs no new metric | 2 docs + 1 YAML |
| 4 | `document_multihop_bridge_01` — 2 gold chunks + bridge entity | D3 (T1.4) | needs the set-recall metric (small `core/metrics.py` addition) | metric + 1 YAML |
| 5 | `document_precision_scope_01` — deprecated + out-of-scope + other-tenant decoys | D2 (T2.1) | makes `must_not_retrieve_*` a real metric and tests abstention | 4 docs + 1 YAML |
| 6 | `document_injection_01` — adversarial memo | D2 (T2.2) | extends an existing, well-understood scenario to the document channel | 1 doc + 1 YAML |
| 7 | `document_provenance_attribution_01` — in-text provenance, two disagreeing sources | D2 (T1.6) | forces the provenance question the harness currently hides | 2 docs + 1 YAML |
| 8 | packaging sweep · alias/cousin entity · time-scoped validity | D2/D3 (T2.4–T2.6) | second wave | 3 doc sets |

Cells 1–6 convert the document arm from *"did the chunk arrive"* into *"did the agent pick the right
claim, from the right source, for the right reason"* — which is the actual research question.

---

## 15. Coverage by failure risk (the checklist to use instead of tiers)

Tier labels are bookkeeping. What matters is: **for every way memory can fail, is there at least one cell
that can detect it?** This table is the short list; it is independent of D1/D2/D3 and of L0–L4.

| # | Failure risk | Covered today? | Detected by an assertion? | Minimum action |
|---|---|---|---|---|
| **R1** | Fact never stored (write path) | ✅ all scenarios | ✅ must_contain | — |
| **R2** | Stored but not recalled later (cross-session) | ✅ `cross_thread_01`, document cells | ✅ | — |
| **R3** | Recalled the **wrong** similar item | ⚠️ `document_dilution_01` (354 w, 9 memos), `entity_binding_01` (chat only) | ⚠️ text guards only; **no retrieval-precision metric** | `document_conflict_dilution_01` + `must_not_retrieve_chunk` on decoys |
| **R4** | Kept the **stale** value (no revision) | ⚠️ `document_effective_dates_01`, `document_prevalence_…`, `document_addendum_revision_01` | ✅ positives + stale guards | `document_prevalence_trap_01` (majority-stale) |
| **R5** | **Could not compose** two facts from different sources | ❌ **nothing** | ❌ | `document_multihop_bridge_01` (2 gold chunks) |
| **R6** | Evidence **severed** by chunking / table header lost | ❌ **nothing** | ❌ | `document_split_evidence_01` |
| **R7** | Guessed instead of **abstaining** | ⚠️ `document_dilution_01` T5/T6, `adversarial_forget_*` | ⚠️ negatives only (amnesia passes) | abstention probe with a positive corpus-exclusive signal |
| **R8** | Obeyed an **injected instruction** held in memory | ⚠️ `memory_injection_01` (chat channel only) | ✅ there, ❌ for documents | `document_injection_01` |
| **R9** | **Attributed** a value to the wrong source / leaked cross-scope data | ❌ (prompt carries no provenance) | ❌ | in-text provenance + `must_not_retrieve_chunk` for other-tenant docs |
| **R10** | **Overwrote** an unrelated fact while updating one | ⚠️ `derived_belief_revision_01` T13, `document_prevalence_…` T9/T10 | ✅ text only | `must_store` (audit step H2) |
| **R11** | Paid an unreasonable **cost** for memory | ✅ `overhead_baseline_01` | ✅ tokens/latency | add L2b call-count column |
| **R12** | Retrieved **decoys** (precision) | ❌ 0 `must_not_retrieve_*` rules in the whole dataset | ❌ | M1: add precision pins |

**Reading of the table:** six risks (R3, R4, R7, R8, R10, R12) are *partially* covered and four (R5, R6, R9,
R12-metric) are open. The three pre-L2b cells close R3, R4 and start R12; R5 is the one gap that no amount
of extra documents fixes without a second query — which is precisely why it belongs before L2b.

**Coverage is claim-driven, not matrix-driven.** For each thesis claim, name the cell that would falsify it:

| Claim you will want to make | Cell that can falsify it |
|---|---|
| "Cross-session recall works" | `cross_thread_01`, `document_cross_thread_01` |
| "The store revises instead of accumulating" | `document_addendum_revision_01` + `must_store` (H2) |
| "Current value beats stale value" | `document_effective_dates_01`, `document_prevalence_trap_01` |
| "Picks the right item among similar ones" | `document_conflict_dilution_01`, `document_dilution_01` |
| "Composes facts across sources" | `document_multihop_bridge_01` |
| "Robust to how the evidence is chunked" | `document_split_evidence_01` |
| "Refuses to answer beyond the corpus" | `document_precision_scope_01` |
| "Document-borne attacks do not steer it" | `document_injection_01` |
| "Agentic retrieval beats one-shot" | bridge + conflict (Phase 1), with prevalence as the negative control |
| "Memory costs X tokens / Y ms" | `overhead_baseline_01` + call counts |

If a claim has no cell in the right-hand column, that is the next test to write — and if a cell serves no
claim, it is a candidate for deletion. That rule keeps the suite from growing without adding discrimination.




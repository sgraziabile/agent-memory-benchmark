# Plan — L2 document **contradiction & temporal validity** (D2 / D3a / D9)

> **Status:** PLAN (2026-11-05) — nothing implemented. Self-contained: every fixture, token,
> assertion and command is specified below.
> **Scope.** Build the three L-D document scenarios that
> [`phase-2-l2-document-stress-plan.md`](phase-2-l2-document-stress-plan.md) §5.2 specifies but never
> produced (`D2` competing authority, `D3a` file modification, `D9` temporal validity), instrument
> them with a resolution report, and run the `k` ablation that separates **retrieval** from
> **resolution**.
> **Out of scope.** `D3b` (same `doc_id` + different content → the backend `ValueError`; belongs in
> `pytest`, not a scored run), `D4` premise ablation (needs the `strict_epistemic` prompt arm), L3
> `UPDATE`/`DELETE`, BM25 backend, LLM-as-judge.
> **Why now.** The L-R axis is closed up to 60 documents
> ([`notes/2026-11-05_l2_document_corpus_size_curve.md`](../notes/2026-11-05_l2_document_corpus_size_curve.md)):
> `Recall@2 = Recall@5 = 100 %`, `P(correct given chunk hit) = 100 %`, retrieval deterministic.
> Adding D16 needles only narrows a CI on a number that is already 100 % and makes the fixture
> harder by construction (§10.1 of that note). The open question is no longer *"can it retrieve?"*
> but *"can it choose?"*.

---

## 0. Checklist (one pass, top to bottom)

| # | task | est. | files touched |
|--:|---|---:|---|
| 0 | **Pre-work: close the L-R axis** (§2) | 30–40 min (unattended) | — |
| 1 | Write the 4 fixtures (§3) | 45 min | `datasets/documents/temporal/*.md`, `datasets/documents/ops_v2/ops_handbook.md`, `datasets/documents/runbook_teamb.md` |
| 2 | Write the 3 scenario YAMLs (+1 control) (§4) | 60 min | `datasets/conversations/logical_consistency/**` |
| 3 | Sanity: load + L0/L1 ablation run (§4.5) | 10 min | — |
| 4 | Run the matrix (§6) | ~15 min compute | `outputs/runs/**` |
| 5 | Write `scripts/resolution_report.py` (§7) | 45 min | new script |
| 6 | Write the note `notes/2026-11-XX_l2_document_conflict_expiry.md` | 45 min | new note |
| 7 | Index the fixtures in `datasets/documents/README.md` | 10 min | README |

Definition of done: §10.

---

## 1. What this axis tests, and why L2 must fail it *interesting*

Two architectural facts decide the outcome before any run:

| fact | code | consequence |
|---|---|---|
| **L2 is append-only**: a `chunk_id` cannot be mutated | `agents/level_2_episodic_rag/backends/vector.py:490-491` — *"already stored with different content — L2 is append-only; mutation is L3 territory"* | every revision is a **new chunk**; both versions coexist and compete for the same `top_k` slots |
| **The memory block carries no provenance** | `agents/level_2_episodic_rag/graph.py:69, 92-106` — content only, no `rank`/`chunk_id`/`session`/`turn`/`source`/`score` | the model **cannot see which chunk is newer**; precedence can only come from text it can read (dates, revision numbers, owners) |

So the hypothesis family is:

- **H1 (retrieval vs resolution).** At `k=1` only one version fits the window, so a wrong answer is
  a *retrieval* failure. At `k≥2` both fit and a wrong answer is a *resolution* failure. The `k`
  ablation tells them apart, and it is only interpretable if **each document is exactly one chunk**
  (§3 budget).
- **H2 (date reasoning).** With three revisions (`expired` / `in force` / `future`) and three
  different "today" values, a model that *reasons* gives three different answers. The predicted L2
  failure is D9's: **it answers the newest date** (`7491`), which is right for the future probe and
  wrong for the other two.
- **H3 (merge).** The predicted D3a failure is not "wrong value" but "both values": *"the port was
  `7443` and is now `7444`"*. That satisfies `must_contain` and violates `must_not_contain` — the
  "right answer, wrong channel" cell that X9 declared an invalid probe and that the 2026-11-03
  top-`k` note re-found.
- **H4 (authority).** Given an unsigned 2025 team runbook that says `Level-1` is enough, the 2024
  policy, and the 2026 CAB addendum, the model must prefer the signed, dated, owner-bearing source.
  L2 has no primitive for authority, only similarity — and the runbook is written in the
  *operational vocabulary of the question*, which is exactly where similarity beats correctness.
- **H5 (leak).** Any stale/expired value that survives into a chit-chat turn is **ungraded**
  (D16-10: 2 of 3 retrieval misses volunteered a neighbouring needle's code). The canary turn (§4)
  exists to *observe* that, not to grade it.

---

## 2. Pre-work — close the L-R axis first (unattended, 30–40 min)

Run these **before** writing fixtures; they finish on their own while you write Markdown.

**PowerShell — one line per command, no `\` continuations** (a `\` is a literal argument and argparse
aborts before the banner; and a lost `--embed-cache` silently produces `embed_cache=off`):

```powershell
python run.py --model glm-5.3-flash --scenario document_corpus_size_200_01 --level 2 --backend vector --embedding-model qwen3-embedding-0.6b --top-k 5 --chunk-size 1200 --chunk-overlap 200 --embed-cache outputs/embed_cache
python run.py --model glm-5.3-flash --scenario document_corpus_size_200_01 --level 2 --backend vector --embedding-model qwen3-embedding-0.6b --top-k 1 --chunk-size 1200 --chunk-overlap 200 --embed-cache outputs/embed_cache
python run.py --model glm-5.3-flash --scenario document_corpus_size_008_01 --level 2 --backend vector --embedding-model qwen3-embedding-0.6b --top-k 2 --chunk-size 1200 --chunk-overlap 200 --embed-cache outputs/embed_cache
python run.py --model glm-5.3-flash --scenario document_corpus_size_060_01 --level 2 --backend vector --embedding-model qwen3-embedding-0.6b --top-k 2 --chunk-size 1200 --chunk-overlap 200 --embed-cache outputs/embed_cache --no-documents
```

| run | what it closes | acceptance |
|---|---|---|
| 200 docs, `k=5` | the third curve point (pre-registered in the note §8.2) | `Recall@5 ∈ [62.5 %, 100 %]` → L-R can be closed for this thesis; ≤ 50 % → a knee exists: report it and **stop scaling the corpus** |
| 200 docs, `k=1` | the matched `k=1` curve 8 → 60 → 200 | central estimate ≈ 48 % |
| 8 docs, `k=2` | turns the *inferred* "min viable `k=2` at 8 docs" into measured | 16/16 expected |
| 60 docs, `--no-documents` | the absence control (threat T-5) | **0/16**; anything else invalidates the no-inferability claim |

⚠️ If Ollama restarts mid-run the run dies with `[WinError 10054]` and writes **no artifact** (failed
tuples are excluded from the export, leaving an empty run dir). Ingestion is cache-resumable: just
re-run. Check `Get-Content $env:TEMP\<name>.err -Tail 5` when a run produces nothing.

---

## 3. Fixtures to write (3 new files + 1 modified copy)

### 3.1 Budget: every document must be **exactly one chunk**

`k` counts **chunks**, not documents (D16-3 (3): at `k=2`, 2 probes spent both slots on a single
document). For the ablation to mean "how many *versions* are in the window", each new fixture must be
shorter than `chunk_size`, so it stays a single chunk at 1200:

| fixture | size (measured) | chunks @1200 | chunks @600 |
|---|---:|---:|---:|
| `temporal/port_register_r{1,2,3}.md` | **562 chars** | 1 each | 1 each |
| `runbook_teamb.md` | **561 chars** | 1 | 1 |
| `ops_v2/ops_handbook.md` (copy of the 3 378-char handbook) | unchanged | 3 | 6 |

Consequence: at `k=1` exactly one revision fits, at `k=2` exactly two, at `k=3` all three — the
staircase the experiment needs. And because 562 < 600, the new fixtures are **single-chunk at both
`chunk_size ∈ {600, 1200}`**, so the staircase does not depend on the chunking arm.

### 3.2 `datasets/documents/temporal/port_register_r1.md` (expired)

```markdown
# Northwind Telemetry Port Register (Revision 1)

Owner: Platform Security
Effective: 2025-01-01

## 1. Ingress

The ingestion gateway for the telemetry pipeline listens on port 7443. The
gateway accepts only mTLS traffic from the collector tier.

## 2. Retention

Compacted segments are kept for 30 days before they move to the cold archive.
The retention compactor runs every six hours.

## 3. Change control

Port assignments are owned by Platform Security. A port is never reused across
environments; the service registry enforces uniqueness per environment.
```

### 3.3 `temporal/port_register_r2.md` (in force as of "today" = 2026-11-05)

Same structure, four edits: `Revision 2`, `Effective: 2026-09-14`, port **7444**, retention
**45 days**, plus one clause only r2 has (keeps T4 negation-robust):
*"The gateway rejects plaintext clients with a 421 response."*

### 3.4 `temporal/port_register_r3.md` (future — not yet effective)

Same structure: `Revision 3`, `Effective: 2027-01-01`, port **7491**, retention **60 days**.

**No `Superseded by:` line in any revision** — precedence must be derived from `Effective:` alone.
That is the whole point of H2 (and what makes the twin-date probes meaningful).

### 3.5 `datasets/documents/runbook_teamb.md` (D2's competing authority)

```markdown
# Team B On-Call Runbook (draft)

Document ID: TMB-RB-07
Last edited: 2025-03-11
Maintained by: Team B on-call rotation (no formal owner)

## Reboots

When a production-tier service needs a reboot, the on-call lead signs off in the
incident channel and then reboots it. That is enough: Level-1 is sufficient for a
routine reboot, and waiting for the change calendar only lengthens the outage.

## Escalation

Page the on-call lead first. If the reboot does not bring the service back
within ten minutes, page the platform manager and open a Severity 2 incident.
```

`TMB-RB-07` is the **runbook-exclusive token** used for `must_not_contain` — never `Level-1`
(rule **R1**: a correct answer may legitimately say *"Level-1 was superseded"*).

### 3.6 `datasets/documents/ops_v2/ops_handbook.md` (D3a — the modified copy)

Same **basename** as the preloaded handbook, so the re-ingested chunk id becomes
`doc_ops_handbook_t3*` and nothing collides. Create it deterministically:

```powershell
New-Item -ItemType Directory -Force datasets/documents/ops_v2 | Out-Null
Copy-Item datasets/documents/ops_handbook.md datasets/documents/ops_v2/ops_handbook.md -Force
(Get-Content datasets/documents/ops_v2/ops_handbook.md -Raw) -replace '7443','7444' -replace 'Rostova','Ferreira' | Set-Content datasets/documents/ops_v2/ops_handbook.md -NoNewline
Select-String -Path datasets/documents/ops_v2/ops_handbook.md -Pattern '7444|Ferreira|7443|Rostova'
```

Expected: exactly one `7444`, one `Ferreira`, and **zero** `7443`/`Rostova`.

> **Known artifact — report it, don't fight it.** Two of the three re-ingested chunks are
> byte-identical to the preloaded ones, so an append-only store can spend `k` slots on the *same text
> twice* (`doc_ops_handbook_c0` **and** `doc_ops_handbook_t3_c0`). §7 therefore adds a
> **duplicate-slot rate** metric: it is a genuine property of append-only memory, not a fixture bug.

---

## 4. Scenarios (3 new YAMLs + 1 control)

All three go in `datasets/conversations/logical_consistency/document_revision/`, `category:
document_revision`, `target_research_question: P2` — one category keeps `scripts/organize_datasets.py`
and `--list-scenarios` happy; the semantics live in `name`/`description`.

### 4.1 `test_document_temporal_validity_01.yaml` (D9 — three revisions, three "todays")

`documents:` → `port_register_r1` (`temporal/port_register_r1.md`), `port_register_r2`,
`port_register_r3`. Chunk ids become `doc_port_register_r{1,2,3}` (one chunk each, §3.1).

| turn | session | role | content | assertions |
|---:|---|---|---|---|
| 1 | s1 | user | *"Today is 2026-11-05. Which port does the telemetry ingestion gateway listen on?"* | `must_contain 7444` · `must_retrieve_chunk doc_port_register_r2` · **aux** `must_not_contain 7491` |
| 2 | s1 | user | *"Thanks, nothing else for now."* | **none** (canary, H5: does it volunteer 7443/7491?) |
| 3 | s2 | user | *"New session. Which port does the telemetry ingestion gateway listen on?"* | same as T1 (echo control) |
| 4 | s2 | user | *"How long are compacted segments kept?"* | `must_contain 45 days` · `must_retrieve_chunk doc_port_register_r2` · **aux** `must_not_contain 60 days` |
| 5 | s1 | user | *"As of 2026-03-01, which port did the ingestion gateway listen on?"* | `must_contain 7443` · `must_retrieve_chunk doc_port_register_r1` |
| 6 | s1 | user | *"And as of 2027-06-01?"* | `must_contain 7491` · `must_retrieve_chunk doc_port_register_r3` |

Why these tokens: **T4's `45 days` exists only in r2** (r1 says 30, r3 says 60) → negation-robust, no
`must_not_contain` needed to gate it. **T5/T6 are the twin dates**: same question, two "todays", two
different correct answers. A model with recency bias answers `7444` to T5 and passes T6 by accident.

- `metadata.expected_level_2_behavior`: *"T6 passes trivially (newest = correct). T1/T3/T4/T5 require
  reading `Effective:`; predicted failure = answering 7491 everywhere (4/6 wrong)."*
- `metadata.expected_level_1_behavior`: *"0/6 — no document store; the registers never enter the thread."*
- `ablation contract`: at `top_k=1` T1/T3/T4 may fail **for retrieval reasons** (only one register
  fits); the resolution test only starts at `k=2`.

### 4.2 `test_document_modification_reingest_01.yaml` (D3a — same file, corrected values)

`documents:` → `ops_handbook` (`ops_handbook.md`, 3 chunks). The revised copy is attached mid-run.

| turn | session | role | content | assertions |
|---:|---|---|---|---|
| 1 | s1 | user | *"Which port does the telemetry ingestion gateway listen on?"* | `must_contain 7443` · `must_retrieve_chunk doc_ops_handbook` |
| 2 | s1 | **system_event** | *"Operations re-published the operations handbook with a corrected port allocation for the telemetry pipeline."* `attachments: [ops_v2/ops_handbook.md]` | — |
| 3 | s1 | user | *"Remind me: which port does the telemetry ingestion gateway use now?"* | `must_contain 7444` · `must_retrieve_chunk doc_ops_handbook_t3` · **aux** `must_not_contain 7443` |
| 4 | s2 | user | *"New session. Which port does the telemetry ingestion gateway use?"* | same as T3 |
| 5 | s2 | user | *"Who is the escalation lead for the Rosario facility?"* | `must_contain Ferreira` · `must_retrieve_chunk doc_ops_handbook_t3` |

The system_event text is **deliberately neutral** ("re-published with a corrected allocation") — no
*"supersedes"*, no *"with immediate effect"*. Precedence must come from the content, which is exactly
where the merge is predicted.

`must_not_contain 7443` is **auxiliary (report-only)** here — a documented deviation from D3a's
letter: R1 (the X3 lesson) says a legitimate diagnostic answer (*"it was 7443, now 7444"*) must not be
scored as a failure. The report prints the sentence containing `7443` so §7 can classify it as
`merge` vs `history mention`; promote or demote after the first run based on what the model does.

### 4.3 `test_document_competing_authority_01.yaml` (D2 — policy vs CAB addendum vs team runbook)

`documents:` → `policy_v1` (`policy_v1.md`) + **`runbook_teamb`** (`runbook_teamb.md`, preloaded);
`addendum_1` arrives via `system_event`.

| turn | session | role | content | assertions |
|---:|---|---|---|---|
| 1 | s1 | user | *"According to our policy, what approval level does a production-tier reboot require?"* | `must_contain Level-1` · `must_retrieve_chunk doc_policy_v1` |
| 2 | s1 | **system_event** | *"Compliance published Addendum 1 to the access policy. It supersedes section 3.2 with immediate effect."* `attachments: [addendum_1.md]` | — |
| 3 | s1 | user | *"Which approval do production-tier reboots need now?"* | `must_contain Level-2` · `must_retrieve_chunk doc_addendum_1` · **aux** `must_not_contain TMB-RB-07` |
| 4 | s2 | user | *"New session. The on-call lead told me Level-1 is enough — can I go ahead?"* | `must_contain Level-2` · `must_retrieve_chunk doc_addendum_1` · **aux** `must_not_contain TMB-RB-07` |
| 5 | s2 | user | *"Which document are you relying on, and who owns it?"* | `must_contain Change Advisory Board` · **aux** `must_not_contain TMB-RB-07` |
| 6 | s2 | user | *"What does a standard change like a dashboard tweak require?"* | `must_contain Level-0` · `must_retrieve_chunk doc_policy_v1` |

- **T6 is the crowding control**: if the addendum evicts the policy document, a later failure cannot be
  attributed to authority reasoning.
- **T5 is the attribution probe**: it separates "got the right value by luck" from "knows which source
  is authoritative".
- The negative is `TMB-RB-07`, never `Level-1` (R1).

### 4.4 `test_document_competing_authority_no_addendum_01.yaml` (control, rule R7)

Copy of §4.3 **without** the `system_event` turn, with T3/T4 flipped to `must_contain Level-1` and
`must_retrieve_chunk doc_policy_v1`. If the model still answers `Level-2` with the addendum absent, it
is answering from priors, not from the corpus, and §4.3 measures nothing.

### 4.5 Sanity checks before spending API budget

```powershell
python run.py --list-scenarios | Select-String 'temporal_validity|modification_reingest|competing_authority'
python run.py --model glm-5.3-flash --scenario document_temporal_validity_01 --level 1
python run.py --model glm-5.3-flash --scenario document_temporal_validity_01 --level 0
```

`--level 0/1` need **no backend and no embeddings** (≈30 s each) and must score **0/12**: that is the
ablation contract, verified before any L2 money is spent. If L1 passes anything, the fixture is
satisfiable without memory and must be rewritten.

---

## 5. Design rules — the traps, and the rule that avoids each

| trap | rule |
|---|---|
| **R1 / X3**: `must_not_contain` on a token a correct answer may legitimately mention (*"was Level-1, now Level-2"*) | negate only **literal values with no diagnostic use** (`TMB-RB-07`, `7491`), or make the probe **negation-robust** by asserting a fact that exists *only* in the in-force revision (`45 days`, `Ferreira`) |
| **provenance blindness**: the model cannot see `doc_id`/date metadata (`graph.py:69`) | the `Effective:` date and the revision number must be **inside the document text** |
| **"today" is never injected by the harness** | the date must be **in the question** ("Today is 2026-11-05", "As of 2026-03-01") |
| **sibling chunks**: `k` counts chunks, so one document can eat every slot | keep the new fixtures **≤ 700 chars** (one chunk at 1200) |
| **substring collision**: `doc_ops_handbook` matches `doc_ops_handbook_t3_c0` | never use the bare old prefix in a negative retrieval assertion |
| **echo dominance**: the model's own answers become retrievable in later turns (measured 2026-11-03) | every critical probe is repeated in a **fresh session** (`s2`) |
| **ungraded leakage** (D16-10) | one **canary chit-chat turn** with no assertions, recorded but not scored |
| **append-only duplicates**: re-ingesting an unchanged file yields byte-identical chunks | report the **duplicate-slot rate** instead of ignoring it |
| **contrast loss**: a passing `must_contain` hides a failing `must_retrieve_chunk` | pair every text assertion with its retrieval pin and count **dissociations** (§7) |
| **L0/L1 must fail** | declare expectations in `metadata` and verify with `--level 0/1` (§4.5) |

---

## 6. Run matrix (11 runs, ~15 min of compute)

All with `--backend vector --embedding-model qwen3-embedding-0.6b --chunk-size 1200 --chunk-overlap 200 --embed-cache outputs/embed_cache`.

| # | scenario | `k` | model | purpose |
|--:|---|---:|---|---|
| 1 | `document_temporal_validity_01` | 1 | glm-5.3-flash | retrieval-only baseline (one revision fits) |
| 2 | " | **2** | glm-5.3-flash | two revisions in the window → resolution starts |
| 3 | " | 3 | glm-5.3-flash | all three revisions in the window |
| 4 | " | 5 | glm-5.3-flash | any gain past 3? |
| 5 | `document_modification_reingest_01` | 1 | glm-5.3-flash | old chunk only → predictable wrong answer |
| 6 | " | 2 | glm-5.3-flash | old + new → the merge test |
| 7 | " | 5 | glm-5.3-flash | full window |
| 8 | `document_competing_authority_01` | 2 | glm-5.3-flash | prediction: runbook displaces the addendum |
| 9 | " | 5 | glm-5.3-flash | prediction: all three present, authority reasoning required |
| 10 | `..._no_addendum_01` | 2 | glm-5.3-flash | R7 control: must answer `Level-1` |
| 11 | `document_temporal_validity_01` | 2 | **deepseek-v4-pro** | one confirmatory cell on a stronger model |

PowerShell template (one line, no continuations):

```powershell
python run.py --model glm-5.3-flash --scenario document_temporal_validity_01 --level 2 --backend vector --embedding-model qwen3-embedding-0.6b --top-k 2 --chunk-size 1200 --chunk-overlap 200 --embed-cache outputs/embed_cache
```

Cost: ~10 chunks total across the three fixtures → **ingestion is seconds**; the matrix is dominated by
generation latency (~5–25 s per turn).

---

## 7. Reporting — `scripts/resolution_report.py` (new, ~150 lines)

`lr_report.py` already works on these scenarios (it keys off turns that carry both `must_contain` and
`must_retrieve_chunk`), so **recall/MRR/Δcos come for free**:

```powershell
python scripts/lr_report.py --scenario-prefix document_temporal_validity --csv outputs/lr_recall_resolution.csv
```

What it cannot do is grade *resolution*, so add one small script with the same shape (stdlib + YAML
only, no new deps, same CLI style):

```text
python scripts/resolution_report.py --scenario-prefix document_ --csv outputs/resolution.csv --verbose
```

### 7.1 Inputs it reads

| source | used for |
|---|---|
| scenario YAML | `must_contain` (expected), `must_not_contain` (forbidden), `must_retrieve_chunk` (pins), and a new `metadata.value_families` block |
| `results.json` | `turn_results[*].agent_response`, `assertion_details`, `retrieved_memories` (chunk_id, rank, score, content), latency, tokens |
| `metadata.value_families` | the **only** fixture addition required: `{in_force: ["7444"], superseded: ["7443"], future: ["7491"]}` per scenario, so the classifier is data-driven, not hardcoded |

### 7.2 Classification (one label per graded turn)

| label | rule |
|---|---|
| `pass` | every graded assertion passed (aux included) |
| `pass_aux_fail` | all gating assertions passed, an **aux** (`must_not_contain`) failed → `merge_or_history` bucket |
| `retrieval` | a `must_retrieve_chunk` failed and the expected token is **absent** from the response |
| `resolution_stale` | response contains a `superseded` value and not the in-force one |
| `resolution_future` | response contains a `future` value and not the in-force one |
| `merge` | response contains the in-force value **and** a non-in-force value from the same family |
| `abstention` | no expected/forbidden token; response matches a refusal marker (`don't have`, `no information`, `cannot find`, `not in the`) |
| `other` | none of the above (inspect the snippet) |

For every forbidden/alternative token found, print **the 120-char window around it** — that is what
separates a merge (*"the port is 7443 or 7444"*) from legitimate history (*"it was 7443, now 7444"*) in
ten seconds of human reading, without an LLM judge.

### 7.3 CSV columns

`run, scenario, model_id, top_k, turn, session, label, passed, failed, expected, forbidden,
hit_chunks, hit_ranks, dup_slots, ctx_chars, tokens, latency_ms, snippet`

`dup_slots` = number of retrieved chunks whose `content` duplicates another retrieved chunk *in the
same turn* → the append-only pollution metric (§3.6).

### 7.4 Summary table (printed and exported)

per `(scenario, top_k, model)`:

| column | definition |
|---|---|
| `resolution_accuracy` | `pass / graded_turns` |
| `retrieval_failure_rate` | `retrieval / graded_turns` |
| `stale_rate`, `future_rate`, `merge_rate` | as classified |
| `abstention_rate` | `abstention / graded_turns` |
| `dissociation_rate` | turns where `must_contain` passed **and** `must_retrieve_chunk` failed (the X9 cell) |
| `duplicate_slot_rate` | mean `dup_slots / top_k` |
| `ctx_chars_per_turn`, `tokens_per_turn` | cost of the resolution attempt |
| `canary_leaks` | canary turns whose response contains any family value (H5) |

---

## 8. Pre-registered predictions (write them down before the first run)

| id | prediction | falsified if |
|---|---|---|
| **P1** | `document_temporal_validity_01` at `k≥2` fails **≥ 3 of 5 graded turns**, all with `resolution_future` (answers `7491`, the newest date) | ≥ 4 turns pass → the model *does* reason about `Effective:` and the L2 limitation is only about update primitives |
| **P2** | The twin dates **dissociate**: T6 (future → newest) passes while T5 (past → `7443`) fails | both pass (good) or both fail (no date reasoning at all, not recency bias) |
| **P3** | `document_modification_reingest_01` at `k=2` lands in `pass_aux_fail`/`merge` (contains `7444` **and** `7443`) | claims of merge are wrong → the model cleanly prefers the corrected chunk |
| **P4** | At `k=1` the reingest scenario answers `7443` (old chunk only) → `retrieval`, not `resolution` | `k=1` answers `7444` → the ranking already prefers the corrected chunk and the collision story is weaker than assumed |
| **P5** | `document_competing_authority_01` fails at `k=2` (runbook displaces the addendum) and passes at `k=5` | passes at `k=2` → similarity already prefers the signed document |
| **P6** | `..._no_addendum_01` answers `Level-1` in **all** turns | any `Level-2` → the model answers from priors and §4.3 is invalid (R7) |
| **P7** | The canary turn leaks a family value in **≥ 1** run | no leak in any run → the 2026-11-03 pollution finding does not replicate here |
| **P8** | `duplicate_slot_rate > 0` at `k≥2` in the reingest scenario (append-only duplicates) | 0 → re-chunking deduplicated the identical text (would itself be a finding) |

P1/P2 are the thesis-grade ones: they decide whether the L2 limit is *"no notion of validity"* (P1
holds) or *"no update primitive but sound read-time reasoning"* (P1 falsified). Both outcomes are
publishable; only one is a failure story.

---

## 9. Risks and confounds specific to this axis

| risk | why it bites here | mitigation |
|---|---|---|
| **The "today" date is a *user claim*** | the model may distrust or ignore it, and some prompts treat the latest user statement as authoritative (`strict_epistemic` says exactly that) | keep it in the question (no harness change) and record the prompt arm; if P1 holds only for `baseline_react`, report both arms |
| **World priors** | "7443/7444/7491" carry no meaning, but "Level-1 vs Level-2" might be guessable | the R7 control (§4.4) is the guard; if it fails, the authority result is void |
| **Sycophancy in T4 of §4.3** | the user *asserts* Level-1 is enough | that is intentional (D4 territory); report it separately from the neutral T3 |
| **Sibling chunks** | the new fixtures are 562/561 chars → single chunk at both 600 and 1200, so the staircase holds; the risk returns only if someone edits them past ~`chunk_size` | keep the D3a handbook copy in mind (3 chunks) and pin `--chunk-size 1200` for the matrix |
| **Formatting of the assertion token** | `must_contain "45 days"` misses "45-day retention" | substring matching is case-insensitive only; keep the token short (`45`) plus the snippet review, or accept and document |
| **Merge vs history** needs judgement | no assertion can tell *"was 7443, now 7444"* from *"7443 or 7444"* | `pass_aux_fail` bucket + snippet; do **not** invent an LLM judge for a 5-turn scenario |
| **Small n** | 5–6 graded turns per scenario → wide CI on any rate | unlike `Recall@k` (deterministic, T-11), here the graded quantity **is generation**, so **3 warm replicas per cell do add independent evidence** — budget them if a claim rests on a 1-turn difference |
| **Ollama flakiness** | `WinError 10054` kills a run and leaves no artifact | re-run; ingestion is seconds here anyway |
| **`deepseek-v4-pro` availability/cost** | one cell only (row 11) | drop it if the provider fails; nothing else depends on it |

---

## 10. Definition of done

- [ ] 4 fixtures exist and are **byte-stable** (`sha256` recorded in the note)
- [ ] 4 YAMLs load: `--list-scenarios` shows them; `--level 0/1` score **0/12** on each
- [ ] 11 runs in `outputs/runs/`, `error_turns = 0` in all
- [ ] `scripts/resolution_report.py` produces `outputs/resolution.csv` + the summary table
- [ ] `scripts/lr_report.py --scenario-prefix document_` includes the new scenarios without crashing
- [ ] the note `notes/2026-11-XX_l2_document_conflict_expiry.md` records: fixtures, matrix, every P1–P8
      verdict (met / falsified / inconclusive), the `merge` snippets, and the **attribution**
      paragraph (*"what this says about L2 vs L3"*)
- [ ] `datasets/documents/README.md` indexes the 4 new fixtures
- [ ] `notes/2026-11-05_l2_document_corpus_size_curve.md` gets a one-line forward link in its §10

---

## 11. Open decisions (need a call before writing code)

1. **Model set.** `glm-5.3-flash` for the 10-cell matrix + one `deepseek-v4-pro` cell (proposed), or
   all-`pro` (4× the cost, no extra information for a first pass)?
2. **`must_not_contain` policy.** Auxiliary everywhere (proposed, safe) or graded as D3a's letter says
   (sharper, but R1 false positives will appear as soon as the model explains the history)?
3. **`strict_epistemic` arm.** Out of scope here, but T4 of §4.3 is a sycophancy probe: run it now in
   both prompts (4 extra runs) or defer to the D4 plan?
4. **Duplicate-slot artifact.** Keep the 3-chunk handbook copy (realistic, produces P8) or shrink the
   D3a fixture to one chunk (cleaner, loses the append-only duplicate finding)?
5. **Schema touch.** `metadata.value_families` needs **no** schema change (metadata is free-form) —
   confirm the report reads it from YAML rather than from `results.json`, which does not carry it.

---

## 12. What this buys (one paragraph for the thesis)

If P1 and P5 hold, the chapter reads: *a naive RAG memory has no representation of a fact's validity
interval or of a source's authority. Because the store is append-only and the injected context carries
no provenance, every revision coexists with what it replaced, and the model resolves the conflict
from text alone — preferring whichever value the embedding ranks first, or the newest date, or
repeating both. Raising `k` puts the competing values in the same window but does not create a
preference signal: it trades a retrieval error for a resolution error.* That is a **capability
boundary** (L2 → L3), measured on fixtures that cost seconds to ingest, and it is a far stronger
contribution than a fourth recall point on the D16 curve.

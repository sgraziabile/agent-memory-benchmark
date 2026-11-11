# Assertion-Hygiene Pass — Scope and Checks

**What it is.** A bounded set of **dataset-wide invariant tests** over the 33 scenario YAMLs, plus the small
data edits needed to satisfy them. No new scenarios, no schema change, no corpus work. It exists so every
assertion says what it means and every scenario declares what "pass" is allowed to mean — after which each
new cell is checked automatically at authoring time.

**Why it comes before any new document test.** Otherwise the new cells inherit the current defects: vacuous
assertions (a value echoed by its own probe), sub-token assertions (any digit satisfies the check), scenarios
whose pass/fail cannot be interpreted, no retrieval-precision measurement anywhere, and expectations that
exist only as prose.

**Effort:** ~2–3 h. The diagnostic below has already been run — every count is measured, not estimated.

> **Status — implemented.** [`tests/test_assertion_hygiene.py`](../tests/test_assertion_hygiene.py)
> enforces H1–H8 over all 33 YAMLs; the data edits below are in place; the `EXEMPT` ratchet fails
> both when an entry stops being necessary and when an entry is not justified by a declared
> control kind. Three deliberate deviations from the text above:
>
> 1. `probe.query_kind` accepts a third value, **`guard`**, for seed/confirmation turns whose only
>    rules are leakage guards. Labelling those `lookup` or `semantic` would be false, and the
>    schema plus H8 both refuse `guard` on a turn that carries a positive rule.
> 2. H1's "fix" for forced-label probes is implemented as a **guarded-menu rule**, not a data
>    trick: a positive token that appears in its own probe is accepted iff the same turn guards a
>    *rejected* label that also appears in the probe (audit step M6). `backward_modus_tollens_01`
>    therefore gained the guards and needs no exemption; only declared controls are exempt.
> 3. H5 is enforced with no exemption. Two pins measure known context-blending behaviour rather
>    than asserting a healthy retriever — `document_addendum_revision_01` T4 (`ep_s1_t1`, the
>    stale in-session answer) and `document_cross_thread_01` T6 (`ops_handbook`, injected into an
>    unrelated turn) are expected to stay red until the retriever stops mixing channels. Their
>    failure is the finding, not a false negative.

---

## 1. The eight checks

| # | Invariant | Violations today (measured) | Fix |
|---|---|---|---|
| **H1** | A `must_contain` value must not appear in **its own probe turn** (rule R10) | **10 hits / 8 scenarios** | 2 real fixes + 8 documented exemptions |
| **H2** | An assertion value must carry ≥3 characters **or** set `case_sensitive: true` **or** be exempt | **10 hits / 3 scenarios** (`3`, `Au`, `O`, `0`, `6`, `7`, `9`, `24`) | 3 edits + 1 exemption |
| **H3** | Every scenario declares `metadata.kind`, and the kind justifies its assertion mix | **33 / 33 missing** | mechanical |
| **H4** | A scenario with `documents:` must have ≥1 positive retrieval pin | **1** (`document_cross_thread_01`) | add 2 pins |
| **H5** | A scenario with `documents:` must have ≥1 **precision** pin (`must_not_retrieve_*`) | **10 / 10** | 1 pin each |
| **H6** | No `must_not_contain` value may be a substring of a `must_contain` value on the same turn | **0** | insurance |
| **H7** | Every scenario declares machine-readable `metadata.expected` per level | **33 / 33 missing** | seed from prose + evidence |
| **H8** | Every asserting turn declares `probe.query_kind ∈ {lookup, semantic}` | 0 / 113 turns | needed for L2b |

### H1 — no echo (R10)

The existing test (`tests/test_document_ingestion.py::test_no_positive_assertion_echoes_its_own_probe`) covers
only ids starting with `document_` — 10 of 33 scenarios. Generalise it. The 10 current hits split cleanly:

| Hit | Verdict |
|---|---|
| `in_context_control_01` (`Thursday`, `108`), `in_context_needle_01` (`DELTA-7749`), `in_context_multihop_01` (`Berlin`), `in_context_derived_revision_01` (`Level-1`), `in_context_modus_tollens_01` (`NO_EU_CENTRAL`) | **exempt** — an in-context control carries the whole context *in the probe* by construction |
| `overhead_baseline_01` T7 (`O`) | **exempt** — P3 cost control; accuracy is not its claim |
| `backward_modus_tollens_01` T12/T13 (`NO_EU_CENTRAL`, `YES_BASTION`) | **fix** — the value is in the probe only because the probe prints the 3-label menu; add `must_not_contain` guards for the two rejected labels |
| `negative_hallucination_01` T4 (`no launch date`) | **fix** — replace with a corpus-exclusive positive token (the phrase echoes the probe's wording) |

### H2 — entropy floor

`must_contain: "0"` passes on any response containing a zero; `"Au"` passes on "because"; `"O"` passes on
"to". Fixes:

- `attrition_01` T6 and `overhead_baseline_01` T1/T7 → `case_sensitive: true` (a chemical symbol is
  unambiguous when case matters; that alone removes the false positives).
- `belief_revision_01` T2/T5 → assert `"3 years"` instead of `"3"`.
- `overhead_baseline_01`'s numeric answers (`6`, `0`, `7`, `9`, `24`) → **exempt with reason**: the cell is a
  P3 cost control whose pass rate is *designed* to be 100 % at every level; tightening it would only create
  false negatives on a metric it does not claim.

### H3 — declared kind (this replaces "every scenario needs a positive rule")

The naive rule ("every scenario must have a positive assertion") is **wrong**, and the data proves it:
`adversarial_forget_01/02` are *privacy* cells and both state `expected_level_0_behavior: "Passes purely by
amnesia"`. An amnesic pass is legitimate there. What is missing is the declaration:

```yaml
metadata:
  kind: privacy              # recall | revision | privacy | security | scaling | control_p3 | control_in_context
  amnesia_passes: true       # required when kind: privacy
  quality_eligible: false    # excluded from accuracy aggregates
```

| `kind` | Requirement |
|---|---|
| `recall`, `revision` | ≥1 positive rule; `amnesia_passes: false`; quality-eligible |
| `privacy`, `security` | negative-only **iff** `amnesia_passes: true` **and** `quality_eligible: false` |
| `control_p3`, `control_in_context` | exempt from H1/H2; must declare what they control |
| `scaling` | must declare `corpus_difficulty.lexical_baseline_hit_at_5` |

This turns an apparent design flaw into a declared contract — and the aggregate stops mixing "did it
remember" with "did it avoid leaking".

### H4/H5 — document coverage and precision

`document_cross_thread_01` has **0 retrieval pins**, so a retrieval failure is invisible there. Every
document scenario should carry ≥1 positive pin (H4) and ≥1 precision pin (H5). The dataset has **232 rules
and 0 `must_not_retrieve_*`**, although the engine implements them correctly
(`_evaluate_retrieval_assertion`: for `must_not_retrieve_*`, pass iff no chunk matches). H5 is what turns
"context blending" from a hand-written text guard into a metric.


### H6 — no self-contradiction

Cheap insurance: a pair like `must_contain: "UPS-7C42"` + `must_not_contain: "UPS-7C4"` can never pass. Zero
hits today; the check costs nothing and catches a whole authoring class.

### H7 — machine-readable expectations

Replace prose-only `expected_level_N_behavior` with:

```yaml
metadata:
  expected: {level_0: fail, level_1: pass, level_2: pass, level_3: pass, level_4: pass}
```

Allowed values: `pass | fail | partial | control | n/a`. Today all 33 files declare L0 prose (decaying to
21 / 18 / 5 / 2 for L1–L4) and **none** is machine-readable — so a run that contradicts the expectation is
indistinguishable from a broken expectation. Seed the values from `DIFFICULTY_AUDIT.md` Appendix B plus each
file's prose, then review once.

### H8 — probe kind (for L2b)

`probe.query_kind: lookup` (the probe names the opaque key) vs `semantic` (it describes the entity). Without
this field the L2b experiment cannot distinguish "the agent reformulated the query well" from "the probe
handed it the key". 0/113 turns declare it today.

---

## 2. Rollout: diagnostic → ratchet → enforcement

1. **Diagnostic (done).** Counts in §1, from a read-only scan over `datasets/conversations/**`.
2. **Ratchet.** Land `tests/test_assertion_hygiene.py` with an `EXEMPT` table where each entry names the
   check, the scenario, the reason and the finding. The test also asserts each exemption is **still
   necessary** (it fails when the underlying violation is gone), so the allow-list cannot rot.
3. **Enforcement.** Fix the data, delete exemptions as they resolve, require the test green in CI. From then
   on a scenario cannot land with an echo, a 1-character assertion, an undeclared kind, a document cell
   without precision pins, or an undeclared expectation.

---

## 3. What this pass does **not** cover

| Not covered | Where it belongs |
|---|---|
| whether a cell actually *discriminates* (vs being solved by a keyword engine) | V1 lexical-baseline test (`CORPUS_VALIDITY.md` §7) |
| store-level consistency (false updates, duplicated stale bindings) | `must_store` / `must_not_store` — audit step H2 |
| prompt-level confounds (`strict_epistemic` instructing revision) | audit step L3 |
| semantic adequacy of the corpus (claim density, competition) | `CORPUS_VALIDITY.md` §6 |

Those four need code or corpus changes; the hygiene pass is deliberately limited to **invariants a test can
assert about the YAML as it stands today**.


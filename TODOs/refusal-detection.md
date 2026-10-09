# TODO: Refusal Detection for the Benchmark Harness

> **Status:** PROPOSED — awaiting decision. Not implemented.
> **Origin:** Analysis of `explicit_forget_01` runs (2026-10-09), where gpt-oss-20b refuses to
> "store or retain" credentials on turns 1–2 even at level 1 with full context available.
> The refusal is model behavior (safety-training over-refusal), not a memory defect, but the
> harness currently cannot distinguish it from genuine amnesia.

## 1. Problem

`pass_rate` conflates four distinct behaviors behind identical assertion outcomes:

| Behavior | Example | Assertion outcome |
|---|---|---|
| Compliant recall | "Your active Employee ID is EMP-8042" | must_contain → PASS |
| Refusal (policy) | "I can't store or retain that information" | must_contain → FAIL |
| No-info claim (amnesia) | "I don't have any information about Elena's city" | must_contain → FAIL |
| Error | blank response (already exit-code 2) | excluded |

A `must_contain` FAIL is scored identically whether the model *couldn't* remember or
*wouldn't* engage. Refusal detection annotates turns so analysis can tell them apart —
**without changing any existing score**.

## 2. Design

### Tier 1 — inline lexical detector (deterministic, zero API cost)

Runs in `run_single` right after `evaluate_assertions`, annotating **every** turn
(including assertion-less turns like `explicit_forget_01` turns 1 and 3, where an early
refusal predicts later failures).

**A. Normalization:** reuse `_normalize_for_matching()` (NFKC + typographic folding +
casefold). Critical: every observed response uses curly apostrophes (U+2019) —
`can't` must match `can’t`.

**B. Two marker sets:**
- **Refusal** (policy/capability decline): `can[’']t / cannot / unable to / won[’']t` +
  `{store, retain, provide, share, disclose, help with, retrieve}`; also standalone
  patterns (`against my policy`, `not permitted to`).
- **No-info** (claimed ignorance): `don[’']t/do not have {that information, any
  information, access to}`, `wasn[’']t provided`, `you haven[’']t told me`,
  `not mentioned in (our|this) conversation`, `I don[’']t recall`.

A turn can hit both sets → `marker_type: refusal | no_info | both`.

**C. Precision guard (two fields, not one):**
- `refusal_marker` — lexical hit only (raw signal)
- `refusal_detected` — marker present **AND** the turn achieved zero passing
  `must_contain` assertions (or has no assertions)

Proof case: *"I can't store files, but your ID is EMP-8042"* → marker HIT, but the model
complied → `refusal_detected = false`. The marker is evidence; the behavioral outcome is
the verdict. Hedged compliance is still visible via `refusal_marker` for analysis.

Known edge (intentional): refusal on a `must_not_contain` turn (e.g. level-1 turn 4
"I can't provide that information" passing `must_not_contain: 9173`) → `refusal_detected =
true` even though the assertion passed. The guard only consults `must_contain` outcomes.
Keep as designed: that refusal is behaviorally real and is arguably the *ideal* outcome
of `explicit_forget_01` (selective-unlearning compliance) — we want it tagged.

Overlap rule: phrases matching both sets ("I can't recall that information") resolve
toward `refusal` — "can't recall" after seeing the context is still a declination.

### Tier 2 — optional LLM-judge refinement (offline)

Post-processing script (not inline) feeding marker-bearing or failed turns to the judge
(`qwen/qwen3-235b-a22b`, temp 0 — currently inert `judge:` config) with a rubric:
`compliant / refusal / no-info / ambiguous`. Tier 1 output is the input filter, so only
suspicious turns cost quota. Caveats: judge is itself a tested model (self-preference
bias); judge calls consume subscription quota.

## 3. Schema changes

- `TurnResult` += `refusal_marker: str | None` (matched phrase), `refusal_detected: bool = False`
- `BenchmarkRunSummary` += `refusal_turns`, `refusal_rate`,
  `failed_due_to_refusal` (must_contain failures on refusal turns),
  `pass_rate_refusal_adjusted = passed / (total_assertions − failed_due_to_refusal)`
  — **both** rates exported; nothing hidden.

## 4. Export changes

- `summary.csv`: new columns
- `summary.md`: "Refusals" column
- `run.py`: per-run refusal line, same style as error accounting
- Backfill script: re-annotate existing `outputs/runs/*/results.json` offline
  (responses stored verbatim → no API calls needed). Also used to validate Tier 1
  against the 4 known `explicit_forget_01` runs before any sweep.

## 5. Interpretation stays manual (detection is orthogonal to scoring)

Refusal is NOT wired into pass/fail. Meaning depends on context:

| Tag | Level | Meaning |
|---|---|---|
| Refusal on must_contain | any | over-refusal — capability-measurement invalidator |
| Refusal on must_not_contain | level 1 | correct selective-unlearning compliance (ideal outcome) |
| No-info claim | level 0 | honest amnesia (model truthfully can't see history) |
| No-info claim | level 1 | inattention to context, or strategic "can't recall" — the interesting case |

## 6. Tests

Canned-response unit tests per marker class: curly apostrophes, both-set hits,
"complied despite marker" precision-guard case, blank/error turns → no marker.
All offline/deterministic.

## 7. Honest limitations

- Tier 1 is a heuristic: novel phrasings ("I must decline to persist such details") missed
  until Tier 2 exists.
- `no_info`/`refusal` overlap resolved by rule, not by meaning.
- True refusal rate ≥ detected rate → report both `refusal_marker` and `refusal_detected`
  counts.

## 8. Implementation scope (when approved)

`core/schemas.py` (2 + 4 fields), `core/runner.py` (detector + aggregation, ~60 lines),
exporter CSV/MD, `run.py` display, tests, backfill script. Existing `pass_rate` semantics
untouched — all old exports remain comparable.

## 9. Related findings (context for the decision)

- gpt-oss-20b refuses credentials at turn 1 (no memory needed) → the refusal erases the
  level-0/level-1 distinction on this scenario; current scores differ only on turn 5.
- `baseline_react` partially overrides the refusal prior; `minimal` leaves it dominant
  (L1×baseline answered EMP-8042 at turn 5 twice; L1×minimal refused). Real prompt-ablation
  finding.
- Cheap follow-up experiments: run `explicit_forget_01` on the other 7 models (weaker
  privacy-refusal priors expected); try `strict_epistemic` on gpt-oss-20b (its mandatory
  "TRACK all facts" rule may overpower the refusal).

# Deprecated `document_split_evidence_01` runs -- pre strict probe (2026-11-11)

**Status: keep them, the findings stand; archived only because T2's assertion set changed.**
Bundles: `20261111_155622` (Level 2, `top_k=5`, 12 documents, 37 chunks) and
`20261111_155646` (`top_k=15`). Both deepseek-v4.1-flash, baseline ReAct.

## Why they are not comparable with later runs

T2 used to be scored with `must_contain KVN-4C8R-1194` alone, so an answer that listed
*both* codes of the row passed without the model knowing which one was the certified
value. Two runs at k=5 with the guide absent -- identical retrieval shape -- then split
on nothing but answer form:

| Run | T2 answer | Old verdict |
|---|---|---|
| `20261111_154640` (archived pre-BOM fix) | `TST-9F2M-6630` alone | FAIL |
| `20261111_155622` | `KVN-4C8R-1194 (with TST-9F2M-6630, dated 2026-01-22)` | PASS |

T2 now asks for the code only and guards the other field of the row, so the hedge is
caught. The rule count went 9 -> 10.

## Replay of these transcripts against the current rules

`BenchmarkRunner.evaluate_assertions` over the recorded responses, current YAML:

| Run | T2 | T3 | T4 | T5 | Total |
|---|---|---|---|---|---|
| `20261111_154640` | FAIL (wrong code **and** guard) | FAIL | PASS | FAIL | **6/10** |
| `20261111_155622` | FAIL (guard: named the pressure token) | PASS | PASS | FAIL | **8/10** |
| `20261111_155646` | FAIL (guard) | PASS | PASS | FAIL (over-disclosed `MVB-2K9P-6634`) | **8/10** |

Under the strict probe the pair is readable instead of contradictory: `154640` commits to
the wrong field twice, `155622` hedges twice and is caught at T2 while T3 (deliberately
loose) still passes. T5 fails in all three runs -- same cause each time, the export's
title chunk never reaches the window at k=5, and at k=15 the model over-discloses the
other family's value.

## What these runs established and still hold

1. **T5 is a stable failure, not a phrasing effect** -- three runs, three failures, one cause
   (`failure_taxonomy.title_chunk_absent`).
2. **The guide is genuinely absent at k=5** -- no run retrieved `split_legend_01` in the T2/T3
   windows, and at k=15 it appears at rank 12/15 and is then applied correctly. That is the
   cutoff coupling the cell is built on.
3. **T4 is a working solvability control** -- the probe that names the format reaches the guide
   and both family orders are reproduced.
4. **Answer form decided the old T2 verdict.** That is the finding that motivated the strict
   probe; it is worth reporting as a metric-sensitivity observation, not as a model difference.

## Housekeeping

Both bundles keep the 9-rule `results.json` they were produced with -- that is the point.
Re-run to obtain numbers under the current 10 rules.

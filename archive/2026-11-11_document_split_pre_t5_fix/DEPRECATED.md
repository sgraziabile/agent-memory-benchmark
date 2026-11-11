# Deprecated `document_split_evidence_01` run — pre T5 change (2026-11-11)

**Status: keep it, it is the reference failure for this cell.** One bundle,
`20261111_153740` (deepseek-v4.1-flash, Level 2, `top_k=5`, 12 documents, 37 chunks, 6/8 assertions,
`retrieval_recall = 1.0`). It ran against a scenario whose T5 was a negative-only guard
(*"Is VLV-331 part of the north-riser assembly? Answer yes or no."*), which the run showed to be too weak —
see below. T5 now asks *"Which export does VLV-331 belong to?"* and asserts `north-riser`, so the assertion
count went 8 → 9 and `pass_rate` is not comparable with later runs.

T2/T3/T4 are unaffected by that change: their rules, their windows and their verdicts stand, so this bundle
remains valid evidence for the **severance** finding.

## What the run established (all verified against the retrieved windows)

**T2 and T3 failed, and the failure is the cell's target failure — field mis-binding.**

| Turn | Answer | Truth | Retrieved |
|---|---|---|---|
| T2 | `TST-9F2M-6630` ("verified 2026-01-22") | `KVN-4C8R-1194` | all four `split_rows_01` chunks + one `split_rows_02` chunk — **no guide chunk** |
| T3 | `TST-3M6Q-9058` ("verified 2026-01-10") | `KVN-8T2D-4417` | three `split_rows_01` chunks — **no guide chunk** |

The record `VLV-220 | KVN-4C8R-1194 | TST-9F2M-6630 | 2026-01-22` arrived intact and the model bound the
**third** field as the value, correctly identifying the fourth as the date. The same wrong binding recurs on a
second record, so this is a systematic misread, not a coin flip. The guide — the only document that states the
field order — never entered the window: the retrieval pins confirm the records were present and the guide
absent, which is exactly why the failure is attributable to severed evidence rather than to a retriever miss.

**T4 passed and proves solvability.** Probed with *"What order are the fields in for the third-riser assembly
export?"*, the guide ranked 1 and the model reproduced the full order **and** contrasted it with the
north-riser family. So the cell is solvable in two hops and impossible in one.

**T5 passed while the model was wrong.** Asked whether VLV-331 is part of the north-riser assembly, it
answered **"No"** — VLV-331 is in the north-riser export — and the negative-only guard (which only forbade
quoting the value) recorded a pass. `retrieval_recall` stayed 1.0 and the score read 6/8 when the substantive
picture was 2 wrong answers plus one wrong Yes/No.

## The change this run caused

T5 is no longer a guard: it asks for the family by name and asserts the corpus-exclusive token `north-riser`,
keeping the no-disclosure rule as a floor. Under the old assertions this run passed T5; under the new ones it
would fail it — which is the point. The general lesson, recorded in the plan: **a negative-only guard cannot
detect a wrong answer**, only an over-disclosure, so a cell's cheapest probe should not be a guard.

# Deprecated `document_split_evidence_01` run — pre BOM fix (2026-11-11)

**Status: keep it, the verdicts stand; it is only archived because the corpus changed by one byte.**
One bundle, `20261111_154640` (deepseek-v4.1-flash, Level 2, `top_k=5`, 12 documents, 37 chunks, **6/9**,
`retrieval_recall = 1.0`).

## Why it is not comparable with later runs

`split_rows_01.md`, `split_rows_02.md` and `asset_registry_01.md` carried a UTF-8 **BOM** (a leftover from
rewriting them with PowerShell). The BOM sat inside the first chunk of each file, so it was embedded and
reached the prompt as an invisible character. Removing it changed the text of those chunks by one byte, and
because chunk text is the embedding-cache key, their vectors are invalidated. Chunk counts are unchanged
(37 for this cell) and no assertion depends on the byte.

## What the run established — all still valid

| Turn | Verdict | Class | Evidence |
|---|---|---|---|
| T2 | FAIL `KVN-4C8R-1194` (answered `TST-9F2M-6630`) | **F3 field mis-binding** | all four `split_rows_01` chunks retrieved, **no guide chunk** |
| T3 | FAIL `KVN-8T2D-4417` (answered `TST-3M6Q-9058`) | **F3, same binding** | three `split_rows_01` chunks, **no guide chunk** |
| T5 | FAIL `north-riser` | **F2 title chunk absent** | window held `rows_02_c1/c2` and three `rows_01` chunks, never `rows_02_c0` |
| T2 ×2 pins, T3 pin, T4 ×2, T5 guard | PASS | — | T4 retrieved the guide at rank 1 and reproduced both family orders |

Two things this run taught, both now encoded in the scenario and its tests:

1. **T5 is a third instance of the severance, not a field-position test.** The family label exists only in
   chunk 0 of each export; an identifier-bearing probe returns the row chunks and never the title, so the
   records arrive unattributed. Recorded as `failure_taxonomy.title_chunk_absent`, and pinned by
   `test_the_family_label_is_severed_from_its_rows`.
2. **The model's abstention was optimal.** It located the block by its row range (`VLV-319 … VLV-336`) and by
   the dates, said the blocks "aren't titled", and asked for the label. Scoring it as a failure is correct —
   the family name is the deliverable — but the answer text should be read, not just the verdict.

## Housekeeping

The archived bundle still contains the BOM-era `results.json`, which is the point: it is the record of what
the corpus looked like before the cleanup. Re-run to obtain numbers for the current bytes.

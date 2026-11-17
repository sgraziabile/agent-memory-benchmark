# Deprecated `document_split_evidence_01` runs -- pre decisive-T3 (2026-11-11)

**Status: superseded within the same session; kept for the record.**
Bundles: `20261111_161728` (`top_k=5`, **6/10**) and `20261111_162001` (`top_k=15`, **6/10**),
deepseek-v4.1-flash, `qwen3-embedding-0.6b`, 12 documents / 37 chunks.

## Why they are not comparable with later runs

Produced under the 10-rule set of the first R2 pass, where T2 was strict but T3 was still scored loosely
(`must_contain` only). The k=15 bundle showed the guide **in** T2's window (ranks 10 and 13) with the model still
answering `TST-9F2M-6630`, i.e. T2 could not certify a binding on its own, and a loose T3 could be satisfied by
hedging -- so the pair certified nothing. T3 then became the decisive probe with the same strict treatment
(single token + guard), rules 10 -> 11, and these two bundles were re-run as `20261111_163559` (8/11) and
`20261111_163657` (9/11).

## What these runs still establish

| Turn | Evidence | Class |
|---|---|---|
| T2 | both arms answered `TST-9F2M-6630` alone | the pressure reading of "limit" (see `probe_noun_ambiguous`) |
| T3 | both arms answered `TST-3M6Q-9058` alone | **field mis-binding** -- decisive, no term of art in the question |
| T4 | guide retrieved, both family orders reproduced | solvability control works |
| T5 | k=15: named `north-riser` but also disclosed `MVB-2K9P-6634` | over-disclosure |

They are the runs that motivated making T3 decisive: with T2 alone, the same model could be scored as failing
(a wrong guess) or as passing (a hedge) on retrieval that was identical.

## Housekeeping

Both bundles keep the 10-rule `results.json` they were produced with. Re-run to obtain numbers under the
current rules.

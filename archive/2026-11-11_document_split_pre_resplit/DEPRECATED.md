# Deprecated `document_split_evidence_01` runs -- pre re-split (2026-11-11)

**Status: superseded by the two-cell pair; kept because they are the evidence that forced it.**

| Bundle | Cell as it then was | Score | Note |
|---|---|---|---|
| `20261111_163559` | one scenario, 11 rules, T2 → T3 in one thread | 8/11 | T3 passed on T2's answer (see the leak experiment) |
| `20261111_163657` | same, `top_k=15` | 9/11 | guide in T2's window (ranks 11, 14), still mis-answered |
| `20261111_171155` | binding cell, 6 rules, but probing **VLV-247** (third riser) | 6/6 | **positional pass**: the model reports field 2, and field 2 is the certified value in that family |

The last one is the important one. An isolated single-probe cell at k=5, guide **not** in the window, scored
6/6 — which falsified "the record is undecodable in one hop". Two follow-up probes showed why: the model's
default is "report the second field of the line".

| Probe | Row | Answer | Verdict |
|---|---|---|---|
| VLV-247 (third riser) | `VLV-247 \| KVN-… \| TST-… \| date` | `KVN-8T2D-4417` | right, for the wrong reason |
| VLV-320 (north riser, k=5) | `VLV-320 \| TRQ-… \| MVB-… \| date` | `TRQ-HJZS-1262` | position-2 = test pressure |
| VLV-320 (north riser, k=15) | same, **guide in window** (ranks 10, 12) | `TRQ-HJZS-1262` | same answer with the schema present |

Consequence: the decisive probe now targets a **north-riser** record, where the second field is the test
pressure, so the positional default is visibly wrong. Numbers under the current two-cell design
(`document_split_binding_01` 4/6 at k=5 and at k=15; `document_split_evidence_01` 5/8 and 6/8) are in
`TODOs/l2b-sequencing-plan.md` §2.7.

Re-run to compare; these bundles keep the rules and probe wording they were produced with.

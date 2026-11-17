# Experiment: does T2 contaminate T3? (2026-11-11)

Four variants at `top_k=5`, deepseek-v4.1-flash, `qwen3-embedding-0.6b`, same corpus (12 documents / 37 chunks).
Run as throwaway scenario copies (`split_t3_*_scratch_01`, deleted afterwards) so the real cell was untouched.

| Bundle | Variant | T2 answer | T3 window | T3 answer | T3 verdict |
|---|---|---|---|---|---|
| `20261111_163559` (kept in `outputs/runs`) | T2 then T3, one thread | refused | identical | `KVN-8T2D-4417` | **PASS** |
| `20261111_164916` | same cell, second sample | refused | identical | `KVN-8T2D-4417` | **PASS** |
| `20261111_165301` | T3 alone (T2 turn removed) | — | identical | `TST-3M6Q-9058` | FAIL |
| `20261111_165138` | T3 in its own session (s3) | `TST-9F2M-6630` | `ep_s2_t1` (0.7444) + 4 rows chunks | `TST-3M6Q-9058` | FAIL |
| `20261111_165345` | T3 first, T2 after | `TST-9F2M-6630` | identical | `TST-3M6Q-9058` | FAIL |

"identical" = `doc_split_rows_01_c3, _c2, _c1, _c0, doc_split_rows_02_c1` — the same five chunks, in that order,
in every variant that keeps T3 in `s2`. The retrieval evidence is byte-identical across PASS and FAIL; the only
variable is whether T2's answer was in the prompt.

**Verdict: T2 injects the answer key, and it is the only thing that made T3 pass at k=5.** The leak is textual,
not positional: T2's refusal repeats the row's field names in order (`asset identifier, certified value, test
pressure, and verification date`), which is exactly what `split_legend_01` would have supplied. `graph.py` places
the prior thread history in the prompt verbatim, so T3 read the key out of its own conversation.

Two secondary findings:

1. **Session isolation is not a fix.** With T3 moved to `s3`, T2's episode came back as the rank-1 memory
   (`ep_s2_t1`, 0.7444) because L2 retrieves across sessions. Worse, it propagates whatever T2 said: in that
   variant T2 answered the pressure code and T3 copied the pressure code.
2. **T3 is otherwise unanswerable at k=5** — with T2 absent it answered the pressure code in every variant, which
   is the cell behaving as designed (guide at rank 14, row at rank 1).

The real cell was **not** modified by the experiment at the time it ran; the finding is recorded in
`failure_taxonomy.t2_leaks_the_field_order` and the split followed the same day — see
`archive/2026-11-11_document_split_pre_resplit/` for the numbers before and after.

## Follow-up in the same folder: the positional default

`20261111_171339` / `20261111_171416` — a throwaway scenario probing **VLV-320** (a north-riser record) with no
guide in reach, at k=5 and k=15. The isolated third-riser cell had just scored 6/6 (`171155`), and these runs
explain why: the model answers the **second field of the line**, which is the certified value in the third-riser
export and the test pressure in the north-riser one.

| Run | Cut-off | Guide in T2 window | Answer |
|---|---|---|---|
| `20261111_171339` | k=5 | no | `TRQ-HJZS-1262` |
| `20261111_171416` | k=15 | yes (ranks 10, 12) | `TRQ-HJZS-1262` |

That is why the decisive probe now targets the north-riser family.

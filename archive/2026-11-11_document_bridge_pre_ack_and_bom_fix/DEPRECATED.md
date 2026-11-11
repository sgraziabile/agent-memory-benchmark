# Deprecated `document_multihop_bridge_01` runs — pre acknowledgement change and BOM fix (2026-11-11)

**Status: the T2 failures stand and are the cell's intended result; archived only because the prompt and the
corpus changed afterwards.** Two bundles:

| Bundle | Model | `top_k` | Score | Recall | T4 rule in force |
|---|---|---:|---:|---:|---|
| `20261111_140937` | deepseek-v4.1-flash | 5 | 8/10 | 0.75 | `six weeks` (current) |
| `20261111_141036` | deepseek-v4.1-flash | 15 | 8/10 | 0.75 | `six weeks` (current) |

## Why they are not comparable with later runs

1. **The acknowledgement turn changed.** T1 used to enumerate the corpus — *"the asset registry, the limit
   schedule, the maintenance records and the standing policies"*. It now says only *"I have loaded the
   Viaduct facility documentation"*, so a model can no longer learn what documents exist without retrieving.
   That changes the prompt the model sees on turn 1.
2. **`asset_registry_01.md` carried a UTF-8 BOM**, removed with the other calibration documents; the first
   chunk's text changed by one byte and its cached vector is invalidated.

## What the runs established — still valid

Both arms failed the **same two rules, on the composed probe only**:

```
T2  must_contain        QMV-T3R8-4402          FAIL
T2  must_retrieve_chunk calibration_register_02 FAIL
```

T3 (hop 1 in isolation), T4 (the single-hop control, answered *"Every six weeks."*) and both T5 guards passed
in both arms; `retrieval_recall = 0.75` in both, i.e. exactly the hop-2 pin missing. `top_k` 5 → 15 changed
nothing, which is the cell's hypothesis: breadth does not substitute for a second query, because the register
shares no content word with the probe.

These are the last measurements taken under the old acknowledgement prompt. Post-change runs must be compared
against them with that in mind — the composed probe is expected to fail identically, and T4 to keep passing.

# Benchmark Execution Summary

- **Generated at:** 2026-11-11 14:57:25
- **Total Runs:** 1

| Scenario                        | Model             | Prompt           | Level                  | Turns | Pass Rate  | Avg Latency | Total Tokens | Cost (USD) | Errors | Retrieval Recall |
| :------------------------------ | :---------------- | :--------------- | :--------------------- | ----: | ---------: | ----------: | -----------: | ---------: | -----: | ---------------: |
| `document_conflict_dilution_01` | `mistral-small-4` | `baseline_react` | `level_2_episodic_rag` |     5 | **100.0%** |     4998 ms |       16,214 |  $0.006645 |      0 |     3/3 (100.0%) |

`Retrieval Recall` = passed / declared `must_retrieve_chunk` pins (rule-level; a sibling sub-chunk of the gold document satisfies a pin). `*` marks a run where at least one declared probe was never evaluated (infrastructure error) and counts as a miss. Chunk-level `Recall@k` is a different metric: `scripts/lr_report.py`.

# Benchmark Execution Summary

- **Generated at:** 2026-11-11 13:56:57
- **Total Runs:** 1

| Scenario                      | Model           | Prompt           | Level                  | Turns | Pass Rate | Avg Latency | Total Tokens | Cost (USD) | Errors | Retrieval Recall |
| :---------------------------- | :-------------- | :--------------- | :--------------------- | ----: | --------: | ----------: | -----------: | ---------: | -----: | ---------------: |
| `document_multihop_bridge_01` | `mimo-v2.6-pro` | `baseline_react` | `level_2_episodic_rag` |     5 | **80.0%** |    11195 ms |       19,037 |  $0.009248 |      0 |      3/4 (75.0%) |

`Retrieval Recall` = passed / declared `must_retrieve_chunk` pins (rule-level; a sibling sub-chunk of the gold document satisfies a pin). `*` marks a run where at least one declared probe was never evaluated (infrastructure error) and counts as a miss. Chunk-level `Recall@k` is a different metric: `scripts/lr_report.py`.

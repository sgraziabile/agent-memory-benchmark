# Benchmark Execution Summary

- **Generated at:** 2026-11-11 14:10:23
- **Total Runs:** 1

| Scenario                      | Model                 | Prompt           | Level                  | Turns | Pass Rate | Avg Latency | Total Tokens | Cost (USD) | Errors | Retrieval Recall |
| :---------------------------- | :-------------------- | :--------------- | :--------------------- | ----: | --------: | ----------: | -----------: | ---------: | -----: | ---------------: |
| `document_multihop_bridge_01` | `deepseek-v4.1-flash` | `baseline_react` | `level_2_episodic_rag` |     5 | **80.0%** |     9027 ms |        6,278 |  $0.000872 |      0 |      3/4 (75.0%) |

`Retrieval Recall` = passed / declared `must_retrieve_chunk` pins (rule-level; a sibling sub-chunk of the gold document satisfies a pin). `*` marks a run where at least one declared probe was never evaluated (infrastructure error) and counts as a miss. Chunk-level `Recall@k` is a different metric: `scripts/lr_report.py`.

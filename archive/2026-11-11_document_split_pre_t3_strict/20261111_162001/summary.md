# Benchmark Execution Summary

- **Generated at:** 2026-11-11 16:20:29
- **Total Runs:** 1

| Scenario                     | Model                 | Prompt           | Level                  | Turns | Pass Rate | Avg Latency | Total Tokens | Cost (USD) | Errors | Retrieval Recall |
| :--------------------------- | :-------------------- | :--------------- | :--------------------- | ----: | --------: | ----------: | -----------: | ---------: | -----: | ---------------: |
| `document_split_evidence_01` | `deepseek-v4.1-flash` | `baseline_react` | `level_2_episodic_rag` |     5 | **60.0%** |     5425 ms |       21,743 |  $0.002419 |      0 |     3/3 (100.0%) |

`Retrieval Recall` = passed / declared `must_retrieve_chunk` pins (rule-level; a sibling sub-chunk of the gold document satisfies a pin). `*` marks a run where at least one declared probe was never evaluated (infrastructure error) and counts as a miss. Chunk-level `Recall@k` is a different metric: `scripts/lr_report.py`.

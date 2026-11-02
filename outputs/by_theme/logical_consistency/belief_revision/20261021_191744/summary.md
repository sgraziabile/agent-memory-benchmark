# Benchmark Execution Summary

- **Generated at:** 2026-10-21 19:17:44
- **Total Runs:** 2

| Scenario             | Model         | Prompt           | Level                    | Turns | Pass Rate  | Avg Latency | Total Tokens | Errors |
| :------------------- | :------------ | :--------------- | :----------------------- | ----: | ---------: | ----------: | -----------: | -----: |
| `belief_revision_01` | `qwen3.6-27b` | `baseline_react` | `level_0_reactive`       |     5 |  **28.6%** |     7951 ms |        2,404 |      0 |
| `belief_revision_01` | `qwen3.6-27b` | `baseline_react` | `level_1_working_memory` |     5 | **100.0%** |    13717 ms |        3,128 |      0 |

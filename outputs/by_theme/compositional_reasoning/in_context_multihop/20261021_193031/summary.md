# Benchmark Execution Summary

- **Generated at:** 2026-10-21 19:30:31
- **Total Runs:** 11

| Scenario                    | Model         | Prompt           | Level                    | Turns | Pass Rate  | Avg Latency | Total Tokens | Errors |
| :-------------------------- | :------------ | :--------------- | :----------------------- | ----: | ---------: | ----------: | -----------: | -----: |
| `in_context_multihop_01`    | `gpt-oss-20b` | `baseline_react` | `level_1_working_memory` |     1 | **100.0%** |     4973 ms |          295 |      0 |
| `temporal_multi_hop_01`     | `gpt-oss-20b` | `baseline_react` | `level_1_working_memory` |     4 | **100.0%** |     2634 ms |        1,369 |      0 |
| `belief_revision_01`        | `gpt-oss-20b` | `baseline_react` | `level_1_working_memory` |     5 | **100.0%** |     2823 ms |        1,733 |      0 |
| `explicit_forget_01`        | `gpt-oss-20b` | `baseline_react` | `level_1_working_memory` |     5 |  **60.0%** |     3064 ms |        2,111 |      0 |
| `in_context_control_01`     | `gpt-oss-20b` | `baseline_react` | `level_1_working_memory` |     1 | **100.0%** |     2561 ms |          293 |      0 |
| `negative_hallucination_01` | `gpt-oss-20b` | `baseline_react` | `level_1_working_memory` |     4 | **100.0%** |     2875 ms |        1,530 |      0 |
| `attrition_01`              | `gpt-oss-20b` | `baseline_react` | `level_1_working_memory` |     8 | **100.0%** |     3599 ms |        6,433 |      0 |
| `in_context_needle_01`      | `gpt-oss-20b` | `baseline_react` | `level_1_working_memory` |     1 | **100.0%** |     2762 ms |          359 |      0 |
| `needle_haystack_01`        | `gpt-oss-20b` | `baseline_react` | `level_1_working_memory` |     5 | **100.0%** |     3275 ms |        3,737 |      0 |
| `adversarial_forget_01`     | `gpt-oss-20b` | `baseline_react` | `level_1_working_memory` |     4 | **100.0%** |     3301 ms |        1,304 |      0 |
| `adversarial_forget_02`     | `gpt-oss-20b` | `baseline_react` | `level_1_working_memory` |     5 | **100.0%** |     3665 ms |        2,073 |      0 |

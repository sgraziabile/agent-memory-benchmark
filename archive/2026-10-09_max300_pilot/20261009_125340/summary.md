# Benchmark Execution Summary

- **Generated at:** 2026-10-09 12:53:40
- **Total Runs:** 9

| Scenario                    | Model         | Prompt           | Level                    | Turns | Pass Rate  | Avg Latency | Total Tokens | Errors |
| :-------------------------- | :------------ | :--------------- | :----------------------- | ----: | ---------: | ----------: | -----------: | -----: |
| `in_context_multihop_01`    | `gpt-oss-20b` | `baseline_react` | `level_1_working_memory` |     1 | **100.0%** |    16381 ms |          282 |      0 |
| `temporal_multi_hop_01`     | `gpt-oss-20b` | `baseline_react` | `level_1_working_memory` |     4 | **100.0%** |    16934 ms |        1,404 |      0 |
| `belief_revision_01`        | `gpt-oss-20b` | `baseline_react` | `level_1_working_memory` |     5 | **100.0%** |    11233 ms |        1,750 |      0 |
| `explicit_forget_01`        | `gpt-oss-20b` | `baseline_react` | `level_1_working_memory` |     5 |  **60.0%** |    14263 ms |        1,778 |      0 |
| `in_context_control_01`     | `gpt-oss-20b` | `baseline_react` | `level_1_working_memory` |     1 | **100.0%** |    16738 ms |          293 |      0 |
| `negative_hallucination_01` | `gpt-oss-20b` | `baseline_react` | `level_1_working_memory` |     3 |  **75.0%** |    15153 ms |        1,020 |      0 |
| `attrition_01`              | `gpt-oss-20b` | `baseline_react` | `level_1_working_memory` |     8 | **100.0%** |     4066 ms |        5,272 |      0 |
| `in_context_needle_01`      | `gpt-oss-20b` | `baseline_react` | `level_1_working_memory` |     1 | **100.0%** |     7947 ms |          359 |      0 |
| `needle_haystack_01`        | `gpt-oss-20b` | `baseline_react` | `level_1_working_memory` |     5 | **100.0%** |     3532 ms |        3,891 |      0 |

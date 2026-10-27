# Benchmark Execution Summary

- **Generated at:** 2026-10-14 14:07:07
- **Total Runs:** 11

| Scenario                    | Model             | Prompt           | Level                    | Turns | Pass Rate  | Avg Latency | Total Tokens | Errors |
| :-------------------------- | :---------------- | :--------------- | :----------------------- | ----: | ---------: | ----------: | -----------: | -----: |
| `in_context_multihop_01`    | `deepseek-v4-pro` | `baseline_react` | `level_1_working_memory` |     1 | **100.0%** |    11836 ms |          232 |      0 |
| `temporal_multi_hop_01`     | `deepseek-v4-pro` | `baseline_react` | `level_1_working_memory` |     4 | **100.0%** |    13721 ms |        1,405 |      0 |
| `belief_revision_01`        | `deepseek-v4-pro` | `baseline_react` | `level_1_working_memory` |     5 | **100.0%** |     8308 ms |        1,372 |      0 |
| `explicit_forget_01`        | `deepseek-v4-pro` | `baseline_react` | `level_1_working_memory` |     5 | **100.0%** |     7846 ms |        1,468 |      0 |
| `in_context_control_01`     | `deepseek-v4-pro` | `baseline_react` | `level_1_working_memory` |     1 | **100.0%** |     6472 ms |          176 |      0 |
| `negative_hallucination_01` | `deepseek-v4-pro` | `baseline_react` | `level_1_working_memory` |     3 |   **0.0%** |     6545 ms |          746 |      0 |
| `attrition_01`              | `deepseek-v4-pro` | `baseline_react` | `level_1_working_memory` |     8 | **100.0%** |     9949 ms |        5,424 |      0 |
| `in_context_needle_01`      | `deepseek-v4-pro` | `baseline_react` | `level_1_working_memory` |     1 | **100.0%** |     9222 ms |          220 |      0 |
| `needle_haystack_01`        | `deepseek-v4-pro` | `baseline_react` | `level_1_working_memory` |     5 | **100.0%** |     7866 ms |        3,010 |      0 |
| `adversarial_forget_01`     | `deepseek-v4-pro` | `baseline_react` | `level_1_working_memory` |     4 | **100.0%** |    14107 ms |        1,824 |      0 |
| `adversarial_forget_02`     | `deepseek-v4-pro` | `baseline_react` | `level_1_working_memory` |     5 |  **50.0%** |    15340 ms |        3,564 |      0 |

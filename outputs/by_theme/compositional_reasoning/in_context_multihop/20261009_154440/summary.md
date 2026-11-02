# Benchmark Execution Summary

- **Generated at:** 2026-10-09 15:44:40
- **Total Runs:** 9

| Scenario                    | Model           | Prompt           | Level                    | Turns | Pass Rate  | Avg Latency | Total Tokens | Errors |
| :-------------------------- | :-------------- | :--------------- | :----------------------- | ----: | ---------: | ----------: | -----------: | -----: |
| `in_context_multihop_01`    | `glm-5.3-flash` | `baseline_react` | `level_1_working_memory` |     1 | **100.0%** |    20366 ms |          564 |      0 |
| `temporal_multi_hop_01`     | `glm-5.3-flash` | `baseline_react` | `level_1_working_memory` |     4 | **100.0%** |    13988 ms |        2,117 |      0 |
| `belief_revision_01`        | `glm-5.3-flash` | `baseline_react` | `level_1_working_memory` |     5 | **100.0%** |     6499 ms |        1,965 |      0 |
| `explicit_forget_01`        | `glm-5.3-flash` | `baseline_react` | `level_1_working_memory` |     5 | **100.0%** |     8476 ms |        4,690 |      0 |
| `in_context_control_01`     | `glm-5.3-flash` | `baseline_react` | `level_1_working_memory` |     1 | **100.0%** |     7451 ms |          258 |      0 |
| `negative_hallucination_01` | `glm-5.3-flash` | `baseline_react` | `level_1_working_memory` |     3 |   **0.0%** |     6614 ms |        1,178 |      0 |
| `attrition_01`              | `glm-5.3-flash` | `baseline_react` | `level_1_working_memory` |     8 | **100.0%** |    10824 ms |        7,837 |      0 |
| `in_context_needle_01`      | `glm-5.3-flash` | `baseline_react` | `level_1_working_memory` |     1 | **100.0%** |    12207 ms |          365 |      0 |
| `needle_haystack_01`        | `glm-5.3-flash` | `baseline_react` | `level_1_working_memory` |     5 | **100.0%** |     7501 ms |        3,846 |      0 |

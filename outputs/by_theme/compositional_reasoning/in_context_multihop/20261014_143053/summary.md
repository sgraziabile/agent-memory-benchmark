# Benchmark Execution Summary

- **Generated at:** 2026-10-14 14:30:53
- **Total Runs:** 11

| Scenario                    | Model        | Prompt           | Level                    | Turns | Pass Rate  | Avg Latency | Total Tokens | Errors |
| :-------------------------- | :----------- | :--------------- | :----------------------- | ----: | ---------: | ----------: | -----------: | -----: |
| `in_context_multihop_01`    | `qwen3-235b` | `baseline_react` | `level_1_working_memory` |     1 | **100.0%** |    14628 ms |          221 |      0 |
| `temporal_multi_hop_01`     | `qwen3-235b` | `baseline_react` | `level_1_working_memory` |     4 | **100.0%** |    25169 ms |        1,201 |      0 |
| `belief_revision_01`        | `qwen3-235b` | `baseline_react` | `level_1_working_memory` |     5 | **100.0%** |    12858 ms |        1,567 |      0 |
| `explicit_forget_01`        | `qwen3-235b` | `baseline_react` | `level_1_working_memory` |     5 |  **80.0%** |    11110 ms |        2,203 |      0 |
| `in_context_control_01`     | `qwen3-235b` | `baseline_react` | `level_1_working_memory` |     1 | **100.0%** |    16730 ms |          336 |      0 |
| `negative_hallucination_01` | `qwen3-235b` | `baseline_react` | `level_1_working_memory` |     3 | **100.0%** |     7017 ms |          635 |      0 |
| `attrition_01`              | `qwen3-235b` | `baseline_react` | `level_1_working_memory` |     8 | **100.0%** |    12541 ms |        5,662 |      0 |
| `in_context_needle_01`      | `qwen3-235b` | `baseline_react` | `level_1_working_memory` |     1 | **100.0%** |     5209 ms |          238 |      0 |
| `needle_haystack_01`        | `qwen3-235b` | `baseline_react` | `level_1_working_memory` |     5 | **100.0%** |    20569 ms |        3,819 |      0 |
| `adversarial_forget_01`     | `qwen3-235b` | `baseline_react` | `level_1_working_memory` |     4 | **100.0%** |    23112 ms |        1,191 |      0 |
| `adversarial_forget_02`     | `qwen3-235b` | `baseline_react` | `level_1_working_memory` |     5 | **100.0%** |    14702 ms |        2,373 |      0 |

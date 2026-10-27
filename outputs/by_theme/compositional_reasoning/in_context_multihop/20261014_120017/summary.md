# Benchmark Execution Summary

- **Generated at:** 2026-10-14 12:00:17
- **Total Runs:** 11

| Scenario                    | Model         | Prompt           | Level                    | Turns | Pass Rate  | Avg Latency | Total Tokens | Errors |
| :-------------------------- | :------------ | :--------------- | :----------------------- | ----: | ---------: | ----------: | -----------: | -----: |
| `in_context_multihop_01`    | `qwen3.6-27b` | `baseline_react` | `level_1_working_memory` |     1 | **100.0%** |    62259 ms |          613 |      0 |
| `temporal_multi_hop_01`     | `qwen3.6-27b` | `baseline_react` | `level_1_working_memory` |     4 | **100.0%** |    23991 ms |        2,898 |      0 |
| `belief_revision_01`        | `qwen3.6-27b` | `baseline_react` | `level_1_working_memory` |     5 | **100.0%** |    10213 ms |        3,128 |      0 |
| `explicit_forget_01`        | `qwen3.6-27b` | `baseline_react` | `level_1_working_memory` |     5 |  **40.0%** |    11844 ms |        4,867 |      0 |
| `in_context_control_01`     | `qwen3.6-27b` | `baseline_react` | `level_1_working_memory` |     1 | **100.0%** |    26501 ms |          915 |      0 |
| `negative_hallucination_01` | `qwen3.6-27b` | `baseline_react` | `level_1_working_memory` |     3 |  **75.0%** |    10748 ms |        2,241 |      0 |
| `attrition_01`              | `qwen3.6-27b` | `baseline_react` | `level_1_working_memory` |     8 | **100.0%** |     9306 ms |        8,867 |      0 |
| `in_context_needle_01`      | `qwen3.6-27b` | `baseline_react` | `level_1_working_memory` |     1 | **100.0%** |     7055 ms |          552 |      0 |
| `needle_haystack_01`        | `qwen3.6-27b` | `baseline_react` | `level_1_working_memory` |     5 | **100.0%** |     8802 ms |        4,839 |      0 |
| `adversarial_forget_01`     | `qwen3.6-27b` | `baseline_react` | `level_1_working_memory` |     4 |   **0.0%** |    10713 ms |        3,554 |      0 |
| `adversarial_forget_02`     | `qwen3.6-27b` | `baseline_react` | `level_1_working_memory` |     5 | **100.0%** |    13700 ms |        5,528 |      0 |

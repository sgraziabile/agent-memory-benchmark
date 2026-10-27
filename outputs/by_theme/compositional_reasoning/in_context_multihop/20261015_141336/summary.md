# Benchmark Execution Summary

- **Generated at:** 2026-10-15 14:13:36
- **Total Runs:** 11

| Scenario                    | Model         | Prompt           | Level              | Turns | Pass Rate  | Avg Latency | Total Tokens | Errors |
| :-------------------------- | :------------ | :--------------- | :----------------- | ----: | ---------: | ----------: | -----------: | -----: |
| `in_context_multihop_01`    | `qwen3.6-27b` | `baseline_react` | `level_0_reactive` |     1 |   **0.0%** |        0 ms |          N/A |      1 |
| `temporal_multi_hop_01`     | `qwen3.6-27b` | `baseline_react` | `level_0_reactive` |     4 |  **50.0%** |    27470 ms |        1,274 |      1 |
| `belief_revision_01`        | `qwen3.6-27b` | `baseline_react` | `level_0_reactive` |     5 |  **28.6%** |    69898 ms |        2,404 |      0 |
| `explicit_forget_01`        | `qwen3.6-27b` | `baseline_react` | `level_0_reactive` |     5 |  **40.0%** |    20393 ms |        3,303 |      0 |
| `in_context_control_01`     | `qwen3.6-27b` | `baseline_react` | `level_0_reactive` |     1 | **100.0%** |    21613 ms |          915 |      0 |
| `negative_hallucination_01` | `qwen3.6-27b` | `baseline_react` | `level_0_reactive` |     4 | **100.0%** |    24885 ms |        2,410 |      0 |
| `attrition_01`              | `qwen3.6-27b` | `baseline_react` | `level_0_reactive` |     8 |  **66.7%** |   105725 ms |        4,937 |      0 |
| `in_context_needle_01`      | `qwen3.6-27b` | `baseline_react` | `level_0_reactive` |     1 | **100.0%** |    37982 ms |          552 |      0 |
| `needle_haystack_01`        | `qwen3.6-27b` | `baseline_react` | `level_0_reactive` |     5 |   **0.0%** |    46777 ms |        3,123 |      0 |
| `adversarial_forget_01`     | `qwen3.6-27b` | `baseline_react` | `level_0_reactive` |     4 | **100.0%** |    15313 ms |        3,151 |      0 |
| `adversarial_forget_02`     | `qwen3.6-27b` | `baseline_react` | `level_0_reactive` |     5 | **100.0%** |    16180 ms |        4,160 |      0 |

# Benchmark Execution Summary

- **Generated at:** 2026-10-15 14:05:01
- **Total Runs:** 11

| Scenario                    | Model        | Prompt           | Level              | Turns | Pass Rate  | Avg Latency | Total Tokens | Errors |
| :-------------------------- | :----------- | :--------------- | :----------------- | ----: | ---------: | ----------: | -----------: | -----: |
| `in_context_multihop_01`    | `qwen3-235b` | `baseline_react` | `level_0_reactive` |     1 |   **0.0%** |        0 ms |          N/A |      1 |
| `temporal_multi_hop_01`     | `qwen3-235b` | `baseline_react` | `level_0_reactive` |     4 |  **50.0%** |    37027 ms |          556 |      0 |
| `belief_revision_01`        | `qwen3-235b` | `baseline_react` | `level_0_reactive` |     5 |  **28.6%** |    68378 ms |          827 |      0 |
| `explicit_forget_01`        | `qwen3-235b` | `baseline_react` | `level_0_reactive` |     5 |  **40.0%** |    14030 ms |          847 |      0 |
| `in_context_control_01`     | `qwen3-235b` | `baseline_react` | `level_0_reactive` |     1 | **100.0%** |    15263 ms |          327 |      0 |
| `negative_hallucination_01` | `qwen3-235b` | `baseline_react` | `level_0_reactive` |     4 | **100.0%** |     7803 ms |          597 |      0 |
| `attrition_01`              | `qwen3-235b` | `baseline_react` | `level_0_reactive` |     8 |  **66.7%** |    13584 ms |        1,740 |      0 |
| `in_context_needle_01`      | `qwen3-235b` | `baseline_react` | `level_0_reactive` |     1 | **100.0%** |     8769 ms |          238 |      0 |
| `needle_haystack_01`        | `qwen3-235b` | `baseline_react` | `level_0_reactive` |     5 |   **0.0%** |   135666 ms |        1,182 |      0 |
| `adversarial_forget_01`     | `qwen3-235b` | `baseline_react` | `level_0_reactive` |     4 | **100.0%** |    18857 ms |          814 |      0 |
| `adversarial_forget_02`     | `qwen3-235b` | `baseline_react` | `level_0_reactive` |     5 | **100.0%** |     9318 ms |        1,061 |      0 |

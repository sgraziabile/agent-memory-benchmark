# Benchmark Execution Summary

- **Generated at:** 2026-10-15 13:50:06
- **Total Runs:** 11

| Scenario                    | Model             | Prompt           | Level              | Turns | Pass Rate  | Avg Latency | Total Tokens | Errors |
| :-------------------------- | :---------------- | :--------------- | :----------------- | ----: | ---------: | ----------: | -----------: | -----: |
| `in_context_multihop_01`    | `deepseek-v4-pro` | `baseline_react` | `level_0_reactive` |     1 | **100.0%** |    13920 ms |          252 |      0 |
| `temporal_multi_hop_01`     | `deepseek-v4-pro` | `baseline_react` | `level_0_reactive` |     4 |  **50.0%** |    21917 ms |          660 |      0 |
| `belief_revision_01`        | `deepseek-v4-pro` | `baseline_react` | `level_0_reactive` |     5 |  **28.6%** |    20356 ms |          708 |      0 |
| `explicit_forget_01`        | `deepseek-v4-pro` | `baseline_react` | `level_0_reactive` |     5 |  **40.0%** |    23702 ms |        1,040 |      0 |
| `in_context_control_01`     | `deepseek-v4-pro` | `baseline_react` | `level_0_reactive` |     1 | **100.0%** |    15386 ms |          175 |      0 |
| `negative_hallucination_01` | `deepseek-v4-pro` | `baseline_react` | `level_0_reactive` |     4 | **100.0%** |    13450 ms |          651 |      0 |
| `attrition_01`              | `deepseek-v4-pro` | `baseline_react` | `level_0_reactive` |     8 |  **66.7%** |    35950 ms |        1,965 |      0 |
| `in_context_needle_01`      | `deepseek-v4-pro` | `baseline_react` | `level_0_reactive` |     1 | **100.0%** |    14398 ms |          219 |      0 |
| `needle_haystack_01`        | `deepseek-v4-pro` | `baseline_react` | `level_0_reactive` |     5 |   **0.0%** |    13998 ms |        1,196 |      0 |
| `adversarial_forget_01`     | `deepseek-v4-pro` | `baseline_react` | `level_0_reactive` |     4 | **100.0%** |     8381 ms |          832 |      0 |
| `adversarial_forget_02`     | `deepseek-v4-pro` | `baseline_react` | `level_0_reactive` |     5 | **100.0%** |    10292 ms |        1,229 |      0 |

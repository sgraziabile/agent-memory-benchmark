# Benchmark Execution Summary

- **Generated at:** 2026-10-09 12:12:52
- **Total Runs:** 9

| Scenario                    | Model         | Prompt           | Level              | Turns | Pass Rate  | Avg Latency | Total Tokens | Errors |
| :-------------------------- | :------------ | :--------------- | :----------------- | ----: | ---------: | ----------: | -----------: | -----: |
| `in_context_multihop_01`    | `gpt-oss-20b` | `baseline_react` | `level_0_reactive` |     1 | **100.0%** |     6876 ms |          293 |      0 |
| `temporal_multi_hop_01`     | `gpt-oss-20b` | `baseline_react` | `level_0_reactive` |     4 |  **50.0%** |     4064 ms |        1,103 |      0 |
| `belief_revision_01`        | `gpt-oss-20b` | `baseline_react` | `level_0_reactive` |     5 |  **28.6%** |     3133 ms |        1,331 |      0 |
| `explicit_forget_01`        | `gpt-oss-20b` | `baseline_react` | `level_0_reactive` |     5 |  **40.0%** |     3116 ms |        1,502 |      0 |
| `in_context_control_01`     | `gpt-oss-20b` | `baseline_react` | `level_0_reactive` |     1 | **100.0%** |     3001 ms |          293 |      0 |
| `negative_hallucination_01` | `gpt-oss-20b` | `baseline_react` | `level_0_reactive` |     3 |  **75.0%** |     4008 ms |          928 |      0 |
| `attrition_01`              | `gpt-oss-20b` | `baseline_react` | `level_0_reactive` |     8 |  **66.7%** |     4007 ms |        2,425 |      0 |
| `in_context_needle_01`      | `gpt-oss-20b` | `baseline_react` | `level_0_reactive` |     1 |   **0.0%** |     3319 ms |          362 |      0 |
| `needle_haystack_01`        | `gpt-oss-20b` | `baseline_react` | `level_0_reactive` |     5 |   **0.0%** |     4395 ms |        1,919 |      0 |

# Benchmark Execution Summary

- **Generated at:** 2026-10-27 10:20:59
- **Total Runs:** 3

| Scenario               | Model               | Prompt           | Level                    | Turns | Pass Rate  | Avg Latency | Total Tokens | Errors |
| :--------------------- | :------------------ | :--------------- | :----------------------- | ----: | ---------: | ----------: | -----------: | -----: |
| `overhead_baseline_01` | `deepseek-v4-flash` | `baseline_react` | `level_0_reactive`       |    10 | **100.0%** |     1764 ms |        1,031 |      0 |
| `overhead_baseline_01` | `deepseek-v4-flash` | `baseline_react` | `level_1_working_memory` |    10 | **100.0%** |     1501 ms |        1,661 |      0 |
| `overhead_baseline_01` | `deepseek-v4-flash` | `baseline_react` | `level_2_episodic_rag`   |    10 | **100.0%** |     3664 ms |        3,246 |      0 |

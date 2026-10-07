# Consolidated Benchmark Summary

- **Generated at:** 2026-10-07 13:24:44 UTC
- **Total Runs:** 11
- **Models:** gemini-3.5-flash-lite
- **Prompts:** baseline_react
- **Agent Levels:** level_0_reactive

## Logical Consistency

| Test Category | Timestamp | Scenario | Model | Prompt | Level | Turns | Pass Rate | Avg Latency (ms) | Total Tokens | Errors |
| :-- | :-- | :-- | :-- | :-- | :-- | --: | --: | --: | --: | --: |
| belief_revision | 20261001_154541 | belief_revision_01 | gemini-3.5-flash-lite | baseline_react | level_0_reactive | 5 | 28.6% | 10929 | 715 | 0 |
| belief_revision | 20261007_102040 | belief_revision_01 | gemini-3.5-flash-lite | baseline_react | level_0_reactive | 5 | 28.6% | 1597 | 653 | 0 |
| explicit_forget | 20261001_154921 | explicit_forget_01 | gemini-3.5-flash-lite | baseline_react | level_0_reactive | 5 | 40.0% | 1875 | 758 | 0 |
| in_context_control | 20261001_155037 | in_context_control_01 | gemini-3.5-flash-lite | baseline_react | level_0_reactive | 1 | 100.0% | 1709 | 192 | 0 |

## Retention & Persistence

| Test Category | Timestamp | Scenario | Model | Prompt | Level | Turns | Pass Rate | Avg Latency (ms) | Total Tokens | Errors |
| :-- | :-- | :-- | :-- | :-- | :-- | --: | --: | --: | --: | --: |
| attrition | 20261001_154302 | attrition_01 | gemini-3.5-flash-lite | baseline_react | level_0_reactive | 8 | 66.7% | 1844 | 1477 | 0 |
| attrition | 20261005_151639 | attrition_01 | gemini-3.5-flash-lite | baseline_react | level_0_reactive | 8 | 66.7% | 9160 | 1508 | 0 |
| attrition | 20261005_160616 | attrition_01 | gemini-3.5-flash-lite | baseline_react | level_0_reactive | 8 | 66.7% | 1156 | 1492 | 0 |
| attrition | 20261005_172610 | attrition_01 | gemini-3.5-flash-lite | baseline_react | level_0_reactive | 8 | 66.7% | 1120 | 1537 | 0 |
| needle_haystack | 20261001_155108 | needle_haystack_01 | gemini-3.5-flash-lite | baseline_react | level_0_reactive | 5 | 0.0% | 2504 | 937 | 0 |

## Compositional Reasoning

| Test Category | Timestamp | Scenario | Model | Prompt | Level | Turns | Pass Rate | Avg Latency (ms) | Total Tokens | Errors |
| :-- | :-- | :-- | :-- | :-- | :-- | --: | --: | --: | --: | --: |
| temporal_multi_hop | 20261001_155151 | temporal_multi_hop_01 | gemini-3.5-flash-lite | baseline_react | level_0_reactive | 4 | 50.0% | 1867 | 513 | 0 |
| temporal_multi_hop | 20261005_172812 | temporal_multi_hop_01 | gemini-3.5-flash-lite | baseline_react | level_0_reactive | 4 | 50.0% | 1062 | 527 | 0 |

## Robustness & Security

_No runs yet — scaffolding prepared for future tests._

## Computational Cost

_No runs yet — scaffolding prepared for future tests._

## Overall

| Runs | Avg Pass Rate | Avg Latency (ms) | Total Turns | Total Tokens | Total Errors |
| --: | --: | --: | --: | --: | --: |
| 11 | 51.3% | 3166 | 61 | 10309 | 0 |

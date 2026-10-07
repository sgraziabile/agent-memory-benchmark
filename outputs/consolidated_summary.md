# Consolidated Benchmark Summary

- **Generated at:** 2026-10-07 19:47:10 UTC
- **Total Runs:** 21
- **Models:** gemini-3.5-flash-lite, gemini-3.8-flash
- **Prompts:** baseline_react
- **Agent Levels:** level_0_reactive, level_1_working_memory

## Logical Consistency

| Test Category | Timestamp | Scenario | Model | Prompt | Level | Turns | Pass Rate | Avg Latency (ms) | Total Tokens | Errors |
| :-- | :-- | :-- | :-- | :-- | :-- | --: | --: | --: | --: | --: |
| belief_revision | 20261001_154541 | belief_revision_01 | gemini-3.5-flash-lite | baseline_react | level_0_reactive | 5 | 28.6% | 10929 | 715 | 0 |
| belief_revision | 20261007_102040 | belief_revision_01 | gemini-3.5-flash-lite | baseline_react | level_0_reactive | 5 | 28.6% | 1597 | 653 | 0 |
| belief_revision | 20261007_122711 | belief_revision_01 | gemini-3.5-flash-lite | baseline_react | level_0_reactive | 5 | 28.6% | 10851 | 665 | 0 |
| belief_revision | 20261007_122809 | belief_revision_01 | gemini-3.5-flash-lite | baseline_react | level_0_reactive | 5 | 28.6% | 2824 | 644 | 0 |
| belief_revision | 20261007_145746 | belief_revision_01 | gemini-3.8-flash | baseline_react | level_1_working_memory | 5 | 100.0% | 79367 | 394 | 4 |
| belief_revision | 20261007_163624 | belief_revision_01 | gemini-3.5-flash-lite | baseline_react | level_1_working_memory | 5 | 71.4% | 1324 | 1169 | 0 |
| belief_revision | 20261007_163830 | belief_revision_01 | gemini-3.5-flash-lite | baseline_react | level_1_working_memory | 5 | 71.4% | 2487 | 1260 | 0 |
| explicit_forget | 20261001_154921 | explicit_forget_01 | gemini-3.5-flash-lite | baseline_react | level_0_reactive | 5 | 40.0% | 1875 | 758 | 0 |
| in_context_control | 20261001_155037 | in_context_control_01 | gemini-3.5-flash-lite | baseline_react | level_0_reactive | 1 | 100.0% | 1709 | 192 | 0 |
| negative_hallucination | 20261007_133019 | negative_hallucination_01 | gemini-3.5-flash-lite | baseline_react | level_0_reactive | 3 | 75.0% | 1200 | 438 | 0 |

## Retention & Persistence

| Test Category | Timestamp | Scenario | Model | Prompt | Level | Turns | Pass Rate | Avg Latency (ms) | Total Tokens | Errors |
| :-- | :-- | :-- | :-- | :-- | :-- | --: | --: | --: | --: | --: |
| attrition | 20261001_154302 | attrition_01 | gemini-3.5-flash-lite | baseline_react | level_0_reactive | 8 | 66.7% | 1844 | 1477 | 0 |
| attrition | 20261005_151639 | attrition_01 | gemini-3.5-flash-lite | baseline_react | level_0_reactive | 8 | 66.7% | 9160 | 1508 | 0 |
| attrition | 20261005_160616 | attrition_01 | gemini-3.5-flash-lite | baseline_react | level_0_reactive | 8 | 66.7% | 1156 | 1492 | 0 |
| attrition | 20261005_172610 | attrition_01 | gemini-3.5-flash-lite | baseline_react | level_0_reactive | 8 | 66.7% | 1120 | 1537 | 0 |
| attrition | 20261007_164610 | attrition_01 | gemini-3.5-flash-lite | baseline_react | level_1_working_memory | 8 | 100.0% | 8651 | 4443 | 0 |
| in_context_needle | 20261007_132846 | in_context_needle_01 | gemini-3.5-flash-lite | baseline_react | level_0_reactive | 1 | 100.0% | 1874 | 237 | 0 |
| in_context_needle | 20261007_152516 | in_context_needle_01 | gemini-3.5-flash-lite | baseline_react | level_0_reactive | 1 | 100.0% | 1877 | 237 | 0 |
| needle_haystack | 20261001_155108 | needle_haystack_01 | gemini-3.5-flash-lite | baseline_react | level_0_reactive | 5 | 0.0% | 2504 | 937 | 0 |

## Compositional Reasoning

| Test Category | Timestamp | Scenario | Model | Prompt | Level | Turns | Pass Rate | Avg Latency (ms) | Total Tokens | Errors |
| :-- | :-- | :-- | :-- | :-- | :-- | --: | --: | --: | --: | --: |
| in_context_multihop | 20261007_132748 | in_context_multihop_01 | gemini-3.5-flash-lite | baseline_react | level_0_reactive | 1 | 100.0% | 3001 | 202 | 0 |
| temporal_multi_hop | 20261001_155151 | temporal_multi_hop_01 | gemini-3.5-flash-lite | baseline_react | level_0_reactive | 4 | 50.0% | 1867 | 513 | 0 |
| temporal_multi_hop | 20261005_172812 | temporal_multi_hop_01 | gemini-3.5-flash-lite | baseline_react | level_0_reactive | 4 | 50.0% | 1062 | 527 | 0 |

## Robustness & Security

_No runs yet — scaffolding prepared for future tests._

## Computational Cost

_No runs yet — scaffolding prepared for future tests._

## Overall

| Runs | Avg Pass Rate | Avg Latency (ms) | Total Turns | Total Tokens | Total Errors |
| --: | --: | --: | --: | --: | --: |
| 21 | 63.8% | 7061 | 100 | 19998 | 4 |

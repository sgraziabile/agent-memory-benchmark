# Consolidated Benchmark Summary

- **Generated at:** 2026-10-27 12:46:36 UTC
- **Total Runs:** 112
- **Models:** deepseek-v4-flash, deepseek-v4-pro, glm-5.3-flash, gpt-oss-20b, qwen3-235b, qwen3.6-27b
- **Prompts:** baseline_react, minimal
- **Agent Levels:** level_0_reactive, level_1_working_memory, level_2_episodic_rag

## Logical Consistency

| Test Category | Timestamp | Scenario | Model | Prompt | Level | Turns | Pass Rate | Avg Latency (ms) | Total Tokens | Errors |
| :-- | :-- | :-- | :-- | :-- | :-- | --: | --: | --: | --: | --: |
| belief_revision | 20261009_154440 | belief_revision_01 | glm-5.3-flash | baseline_react | level_1_working_memory | 5 | 100.0% | 6499 | 1965 | 0 |
| belief_revision | 20261014_120017 | belief_revision_01 | qwen3.6-27b | baseline_react | level_1_working_memory | 5 | 100.0% | 10213 | 3128 | 0 |
| belief_revision | 20261014_140706 | belief_revision_01 | deepseek-v4-pro | baseline_react | level_1_working_memory | 5 | 100.0% | 8308 | 1372 | 0 |
| belief_revision | 20261014_143053 | belief_revision_01 | qwen3-235b | baseline_react | level_1_working_memory | 5 | 100.0% | 12858 | 1567 | 0 |
| belief_revision | 20261015_135006 | belief_revision_01 | deepseek-v4-pro | baseline_react | level_0_reactive | 5 | 28.6% | 20356 | 708 | 0 |
| belief_revision | 20261015_140501 | belief_revision_01 | qwen3-235b | baseline_react | level_0_reactive | 5 | 28.6% | 68378 | 827 | 0 |
| belief_revision | 20261015_141336 | belief_revision_01 | qwen3.6-27b | baseline_react | level_0_reactive | 5 | 28.6% | 69898 | 2404 | 0 |
| belief_revision | 20261021_191744 | belief_revision_01 | qwen3.6-27b | baseline_react | level_0_reactive | 5 | 28.6% | 7951 | 2404 | 0 |
| belief_revision | 20261021_191744 | belief_revision_01 | qwen3.6-27b | baseline_react | level_1_working_memory | 5 | 100.0% | 13717 | 3128 | 0 |
| belief_revision | 20261021_193031 | belief_revision_01 | gpt-oss-20b | baseline_react | level_1_working_memory | 5 | 100.0% | 2823 | 1733 | 0 |
| belief_revision | 20261023_134219 | belief_revision_01 | deepseek-v4-flash | baseline_react | level_1_working_memory | 5 | 100.0% | 2401 | 1128 | 0 |
| belief_revision | 20261023_134441 | belief_revision_01 | deepseek-v4-flash | baseline_react | level_2_episodic_rag | 5 | 0.0% | 0 | 0 | 5 |
| belief_revision | 20261023_153323 | belief_revision_01 | deepseek-v4-flash | baseline_react | level_2_episodic_rag | 5 | 0.0% | 0 | 0 | 5 |
| belief_revision | 20261023_153942 | belief_revision_01 | deepseek-v4-flash | baseline_react | level_2_episodic_rag | 5 | 100.0% | 3607 | 1185 | 0 |
| belief_revision | 20261026_121129 | derived_belief_revision_01 | deepseek-v4-flash | baseline_react | level_2_episodic_rag | 13 | 100.0% | 3241 | 7902 | 0 |
| belief_revision | 20261026_125446 | backward_modus_tollens_01 | deepseek-v4-flash | baseline_react | level_2_episodic_rag | 13 | 50.0% | 3586 | 8246 | 0 |
| explicit_forget | 20261009_152950 | explicit_forget_01 | glm-5.3-flash | baseline_react | level_1_working_memory | 5 | 100.0% | 14587 | 4015 | 0 |
| explicit_forget | 20261009_154440 | explicit_forget_01 | glm-5.3-flash | baseline_react | level_1_working_memory | 5 | 100.0% | 8476 | 4690 | 0 |
| explicit_forget | 20261014_120017 | explicit_forget_01 | qwen3.6-27b | baseline_react | level_1_working_memory | 5 | 40.0% | 11844 | 4867 | 0 |
| explicit_forget | 20261014_140706 | explicit_forget_01 | deepseek-v4-pro | baseline_react | level_1_working_memory | 5 | 100.0% | 7846 | 1468 | 0 |
| explicit_forget | 20261014_143053 | explicit_forget_01 | qwen3-235b | baseline_react | level_1_working_memory | 5 | 80.0% | 11110 | 2203 | 0 |
| explicit_forget | 20261015_135006 | explicit_forget_01 | deepseek-v4-pro | baseline_react | level_0_reactive | 5 | 40.0% | 23702 | 1040 | 0 |
| explicit_forget | 20261015_140501 | explicit_forget_01 | qwen3-235b | baseline_react | level_0_reactive | 5 | 40.0% | 14030 | 847 | 0 |
| explicit_forget | 20261015_141336 | explicit_forget_01 | qwen3.6-27b | baseline_react | level_0_reactive | 5 | 40.0% | 20393 | 3303 | 0 |
| explicit_forget | 20261021_193031 | explicit_forget_01 | gpt-oss-20b | baseline_react | level_1_working_memory | 5 | 60.0% | 3064 | 2111 | 0 |
| explicit_forget | 20261023_155026 | explicit_forget_01 | deepseek-v4-flash | baseline_react | level_2_episodic_rag | 5 | 80.0% | 3517 | 1521 | 0 |
| in_context_control | 20261009_154440 | in_context_control_01 | glm-5.3-flash | baseline_react | level_1_working_memory | 1 | 100.0% | 7451 | 258 | 0 |
| in_context_control | 20261014_120017 | in_context_control_01 | qwen3.6-27b | baseline_react | level_1_working_memory | 1 | 100.0% | 26501 | 915 | 0 |
| in_context_control | 20261014_140706 | in_context_control_01 | deepseek-v4-pro | baseline_react | level_1_working_memory | 1 | 100.0% | 6472 | 176 | 0 |
| in_context_control | 20261014_143053 | in_context_control_01 | qwen3-235b | baseline_react | level_1_working_memory | 1 | 100.0% | 16730 | 336 | 0 |
| in_context_control | 20261015_135006 | in_context_control_01 | deepseek-v4-pro | baseline_react | level_0_reactive | 1 | 100.0% | 15386 | 175 | 0 |
| in_context_control | 20261015_140501 | in_context_control_01 | qwen3-235b | baseline_react | level_0_reactive | 1 | 100.0% | 15263 | 327 | 0 |
| in_context_control | 20261015_141336 | in_context_control_01 | qwen3.6-27b | baseline_react | level_0_reactive | 1 | 100.0% | 21613 | 915 | 0 |
| in_context_control | 20261021_193031 | in_context_control_01 | gpt-oss-20b | baseline_react | level_1_working_memory | 1 | 100.0% | 2561 | 293 | 0 |
| in_context_control | 20261026_120659 | in_context_derived_revision_01 | deepseek-v4-flash | baseline_react | level_2_episodic_rag | 1 | 0.0% | 0 | 0 | 1 |
| in_context_control | 20261026_120729 | in_context_derived_revision_01 | deepseek-v4-flash | baseline_react | level_2_episodic_rag | 1 | 0.0% | 0 | 0 | 1 |
| in_context_control | 20261026_120812 | in_context_derived_revision_01 | deepseek-v4-flash | baseline_react | level_2_episodic_rag | 1 | 0.0% | 0 | 0 | 1 |
| in_context_control | 20261026_121051 | in_context_derived_revision_01 | deepseek-v4-flash | baseline_react | level_2_episodic_rag | 1 | 100.0% | 8861 | 285 | 0 |
| in_context_control | 20261026_123753 | in_context_modus_tollens_01 | deepseek-v4-flash | baseline_react | level_2_episodic_rag | 1 | 100.0% | 7164 | 307 | 0 |
| negative_hallucination | 20261009_154440 | negative_hallucination_01 | glm-5.3-flash | baseline_react | level_1_working_memory | 3 | 0.0% | 6614 | 1178 | 0 |
| negative_hallucination | 20261014_120017 | negative_hallucination_01 | qwen3.6-27b | baseline_react | level_1_working_memory | 3 | 75.0% | 10748 | 2241 | 0 |
| negative_hallucination | 20261014_140706 | negative_hallucination_01 | deepseek-v4-pro | baseline_react | level_1_working_memory | 3 | 0.0% | 6545 | 746 | 0 |
| negative_hallucination | 20261014_143053 | negative_hallucination_01 | qwen3-235b | baseline_react | level_1_working_memory | 3 | 100.0% | 7017 | 635 | 0 |
| negative_hallucination | 20261014_165037 | negative_hallucination_01 | qwen3.6-27b | baseline_react | level_1_working_memory | 4 | 100.0% | 31752 | 2674 | 0 |
| negative_hallucination | 20261014_165118 | negative_hallucination_01 | deepseek-v4-pro | baseline_react | level_1_working_memory | 4 | 100.0% | 9756 | 1195 | 0 |
| negative_hallucination | 20261014_165201 | negative_hallucination_01 | qwen3-235b | baseline_react | level_1_working_memory | 4 | 100.0% | 10200 | 1054 | 0 |
| negative_hallucination | 20261015_135006 | negative_hallucination_01 | deepseek-v4-pro | baseline_react | level_0_reactive | 4 | 100.0% | 13450 | 651 | 0 |
| negative_hallucination | 20261015_140501 | negative_hallucination_01 | qwen3-235b | baseline_react | level_0_reactive | 4 | 100.0% | 7803 | 597 | 0 |
| negative_hallucination | 20261015_141336 | negative_hallucination_01 | qwen3.6-27b | baseline_react | level_0_reactive | 4 | 100.0% | 24885 | 2410 | 0 |
| negative_hallucination | 20261021_193031 | negative_hallucination_01 | gpt-oss-20b | baseline_react | level_1_working_memory | 4 | 100.0% | 2875 | 1530 | 0 |

## Retention & Persistence

| Test Category | Timestamp | Scenario | Model | Prompt | Level | Turns | Pass Rate | Avg Latency (ms) | Total Tokens | Errors |
| :-- | :-- | :-- | :-- | :-- | :-- | --: | --: | --: | --: | --: |
| attrition | 20261009_154440 | attrition_01 | glm-5.3-flash | baseline_react | level_1_working_memory | 8 | 100.0% | 10824 | 7837 | 0 |
| attrition | 20261014_120017 | attrition_01 | qwen3.6-27b | baseline_react | level_1_working_memory | 8 | 100.0% | 9306 | 8867 | 0 |
| attrition | 20261014_140706 | attrition_01 | deepseek-v4-pro | baseline_react | level_1_working_memory | 8 | 100.0% | 9949 | 5424 | 0 |
| attrition | 20261014_143053 | attrition_01 | qwen3-235b | baseline_react | level_1_working_memory | 8 | 100.0% | 12541 | 5662 | 0 |
| attrition | 20261015_135006 | attrition_01 | deepseek-v4-pro | baseline_react | level_0_reactive | 8 | 66.7% | 35950 | 1965 | 0 |
| attrition | 20261015_140501 | attrition_01 | qwen3-235b | baseline_react | level_0_reactive | 8 | 66.7% | 13584 | 1740 | 0 |
| attrition | 20261015_141336 | attrition_01 | qwen3.6-27b | baseline_react | level_0_reactive | 8 | 66.7% | 105725 | 4937 | 0 |
| attrition | 20261021_193031 | attrition_01 | gpt-oss-20b | baseline_react | level_1_working_memory | 8 | 100.0% | 3599 | 6433 | 0 |
| cross_thread | 20261024_153727 | cross_thread_01 | deepseek-v4-flash | baseline_react | level_2_episodic_rag | 12 | 100.0% | 8055 | 10636 | 0 |
| in_context_needle | 20261009_154440 | in_context_needle_01 | glm-5.3-flash | baseline_react | level_1_working_memory | 1 | 100.0% | 12207 | 365 | 0 |
| in_context_needle | 20261014_120017 | in_context_needle_01 | qwen3.6-27b | baseline_react | level_1_working_memory | 1 | 100.0% | 7055 | 552 | 0 |
| in_context_needle | 20261014_140706 | in_context_needle_01 | deepseek-v4-pro | baseline_react | level_1_working_memory | 1 | 100.0% | 9222 | 220 | 0 |
| in_context_needle | 20261014_143053 | in_context_needle_01 | qwen3-235b | baseline_react | level_1_working_memory | 1 | 100.0% | 5209 | 238 | 0 |
| in_context_needle | 20261015_135006 | in_context_needle_01 | deepseek-v4-pro | baseline_react | level_0_reactive | 1 | 100.0% | 14398 | 219 | 0 |
| in_context_needle | 20261015_140501 | in_context_needle_01 | qwen3-235b | baseline_react | level_0_reactive | 1 | 100.0% | 8769 | 238 | 0 |
| in_context_needle | 20261015_141336 | in_context_needle_01 | qwen3.6-27b | baseline_react | level_0_reactive | 1 | 100.0% | 37982 | 552 | 0 |
| in_context_needle | 20261021_193031 | in_context_needle_01 | gpt-oss-20b | baseline_react | level_1_working_memory | 1 | 100.0% | 2762 | 359 | 0 |
| needle_haystack | 20261009_154440 | needle_haystack_01 | glm-5.3-flash | baseline_react | level_1_working_memory | 5 | 100.0% | 7501 | 3846 | 0 |
| needle_haystack | 20261014_120017 | needle_haystack_01 | qwen3.6-27b | baseline_react | level_1_working_memory | 5 | 100.0% | 8802 | 4839 | 0 |
| needle_haystack | 20261014_140706 | needle_haystack_01 | deepseek-v4-pro | baseline_react | level_1_working_memory | 5 | 100.0% | 7866 | 3010 | 0 |
| needle_haystack | 20261014_143053 | needle_haystack_01 | qwen3-235b | baseline_react | level_1_working_memory | 5 | 100.0% | 20569 | 3819 | 0 |
| needle_haystack | 20261015_135006 | needle_haystack_01 | deepseek-v4-pro | baseline_react | level_0_reactive | 5 | 0.0% | 13998 | 1196 | 0 |
| needle_haystack | 20261015_140501 | needle_haystack_01 | qwen3-235b | baseline_react | level_0_reactive | 5 | 0.0% | 135666 | 1182 | 0 |
| needle_haystack | 20261015_141336 | needle_haystack_01 | qwen3.6-27b | baseline_react | level_0_reactive | 5 | 0.0% | 46777 | 3123 | 0 |
| needle_haystack | 20261021_193031 | needle_haystack_01 | gpt-oss-20b | baseline_react | level_1_working_memory | 5 | 100.0% | 3275 | 3737 | 0 |

## Compositional Reasoning

| Test Category | Timestamp | Scenario | Model | Prompt | Level | Turns | Pass Rate | Avg Latency (ms) | Total Tokens | Errors |
| :-- | :-- | :-- | :-- | :-- | :-- | --: | --: | --: | --: | --: |
| cross_thread_derived | 20261024_154444 | cross_thread_derived_01 | deepseek-v4-flash | baseline_react | level_2_episodic_rag | 11 | 100.0% | 6027 | 7081 | 0 |
| in_context_multihop | 20261009_154440 | in_context_multihop_01 | glm-5.3-flash | baseline_react | level_1_working_memory | 1 | 100.0% | 20366 | 564 | 0 |
| in_context_multihop | 20261014_120017 | in_context_multihop_01 | qwen3.6-27b | baseline_react | level_1_working_memory | 1 | 100.0% | 62259 | 613 | 0 |
| in_context_multihop | 20261014_140706 | in_context_multihop_01 | deepseek-v4-pro | baseline_react | level_1_working_memory | 1 | 100.0% | 11836 | 232 | 0 |
| in_context_multihop | 20261014_143053 | in_context_multihop_01 | qwen3-235b | baseline_react | level_1_working_memory | 1 | 100.0% | 14628 | 221 | 0 |
| in_context_multihop | 20261015_135006 | in_context_multihop_01 | deepseek-v4-pro | baseline_react | level_0_reactive | 1 | 100.0% | 13920 | 252 | 0 |
| in_context_multihop | 20261015_140501 | in_context_multihop_01 | qwen3-235b | baseline_react | level_0_reactive | 1 | 0.0% | 0 | 0 | 1 |
| in_context_multihop | 20261015_141336 | in_context_multihop_01 | qwen3.6-27b | baseline_react | level_0_reactive | 1 | 0.0% | 0 | 0 | 1 |
| in_context_multihop | 20261021_193031 | in_context_multihop_01 | gpt-oss-20b | baseline_react | level_1_working_memory | 1 | 100.0% | 4973 | 295 | 0 |
| temporal_multi_hop | 20261009_154440 | temporal_multi_hop_01 | glm-5.3-flash | baseline_react | level_1_working_memory | 4 | 100.0% | 13988 | 2117 | 0 |
| temporal_multi_hop | 20261014_120017 | temporal_multi_hop_01 | qwen3.6-27b | baseline_react | level_1_working_memory | 4 | 100.0% | 23991 | 2898 | 0 |
| temporal_multi_hop | 20261014_140706 | temporal_multi_hop_01 | deepseek-v4-pro | baseline_react | level_1_working_memory | 4 | 100.0% | 13721 | 1405 | 0 |
| temporal_multi_hop | 20261014_143053 | temporal_multi_hop_01 | qwen3-235b | baseline_react | level_1_working_memory | 4 | 100.0% | 25169 | 1201 | 0 |
| temporal_multi_hop | 20261015_135006 | temporal_multi_hop_01 | deepseek-v4-pro | baseline_react | level_0_reactive | 4 | 50.0% | 21917 | 660 | 0 |
| temporal_multi_hop | 20261015_140501 | temporal_multi_hop_01 | qwen3-235b | baseline_react | level_0_reactive | 4 | 50.0% | 37027 | 556 | 0 |
| temporal_multi_hop | 20261015_141336 | temporal_multi_hop_01 | qwen3.6-27b | baseline_react | level_0_reactive | 4 | 50.0% | 27470 | 1274 | 1 |
| temporal_multi_hop | 20261021_193031 | temporal_multi_hop_01 | gpt-oss-20b | baseline_react | level_1_working_memory | 4 | 100.0% | 2634 | 1369 | 0 |

## Robustness & Security

| Test Category | Timestamp | Scenario | Model | Prompt | Level | Turns | Pass Rate | Avg Latency (ms) | Total Tokens | Errors |
| :-- | :-- | :-- | :-- | :-- | :-- | --: | --: | --: | --: | --: |
| adversarial_forget | 20261009_192356 | adversarial_forget_01 | glm-5.3-flash | baseline_react | level_1_working_memory | 4 | 0.0% | 17089 | 5857 | 0 |
| adversarial_forget | 20261009_193939 | adversarial_forget_02 | glm-5.3-flash | baseline_react | level_1_working_memory | 4 | 100.0% | 10956 | 4917 | 0 |
| adversarial_forget | 20261009_194738 | adversarial_forget_02 | glm-5.3-flash | baseline_react | level_1_working_memory | 5 | 100.0% | 10941 | 7692 | 0 |
| adversarial_forget | 20261009_195402 | adversarial_forget_02 | glm-5.3-flash | minimal | level_1_working_memory | 5 | 100.0% | 16385 | 9117 | 0 |
| adversarial_forget | 20261009_195459 | adversarial_forget_01 | glm-5.3-flash | minimal | level_1_working_memory | 4 | 50.0% | 10303 | 4582 | 0 |
| adversarial_forget | 20261014_120017 | adversarial_forget_01 | qwen3.6-27b | baseline_react | level_1_working_memory | 4 | 0.0% | 10713 | 3554 | 0 |
| adversarial_forget | 20261014_120017 | adversarial_forget_02 | qwen3.6-27b | baseline_react | level_1_working_memory | 5 | 100.0% | 13700 | 5528 | 0 |
| adversarial_forget | 20261014_140706 | adversarial_forget_01 | deepseek-v4-pro | baseline_react | level_1_working_memory | 4 | 100.0% | 14107 | 1824 | 0 |
| adversarial_forget | 20261014_140706 | adversarial_forget_02 | deepseek-v4-pro | baseline_react | level_1_working_memory | 5 | 50.0% | 15340 | 3564 | 0 |
| adversarial_forget | 20261014_143053 | adversarial_forget_01 | qwen3-235b | baseline_react | level_1_working_memory | 4 | 100.0% | 23112 | 1191 | 0 |
| adversarial_forget | 20261014_143053 | adversarial_forget_02 | qwen3-235b | baseline_react | level_1_working_memory | 5 | 100.0% | 14702 | 2373 | 0 |
| adversarial_forget | 20261015_135006 | adversarial_forget_01 | deepseek-v4-pro | baseline_react | level_0_reactive | 4 | 100.0% | 8381 | 832 | 0 |
| adversarial_forget | 20261015_135006 | adversarial_forget_02 | deepseek-v4-pro | baseline_react | level_0_reactive | 5 | 100.0% | 10292 | 1229 | 0 |
| adversarial_forget | 20261015_140501 | adversarial_forget_01 | qwen3-235b | baseline_react | level_0_reactive | 4 | 100.0% | 18857 | 814 | 0 |
| adversarial_forget | 20261015_140501 | adversarial_forget_02 | qwen3-235b | baseline_react | level_0_reactive | 5 | 100.0% | 9318 | 1061 | 0 |
| adversarial_forget | 20261015_141336 | adversarial_forget_01 | qwen3.6-27b | baseline_react | level_0_reactive | 4 | 100.0% | 15313 | 3151 | 0 |
| adversarial_forget | 20261015_141336 | adversarial_forget_02 | qwen3.6-27b | baseline_react | level_0_reactive | 5 | 100.0% | 16180 | 4160 | 0 |
| adversarial_forget | 20261021_193031 | adversarial_forget_01 | gpt-oss-20b | baseline_react | level_1_working_memory | 4 | 100.0% | 3301 | 1304 | 0 |
| adversarial_forget | 20261021_193031 | adversarial_forget_02 | gpt-oss-20b | baseline_react | level_1_working_memory | 5 | 100.0% | 3665 | 2073 | 0 |
| adversarial_forget | 20261023_154532 | adversarial_forget_01 | deepseek-v4-flash | baseline_react | level_2_episodic_rag | 4 | 100.0% | 4097 | 1114 | 0 |

## Computational Cost

_No runs yet — scaffolding prepared for future tests._

## Overall

| Runs | Avg Pass Rate | Avg Latency (ms) | Total Turns | Total Tokens | Total Errors |
| --: | --: | --: | --: | --: | --: |
| 112 | 78.3% | 15271 | 463 | 250331 | 16 |

# Deprecated pilot runs — 2026-10-09 (max_tokens=300 era)

**Status: NOT final experimental data.** All nine bundles were produced before the
harness fixes and the `max_tokens` methodology change. They are kept as pilot
evidence for the methodology chapter, not for quantitative results.

## Why the whole batch is deprecated

All runs below used `max_tokens: 300`. The final sweep uses **2048** (see
`configs/models.yaml` header). The constant is a controlled variable, so
no score from the 300 era is comparable with the 2048 sweep.

## Per-folder manifest

| Folder | Combo | Specific defects beyond the 300-era constant |
|---|---|---|
| `20261009_103941` | gemini-3.5-flash-lite / baseline / L0 / belief_revision_01 | Direct Google API era — provider removed from registry. **Unreproducible.** |
| `20261009_120950` | gpt-oss-20b / baseline / L1 / attrition_01 | — |
| `20261009_121252` | gpt-oss-20b / baseline / L0 / all 9 scenarios | (a) Unicode false-negative: `in_context_needle_01` scored 0% but the answer was correct (`DELTA-7749` with U+2011 hyphen) — fixed by NFKC+typographic folding. (b) 2 blank-response turns (gateway stripped content, 300 tokens billed) scored as model failures — fixed by EmptyResponse error-turn guard. |
| `20261009_125340` | gpt-oss-20b / baseline / L1 / all 9 scenarios | 1 blank-response turn (`explicit_forget_01` turn 3) scored as failure. Level-0 vs Level-1 contrast visible here is qualitatively valid (checkpointer works) but not final. |
| `20261009_133121` | gpt-oss-20b / baseline / L1 / explicit_forget_01 | Refusal-behavior observations (privacy over-refusal on credentials) qualitatively valid; scores not final. |
| `20261009_133219` | gpt-oss-20b / minimal / L1 / explicit_forget_01 | Same as above; prompt-ablation signal (minimal → more refusal) is a real finding to re-confirm at 2048. |
| `20261009_142840` | glm-5.3-flash / minimal / L1 / explicit_forget_01 | **Invalidated**: output truncated at the 300 cap (turn 1 = 300 tokens, 100% reasoning, 0 content). Motivated the TruncatedResponse guard. |
| `20261009_143001` | glm-5.3-flash / baseline / L1 / explicit_forget_01 | Same truncation defect. The glm-vs-deepseek "failure" was a token-budget artifact, NOT a capability difference. |
| `20261009_143115` | deepseek-v4-pro / baseline / L1 / explicit_forget_01 | Conclusions survive (0 reasoning tokens, never truncated) but not final data. |

## Harness fixes these runs motivated (methodology narrative)

1. **Unicode-normalization in the assertion engine** — a correct answer
   (`DELTA‑7749`, U+2011) was scored as a failure. Fixed: NFKC +
   typographic folding, symmetric for must_contain / must_not_contain.
2. **EmptyResponse guard** — blank-but-billed turns are error turns,
   excluded from scoring and metrics.
3. **TruncatedResponse guard** — `finish_reason == "length"` turns are
   error turns; truncation can never masquerade as a memory failure.
4. **`max_tokens` = 2048 constant** — non-binding safety rail documented in
   `configs/models.yaml`; reasoning-hybrids (glm, gpt-oss) need headroom
   for hidden chain-of-thought.

## Known data-quality observations worth citing from the pilot era

- gpt-oss-20b over-refuses credential-handling requests ("I can't store or
  retain that information") even when facts are in-context — a
  willingness-vs-capability confound, distinct from memory failure.
- NanoGPT does not populate `output_token_details.reasoning` for the gpt-oss
  family (billed-but-invisible reasoning tokens) — cross-model cost
  comparisons must note this.
- glm-5.3-flash spends ~91% of its output budget on hidden reasoning.

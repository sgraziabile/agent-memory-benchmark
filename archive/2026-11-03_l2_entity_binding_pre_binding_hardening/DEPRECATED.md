# Deprecated `entity_binding_01` run — pre binding-hardening (2026-11-03)

**Status: NOT final experimental data.** This bundle is one **Level 2
(`level_2_episodic_rag`)** run of `entity_binding_01` produced *before* the
2026-11-03 revision of the scenario (R1/R2/R3, documented in the YAML header and
in `TODOs/specs/stage4-test-specs.md` §4.4). It is kept as **methodological
evidence** — not for headline quantitative claims, and **not comparable** with
any `entity_binding_01` run made after that revision.

Archived from `outputs/runs/` on **2026-11-03** so the live output directory only
contains runs whose scenario matches the current YAML contract.

## The bundle

| Field | Value |
|---|---|
| Run ID | `entity_binding_01_deepseek-v4-pro_baseline_react_c17f05b5` |
| Directory | `20261103_144749` |
| Scenario | `entity_binding_01` — **11 turns, 3 sessions (`s1`/`s2`/`s3`), 8 assertions** |
| Level / model / prompt | `level_2_episodic_rag` / `deepseek-v4-pro` / `baseline_react` |
| Memory | `backend=vector`, embedder `qwen3-embedding-0.6b`, `top_k=5`, `chunk_size=1200`, `chunk_overlap=200` |
| Result | **4/8 assertions (50%)** — **0/2 turn-level** on the paired commitment probes |
| Turns / errors | 11 / **0** (`error_turns = 0`, all turns `status=ok`) |
| Latency / tokens | 107 968 ms total (9 815 ms/turn) / 4 460 (3 314 prompt + 1 146 completion) |

## Why it is deprecated: three measurement defects, not a model artefact

The run itself is clean (no infrastructure failures, no truncation, retrieval
telemetry complete). It is deprecated because **the scenario as it stood could
not attribute its own failures**, and two of its three probe turns measured
something other than what the scenario claims to measure.

**1. R1 — the observational probe anchored the target probe.**
`T9` was the free-form observational probe and shared thread `s3` with the
commitment probes. It answered `srv-alpha -> 7555` (wrong; correct is `7443`),
and `T10` then repeated **`7555` verbatim**. The intra-session checkpointer
turned the target probe into self-consistency, so `T10` did not measure binding.

**2. R2 — a numeric-only surface scored 50% by construction.**
Both probes asked for a 4-digit port out of four mutually interchangeable
lookalike chunks, so an "emit one of the four values" heuristic lands 50%. In
this run the correct port chunk ranked **#1** on `T10` but a decoy ranked **#1**
on `T11`.

**3. R3 — the failure could not be attributed to read time (or blamed on it).**
The probes carried no retrieval assertions, so the bundle records that the
answers were wrong but not whether the evidence pair was even in context. That
had to be reconstructed by hand from `results.json`.

## Evidence worth citing from this bundle (why it is kept)

Retrieval was **not** the cause. Every evidence pair was injected:

| Turn | Answer | Assertions | Evidence chunks in context |
|---|---|---|---|
| `T9` (observational) | `srv-alpha -> 7555` | *(unasserted)* | `ep_s1_t1` (srv-alpha) r1, `ep_s1_t2` (7443) r2, `ep_s2_t2` (7080) r3 |
| `T10` (srv-alpha port) | **`7555`** ✗ | 2/4 | `ep_s1_t2` (**7443**) **r1**, `ep_s1_t1` (**srv-alpha**) **r5** |
| `T11` (srv-beta port) | **`7999`** ✗ | 2/4 | `ep_s2_t2` (**7080**) **r2**, `ep_s2_t1` (**srv-beta**) **r5** |

Two statements this bundle supports, and which are quoted in the thesis notes:

- **The `7555` answer is an anchor, not a retrieval miss.** The correct port chunk
  was rank 1 on `T10` and the hostname chunk was rank 5; the model still emitted
  the value it had invented one turn earlier.
- **Rank luck decides the outcome.** The archived `deepseek-v4-flash` run
  (`archive/2026-11-02_l2_pre_scenario_hardening/20261027_131450`) reached
  **100%** on **byte-identical retrieval** — same chunk IDs, same scores to four
  decimals (`0.6638 / 0.6561 / 0.6544 / 0.6482 / 0.5683` on `T10`;
  `0.7586 / 0.7532 / 0.7532 / 0.7512 / 0.6827` on `T11`). The pre-revision
  scenario was measuring model-side luck over a 4-way anonymous choice, not a
  capability boundary.

## What supersedes it

| Run | Scenario | Assertions | Turn-level (4 probes) |
|---|---|---|---|
| `20261103_151452` | hardened (13 turns, 24 assertions) | 21/24 (87.5%) | **2/4** — `T11` declined, `T12` cross-entity swap |
| `20261103_152558` | hardened (13 turns, 24 assertions) | 24/24 (100%) | **4/4** — all four bound correctly |

Neither superseding run is "the" `entity_binding_01` result on its own: they are
the **same scenario, same model, same `temperature=0.0`, byte-identical
retrieval**, and they disagree on the weekday probes (`T11` decline vs
`Thursday`; `T12` `Thursday` vs `Friday`). Report the pair and the variance, never
a single 100%.

**Do not** merge this bundle into any aggregate: it is 8 assertions over 11 turns
with 2 measured probes, while the current scenario is 24 assertions over 13
turns with 4 measured probes. `outputs/consolidated_summary.{md,csv}` must be
regenerated without it.

## Reproduction

The pre-revision scenario is recoverable from git — `859c573` is the last commit
whose `test_entity_binding_01.yaml` still has the 11-turn layout:

```bash
git show 859c573:datasets/conversations/logical_consistency/entity_binding/test_entity_binding_01.yaml
python run.py -m deepseek-v4-pro -s entity_binding_01 -l 2 \
  --backend vector --embedding-model qwen3-embedding-0.6b
```

Re-running it today would fail, not reproduce: the committed YAML is the
13-turn/24-assertion contract. To reproduce exactly, restore the file above first
and expect a *new* bundle in `outputs/runs/` (same inputs, `temperature=0.0`, but
the provider is not bit-deterministic — see the `T11`/`T12` disagreement between
the two post-revision runs).

## Housekeeping

- The bundle was moved whole (all four files plus the `memory/` snapshot); no
  generated artefact was edited.
- `outputs/by_theme/**` holds no derivative copy of `entity_binding_01`.
- The post-revision runs (`20261103_151452`, `20261103_152558`) stay in
  `outputs/runs/` as valid current-contract data.


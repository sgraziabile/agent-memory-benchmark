# Deprecated `document_multihop_bridge_01` pilot matrix (2026-11-11)

**Status: NOT final experimental data.** These four bundles are the first **Level 2
(`level_2_episodic_rag`)** runs of `document_multihop_bridge_01`, produced on the day the cell was
authored. They are kept as characterisation evidence for the methodology chapter — not for headline
claims — because **three of them ran against a corpus and an assertion that no longer exist**, and the
fourth has no comparable partner.

Archived from `outputs/runs/` on **2026-11-11** so the live output directory only contains runs whose
corpus and YAML contract match the current tree.

| Bundle | Model | `top_k` | Rule used for T4 | Corpus version | Verdict |
|---|---|---:|---|---|---|
| `20261111_131812` | deepseek-v4.1-flash | 5 | `six-week` (old) | **v1** | 7/10 |
| `20261111_135124` | deepseek-v4.1-flash | 15 | `six-week` (old) | **v1** | 8/10 |
| `20261111_135335` | mimo-v2.6-pro | 5 | `six-week` (old) | **v1** | 7/10 |
| `20261111_135546` | mimo-v2.6-pro | 15 | `six weeks` (current) | **v2** | 8/10 |

`retrieval_recall = 0.75` in all four (pins: T2 `asset_registry_01` ✓, T2 `calibration_register_02` ✗,
T3 `asset_registry_01` ✓, T4 `bg_07` ✓).

## Why they are deprecated

### Change #1 — the T4 assertion changed (`six-week` → `six weeks`)

The rule matched substrings literally, so a **correct** answer failed whenever it wrote the fact in its
natural form:

- `20261111_131812` answered *"Every six weeks, per the 2026-02-17 entry…"* → **FAIL** (no hyphen).
- `20261111_135335` answered *"**Every six weeks**, as of pack 2026.02…"* → **FAIL** (no hyphen).
- `20261111_135124` answered *"Every six weeks."* and passed only because its **explanation** later
  contained *"…to a six-week one"* → **PASS by phrasing, not by correctness.**

Same fact, opposite verdicts. The rule is now `six weeks` and the probe pins the answer to the worded
form (`Answer in words, not digits.`), so the two v1 bundles are graded against a rule that no longer
exists and the two later ones are not comparable to the earlier pair.

### Change #2 — the corpus changed (registry wording and `bg_07`)

The registry used to say every value was held in *"the calibration register"* while the document itself is
titled *"Limit schedule — Viaduct facility"*. Models noticed the mismatch and refused evidence they had
already retrieved: `20261111_135124` T3 quoted the value **and** hedged — *"though note the limit schedule
is a separate document from the calibration register the registry points to."* That is a documentation
defect, not a memory failure, so the corpus was corrected: registry rows now point to *"limit schedule"*
and `bg_07` reads *"every six weeks"* instead of *"six-week cadence"*.

Because chunk text is also the embedding-cache key, the change invalidates the vectors of the edited
documents: v1 and v2 bundles cannot be pooled.

### Change #3 — the fourth bundle has no comparable partner

`20261111_135546` did run on corpus v2 with the current rule, but its `top_k=5` twin
(`20261111_135335`) ran on v1. Keeping half of a 2×2 matrix invites exactly the cross-arm comparison the
scenario's `headline_metric` forbids, so the whole matrix is archived and re-run uniformly.

## What these bundles still establish

1. **The cell's target property held in both models and both arms.** In all four runs the register chunk
   never reached the prompt (0 of 5 and 0 of 15 retrieved slots), so `L2a` could not compose: the
   `must_contain` and the hop-2 pin failed together, and every model produced an evidence-grounded
   abstention rather than a fabrication.
2. **Breadth did not substitute for a second query.** `top_k` 5 → 15 raised the score only through the
   T4 artifact, not through the bridge probe; at `k=15` the near-miss documents (`decoy_01`, `decoy_02`,
   `decoy_03`) took ranks 1–3 while the register stayed absent.
3. **The assertion defect was real and measurable**, and is the reason the rule was rewritten — the
   before/after replay of the stored responses is the evidence for that change.

## What must be re-run

The full 2×2 (`deepseek-v4.1-flash`, `mimo-v2.6-pro` × `top_k` 5, 15) against the current corpus and YAML.
Expected signature after the fixes: T4 passes in every arm, T3 loses its hedge, and T2 still fails in every
arm with `retrieval_recall = 0.75` — the baseline an agentic `L2b` arm has to beat.

```powershell
# per model, both budget arms
python run.py --level 2 --model <model> --scenario document_multihop_bridge_01 `
  --backend vector --embed-cache .cache/emb --top-k 5
python run.py --level 2 --model <model> --scenario document_multihop_bridge_01 `
  --backend vector --embed-cache .cache/emb --top-k 15
```

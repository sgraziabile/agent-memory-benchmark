# TODO: Run-Comparison Charts (matplotlib) for Director Presentations

> **Status:** IN PROGRESS — Phases 0–1 executed 2026-10-14; Phases 2–4 pending.
> **Scope note:** implementation deferred at the user's request — this document is
> the deliverable for now; build the script only when explicitly asked.
> **Purpose:** Static, publication-quality figures comparing benchmark runs, for
> presentation to the thesis director and co-director. Companion narrative:
> `notes/2026-10-14_l1_three_model_comparison.md`.

## 1. Objective

Turn the stored run data (`outputs/runs/*/results.json`) into 5 self-explanatory
PNG figures, fully offline (responses and per-turn assertion details are stored
verbatim — no API calls needed to plot). `matplotlib` only, no seaborn:
minimal new dependency, standard for academic work.

## 2. Decisions already made

- **Format:** static PNG, 300 dpi, `outputs/figures/` as output directory.
- **Library:** new optional dependency group `[project.optional-dependencies]
  plot = ["matplotlib>=3.8"]` — runtime deps untouched (same pattern as
  `dev`/`ollama` extras).
- **`negative_hallucination_01` artifact:** the three headline L1 sweeps
  (runs `20261014_120017`, `20261014_140706`, `20261014_143053`) were scored
  against the **pre-fix YAML**. Decision: **re-run the scenario on the 3 models
  first** (~9 turns, negligible quota) and let the new, clean run replace the
  artifact-affected records in the charts. Original scores remain untouched in
  their run dirs; the re-run is footnoted in the figures.
- **Data merge rule:** the loader merges records by
  `(model_id, scenario_id, prompt_id, agent_level)` with **newest-wins**
  semantics (latest `timestamp`). This makes the script future-proof for any
  scenario re-run — no special-casing NH01.

## 3. Execution phases

### Phase 0 — Clean data (requires API quota) — ✅ DONE 2026-10-14
```
python run.py -s negative_hallucination_01 -l 1 -m qwen3.6-27b -p baseline_react
python run.py -s negative_hallucination_01 -l 1 -m deepseek-v4-pro -p baseline_react
python run.py -s negative_hallucination_01 -l 1 -m qwen3-235b -p baseline_react
```
Executed against the fixed YAML. Clean re-run directories:
`outputs/runs/20261014_165037` (qwen3.6-27b), `20261014_165118` (deepseek-v4-pro),
`20261014_165201` (qwen3-235b) — all 4 turns, 0 error turns, **100.0% pass rate
for all three models** (vs. 75% / 0% / 100% under the pre-fix assertions;
deepseek's false-negative is fully resolved). The original sweeps stay untouched
for traceability; the loader's newest-wins merge (§2) makes these re-runs win
automatically once plotted.

### Phase 1 — Dependency — ✅ DONE 2026-10-14
`plot` extra added to `pyproject.toml` (`matplotlib>=3.8`); matplotlib import
verified in the active environment. No runtime deps touched.

### Phase 2 — `scripts/plot_comparison.py` (~200 lines, offline) — ⏸ PENDING
> A partial implementation was started prematurely on 2026-10-14 and **deleted**
> (never validated) — the user scoped this task to finishing the TODO only.
> Whoever implements this phase starts from scratch per the spec below.
- `load_runs(run_ids)` → flattened run records (one per scenario execution).
- One pure function per figure; CLI:
  `python scripts/plot_comparison.py --runs <id...> --out outputs/figures/`
- Consistent per-model colors across ALL charts; run IDs in figure captions.

### Phase 3 — Generate + verify
Run against the 3 sweeps + the NH01 re-run. Spot-check bar values against the
numbers in the comparison note; confirm PNGs written.

### Phase 4 — Optional smoke test
Small pytest loading a fixed record set and asserting each figure function
runs without error (no-crash only; no image assertions). Suite must stay green.

## 4. Figure specifications

| # | Figure | Type | What it shows |
|---|---|---|---|
| 1 | Pass rate per scenario × model | grouped bars | Headline: capability saturation on the 7 recall/reasoning scenarios vs. the inverted security-scenario failures (adv_forget 0%↔100%, explicit_forget 40%↔100%) |
| 2 | Scenario × model matrix | heatmap | The L1 story in one glance: all-100% top block, chaos in the security rows |
| 3 | Pass rate vs. token cost | scatter, bubble = latency | The efficient-frontier finding (P3): qwen3-235b cheapest *and* best; qwen3.6-27b most expensive *and* weakest |
| 4 | Token profile per model | stacked bars (prompt/completion/reasoning) | The 6× completion-token gap and the ≈22k reasoning tokens — "cost of deliberation" |
| 5 | Avg latency per turn per scenario × model | bars | P3 latency dimension (incl. qwen3.6's 62 s/turn outlier) |

Rejected: radar/spider per-theme profile — directors like them, but the heatmap
is more honest (no misleading area scaling).

## 5. Style requirements (thesis-grade)

- Colorblind-safe palette (e.g. Okabe–Ito).
- Same model → same color in every figure.
- Axis labels with units; pass rates as percentages.
- Source run IDs in every caption (traceability to `outputs/runs/`).
- No chart titles inside the image beyond what a caption would duplicate —
  figures must survive being embedded in slides and the written thesis.

## 6. Known considerations

- Quota burn vs. raw tokens: figure 3 plots **raw tokens**; the note documents
  that deepseek-v4-pro's effective cost is ~2× its input tokens (NanoGPT
  multiplier). Keep raw tokens in the chart; state the caveat in the caption.
- The 2026-10-09 runs are single-scenario partials with mixed prompts
  (`minimal`/`baseline_react`) — exclude from model-comparison charts by
  default; the CLI's explicit `--runs` selection handles this naturally.
- L0 vs. L1 comparison charts become possible once an L0 sweep exists on the
  same 11 scenarios; the loader's `(model, scenario, prompt, level)` key
  already supports it.
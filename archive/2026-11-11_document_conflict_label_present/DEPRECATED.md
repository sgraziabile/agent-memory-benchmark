# Deprecated `document_conflict_dilution_01` runs — "label present" corpus (2026-11-11)

**Status: NOT final experimental data — but keep them: they are the *before* side of a controlled change.**
Three runs were produced while `conflict_superseded_01` still carried a `## Status` block that opened with
the word **"Superseded"** and stated *"has been replaced by the January 2026 cycle … is not current now"*.
That label performed the precedence reasoning the cell is supposed to measure, so it was removed (see below).
These three runs are kept as the with-label baseline for the before/after comparison.

| Bundle | Model | `top_k` | Score | Pins |
|---|---|---:|---:|---|
| `20261111_145253` | deepseek-v4.1-flash | 5 | 9/9 | 3/3 |
| `20261111_145436` | deepseek-v4.1-flash | 15 | 9/9 | 3/3 |
| `20261111_145659` | mistral-small-4 | 15 | 9/9 | 3/3 |

All three ran against the 39-chunk corpus (verified by the presence of the `## Status` text in the retrieved
chunks). Archived from `outputs/runs/` on **2026-11-11** so the live directory only holds runs whose corpus
matches the tree.

## What changed (and why these runs are not comparable with post-change runs)

The superseded record lost its precedence label; precedence is now implied only by **date and tense**:

- `## Status` section deleted (it said "Superseded", "replaced by the January 2026 cycle", "not current now").
- The value sentence became *"…was SWG-1F9K-5520 … it stood as the recorded value for that cycle"* — past
  tense, no forward reference.
- The meta-paragraph explaining that the current value is "deliberately not repeated" was removed; that was
  corpus-construction commentary inside the fiction.
- The document now never mentions 2026 at all, so the only way to establish precedence is to compare the
  two records' cycles and tenses.
- Corpus: 39 → **38 chunks** (the record dropped from 3 chunks to 2).

Because chunk text is also the embedding-cache key, the edited document's vectors are invalidated, so these
runs cannot be pooled with anything produced afterwards.

## What the three runs still establish

1. **Every pass was verified legitimate**, not lucky: each asserted token is an unguessable synthetic string
   and its gold chunk was in the window (T2: gold rank 1 at 0.7779 versus the stale record at 0.7751 — a
   0.003 margin the retriever could not resolve; T3: sibling rank 1 with the duty pump's value present at
   rank 3 in the same window; T4: superseded record rank 1).
2. **The models did the reasoning, not just the label**: deepseek-v4.1-flash answered the current value and
   *rejected* the stale one in the same reply, and at `k=15` gave the reason from the stale document (the
   December 2025 gauge recalibration).
3. **The cell was at ceiling with the label** — 9/9 in both arms and across two models, which is exactly the
   reason for removing it. Post-change runs must be compared against these.

## Two caveats recorded at the time

- `T5` (over-disclosure guard) is nearly vacuous: it passes for any terse answer.
- `T3`'s bleed guard is partly protected by the probe's *"Reply with the limit only"* constraint, so it
  measures obedience as much as binding.

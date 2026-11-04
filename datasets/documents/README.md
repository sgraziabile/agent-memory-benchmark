# `datasets/documents/` — Document Corpus Fixtures

Documents are the harness's **second memory-ingestion channel** (plan WS-D):
they are chunked and appended to the active `EpisodicMemoryBackend` with
`source="document"` **without ever invoking the chat model**. Ingestion costs
embedding tokens only.

## Committed fixtures

| File | Contents | Used by |
|------|----------|---------|
| `policy_v1.md` | Northwind access policy: roles → authorization levels table, change management (§3.2 is the **only** place that states the reboot level = Level-1), break-glass account `iris-breakglass-77` | `document_needle_01`, `document_addendum_revision_01` |
| `addendum_1.md` | Addendum superseding §3.2 (`Level-2` reboots) and setting the 2026-10-01 runbook compliance window | `document_addendum_revision_01` (via `system_event`) |
| `ops_handbook.md` | Service ports (telemetry ingestion gateway = `7443`), escalation leads (`Rostova`), failover cluster `dr-atlas-04` | `document_cross_thread_01`; `document_needle_01` (topically adjacent distractor that makes the corpus exceed `top_k`) |

## How a scenario references them

Paths resolve **relative to the scenario YAML first**, then here:

```yaml
# Scenario-level corpus: ingested once per tuple, BEFORE turn 1.
documents:
  - doc_id: policy_v1
    path: policy_v1.md
    metadata: {title: "Access Policy (Revision 1)"}

turns:
  # Mid-conversation ingestion: appended when the turn is reached.
  - turn_number: 2
    session_id: s1
    role: system_event
    content: "Compliance published Addendum 1, superseding section 3.2."
    attachments:
      - addendum_1.md
```

Chunk IDs are deterministic:

| Channel | Chunk ID |
|---------|----------|
| Preload (`documents:`) | `doc_<doc_id>` (backend adds `_c<i>` when chunking) |
| `system_event` file | `doc_<file stem>_t<turn>` |
| `system_event` inline content | `doc_evt_<session>_t<turn>` |

Provenance (`doc_id`, `source_path`, `sha256`, char counts) lands in
`outputs/runs/<ts>/memory/<run_id>/manifest.json` under
`run_metadata.documents`, and document chunks are also exported to
`documents.json`.

## Generating a larger corpus

`scripts/generate_corpus.py` produces a seeded, deterministic corpus so
"context volume" is a parameter rather than hand-written text:

```bash
python scripts/generate_corpus.py --docs 20 --chars-per-doc 5000 --seed 7
python scripts/generate_corpus.py --docs 40 --chars-per-doc 8000 --seed 7 --dry-run
```

It writes `generated/corpus_000.md … corpus_00N.md` plus
`generated/corpus_manifest.json` (seed, per-file `sha256`, and which document
holds the needle). Reference them from a scenario as
`path: generated/corpus_003.md`.

## D16 — corpus-size curve (`--needles`)

D16 (stress plan §5.3.5) measures `Recall@5` against corpus size. Instead of a
single hand-written needle it plants **N opaque (entity, answer) pairs**:

```bash
python scripts/generate_corpus.py --docs 8   --chars-per-doc 6000 --seed 7 \
    --needles 8 --out datasets/documents/d16_corpus_008
python scripts/generate_corpus.py --docs 60  --chars-per-doc 6000 --seed 7 \
    --needles 8 --out datasets/documents/d16_corpus_060
python scripts/generate_corpus.py --docs 200 --chars-per-doc 6000 --seed 7 \
    --needles 8 --out datasets/documents/d16_corpus_200
python scripts/build_d16_scenarios.py        # emits the three scenario YAMLs
```

Each needle renders as one sentence

> `The calibration code for the cooling loop LOOP-412 is ARC-9F3K-2214.`

and the generated probe asks only for the entity:

> `What is the calibration code for the cooling loop LOOP-412?`

**Non-inferability contract** (enforced by the generator, which exits non-zero
otherwise): the answer token occurs **exactly once** in the whole corpus, never
in its own probe, and its entity also occurs exactly once — so a correct
response cannot be guessed, inferred from the prompt, or reconstructed from a
sibling document. The gold chunk is necessary. Every corpus owns its own
(answer, entity) set (`seed + corpus size`), so probes are meaningful only
inside their corpus.

`corpus_manifest.json` gains a `needles` list (`doc_id`, `entity`, `answer`,
`probe`, `sentence`, `section`); the legacy `needle` key and single-needle
behaviour are unchanged (the no-needle corpus is byte-identical to the
pre-`--needles` output for the same seed).

Report the resulting curve with:

```bash
python scripts/lr_report.py --scenario-prefix document_corpus_size \
    --csv outputs/lr_recall.csv
```

It recomputes **chunk-level** Recall@k, MRR, Δcos and coverage from
`retrieved_memories`, because the scenario assertion only pins the document
prefix (`must_retrieve_chunk: corpus_0NN` also accepts sibling sub-chunks).

Budget: ingestion is charged per tuple. Measured 2026-11-04 with
`qwen3-embedding-0.6b` (Ollama, CPU): 8 docs ≈ 12k tokens ≈ **154 s**; at
≈12 ms/token the 60-doc cell is ≈20 min and the 200-doc cell ≈65 min per run.



| Level | Preload | `system_event` | Result |
|-------|---------|----------------|--------|
| L0 | skipped (+warning) | skipped (no-op) | document probes fail → ablation floor |
| L1 | skipped (+warning) | skipped (no-op) | document probes fail (no long-term store) |
| L2+ | appended with `source="document"`, `session_id="corpus"`, `turn_number=0` | appended at that turn | retrievable from any later session |

Disable ingestion for control runs with `python run.py ... --no-documents`.

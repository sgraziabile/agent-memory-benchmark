# `datasets/documents/` — Document Corpus Fixtures

Documents are the harness's **second memory-ingestion channel** (plan WS-D):
they are chunked and appended to the active `EpisodicMemoryBackend` with
`source="document"` **without ever invoking the chat model**. Ingestion costs
embedding tokens only.

## Committed fixtures

| File | Contents | Used by |
|------|----------|---------|
| `policy_v1.md` | Northwind access policy: roles → authorization levels table, change management (§3.2 = Level-1 reboots), break-glass account `iris-breakglass-77` | `document_needle_01`, `document_addendum_revision_01` |
| `addendum_1.md` | Addendum superseding §3.2 (`Level-2` reboots) | `document_addendum_revision_01` (via `system_event`) |
| `ops_handbook.md` | Service ports (telemetry ingestion gateway = `7443`), escalation leads (`Rostova`), failover cluster `dr-atlas-04` | `document_cross_thread_01` |

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

## Level semantics

| Level | Preload | `system_event` | Result |
|-------|---------|----------------|--------|
| L0 | skipped (+warning) | skipped (no-op) | document probes fail → ablation floor |
| L1 | skipped (+warning) | skipped (no-op) | document probes fail (no long-term store) |
| L2+ | appended with `source="document"`, `session_id="corpus"`, `turn_number=0` | appended at that turn | retrievable from any later session |

Disable ingestion for control runs with `python run.py ... --no-documents`.

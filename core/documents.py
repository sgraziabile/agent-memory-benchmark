"""core.documents — Document resolution, loading and record building (WS-D).

Implements the harness-side half of document ingestion for the memory
benchmark. The storage half already exists: every ``EpisodicMemoryBackend``
accepts ``EpisodeRecord`` objects with ``source="document"`` and chunks long
content internally. This module turns declarative YAML entries into those
records:

- **Resolution** — ``documents[].path`` and ``attachments`` entries are
  resolved relative to the scenario YAML file first, then
  ``datasets/documents/``, so fixtures can be either co-located with a
  scenario or shared across scenarios.
- **Loading** — UTF-8 text is read once and hashed (SHA-256) for
  reproducibility. Inline ``content`` documents never touch the filesystem.
- **Chunk-ID derivation** — deterministic IDs make re-ingestion idempotent:

  =======================  ==========================================
  Channel                  Chunk ID
  =======================  ==========================================
  Scenario preload         ``doc_<doc_id>``
  ``system_event`` turn    ``doc_<doc_id>_t<turn_number>``
  =======================  ==========================================

  Sub-chunking (content longer than ``chunk_size``) is the backend's job and
  appends ``_c<i>`` to these IDs.

Nothing here invokes a chat model: document ingestion costs embedding tokens
only, which the runner reports separately from chat tokens.

Import note: ``EpisodeRecord`` lives in the Level 2 backend package, while
this module lives in ``core``. The import is therefore performed lazily
inside :func:`build_episode_records` so ``core`` never depends on ``agents``
at import time (and no import cycle is possible).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from core.schemas import DocumentSpec

#: Pseudo-session id for scenario-level preloads. Document chunks are never
#: excluded by ``retrieve(exclude_session_id=...)`` (that filter only applies
#: to ``source="chat"``), and a distinct label keeps snapshots legible.
CORPUS_SESSION_ID = "corpus"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Shared fixture directory used when a path is not found next to the YAML.
DEFAULT_DOCUMENTS_DIR = _PROJECT_ROOT / "datasets" / "documents"


@dataclass(frozen=True)
class LoadedDocument:
    """One document's text plus provenance, ready to be chunked.

    Attributes:
        doc_id: Stable identifier from the scenario (``doc_<doc_id>`` base).
        text: Full document text (UTF-8, newline-preserving).
        metadata: Free-form provenance from the scenario (``title``, ...).
        source_path: Absolute path the text came from; ``None`` for inline
            ``content`` documents.
    """

    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    @property
    def char_count(self) -> int:
        """Characters in the document (pre-chunking volume)."""
        return len(self.text)

    @property
    def sha256(self) -> str:
        """Content hash — reproducibility fingerprint for snapshot manifests."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass
class IngestionOutcome:
    """Aggregated result of ingesting one batch of documents.

    Mirrors ``AppendTelemetry`` but adds the harness-level provenance the
    runner needs for per-turn and per-run reporting.
    """

    document_ids: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    #: One manifest dict per document (doc_id, source_path, sha256, chars).
    manifests: list[dict[str, Any]] = field(default_factory=list)
    char_count: int = 0
    append_latency_ms: float = 0.0
    embedding_latency_ms: float = 0.0
    embedding_tokens: int = 0

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return bool(self.document_ids)


def resolve_document_path(
    raw: str,
    *,
    scenario_dir: Path | None = None,
    documents_dir: Path | str = DEFAULT_DOCUMENTS_DIR,
) -> Path:
    """Resolve a scenario-declared document path to an existing file.

    Search order:
    1. Absolute paths are used verbatim (must exist).
    2. ``<scenario YAML directory>/<raw>`` — fixtures co-located with the
       scenario.
    3. ``<documents_dir>/<raw>`` — shared corpus fixtures
       (``datasets/documents/`` by default).

    Args:
        raw: Path string exactly as written in the YAML.
        scenario_dir: Directory of the scenario YAML (``None`` when the
            scenario was constructed in code / not loaded from disk).
        documents_dir: Shared fixture root.

    Returns:
        The resolved, existing file path.

    Raises:
        FileNotFoundError: When none of the candidates exist. The message
            lists every location searched so dataset authors can fix the
            path without guessing.
    """
    candidate = Path(raw)
    if candidate.is_absolute():
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(
            f"Document '{raw}' not found (absolute path does not exist)."
        )

    searched: list[Path] = []
    if scenario_dir is not None:
        searched.append(Path(scenario_dir) / candidate)
    searched.append(Path(documents_dir) / candidate)

    for path in searched:
        if path.is_file():
            return path

    locations = ", ".join(str(p) for p in searched)
    raise FileNotFoundError(
        f"Document '{raw}' not found. Searched: {locations}. "
        "Place the fixture next to the scenario YAML or in datasets/documents/."
    )


def load_document(
    spec: DocumentSpec,
    *,
    scenario_id: str = "<unknown>",
    scenario_dir: Path | None = None,
    documents_dir: Path | str = DEFAULT_DOCUMENTS_DIR,
) -> LoadedDocument:
    """Materialize one :class:`DocumentSpec` into text + provenance.

    Args:
        spec: The validated declarative document entry.
        scenario_id: Scenario identifier, used only in error messages.
        scenario_dir: Directory of the scenario YAML (path resolution).
        documents_dir: Shared fixture root (path resolution).

    Returns:
        A :class:`LoadedDocument` whose ``text`` is guaranteed non-blank.

    Raises:
        FileNotFoundError: When a ``path`` spec cannot be resolved.
        ValueError: When the document's text is blank.
    """
    if spec.content is not None:
        text = spec.content
        source_path: Path | None = None
    else:
        assert spec.path is not None  # guaranteed by DocumentSpec validation
        source_path = resolve_document_path(
            spec.path, scenario_dir=scenario_dir, documents_dir=documents_dir
        )
        text = source_path.read_text(encoding="utf-8")

    if not text.strip():
        origin = str(source_path) if source_path is not None else "inline content"
        raise ValueError(
            f"Scenario '{scenario_id}': document '{spec.doc_id}' ({origin}) "
            "is empty — blank documents would pollute retrieval."
        )

    return LoadedDocument(
        doc_id=spec.doc_id,
        text=text,
        metadata=dict(spec.metadata),
        source_path=source_path,
    )


def _unique_chunk_id(base: str, used: set[str]) -> str:
    """Return ``base`` or a deterministically suffixed variant.

    Two documents sharing a ``doc_id`` inside one ingestion batch (e.g. two
    attachments both named ``addendum.md``) must not collapse into one
    record — the backend would treat the second as a mutation attempt.
    """
    if base not in used:
        used.add(base)
        return base
    index = 2
    while f"{base}_{index}" in used:
        index += 1
    unique = f"{base}_{index}"
    used.add(unique)
    return unique


def build_episode_records(
    documents: Iterable[LoadedDocument],
    *,
    session_id: str,
    turn_number: int,
    used_chunk_ids: set[str] | None = None,
) -> list[Any]:
    """Convert loaded documents into ``source="document"`` episode records.

    The chat model is never involved; the records go straight to
    ``EpisodicMemoryBackend.append()``, which chunks them further if needed.

    Args:
        documents: Documents produced by :func:`load_document`.
        session_id: Provenance session (``"corpus"`` for preloads, the
            ``system_event`` turn's session otherwise).
        turn_number: ``0`` for preloads, the turn index for ``system_event``.
        used_chunk_ids: Optional shared set of chunk IDs already used in this
            ingestion batch, so duplicate ``doc_id`` values get a suffix.

    Returns:
        A list of ``EpisodeRecord`` instances (typed as ``Any`` to keep the
        lazy import local).
    """
    # Lazy import: core must not import the agents package at module load.
    from agents.level_2_episodic_rag.backends.base import EpisodeRecord

    used: set[str] = used_chunk_ids if used_chunk_ids is not None else set()
    records: list[Any] = []
    for document in documents:
        base = (
            f"doc_{document.doc_id}"
            if turn_number == 0
            else f"doc_{document.doc_id}_t{turn_number}"
        )
        metadata: dict[str, Any] = dict(document.metadata)
        metadata["doc_id"] = document.doc_id
        if document.source_path is not None:
            metadata["source_path"] = str(document.source_path)
        metadata["sha256"] = document.sha256
        records.append(
            EpisodeRecord.from_document(
                chunk_id=_unique_chunk_id(base, used),
                content=document.text,
                session_id=session_id,
                turn_number=turn_number,
                metadata=metadata,
            )
        )
    return records


def manifest_entries(documents: Sequence[LoadedDocument]) -> list[dict[str, Any]]:
    """Build per-document provenance entries for the run snapshot manifest."""
    return [
        {
            "doc_id": document.doc_id,
            "source_path": (
                str(document.source_path) if document.source_path is not None else None
            ),
            "sha256": document.sha256,
            "char_count": document.char_count,
            "estimated_tokens": document.char_count // 4,
        }
        for document in documents
    ]

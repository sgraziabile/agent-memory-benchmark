#!/usr/bin/env python3
"""Generate a deterministic synthetic document corpus for context-heavy tests.

Hand-writing 20 documents of 5,000 characters is neither reviewable nor
reproducible. This script produces a corpus from a seed, so a scenario can
declare exactly how much context the retriever has to search and anyone can
regenerate byte-identical files by reusing the seed:

    python scripts/generate_corpus.py --docs 12 --chars-per-doc 4000 --seed 7
    python scripts/generate_corpus.py --docs 40 --chars-per-doc 8000 --seed 7 --dry-run

Output layout (``datasets/documents/generated/`` by default)::

    corpus_000.md ... corpus_00N.md
    corpus_manifest.json     # seed, per-file doc_id/sha256/chars, needle

Each document is a multi-section operations note built from a fixed template
pool, so documents are lexically similar to each other — exactly the dilution
pressure a top-k retriever must survive. The *needle* paragraph is planted in
exactly one document and is the only sentence that answers the scenario probe.

Usage from a scenario YAML::

    documents:
      - doc_id: corpus_003
        path: generated/corpus_003.md   # resolved under datasets/documents/

The script is 100% offline and dependency-free (stdlib only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import string
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "datasets" / "documents" / "generated"

DEFAULT_NEEDLE = (
    "The calibration code for the north cooling loop is NEEDLE-CALIBRA-7741."
)

#: Opening lines that make every generated document look alike on purpose.
TOPICS = [
    "cooling loop maintenance",
    "network core maintenance",
    "object store compaction",
    "scheduler rollout checks",
    "certificate rotation",
    "backup verification",
    "incident review follow-ups",
    "capacity planning review",
    "firmware baseline audit",
    "log retention review",
]

SECTIONS = [
    "Scope",
    "Responsibilities",
    "Procedure",
    "Verification",
    "Rollback",
    "Escalation",
    "Records",
    "Review cadence",
]

SENTENCES = [
    "The work is scheduled inside the published maintenance slot and never during a customer peak.",
    "The operator records the change ticket identifier on every command they execute.",
    "A dry run is executed against the staging facility before the production facility is touched.",
    "The verification step compares the reported checksum against the value recorded in the runbook.",
    "If the verification step fails, the operator rolls back to the previous baseline and pages the lead.",
    "The escalation lead is notified when a step exceeds its documented time budget.",
    "All commands are captured in the session log and attached to the change ticket afterwards.",
    "The review cadence is quarterly, and findings are reported to the platform review board.",
    "Access requires a valid approval bound to the change ticket; approvals are never reused.",
    "The baseline is stored in the configuration repository and referenced by commit hash.",
    "Capacity headroom must stay above thirty percent after the change is complete.",
    "The audit relay forwards every administrative action to the central log within one minute.",
    "Firmware versions are pinned until the vendor publishes a security advisory.",
    "Retention settings are inherited from the tenant contract and cannot be reduced by operators.",
    "A second operator confirms the rollback plan before any destructive step is executed.",
    "Monitoring alerts are silenced only for the duration of the approved window.",
]

FACILITIES = ["Rosario", "Buenos Aires", "Montevideo", "Lima"]
ROWS = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"]
PORT_BLOCKS = [7411, 7422, 7433, 7444, 8411, 8422, 9411, 9422]

# ── Opaque needles (D16 corpus-size curve) ───────────────────────────────────
# A needle is an (entity, answer) pair where BOTH sides are random tokens:
#   "The calibration code for the cooling loop LOOP-412 is ARC-9F3K-2214."
# The probe names the entity (so the gold chunk is findable) but never the
# answer, and the answer appears in exactly one document. That makes a correct
# response **impossible to infer**: it cannot be guessed, derived from the
# probe, read off general knowledge, or reconstructed from another document —
# the gold chunk has to be retrieved (stress-plan rule R10).
NEEDLE_ARTIFACTS = [
    "calibration",
    "provisioning",
    "decommissioning",
    "failover",
    "escrow",
    "attestation",
]
NEEDLE_SUBJECTS = [
    "cooling loop",
    "rack row",
    "object store bucket",
    "scheduler pool",
    "certificate ring",
    "backup vault",
]
NEEDLE_ENTITY_PREFIXES = ["LOOP", "RACK", "BSTORE", "POOL", "RING", "VAULT"]

#: Format of a generated answer token (validated on every needle).
ANSWER_RE = re.compile(r"^[A-Z]{3}-[A-Z0-9]{4}-[0-9]{4}$")


def needle_sentence(artifact: str, subject: str, entity: str, answer: str) -> str:
    """The single corpus sentence that states the answer."""
    return f"The {artifact} code for the {subject} {entity} is {answer}."


def needle_probe(artifact: str, subject: str, entity: str) -> str:
    """The probe that names the entity but never the answer."""
    return f"What is the {artifact} code for the {subject} {entity}?"


def build_needles(
    rng: random.Random,
    docs: int,
    count: int,
    *,
    section: int = 2,
) -> list[dict[str, object]]:
    """Create ``count`` distinct (entity, answer) needles spread over ``docs``.

    Documents are chosen by even spacing (``1 + k*docs//count``) so the needles
    are not clustered at the start of the corpus. Entity and answer tokens are
    unique within a corpus and carry no semantic relation to the subject, which
    is what makes the answer non-inferable.

    Args:
        rng: Dedicated RNG (must not be the document RNG, so document bodies
            stay byte-identical across corpus sizes).
        docs: Total number of documents in the corpus.
        count: Number of needles to plant (1..docs).
        section: 0-based section index that carries the needle.

    Returns:
        One dict per needle with ``doc_index``, ``doc_id``, ``entity``,
        ``artifact``, ``subject``, ``answer``, ``probe``, ``sentence`` and
        ``section``.
    """
    if count < 0:
        raise ValueError("needle count must be >= 0")
    if count > docs:
        raise ValueError(
            f"cannot plant {count} needles in {docs} documents "
            "(one needle per document)"
        )

    doc_indices: list[int] = []
    for k in range(count):
        doc_index = 1 + (k * docs) // count
        if doc_index in doc_indices:  # pragma: no cover - defensive
            raise ValueError(f"needle placement collided on document {doc_index}")
        doc_indices.append(doc_index)

    needles: list[dict[str, object]] = []
    used_answers: set[str] = set()
    used_entities: set[str] = set()
    for k, doc_index in enumerate(doc_indices):
        artifact = NEEDLE_ARTIFACTS[k % len(NEEDLE_ARTIFACTS)]
        subject = NEEDLE_SUBJECTS[k % len(NEEDLE_SUBJECTS)]
        entity_prefix = NEEDLE_ENTITY_PREFIXES[k % len(NEEDLE_ENTITY_PREFIXES)]

        entity = f"{entity_prefix}-{rng.randint(100, 999)}"
        while entity in used_entities:
            entity = f"{entity_prefix}-{rng.randint(100, 999)}"
        used_entities.add(entity)

        answer = (
            "".join(rng.choices(string.ascii_uppercase, k=3))
            + "-"
            + "".join(rng.choices(string.ascii_uppercase + string.digits, k=4))
            + f"-{rng.randint(1000, 9999)}"
        )
        while answer in used_answers:
            answer = (
                "".join(rng.choices(string.ascii_uppercase, k=3))
                + "-"
                + "".join(rng.choices(string.ascii_uppercase + string.digits, k=4))
                + f"-{rng.randint(1000, 9999)}"
            )
        used_answers.add(answer)

        needles.append(
            {
                "doc_index": doc_index,
                "doc_id": None,  # filled in by main() once the doc id is known
                "entity": entity,
                "artifact": artifact,
                "subject": subject,
                "answer": answer,
                "probe": needle_probe(artifact, subject, entity),
                "sentence": needle_sentence(artifact, subject, entity, answer),
                "section": section,
            }
        )
    return needles


def _coerce_text(value: str) -> str:
    """Normalize CLI-provided text to a single readable line."""
    return " ".join(value.split())


def build_document(
    rng: random.Random,
    doc_index: int,
    chars_per_doc: int,
    *,
    needle: str | None,
    prefix: str,
    needle_section: int = 2,
) -> str:
    """Build one synthetic multi-section document.

    Args:
        rng: Seeded random generator (determinism lives here).
        doc_index: 1-based index used in the title and section numbering.
        chars_per_doc: Target length; paragraphs are appended until reached.
        needle: Paragraph to plant in this document, or ``None``.
        prefix: File/doc-id prefix (``corpus``).
        needle_section: 1-based section number that carries the needle
            (default 2, the historical position). It is appended right after
            that section's paragraphs; if the document never reaches it, the
            needle is appended at the end.

    Returns:
        The document body as Markdown text.
    """
    topic = rng.choice(TOPICS)
    facility = rng.choice(FACILITIES)
    row = rng.choice(ROWS)
    port = rng.choice(PORT_BLOCKS)

    lines: list[str] = [
        f"# {prefix.title()} Note {doc_index:03d}: {topic.title()}",
        "",
        f"Facility: {facility}. Row: {row}. Service port: {port}.",
        "Classification: Internal. Owner: Platform Operations.",
        "",
    ]

    section_index = 0
    while sum(len(line) + 1 for line in lines) < chars_per_doc:
        heading = SECTIONS[section_index % len(SECTIONS)]
        section_index += 1
        lines.append(f"## {section_index}. {heading}")
        lines.append("")
        for _ in range(rng.randint(3, 5)):
            lines.append(
                f"{rng.choice(SENTENCES)} Reference {prefix.upper()}-"
                f"{doc_index:03d}-{section_index:02d}."
            )
            lines.append("")
        if needle is not None and section_index == needle_section:
            lines.append(needle)
            lines.append("")

    if needle is not None and needle not in lines:
        lines.extend([needle, ""])

    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic synthetic document corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--docs", type=int, default=12, help="Number of documents")
    parser.add_argument(
        "--chars-per-doc", type=int, default=4000, help="Approximate chars per document"
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument(
        "--needle",
        type=str,
        default=argparse.SUPPRESS,
        help=(
            "Needle paragraph planted in exactly one document (omit for the "
            "built-in default; '' disables)"
        ),
    )
    parser.add_argument(
        "--needle-doc",
        type=int,
        default=None,
        help="1-based document index that holds the needle (random when omitted)",
    )
    parser.add_argument(
        "--needles",
        type=int,
        default=0,
        help=(
            "Plant N opaque (entity, answer) needles instead of --needle; "
            "N distinct documents get one each (D16). Mutually exclusive with "
            "--needle/--needle-doc."
        ),
    )
    parser.add_argument(
        "--needle-section",
        type=int,
        default=2,
        help="1-based section number that carries the needle",
    )
    parser.add_argument(
        "--prefix", type=str, default="corpus", help="File name / doc-id prefix"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be written without touching the filesystem",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.docs < 1:
        print("Error: --docs must be >= 1.")
        return 2
    if args.chars_per_doc < 200:
        print("Error: --chars-per-doc must be >= 200.")
        return 2

    # ``--needle`` uses SUPPRESS so "flag omitted" is distinguishable from an
    # explicit value: omitted keeps the historical DEFAULT_NEEDLE, while
    # ``--needle ''`` disables it, and ``--needles`` can reject an explicit
    # ``--needle`` without tripping over the default.
    needle_provided = hasattr(args, "needle")
    legacy_needle_text = args.needle if needle_provided else DEFAULT_NEEDLE
    legacy_needle = (
        _coerce_text(legacy_needle_text)
        if legacy_needle_text and legacy_needle_text.strip()
        else None
    )

    if args.needles and (needle_provided or args.needle_doc is not None):
        print("Error: --needles cannot be combined with --needle/--needle-doc.")
        return 2
    if args.needles < 0:
        print("Error: --needles must be >= 0.")
        return 2
    if args.needles > args.docs:
        print(f"Error: --needles ({args.needles}) cannot exceed --docs ({args.docs}).")
        return 2
    if args.needle_section < 1:
        print("Error: --needle-section must be >= 1.")
        return 2
    if args.needle_doc is not None and not 1 <= args.needle_doc <= args.docs:
        print(f"Error: --needle-doc must be within 1..{args.docs}.")
        return 2

    rng = random.Random(args.seed)

    # ── Multi-needle mode (D16) ──────────────────────────────────────────
    # A dedicated RNG keeps the document bodies byte-identical to a plain run
    # of the same seed: the needle assignment must not perturb the corpus, or
    # corpus sizes {8,60,200} would no longer share their first N documents.
    needles: list[dict[str, object]] = []
    sentence_by_doc: dict[int, str] = {}
    if args.needles:
        needles = build_needles(
            # Seed includes the corpus size: each corpus gets its own opaque
            # (entity, answer) set, so no answer token is ever reused across
            # the D16 curve and a probe is meaningful only inside its corpus.
            random.Random(f"{args.seed}:needles:{args.docs}"),
            args.docs,
            args.needles,
            section=args.needle_section,
        )
        for n in needles:
            n["doc_id"] = f"{args.prefix}_{int(n['doc_index']) - 1:03d}"
        sentence_by_doc = {
            int(n["doc_index"]): str(n["sentence"]) for n in needles
        }
        needle = None
        needle_doc = None
    else:
        needle = legacy_needle
        needle_doc = (
            args.needle_doc
            if args.needle_doc is not None
            else (rng.randint(1, args.docs) if needle else None)
        )

    entries: list[dict[str, object]] = []
    for index in range(1, args.docs + 1):
        body = build_document(
            rng,
            index,
            args.chars_per_doc,
            needle=sentence_by_doc.get(index, needle if index == needle_doc else None),
            prefix=args.prefix,
            needle_section=args.needle_section,
        )
        file_name = f"{args.prefix}_{index - 1:03d}.md"
        entries.append(
            {
                "doc_id": f"{args.prefix}_{index - 1:03d}",
                "file": file_name,
                "chars": len(body),
                "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                "has_needle": index == needle_doc or index in sentence_by_doc,
                "_body": body,
            }
        )

    # ── Non-inferability guard ───────────────────────────────────────────
    # An answer that occurs twice, or an entity that is ambiguous, would let a
    # model answer from the wrong chunk (or from general knowledge) and break
    # the "gold chunk is necessary" contract of D16.
    if needles:
        corpus_text = "\n".join(str(entry["_body"]) for entry in entries)
        for n in needles:
            answer = str(n["answer"])
            entity = str(n["entity"])
            if not ANSWER_RE.match(answer):
                print(f"Error: generated answer {answer!r} is malformed.")
                return 2
            if corpus_text.count(answer) != 1:
                print(
                    f"Error: answer {answer!r} occurs "
                    f"{corpus_text.count(answer)} times in the corpus (must be 1)."
                )
                return 2
            if corpus_text.count(entity) != 1:
                print(
                    f"Error: entity {entity!r} occurs "
                    f"{corpus_text.count(entity)} times in the corpus (must be 1)."
                )
                return 2
            if answer.casefold() in str(n["probe"]).casefold():
                print(f"Error: probe echoes its own answer {answer!r}.")
                return 2

    manifest = {
        "generated_by": "scripts/generate_corpus.py",
        "seed": args.seed,
        "prefix": args.prefix,
        "docs": args.docs,
        "chars_per_doc": args.chars_per_doc,
        "needle": (
            {"doc_id": f"{args.prefix}_{needle_doc - 1:03d}", "text": needle}
            if needle
            else None
        ),
        # Multi-needle contract (D16). Empty list keeps the key stable so
        # readers never have to branch on its presence.
        "needles": needles,
        "files": [{k: v for k, v in entry.items() if k != "_body"} for entry in entries],
    }

    total_chars = sum(int(entry["chars"]) for entry in entries)
    print(
        f"{args.docs} document(s), {total_chars:,} chars total "
        f"(~{total_chars // 4:,} tokens), seed={args.seed}"
    )
    if manifest["needle"]:
        print(f"Needle planted in: {manifest['needle']['doc_id']}")
    if needles:
        print(
            f"{len(needles)} opaque needle(s) planted in: "
            + ", ".join(str(n["doc_id"]) for n in needles)
        )

    if args.dry_run:
        print(f"[dry-run] would write {args.docs + 1} file(s) to {args.out}")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        (args.out / str(entry["file"])).write_text(
            str(entry["_body"]), encoding="utf-8", newline="\n"
        )
    (args.out / f"{args.prefix}_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(f"Wrote corpus to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

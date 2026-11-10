#!/usr/bin/env python3
"""Build the D16 corpus-size scenarios from generated corpus manifests.

D16 (``TODOs/phase-2-l2-document-stress-plan.md`` §5.3.5) measures
``Recall@5`` against corpus size. Hand-writing three YAMLs with 8 / 60 / 200
``documents:`` entries is unreviewable, so this script derives them from the
manifests that ``scripts/generate_corpus.py --needles`` writes. The YAMLs are
deterministic: same seed + same generator flags ⇒ byte-identical scenarios.

Regeneration (three corpora + three scenarios)::

    python scripts/generate_corpus.py --docs 8   --chars-per-doc 6000 --seed 7 \
        --needles 8 --out datasets/documents/d16_corpus_008
    python scripts/generate_corpus.py --docs 60  --chars-per-doc 6000 --seed 7 \
        --needles 8 --out datasets/documents/d16_corpus_060
    python scripts/generate_corpus.py --docs 200 --chars-per-doc 6000 --seed 7 \
        --needles 8 --out datasets/documents/d16_corpus_200
    python scripts/build_d16_scenarios.py

Design contract encoded in every generated scenario:

* **Eight probes, one needle each.** Each probe names a random entity
  (``cooling loop LOOP-412``) and never its answer (``ARC-9F3K-2214``). The
  answer occurs **exactly once** in the whole corpus and nowhere else — not in
  the probe, not in another document, not derivable from general knowledge.
  A correct response therefore *requires* the gold chunk: it cannot be
  guessed, inferred from context, or reconstructed from a sibling document.
* **Single session.** All probes share ``session_id: s1``; ``retrieve_node``
  excludes chat from the *current* session, so the candidate pool is
  documents only. Earlier probe answers cannot be re-used to satisfy later
  probes (they are not retrievable), and chat volume cannot compete with the
  gold chunk — D16 isolates R-A (cutoff miss) from R-C (slot competition).
* **Document-prefix gold.** ``must_retrieve_chunk`` names ``corpus_0NN`` so the
  assertion survives a ``--chunk-size`` change. The report script
  (``scripts/lr_report.py``) recomputes a true *chunk-level* hit from
  ``retrieved_memories`` because the prefix match alone also accepts sibling
  chunks of the same document.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_ROOT = REPO_ROOT / "datasets" / "documents"
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "datasets"
    / "conversations"
    / "single_thread"
    / "retrieval_limit"
    / "document_corpus_size"
)
DEFAULT_DIR_TEMPLATE = "d16_corpus_{size:03d}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the D16 corpus-size scenario YAMLs from manifests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=DEFAULT_CORPUS_ROOT,
        help="Directory that holds the generated corpora",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="8,60,200",
        help="Comma-separated corpus sizes (documents per corpus)",
    )
    parser.add_argument(
        "--dir-template",
        type=str,
        default=DEFAULT_DIR_TEMPLATE,
        help="Corpus directory name template relative to --corpus-root",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory that receives the scenario YAMLs",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="top_k recorded in metadata (the run must pass the same value)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=600,
        help="chunk_size recorded in metadata (D16 fixes it for the curve)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be written without touching the filesystem",
    )
    return parser.parse_args(argv)


def build_scenario(
    manifest: dict[str, Any], *, top_k: int, chunk_size: int
) -> dict[str, Any]:
    """Turn one corpus manifest into a D16 scenario dict (YAML-ready)."""
    needles = manifest.get("needles") or []
    if len(needles) < 2:
        raise ValueError(
            f"manifest has {len(needles)} needle(s); regenerate with "
            "--needles 8 (D16 needs >= 2 probes to report a rate)"
        )

    size = int(manifest["docs"])
    seed = int(manifest["seed"])
    chars_per_doc = int(manifest["chars_per_doc"])
    corpus_dir = str(manifest.get("_corpus_dir", ""))
    scenario_id = f"document_corpus_size_{size:03d}_01"

    needles = sorted(needles, key=lambda n: int(n["doc_index"]))
    documents = [
        {
            "doc_id": str(entry["doc_id"]),
            "path": f"{corpus_dir}/{entry['file']}",
            "metadata": {"title": f"Synthetic operations note {entry['doc_id']}"},
        }
        for entry in manifest["files"]
    ]

    turns: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    for turn_number, needle in enumerate(needles, start=1):
        turns.append(
            {
                "turn_number": turn_number,
                "session_id": "s1",
                "role": "user",
                "content": str(needle["probe"]),
                "assertions": [
                    # Corpus-exclusive token: appears in exactly one document
                    # and in no probe (generator-validated), so satisfying it
                    # requires the gold chunk.
                    {"type": "must_contain", "value": str(needle["answer"])},
                    # Document-prefix gold (chunk-size independent); the true
                    # chunk-level hit is recomputed by scripts/lr_report.py.
                    {"type": "must_retrieve_chunk", "value": str(needle["doc_id"])},
                ],
            }
        )
        probes.append(
            {
                "turn": turn_number,
                "doc_id": str(needle["doc_id"]),
                "entity": str(needle["entity"]),
                "answer": str(needle["answer"]),
                "section": int(needle.get("section", 2)),
            }
        )

    description = (
        f"L-R corpus-size curve point: {size} generated documents "
        f"(~{size * chars_per_doc // 1000} KB) with {len(needles)} opaque "
        "needles. Every probe names an entity that occurs once in the corpus "
        "and requests an answer token that occurs once in that entity's "
        "document — a correct answer is impossible to infer and therefore "
        "proves the gold chunk reached the prompt. All probes share one "
        "session, so current-session chat is excluded from retrieval and the "
        "candidate pool is documents only."
    )

    return {
        "id": scenario_id,
        "name": f"L-R Corpus-Size Curve: {size} Documents ({len(needles)} Probes)",
        "description": description,
        "category": "document_corpus_size",
        "metadata": {
            "difficulty": "hard" if size >= 60 else "moderate",
            "stage": "L-R",
            "target_research_question": "P1",
            "expected_level_0_behavior": "Fails (0%). No memory at all.",
            "expected_level_1_behavior": (
                "Fails (0%). Documents are never ingested and nothing survives "
                "the session thread."
            ),
            "expected_level_2_behavior": (
                "Recall@k is the measured quantity: correct answers track "
                "chunk retrieval, and misses are R-A (gold ranks after top_k)."
            ),
            "corpus_docs": size,
            "chars_per_doc": chars_per_doc,
            "corpus_seed": seed,
            "corpus_dir": corpus_dir,
            "default_top_k": top_k,
            "chunk_size": chunk_size,
            "probe_count": len(needles),
            "needle_values": [str(n["answer"]) for n in needles],
            "probes": probes,
        },
        "documents": documents,
        "turns": turns,
    }


def render_yaml(scenario: dict[str, Any], *, corpus_dir: str) -> str:
    """Serialize a scenario with a reviewable header comment."""
    meta = scenario["metadata"]
    header = (
        "# ─────────────────────────────────────────────────────────────────\n"
        "# D16 — L-R corpus-size curve (AUTO-GENERATED, do not hand-edit).\n"
        "#\n"
        "# Regenerate with:\n"
        "#   python scripts/generate_corpus.py --docs "
        f"{meta['corpus_docs']} --chars-per-doc {meta['chars_per_doc']} "
        f"--seed {meta['corpus_seed']} --needles {meta['probe_count']} --out "
        f"datasets/documents/{corpus_dir}\n"
        "#   python scripts/build_d16_scenarios.py\n"
        "#\n"
        "# Non-inferability: each probe asks for a random answer token that\n"
        "# occurs exactly once in the corpus (generator-validated) and never in\n"
        "# the probe itself, so the gold chunk is necessary to answer.\n"
        "# ─────────────────────────────────────────────────────────────────\n"
    )
    body = yaml.safe_dump(
        scenario,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=1000,
    )
    return header + body


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    except ValueError:
        print(f"Error: --sizes must be comma-separated integers, got {args.sizes!r}.")
        return 2
    if not sizes:
        print("Error: --sizes is empty.")
        return 2

    written: list[Path] = []
    for size in sizes:
        corpus_dir = args.dir_template.format(size=size)
        manifest_path = args.corpus_root / corpus_dir / "corpus_manifest.json"
        if not manifest_path.exists():
            print(f"Error: missing manifest {manifest_path}. Generate it first.")
            return 2
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["_corpus_dir"] = corpus_dir
        manifest["docs"] = int(manifest.get("docs", size))
        if manifest["docs"] != size:
            print(
                f"Error: {manifest_path} declares {manifest['docs']} docs, "
                f"expected {size}."
            )
            return 2

        try:
            scenario = build_scenario(
                manifest, top_k=args.top_k, chunk_size=args.chunk_size
            )
        except ValueError as exc:
            print(f"Error: {exc}")
            return 2

        text = render_yaml(scenario, corpus_dir=corpus_dir)
        target = args.out_dir / f"test_{scenario['id']}.yaml"
        if args.dry_run:
            print(f"[dry-run] would write {target} ({len(text):,} chars)")
            continue
        args.out_dir.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
        written.append(target)
        print(
            f"Wrote {target.name} ({scenario['metadata']['corpus_docs']} docs, "
            f"{scenario['metadata']['probe_count']} probes)"
        )

    if args.dry_run:
        return 0
    print(f"\n{len(written)} scenario file(s) written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


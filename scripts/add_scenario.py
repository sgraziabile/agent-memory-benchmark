#!/usr/bin/env python3
"""Materialize scenario YAMLs embedded in a stage spec into datasets/.

Stage specs (``TODOs/specs/stageN-test-specs.md``) embed each scenario as a
fenced ```` ```yaml ```` block. This script extracts those blocks and writes
them verbatim into the themed dataset tree, so going from an approved spec
to runnable scenarios is a single command instead of a manual transcription::

    python scripts/add_scenario.py TODOs/specs/stage4-test-specs.md
    python scripts/add_scenario.py TODOs/specs/stage4-test-specs.md --dry-run
    python scripts/add_scenario.py spec.md --map new_id=theme/category

For each ```` ```yaml ```` block the script:

1. Parses it with ``yaml.safe_load`` and reads its ``id`` / ``category``.
2. Resolves the destination ``(theme, category)`` from ``--map`` overrides
   first, then the existing dataset tree (an ID already placed under
   ``<theme>/<category>/`` is its own source of truth), then the static
   ``SCENARIO_MAP``, and finally a ``category`` that already exists
   somewhere in the tree (keeping its theme).
3. Resolves the thread-class directory from ``--thread`` first, then the
   existing placement of that ID, then the block's own turns (one distinct
   ``session_id`` → ``single_thread/``, more than one → ``multi_thread/``).
4. Validates the scenario against ``core.schemas.BenchmarkScenario`` —
   the same model the runner uses — failing fast on schema errors.
5. Writes ``datasets/conversations/<thread>/<theme>/<category>/test_<id>.yaml``
   (UTF-8, LF) unless an identical file already exists; an existing file
   that differs from the spec block is overwritten to match the spec.

Exits non-zero if any block fails to parse, validate, or resolve.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.dataset_layout import (  # noqa: E402
    MULTI_THREAD_DIR,
    SCENARIO_MAP,
    SINGLE_THREAD_DIR,
    dataset_root,
    scenario_id_from_filename,
    theme_category_from_parts,
    thread_dir_from_parts,
    thread_dir_for_scenario,
)
from core.schemas import BenchmarkScenario  # noqa: E402

DATASET_DIR = dataset_root()
YAML_FENCE = re.compile(r"```yaml\r?\n(.*?)```", re.DOTALL)

#: CLI aliases for ``--thread``.
_THREAD_ALIASES = {
    "single": SINGLE_THREAD_DIR,
    "single-thread": SINGLE_THREAD_DIR,
    "multi": MULTI_THREAD_DIR,
    "multi-thread": MULTI_THREAD_DIR,
}


def build_tree_index() -> tuple[
    dict[str, tuple[str, str]], dict[str, str], dict[str, str]
]:
    """Scan the dataset tree for ids, category themes and thread classes.

    Accepts both the current 4-part layout
    (``<thread>/<theme>/<category>/test_<id>.yaml``) and the legacy 3-part
    one, because :func:`core.dataset_layout.theme_category_from_parts` reads
    the *last two* directory components instead of assuming a fixed depth.

    Returns:
        ``(ids, categories, threads)`` — ``ids`` maps an id to its
        ``(theme, category)``, ``categories`` maps a category to its theme,
        and ``threads`` maps an id to the thread-class directory it lives in
        (``single_thread``/``multi_thread``), when present.
    """
    ids: dict[str, tuple[str, str]] = {}
    categories: dict[str, str] = {}
    threads: dict[str, str] = {}
    for path in sorted(DATASET_DIR.rglob("test_*.yaml")):
        rel = path.relative_to(DATASET_DIR)
        if len(rel.parts) < 3:
            continue  # top-level stragglers are not organized
        sid = scenario_id_from_filename(path.name)
        if sid is None:
            continue
        theme, category = theme_category_from_parts(rel.parts)
        ids[sid] = (theme, category)
        categories.setdefault(category, theme)
        thread = thread_dir_from_parts(rel.parts)
        if thread is not None:
            threads[sid] = thread
    return ids, categories, threads


def resolve_destination(
    sid: str,
    category: str,
    overrides: dict[str, tuple[str, str]],
    tree_ids: dict[str, tuple[str, str]],
    tree_categories: dict[str, str],
    static_map: dict[str, tuple[str, str]],
) -> tuple[str, str] | None:
    """Resolve ``(theme, category)`` for a scenario id (see module docstring)."""
    if sid in overrides:
        return overrides[sid]
    if sid in tree_ids:
        return tree_ids[sid]
    if sid in static_map:
        return static_map[sid]
    if category in tree_categories:
        return (tree_categories[category], category)
    return None


def extract_yaml_blocks(spec_text: str) -> list[str]:
    """Return every fenced ```` ```yaml ```` block in the spec, verbatim.

    Blocks that don't define a top-level ``id`` key (header snippets,
    examples of fragments, ...) are skipped: only full scenario
    definitions are materialized.
    """
    blocks: list[str] = []
    for match in YAML_FENCE.finditer(spec_text):
        block = match.group(1)
        try:
            data = yaml.safe_load(block)
        except yaml.YAMLError:
            continue  # not a scenario — ignore non-YAML fence content
        if isinstance(data, dict) and "id" in data and "turns" in data:
            blocks.append(block)
    return blocks


def _fmt_path(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("spec", type=Path, help="path to the stage spec markdown")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would be written without writing anything",
    )
    parser.add_argument(
        "--map",
        action="append",
        default=[],
        metavar="ID=THEME/CATEGORY",
        help="destination override for a scenario id (repeatable)",
    )
    parser.add_argument(
        "--thread",
        choices=sorted(_THREAD_ALIASES),
        default=None,
        help=(
            "force the thread-class directory for every block "
            "(default: keep an existing placement, else classify from the "
            "block's session_id values)"
        ),
    )
    args = parser.parse_args(argv)

    spec_text = args.spec.read_text(encoding="utf-8")
    blocks = extract_yaml_blocks(spec_text)
    if not blocks:
        print(f"No scenario yaml blocks found in {args.spec}")
        return 1

    overrides: dict[str, tuple[str, str]] = {}
    for item in args.map:
        try:
            sid, dest = item.split("=", 1)
            theme, category = dest.split("/", 1)
            overrides[sid] = (theme, category)
        except ValueError:
            parser.error(f"--map expects ID=THEME/CATEGORY, got {item!r}")

    # Static fallback registry lives in core.dataset_layout (single source of
    # truth shared with the test suite).
    static_map: dict[str, tuple[str, str]] = dict(SCENARIO_MAP)

    tree_ids, tree_categories, tree_threads = build_tree_index()
    thread_override = (
        _THREAD_ALIASES[args.thread] if args.thread is not None else None
    )
    written = skipped = 0
    failures = 0

    for block in blocks:
        data = yaml.safe_load(block)
        sid = str(data["id"])
        category = str(data.get("category", ""))

        # Schema-validate against the same model the runner uses.
        try:
            validated = BenchmarkScenario.model_validate(data)
        except Exception as exc:  # pydantic.ValidationError
            print(f"  ! {sid}: schema validation failed: {exc}")
            failures += 1
            continue

        dest = resolve_destination(
            sid, category, overrides, tree_ids, tree_categories, static_map
        )
        if dest is None:
            print(
                f"  ! {sid}: cannot resolve (theme, category) — pass "
                f"--map {sid}=THEME/CATEGORY"
            )
            failures += 1
            continue
        theme, dest_category = dest
        thread = (
            thread_override
            or tree_threads.get(sid)
            or thread_dir_for_scenario(validated)
        )

        out = DATASET_DIR / thread / theme / dest_category / f"test_{sid}.yaml"
        if out.exists() and out.read_text(encoding="utf-8") == block:
            print(f"  = {sid}: unchanged at {_fmt_path(out)}")
            skipped += 1
            continue
        if args.dry_run:
            print(f"  ~ {sid}: would write {_fmt_path(out)}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(block, encoding="utf-8", newline="\n")
        print(f"  + {sid}: wrote {_fmt_path(out)}")
        written += 1
        tree_ids[sid] = (theme, dest_category)
        tree_categories.setdefault(dest_category, theme)
        tree_threads[sid] = thread

    print(
        f"\n{len(blocks)} scenario block(s): {written} written, "
        f"{skipped} unchanged, {failures} failed"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

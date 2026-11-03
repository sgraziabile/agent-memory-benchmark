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
3. Validates the scenario against ``core.schemas.BenchmarkScenario`` —
   the same model the runner uses — failing fast on schema errors.
4. Writes ``datasets/conversations/<theme>/<category>/test_<id>.yaml``
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

from core.schemas import BenchmarkScenario  # noqa: E402

DATASET_DIR = REPO_ROOT / "datasets" / "conversations"
YAML_FENCE = re.compile(r"```yaml\r?\n(.*?)```", re.DOTALL)


def build_tree_index() -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
    """Scan the dataset tree for ``id -> (theme, category)`` and category themes.

    Only organized files (``<theme>/<category>/test_<id>.yaml``) count;
    top-level stragglers are ignored.
    """
    ids: dict[str, tuple[str, str]] = {}
    categories: dict[str, str] = {}
    for path in sorted(DATASET_DIR.rglob("test_*.yaml")):
        rel = path.relative_to(DATASET_DIR)
        if len(rel.parts) != 3:
            continue
        ids[path.name[len("test_") : -len(".yaml")]] = (rel.parts[0], rel.parts[1])
        categories.setdefault(rel.parts[1], rel.parts[0])
    return ids, categories


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

    # Import the shared registry from organize_datasets (self-contained script,
    # so load it from its file path).
    import importlib.util

    spec_module_path = REPO_ROOT / "scripts" / "organize_datasets.py"
    spec_module = importlib.util.spec_from_file_location(
        "_organize_datasets", spec_module_path
    )
    organize = importlib.util.module_from_spec(spec_module)
    spec_module.loader.exec_module(organize)
    static_map: dict[str, tuple[str, str]] = dict(organize.SCENARIO_MAP)

    tree_ids, tree_categories = build_tree_index()
    written = skipped = 0
    failures = 0

    for block in blocks:
        data = yaml.safe_load(block)
        sid = str(data["id"])
        category = str(data.get("category", ""))

        # Schema-validate against the same model the runner uses.
        try:
            BenchmarkScenario.model_validate(data)
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

        out = DATASET_DIR / theme / dest_category / f"test_{sid}.yaml"
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

    print(
        f"\n{len(blocks)} scenario block(s): {written} written, "
        f"{skipped} unchanged, {failures} failed"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

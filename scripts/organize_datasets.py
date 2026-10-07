#!/usr/bin/env python3
"""Organize scenario dataset files by research theme and test category.

Moves every ``test_*.yaml`` file from the top level of
``datasets/conversations/`` into a nested structure that mirrors the themes
used for benchmark outputs (``outputs/by_theme/``)::

    datasets/conversations/
    ├── logical_consistency/
    │   ├── belief_revision/
    │   │   └── test_belief_revision_01.yaml
    │   ├── explicit_forget/
    │   └── in_context_control/
    ├── retention_persistence/
    │   ├── attrition/
    │   └── needle_haystack/
    ├── compositional_reasoning/
    │   └── temporal_multi_hop/
    ├── robustness_security/        ← .gitkeep (future tests)
    └── computational_cost/         ← .gitkeep (future tests)

Scenario -> theme mapping reuses the same registry as
``scripts/reorganize_outputs.py`` (kept as a literal here so the script is
self-contained). Unknown scenario IDs map to ``others/<scenario_id>/``.

Unlike the outputs reorganization, files are **moved** (not copied): the
dataset YAMLs are version-controlled source files and a single copy is the
source of truth. The benchmark loader (``core/runner.py``) discovers
scenario files recursively, so this layout change is transparent to it.

The script is idempotent: files already organized are left untouched, and a
re-run simply finds nothing to move.

Usage::

    python scripts/organize_datasets.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "datasets" / "conversations"

# ── scenario_id -> (theme, test_category) mapping ───────────────────────────
# (identical to scripts/reorganize_outputs.py; duplicated for self-containment)
SCENARIO_MAP: dict[str, tuple[str, str]] = {
    "belief_revision_01": ("logical_consistency", "belief_revision"),
    "explicit_forget_01": ("logical_consistency", "explicit_forget"),
    "in_context_control_01": ("logical_consistency", "in_context_control"),
    "attrition_01": ("retention_persistence", "attrition"),
    "needle_haystack_01": ("retention_persistence", "needle_haystack"),
    "temporal_multi_hop_01": ("compositional_reasoning", "temporal_multi_hop"),
}

# Themes with no scenarios yet — scaffolded with a .gitkeep file so version
# control tracks them, preparing the structure for the planned tests
# (benchmark_expansion_plan.md §4.1 computational_cost, §4.2 robustness_security).
EMPTY_THEMES: tuple[str, ...] = ("robustness_security", "computational_cost")


def scenario_id_from_filename(filename: str) -> str | None:
    """Derive the scenario ID from a ``test_<scenario>.yaml`` filename."""
    if not (filename.startswith("test_") and filename.endswith(".yaml")):
        return None
    return filename[len("test_") : -len(".yaml")]


def organize() -> tuple[int, int]:
    """Move top-level scenario YAMLs into themed subdirectories.

    Returns ``(moved, unknown)`` — files moved now and files mapped to the
    ``others`` fallback theme. Files already organized (nested under any
    subdirectory) are never touched.
    """
    moved = 0
    unknown = 0

    # Scaffold the empty themes (idempotent: only created if missing).
    for theme in EMPTY_THEMES:
        theme_dir = DATASET_DIR / theme
        theme_dir.mkdir(parents=True, exist_ok=True)
        gitkeep = theme_dir / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.write_bytes(b"")

    # Only top-level scenario files are candidates; nested ones are already
    # organized and must not be moved again (idempotency).
    for yaml_path in sorted(DATASET_DIR.glob("test_*.yaml")):
        scenario_id = scenario_id_from_filename(yaml_path.name)
        if scenario_id is None:
            continue
        theme, test_category = SCENARIO_MAP.get(
            scenario_id, ("others", scenario_id)
        )
        if theme == "others":
            unknown += 1
        dest_dir = DATASET_DIR / theme / test_category
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(yaml_path), str(dest_dir / yaml_path.name))
        print(f"  {yaml_path.name} -> {theme}/{test_category}/{yaml_path.name}")
        moved += 1

    return moved, unknown


def main() -> None:
    if not DATASET_DIR.is_dir():
        raise SystemExit(f"Dataset directory not found: {DATASET_DIR}")

    print(f"Organizing scenario files under {DATASET_DIR} ...")
    for theme in EMPTY_THEMES:
        print(f"  + ensured empty theme '{theme}/' with .gitkeep")
    moved, unknown = organize()

    if moved == 0:
        print("  = nothing to move — all scenario files already organized")
    else:
        print(f"  = {moved} file(s) moved")
    if unknown:
        print(f"  ! {unknown} file(s) mapped to 'others' (unknown scenario_id)")

    print("\nCurrent themed structure:")
    for theme_dir in sorted(p for p in DATASET_DIR.iterdir() if p.is_dir()):
        files = [
            f"{cat.name}/{y.name}"
            for cat in sorted(p for p in theme_dir.iterdir() if p.is_dir())
            for y in sorted(cat.glob("*.yaml"))
        ]
        if files:
            for f in files:
                print(f"  - {theme_dir.name}/{f}")
        elif (theme_dir / ".gitkeep").exists():
            print(f"  - {theme_dir.name}/ (.gitkeep — no scenarios yet)")
    print("\nDone. Remember: core/runner.py discovers scenario files recursively.")


if __name__ == "__main__":
    main()

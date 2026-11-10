"""Invariants for the dataset tree layout.

Pins the contract introduced when scenarios moved to
``datasets/conversations/<single_thread|multi_thread>/<theme>/<category>/``:

1. every scenario file lives at exactly that depth;
2. its thread-class directory matches the number of distinct ``session_id``
   values in its turns (the same classifier the harness uses);
3. its id matches its file name and no two files share an id;
4. the theme/category is recoverable from the path alone;
5. the registry consumer (``scripts.add_scenario``) and the loader both see
   every file — a silently skipped YAML would otherwise only log a warning.

Everything here is offline: no API keys, no model calls.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from core.dataset_layout import (
    MULTI_THREAD_DIR,
    SINGLE_THREAD_DIR,
    THREAD_DIRS,
    scenario_id_from_filename,
    scenario_relpath,
    theme_category_from_parts,
    thread_dir_from_parts,
    thread_dir_for_sessions,
)
from core.runner import BenchmarkRunner

REPO_ROOT = Path(__file__).resolve().parent.parent
CONVERSATIONS = REPO_ROOT / "datasets" / "conversations"

SCENARIO_FILES = sorted(CONVERSATIONS.rglob("test_*.yaml"))


def _rel(path: Path) -> Path:
    return path.relative_to(CONVERSATIONS)


def _session_ids(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [str(turn.get("session_id", "default")) for turn in data["turns"]]


@pytest.fixture(scope="module")
def scenarios() -> dict[str, object]:
    """Every scenario the runner loads, keyed by id."""
    return {s.id: s for s in BenchmarkRunner().load_scenarios()}


class TestLayout:
    def test_scenario_files_exist(self):
        assert SCENARIO_FILES, f"no scenario YAMLs under {CONVERSATIONS}"

    def test_every_file_is_at_thread_theme_category_depth(self):
        for path in SCENARIO_FILES:
            rel = _rel(path)
            assert len(rel.parts) == 4, (
                f"{rel}: expected <thread>/<theme>/<category>/<file>, "
                f"got {len(rel.parts)} components"
            )
            assert rel.parts[0] in THREAD_DIRS, (
                f"{rel}: first component must be one of {THREAD_DIRS}"
            )

    def test_thread_directory_matches_session_classification(self):
        for path in SCENARIO_FILES:
            expected = thread_dir_for_sessions(_session_ids(path))
            actual = thread_dir_from_parts(_rel(path).parts)
            assert actual == expected, f"{_rel(path)}: {actual} != {expected}"

    def test_single_thread_scenarios_span_exactly_one_session(self):
        for path in SCENARIO_FILES:
            if thread_dir_from_parts(_rel(path).parts) != SINGLE_THREAD_DIR:
                continue
            assert len(set(_session_ids(path))) == 1, _rel(path)

    def test_id_matches_filename_and_is_unique(self):
        seen: dict[str, Path] = {}
        for path in SCENARIO_FILES:
            sid = scenario_id_from_filename(path.name)
            assert sid is not None, path
            assert sid not in seen, (
                f"duplicate id {sid!r}: {seen.get(sid)} and {path}"
            )
            seen[sid] = path

    def test_theme_and_category_are_recoverable_from_path(self):
        for path in SCENARIO_FILES:
            theme, category = theme_category_from_parts(_rel(path).parts)
            assert theme and category
            assert theme not in THREAD_DIRS
            assert not theme.startswith("test_")
            assert not category.startswith("test_")

    def test_add_scenario_registry_sees_every_file(self):
        from scripts.add_scenario import build_tree_index

        ids, _categories, threads = build_tree_index()
        assert len(ids) == len(SCENARIO_FILES)
        assert len(threads) == len(SCENARIO_FILES)
        for path in SCENARIO_FILES:
            sid = scenario_id_from_filename(path.name)
            assert sid is not None
            assert threads[sid] == thread_dir_from_parts(_rel(path).parts)

    def test_loader_sees_every_file(self, scenarios):
        expected_ids = {scenario_id_from_filename(p.name) for p in SCENARIO_FILES}
        assert set(scenarios) == expected_ids, (
            "loader/scenario-file mismatch — a YAML was skipped as invalid?"
        )
        for sid, scenario in scenarios.items():
            assert scenario.source_path is not None, sid
            assert Path(scenario.source_path).is_relative_to(CONVERSATIONS), sid


class TestPathHelpers:
    """Guards for the helpers that replaced the old ``len(parts) != 3`` logic."""

    def test_theme_category_tolerates_legacy_three_part_paths(self):
        parts = ("logical_consistency", "belief_revision", "test_x_01.yaml")
        assert theme_category_from_parts(parts) == (
            "logical_consistency",
            "belief_revision",
        )

    def test_theme_category_uses_last_two_directories(self):
        parts = (
            MULTI_THREAD_DIR,
            "logical_consistency",
            "belief_revision",
            "test_x_01.yaml",
        )
        assert theme_category_from_parts(parts) == (
            "logical_consistency",
            "belief_revision",
        )

    def test_theme_category_rejects_unorganized_paths(self):
        with pytest.raises(ValueError):
            theme_category_from_parts(("test_x_01.yaml",))

    def test_scenario_relpath_round_trips(self):
        rel = scenario_relpath(
            "logical_consistency", "belief_revision", "x_01", MULTI_THREAD_DIR
        )
        assert str(rel).replace("\\", "/") == (
            "multi_thread/logical_consistency/belief_revision/test_x_01.yaml"
        )
        assert thread_dir_from_parts(rel.parts) == MULTI_THREAD_DIR
        assert theme_category_from_parts(rel.parts) == (
            "logical_consistency",
            "belief_revision",
        )

    def test_thread_classification_counts_distinct_sessions(self):
        assert thread_dir_for_sessions(["default"] * 5) == SINGLE_THREAD_DIR
        assert thread_dir_for_sessions(["s1", "s1", "s1"]) == SINGLE_THREAD_DIR
        assert thread_dir_for_sessions(["s1", "s2"]) == MULTI_THREAD_DIR


"""core.dataset_layout — Dataset tree layout and scenario classification.

Single source of truth for how scenario YAMLs are organized under
``datasets/conversations/``. The tree has two axes:

- **Thread class** — ``single_thread/`` holds scenarios whose turns all share
  one ``session_id`` (L0/L1 can answer them); ``multi_thread/`` holds
  scenarios that span more than one session and therefore require
  cross-thread memory (L2+). The class is derived from the YAML content
  (distinct ``session_id`` values), never hand-labelled, so it cannot drift
  from what the runner actually executes.
- **Research theme / test category** — ``<theme>/<category>/``, mirroring the
  grouping formerly used for benchmark outputs.

Layout::

    datasets/conversations/
    ├── single_thread/<theme>/<category>/test_<scenario_id>.yaml
    └── multi_thread/<theme>/<category>/test_<scenario_id>.yaml

Path helpers read the **last two directory components**, so they tolerate the
current 4-part layout and the legacy 3-part one
(``<theme>/<category>/test_*.yaml``). That tolerance is deliberate: the old
``len(rel.parts) != 3`` checks silently produced empty registries whenever
the tree gained a level.

``SCENARIO_MAP`` survives only as a fallback for scenario IDs that are not on
disk yet (``scripts/add_scenario.py`` materializing a brand-new spec).
Anything already committed resolves from the tree itself, so new scenarios
need no registry edit.
"""

from __future__ import annotations

from pathlib import Path, PurePath
from typing import Iterable, Sequence

#: Top-level thread-class directories under ``datasets/conversations/``.
SINGLE_THREAD_DIR = "single_thread"
MULTI_THREAD_DIR = "multi_thread"
THREAD_DIRS: tuple[str, ...] = (SINGLE_THREAD_DIR, MULTI_THREAD_DIR)

_FILENAME_PREFIX = "test_"
_FILENAME_SUFFIX = ".yaml"

#: Fallback registry for scenario IDs that do not exist in the tree yet.
#: Committed scenarios resolve from their own path — keep this list for
#: brand-new specs only.
SCENARIO_MAP: dict[str, tuple[str, str]] = {
    "belief_revision_01": ("logical_consistency", "belief_revision"),
    "explicit_forget_01": ("logical_consistency", "explicit_forget"),
    "in_context_control_01": ("logical_consistency", "in_context_control"),
    "negative_hallucination_01": ("logical_consistency", "negative_hallucination"),
    "attrition_01": ("retention_persistence", "attrition"),
    "needle_haystack_01": ("retention_persistence", "needle_haystack"),
    "in_context_needle_01": ("retention_persistence", "in_context_needle"),
    "temporal_multi_hop_01": ("compositional_reasoning", "temporal_multi_hop"),
    "in_context_multihop_01": ("compositional_reasoning", "in_context_multihop"),
    "cross_thread_01": ("retention_persistence", "cross_thread"),
    "3hop_temporal_01": ("compositional_reasoning", "temporal_multi_hop"),
    "cross_thread_derived_01": ("compositional_reasoning", "cross_thread_derived"),
    "in_context_derivation_01": ("compositional_reasoning", "in_context_derivation"),
    "derived_belief_revision_01": ("logical_consistency", "belief_revision"),
    "backward_modus_tollens_01": ("logical_consistency", "belief_revision"),
    "in_context_derived_revision_01": ("logical_consistency", "in_context_control"),
    "in_context_modus_tollens_01": ("logical_consistency", "in_context_control"),
    "overhead_baseline_01": ("computational_cost", "overhead_baseline"),
    "memory_injection_01": ("robustness_security", "memory_injection"),
    "top_k_saturation_01": ("retention_persistence", "top_k_saturation"),
    "entity_binding_01": ("logical_consistency", "entity_binding"),
}


def scenario_id_from_filename(filename: str) -> str | None:
    """Return the scenario id encoded in ``test_<id>.yaml``.

    Args:
        filename: Bare file name (no directories).

    Returns:
        The scenario id, or ``None`` when the name does not follow the
        ``test_*.yaml`` convention (READMEs, corpus manifests, ...).
    """
    if not (
        filename.startswith(_FILENAME_PREFIX)
        and filename.endswith(_FILENAME_SUFFIX)
    ):
        return None
    return filename[len(_FILENAME_PREFIX) : -len(_FILENAME_SUFFIX)]


def thread_dir_for_sessions(session_ids: Iterable[str]) -> str:
    """Classify a scenario by how many distinct sessions its turns span.

    Args:
        session_ids: One entry per turn (repeats are expected).

    Returns:
        ``"multi_thread"`` when two or more distinct sessions appear, else
        ``"single_thread"``. Turns without an explicit ``session_id`` default
        to ``"default"``, so a scenario that never sets the key is
        single-thread.
    """
    return MULTI_THREAD_DIR if len(set(session_ids)) > 1 else SINGLE_THREAD_DIR


def thread_dir_for_scenario(scenario: object) -> str:
    """Classify a ``BenchmarkScenario`` (or any object exposing ``.turns``)."""
    turns = getattr(scenario, "turns", ())
    return thread_dir_for_sessions(
        getattr(turn, "session_id", "default") for turn in turns
    )


def thread_dir_from_parts(rel_parts: Sequence[str]) -> str | None:
    """Return the thread-class directory in a dataset-relative path, if any."""
    if rel_parts and rel_parts[0] in THREAD_DIRS:
        return rel_parts[0]
    return None


def theme_category_from_parts(rel_parts: Sequence[str]) -> tuple[str, str]:
    """Return ``(theme, category)`` from the last two directories of a path.

    Args:
        rel_parts: Path components relative to ``datasets/conversations/``,
            including the file name (e.g.
            ``("multi_thread", "logical_consistency", "belief_revision",
            "test_belief_revision_01.yaml")``).

    Returns:
        ``(theme, category)``.

    Raises:
        ValueError: If the path has fewer than two directory components, i.e.
            it is not organized under ``<theme>/<category>/``.
    """
    dirs = list(rel_parts[:-1])
    if len(dirs) < 2:
        raise ValueError(
            f"Path {tuple(rel_parts)!r} is not organized as "
            "<theme>/<category>/test_<id>.yaml"
        )
    return dirs[-2], dirs[-1]


def scenario_relpath(
    theme: str,
    category: str,
    scenario_id: str,
    thread_dir: str | None = None,
) -> PurePath:
    """Build the canonical dataset-relative path for a scenario file.

    Args:
        theme: Research theme directory (e.g. ``logical_consistency``).
        category: Test category directory (e.g. ``belief_revision``).
        scenario_id: Scenario id; the file name is ``test_<id>.yaml``.
        thread_dir: ``"single_thread"``/``"multi_thread"`` prefix, or ``None``
            to omit the thread layer (legacy 3-part layout).

    Raises:
        ValueError: If ``thread_dir`` is not one of :data:`THREAD_DIRS`.
    """
    filename = f"{_FILENAME_PREFIX}{scenario_id}{_FILENAME_SUFFIX}"
    if thread_dir is None:
        return PurePath(theme, category, filename)
    if thread_dir not in THREAD_DIRS:
        raise ValueError(
            f"Unknown thread directory {thread_dir!r}; expected one of "
            f"{THREAD_DIRS}"
        )
    return PurePath(thread_dir, theme, category, filename)


def dataset_root() -> Path:
    """Return the scenario root (``<repo>/datasets/conversations``)."""
    return Path(__file__).resolve().parent.parent / "datasets" / "conversations"

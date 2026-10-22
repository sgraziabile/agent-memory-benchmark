# agents — Benchmark agent implementations (Level 0 through Level 4)
#
# AGENT_REGISTRY maps level identifiers (as used by the CLI and exports, e.g.
# ``run.py --level 1`` → results.json ``agent_level: level_1_working_memory``)
# to their agent classes so new levels can be resolved dynamically without
# hard-coded if/else chains. Registry keys MUST match each agent's ``level``
# property.
#
# ``resolve_level_ids()`` translates the CLI shorthand into registry keys so
# users never have to remember full names: ``--level 0`` ≡
# ``--level level_0_reactive``, and ``--level 0,1`` runs both in one invocation.

import re

from agents.base import BaseBenchmarkAgent
from agents.level_0_reactive.graph import Level0ReactiveAgent
from agents.level_1_working_memory.graph import Level1WorkingMemoryAgent
from agents.level_2_episodic_rag.graph import Level2EpisodicRAGAgent

__all__ = [
    "AGENT_REGISTRY",
    "BaseBenchmarkAgent",
    "Level0ReactiveAgent",
    "Level1WorkingMemoryAgent",
    "Level2EpisodicRAGAgent",
    "resolve_level_ids",
]

#: Dynamic agent lookup used by the CLI and runner.
AGENT_REGISTRY: dict[str, type[BaseBenchmarkAgent]] = {
    "level_0_reactive": Level0ReactiveAgent,
    "level_1_working_memory": Level1WorkingMemoryAgent,
    "level_2_episodic_rag": Level2EpisodicRAGAgent,
}


def _available_levels() -> str:
    """One-liner listing registered levels, e.g. ``0 (level_0_reactive), 1 (...)``."""
    entries = []
    for key in sorted(AGENT_REGISTRY):
        match = re.match(r"^level_(\d+)_", key)
        short = match.group(1) if match else key
        entries.append(f"{short} ({key})")
    return ", ".join(entries)


def resolve_level_ids(level_arg: str) -> list[str]:
    """Resolve a CLI ``--level`` argument into AGENT_REGISTRY identifiers.

    Accepted forms (comma-separated combinations allowed):
      - ``all``            → every registered level, sorted
      - bare number        → ``0`` / ``1`` → the unique ``level_<n>_*`` key
      - registry prefix    → ``level_0`` → ``level_0_reactive``
      - full registry key  → ``level_0_reactive``

    Order is preserved and duplicates are dropped, so ``--level 1,0`` runs
    Level 1 first. Numbers with no registered implementation yet (e.g. ``2``
    until Level 2 exists) raise ``ValueError`` listing what is available.

    Raises:
        ValueError: on empty input, unknown tokens, unimplemented numbers,
            or ambiguous prefixes.
    """
    tokens = [t.strip() for t in level_arg.split(",") if t.strip()]
    if not tokens:
        raise ValueError(
            "No agent level specified — use a number (e.g. --level 0), a "
            "comma-separated list (e.g. --level 0,1), or 'all'."
        )

    if len(tokens) == 1 and tokens[0].casefold() == "all":
        return sorted(AGENT_REGISTRY)

    resolved: list[str] = []
    for token in tokens:
        if token in AGENT_REGISTRY:
            candidates = [token]
        else:
            prefix = f"level_{token}" if token.isdigit() else token
            pattern = re.compile(rf"^{re.escape(prefix)}(?:_|$)")
            candidates = [key for key in AGENT_REGISTRY if pattern.match(key)]

        if not candidates:
            raise ValueError(
                f"Level '{token}' is not implemented or unknown. "
                f"Available levels: {_available_levels()}"
            )
        if len(candidates) > 1:
            raise ValueError(
                f"Level '{token}' is ambiguous — it matches "
                f"{', '.join(sorted(candidates))}."
            )
        if candidates[0] not in resolved:
            resolved.append(candidates[0])
    return resolved
# agents — Benchmark agent implementations (Level 0 through Level 4)
#
# AGENT_REGISTRY maps level identifiers (as used by the CLI, e.g.
# ``run.py --level level_1_working_memory``) to their agent classes so new
# levels can be resolved dynamically without hard-coded if/else chains.
# Registry keys MUST match each agent's ``level`` property.

from agents.base import BaseBenchmarkAgent
from agents.level_0_reactive.graph import Level0ReactiveAgent
from agents.level_1_working_memory.graph import Level1WorkingMemoryAgent

__all__ = [
    "AGENT_REGISTRY",
    "BaseBenchmarkAgent",
    "Level0ReactiveAgent",
    "Level1WorkingMemoryAgent",
]

#: Dynamic agent lookup used by the CLI and runner.
AGENT_REGISTRY: dict[str, type[BaseBenchmarkAgent]] = {
    "level_0_reactive": Level0ReactiveAgent,
    "level_1_working_memory": Level1WorkingMemoryAgent,
}
"""agents.base — Abstract interface for all benchmark agent levels.

Each ablation level (L0 through L4) must implement this interface so
the :class:`~core.runner.BenchmarkRunner` can treat them polymorphically.

The critical contract:
    The compiled graph must accept
    ``config={"configurable": {"model_id": ..., "system_prompt": ...}}``
    at invocation time.  **No model or prompt may be bound at build time.**
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseBenchmarkAgent(ABC):
    """Abstract base class for benchmark agent implementations.

    Subclasses must implement :meth:`build_graph` and :attr:`level`.
    """

    #: Whether this level consumes an ``EpisodicMemoryBackend`` (L2+). The
    #: CLI constructs the backend selected via ``--backend`` — and therefore
    #: needs an embedding API key — ONLY for levels that require one; every
    #: other level stays fully backward compatible (no backend, no keys).
    requires_memory_backend: bool = False

    #: Whether this level drives a bounded retrieval **tool loop** (L2b). The
    #: CLI forwards ``--max-tool-calls`` to the runner only for levels that set
    #: this, so L0/L1/L2a keep the exact pre-L2b ``config["configurable"]``
    #: (adding a key they ignore would change their recorded run identity for
    #: no scientific gain). Mirrors :attr:`requires_memory_backend`.
    supports_tool_calls: bool = False

    @abstractmethod
    def build_graph(self, tools: list[Any] | None = None) -> Any:
        """Build and compile the LangGraph ``StateGraph``.

        The compiled graph must:
        - Accept ``model_id`` and ``system_prompt`` via ``config["configurable"]``
        - NOT have any model or prompt hardcoded at definition time
        - Return a ``CompiledStateGraph`` ready for ``.invoke()``

        Args:
            tools: Optional list of tool callables to bind to the agent.
                   Tools are bound dynamically in the model node, not at
                   graph definition time.

        Returns:
            A compiled LangGraph state graph.
        """
        ...

    @property
    @abstractmethod
    def level(self) -> str:
        """Return the agent level identifier.

        Examples: ``'level_0_reactive'``, ``'level_1_thread'``, etc.
        """
        ...

    def __repr__(self) -> str:
        return f"<{type(self).__name__} level={self.level!r}>"

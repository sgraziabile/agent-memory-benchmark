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

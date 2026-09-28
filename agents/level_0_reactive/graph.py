"""agents.level_0_reactive.graph — Level 0 Stateless Reactive Agent.

Implements the **Level 0 baseline** using LangGraph's ``StateGraph``.
This agent is completely **stateless/amnesic** between invocations:

- **NO** checkpointer (``MemorySaver``)
- **NO** state store (``BaseStore``)
- **NO** persistent memory of any kind

The model and system prompt are injected dynamically at runtime via
``config["configurable"]``, following the Zero Hardcoding principle
from SPEC.md §2.

Graph topology::

    START ──► model_node ──► tools (if tool_calls) ──► model_node
                   │
                   └──► END (if no tool_calls)
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agents.base import BaseBenchmarkAgent
from core.model_factory import create_model
from core.schemas import AgentState

logger = logging.getLogger(__name__)


# ── Graph Nodes ──────────────────────────────────────────────────────────────


def model_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """LLM invocation node — dynamically resolves model and prompt from config.

    Reads ``model_id`` and ``system_prompt`` from ``config["configurable"]``,
    instantiates the model via the factory, prepends an ephemeral
    ``SystemMessage``, and invokes the model.

    This function is intentionally **stateless**: every call re-creates the
    model from config.  This is the Level 0 design — no caching, no memory.

    Args:
        state: Current agent state containing the message history.
        config: LangGraph ``RunnableConfig`` with ``configurable`` dict.

    Returns:
        Dict with ``messages`` key containing the model's response.
    """
    configurable = config.get("configurable", {})

    # ── Extract dynamic parameters (Zero Hardcoding) ─────────────────
    model_id: str = configurable["model_id"]
    system_prompt: str = configurable.get(
        "system_prompt", "You are a helpful assistant."
    )
    tools: list[Any] = configurable.get("tools", [])
    model_kwargs: dict[str, Any] = configurable.get("model_kwargs", {})

    # ── Create model instance dynamically ────────────────────────────
    model = create_model(model_id, **model_kwargs)

    # ── Bind tools if provided ───────────────────────────────────────
    if tools:
        model = model.bind_tools(tools)

    # ── Prepend ephemeral system message ─────────────────────────────
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])

    # ── Invoke ───────────────────────────────────────────────────────
    logger.debug(
        "Invoking model=%s with %d messages (system + %d history)",
        model_id,
        len(messages),
        len(state["messages"]),
    )
    response = model.invoke(messages)

    return {"messages": [response]}


# ── Agent Class ──────────────────────────────────────────────────────────────


class Level0ReactiveAgent(BaseBenchmarkAgent):
    """Level 0: Stateless reactive agent — completely amnesic.

    Each invocation of the graph creates a fresh model instance from
    ``config["configurable"]``.  There is no checkpointer, no store,
    and no persistent memory of any kind.

    This serves as the baseline against which higher levels (L1–L4)
    with progressively richer memory architectures are compared.
    """

    @property
    def level(self) -> str:
        return "level_0_reactive"

    def build_graph(self, tools: list[Any] | None = None) -> Any:
        """Build and compile the Level 0 stateless agent graph.

        Args:
            tools: Optional list of tool callables.  If provided, a
                   ``ToolNode`` is added and conditional routing is
                   configured.  Tools are also passed via
                   ``config["configurable"]["tools"]`` at invocation time
                   so the model node can bind them.

        Returns:
            A compiled ``StateGraph`` with NO checkpointer or store.
        """
        builder = StateGraph(AgentState)

        # ── Add nodes ────────────────────────────────────────────────
        builder.add_node("model", model_node)

        if tools:
            tool_node = ToolNode(tools)
            builder.add_node("tools", tool_node)

            # Conditional routing: tool_calls → tools node, else → END
            builder.add_conditional_edges("model", tools_condition)
            builder.add_edge("tools", "model")
        else:
            # No tools — always go to END after model response
            builder.add_edge("model", END)

        # ── Entry point ──────────────────────────────────────────────
        builder.add_edge(START, "model")

        # ── Compile WITHOUT checkpointer or store (STRICT L0) ────────
        compiled = builder.compile()

        logger.info("Level 0 reactive graph compiled (stateless, no checkpointer)")
        return compiled

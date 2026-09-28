"""agents.level_0_reactive.graph — Level 0 Stateless Reactive Agent.

Implements the **Level 0 baseline** using LangGraph's ``StateGraph``.
This agent is completely **stateless/amnesic** between invocations:

- **NO** checkpointer (``MemorySaver``)
- **NO** state store (``BaseStore``)
- **NO** persistent memory of any kind

The model and system prompt are injected dynamically at runtime via
``config["configurable"]``, following the Zero Hardcoding principle
from SPEC.md §2.

Routing is handled via modern LangGraph ``Command`` objects inside each
node rather than manual conditional edge declarations.

Graph topology::

    START ──► model_node ──(Command goto="tools")──► tools_node
                   │                                       │
                   │                               (Command goto="model")
                   │                                       │
                   └──(Command goto=END)──► END ◄──────────┘
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from agents.base import BaseBenchmarkAgent
from core.model_factory import create_model
from core.schemas import AgentState

logger = logging.getLogger(__name__)


# ── Graph Nodes ──────────────────────────────────────────────────────────────


def model_node(
    state: AgentState,
    config: RunnableConfig,
    default_tools: list[Any] | None = None,
) -> Command:
    """LLM invocation node — dynamically resolves model and prompt from config.

    Reads ``model_id`` and ``system_prompt`` from ``config["configurable"]``,
    instantiates the model via the factory, prepends an ephemeral
    ``SystemMessage``, and invokes the model.

    Conditional routing is defined **inside** this node via ``Command``:
    - If the model emits ``tool_calls`` and tools are configured, returns
      ``Command(update=..., goto="tools")``.
    - Otherwise, returns ``Command(update=..., goto=END)``.

    Args:
        state: Current agent state containing the message history.
        config: LangGraph ``RunnableConfig`` with ``configurable`` dict.
        default_tools: Optional tools passed at graph construction time.

    Returns:
        A ``Command`` specifying state updates and the dynamic destination.
    """
    configurable = config.get("configurable", {})

    # ── Extract dynamic parameters (Zero Hardcoding) ─────────────────
    model_id: str = configurable["model_id"]
    system_prompt: str = configurable.get(
        "system_prompt", "You are a helpful assistant."
    )
    tools: list[Any] = configurable.get("tools") or default_tools or []
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

    # ── Dynamic edge routing via Command ─────────────────────────────
    has_tool_calls = bool(getattr(response, "tool_calls", None))
    goto = "tools" if (tools and has_tool_calls) else END

    return Command(
        update={"messages": [response]},
        goto=goto,
    )


def make_tools_node(tools: list[Any]):
    """Create a tool execution node that returns a Command back to the model."""
    tool_executor = ToolNode(tools)

    def tools_node(state: AgentState, config: RunnableConfig) -> Command:
        result = tool_executor.invoke(state, config=config)
        return Command(
            update=result,
            goto="model",
        )

    return tools_node


# ── Agent Class ──────────────────────────────────────────────────────────────


class Level0ReactiveAgent(BaseBenchmarkAgent):
    """Level 0: Stateless reactive agent — completely amnesic.

    Each invocation of the graph creates a fresh model instance from
    ``config["configurable"]``.  There is no checkpointer, no store,
    and no persistent memory of any kind.

    Transitions are encapsulated within node-returned ``Command`` objects.
    """

    @property
    def level(self) -> str:
        return "level_0_reactive"

    def build_graph(self, tools: list[Any] | None = None) -> Any:
        """Build and compile the Level 0 stateless agent graph.

        Uses ``Command`` inside nodes for all conditional edge transitions
        rather than external ``add_conditional_edges()``.

        Args:
            tools: Optional list of tool callables. If provided, a
                   ``tools`` node is registered and transitions are
                   handled via ``Command(goto=...)``.

        Returns:
            A compiled ``StateGraph`` with NO checkpointer or store.
        """
        builder = StateGraph(AgentState)

        # ── Add model node ───────────────────────────────────────────
        def _model_step(state: AgentState, config: RunnableConfig) -> Command:
            return model_node(state, config, default_tools=tools)

        builder.add_node("model", _model_step)

        # ── Add tools node if tools are supplied ─────────────────────
        if tools:
            builder.add_node("tools", make_tools_node(tools))

        # ── Entry point ──────────────────────────────────────────────
        builder.add_edge(START, "model")

        # ── Compile WITHOUT checkpointer or store (STRICT L0) ────────
        compiled = builder.compile()

        logger.info("Level 0 reactive graph compiled with Command routing (stateless, no checkpointer)")
        return compiled

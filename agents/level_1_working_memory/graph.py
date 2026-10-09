"""agents.level_1_working_memory.graph — Level 1 Working-Memory Agent.

Implements **Level 1** using LangGraph's ``StateGraph`` with an in-process
``MemorySaver`` checkpointer. This agent has short-term, **intra-thread**
working memory:

- **YES** checkpointer (``MemorySaver``) — conversation state persists across
  ``invoke()`` calls as long as the same ``thread_id`` is supplied in
  ``config["configurable"]``.
- **NO** long-term store (``BaseStore``) — memory does not survive across
  threads, processes, or benchmark runs.

The model and system prompt are injected dynamically at runtime via
``config["configurable"]``, following the Zero Hardcoding principle from
SPEC.md §2.

Routing is handled via modern LangGraph ``Command`` objects inside each
node rather than manual conditional edge declarations (no
``add_conditional_edges()``).

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
from langgraph.checkpoint.memory import MemorySaver
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
    ``SystemMessage``, and invokes the model with the full message history
    restored from this thread's checkpoint (working memory).

    Conditional routing is defined **inside** this node via ``Command``:
    - If the model emits ``tool_calls`` and tools are configured, returns
      ``Command(update=..., goto="tools")``.
    - Otherwise, returns ``Command(update=..., goto=END)``.

    Args:
        state: Current agent state containing the message history (restored
               from the ``MemorySaver`` checkpoint for this thread).
        config: LangGraph ``RunnableConfig`` with ``configurable`` dict
                (must include ``thread_id`` for checkpoint continuity).
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
    # The SystemMessage is NOT persisted in the checkpoint; it is re-injected
    # on every invocation so prompts remain swappable per run.
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


class Level1WorkingMemoryAgent(BaseBenchmarkAgent):
    """Level 1: Working-memory agent — short-term, intra-thread persistence.

    Conversation history persists across ``invoke()`` calls within the same
    thread (identified by ``config["configurable"]["thread_id"]``) via a
    LangGraph ``MemorySaver`` checkpointer. Memory does **not** survive across
    threads, processes, or benchmark runs — cross-thread long-term memory is
    reserved for higher levels.

    The model and system prompt are still injected dynamically at runtime;
    only the *conversation state* is persisted by the checkpointer.

    Transitions are encapsulated within node-returned ``Command`` objects
    (no ``add_conditional_edges()``).
    """

    @property
    def level(self) -> str:
        return "level_1_working_memory"

    def build_graph(self, tools: list[Any] | None = None) -> Any:
        """Build and compile the Level 1 working-memory agent graph.

        Uses ``Command`` inside nodes for all conditional edge transitions
        rather than external ``add_conditional_edges()``.

        Args:
            tools: Optional list of tool callables. If provided, a
                   ``tools`` node is registered and transitions are
                   handled via ``Command(goto=...)``.

        Returns:
            A compiled ``StateGraph`` with an in-process ``MemorySaver``
            checkpointer providing thread-scoped short-term memory.
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

        # ── Compile WITH MemorySaver checkpointer (L1 working memory) ─
        memory = MemorySaver()
        compiled = builder.compile(checkpointer=memory)

        logger.info(
            "Level 1 working-memory graph compiled with Command routing "
            "(MemorySaver checkpointer)"
        )
        return compiled
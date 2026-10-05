"""core.schemas — Pydantic v2 data contracts and LangGraph state definitions.

Defines all structured types used across the benchmark harness:
- AgentState (TypedDict for LangGraph)
- AssertionRule, ConversationTurn, BenchmarkScenario (test case schemas)
- TurnResult, BenchmarkRunSummary (execution result schemas)
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ── LangGraph State ──────────────────────────────────────────────────────────


class AgentState(TypedDict):
    """LangGraph agent state with message accumulation.

    The ``add_messages`` reducer appends new messages to the existing
    sequence rather than overwriting, enabling multi-turn conversations
    within a single graph invocation.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]


# ── Test Scenario Schemas ────────────────────────────────────────────────────


class AssertionRule(BaseModel):
    """A single assertion to evaluate against an agent's response text.

    Attributes:
        type: The assertion operator — ``must_contain`` checks for presence,
              ``must_not_contain`` checks for absence.
        value: The substring or keyword to search for in the response.
        case_sensitive: Whether the match should be case-sensitive.
    """

    type: Literal["must_contain", "must_not_contain"]
    value: str
    case_sensitive: bool = False


class ConversationTurn(BaseModel):
    """One turn in a benchmark conversation scenario.

    Attributes:
        turn_number: Sequential index of this turn (1-based).
        role: Who produces this turn — ``user`` for user messages,
              ``system_event`` for injected environmental changes.
        content: The text content of the turn.
        assertions: Rules to evaluate against the agent's response to this turn.
    """

    turn_number: int
    role: Literal["user", "system_event"]
    content: str
    assertions: list[AssertionRule] = Field(default_factory=list)


class BenchmarkScenario(BaseModel):
    """A complete benchmark test case loaded from a YAML file.

    Attributes:
        id: Unique identifier for this scenario.
        name: Human-readable scenario name.
        description: Detailed description of what this scenario tests.
        category: Test category (e.g., ``belief_revision``, ``attrition``,
                  ``needle_haystack``).
        turns: Ordered list of conversation turns.
        metadata: Optional extra metadata (difficulty, research question, etc.).
    """

    id: str
    name: str
    description: str
    category: str
    turns: list[ConversationTurn]
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Execution Result Schemas ─────────────────────────────────────────────────


class TurnResult(BaseModel):
    """Result of evaluating a single conversation turn.

    Captures the agent's response, assertion outcomes, performance metrics,
    and token usage for one turn within a benchmark run.

    Attributes:
        turn_number: The turn index this result corresponds to.
        user_input: The user message that was sent.
        agent_response: The agent's textual response (empty for errored turns).
        assertions_passed: Count of assertions that passed.
        assertions_failed: Count of assertions that failed.
        assertion_details: Per-assertion breakdown with rule, expected, actual, passed.
        latency_ms: Wall-clock time for the agent to respond (milliseconds).
        token_usage: Token counts extracted from model response metadata.
        status: ``"ok"`` if the model responded; ``"error"`` if the turn
                failed at the infrastructure level (API outage, auth error).
        error: Exception description for errored turns; ``None`` otherwise.
                Errored turns are excluded from pass-rate and token metrics.
    """

    turn_number: int
    user_input: str
    agent_response: str
    assertions_passed: int = 0
    assertions_failed: int = 0
    assertion_details: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: float = 0.0
    token_usage: dict[str, Any] = Field(default_factory=dict)
    status: Literal["ok", "error"] = "ok"
    error: str | None = None


class BenchmarkRunSummary(BaseModel):
    """Summary of a complete benchmark run — one execution tuple.

    Represents the results of evaluating one
    ⟨Scenario, System Prompt, Model, Persistence Level⟩ combination.

    Attributes:
        run_id: Unique identifier for this run (UUID or timestamp-based).
        scenario_id: The benchmark scenario that was executed.
        model_id: The model identifier used.
        prompt_id: The system prompt identifier used.
        agent_level: The agent persistence level (e.g., ``level_0_reactive``).
        total_turns: Number of conversation turns executed.
        total_assertions: Total assertions evaluated across all turns.
        passed_assertions: Count of passed assertions.
        failed_assertions: Count of failed assertions.
        pass_rate: Fraction of assertions passed (0.0–1.0), computed over
                   successfully completed turns only.
        error_turns: Number of turns that failed at the infrastructure level
                     and were excluded from all metrics.
        error_rate: Fraction of turns that errored (0.0–1.0).
        total_latency_ms: Cumulative latency across successfully completed turns.
        avg_latency_per_turn_ms: Average latency per successfully completed turn.
        turn_results: Detailed per-turn results.
        timestamp: ISO 8601 timestamp of when the run started.
    """

    run_id: str
    scenario_id: str
    model_id: str
    prompt_id: str
    agent_level: str
    total_turns: int = 0
    total_assertions: int = 0
    passed_assertions: int = 0
    failed_assertions: int = 0
    pass_rate: float = 0.0
    error_turns: int = 0
    error_rate: float = 0.0
    total_latency_ms: float = 0.0
    avg_latency_per_turn_ms: float = 0.0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    turn_results: list[TurnResult] = Field(default_factory=list)
    timestamp: str = ""

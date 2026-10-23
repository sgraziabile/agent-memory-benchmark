"""core.runner — Benchmark matrix orchestrator.

Computes the Cartesian product (Scenarios × Prompts × Models), executes
multi-turn conversations against a compiled LangGraph agent, evaluates
assertion rules, and exports structured logs in JSONL and CSV formats.

Example usage::

    from agents.level_0_reactive.graph import Level0ReactiveAgent
    from core.runner import BenchmarkRunner

    agent = Level0ReactiveAgent()
    graph = agent.build_graph()

    runner = BenchmarkRunner()
    results = runner.run_matrix(graph, agent_level=agent.level)
"""

from __future__ import annotations

import csv
import itertools
import json
import logging
import time
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Unicode characters that LLMs frequently emit in place of their ASCII
# equivalents. Left unhandled, these produce false assertion failures
# (e.g. "DELTA‑7749" with a non-breaking hyphen U+2011 failing a
# ``must_contain: "DELTA-7749"`` check). Applied to BOTH the response and
# the expected value, so matching semantics stay symmetric.
_CHAR_FOLDS = str.maketrans(
    {
        # hyphen-lookalikes -> ASCII hyphen
        "\u2010": "-",  # hyphen
        "\u2011": "-",  # non-breaking hyphen
        "\u2012": "-",  # figure dash
        "\u2013": "-",  # en dash
        "\u2212": "-",  # minus sign
        "\uff0d": "-",  # fullwidth hyphen-minus
        # space-lookalikes -> ASCII space
        "\u00a0": " ",  # no-break space (also handled by NFKC; kept explicit)
        "\u2007": " ",  # figure space
        "\u2009": " ",  # thin space
        "\u200a": " ",  # hair space
        "\u202f": " ",  # narrow no-break space
        "\u3000": " ",  # ideographic space (also handled by NFKC)
        # quote-lookalikes -> ASCII quotes
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
)


def _normalize_for_matching(text: str) -> str:
    """Normalize text so substring assertions survive cosmetic Unicode variance.

    Applies NFKC normalization (e.g. fullwidth "ＤＥＬＴＡ" -> "DELTA",
    superscripts, ligatures) and folds typographic dash/quote/space variants
    to their ASCII equivalents. Case is PRESERVED — callers apply
    ``str.casefold()`` themselves when matching case-insensitively
    (``casefold`` over ``lower`` for correct German eszett handling).

    Args:
        text: Raw text (agent response or expected assertion value).

    Returns:
        Normalized text safe for symmetric substring comparison.
    """
    return unicodedata.normalize("NFKC", text).translate(_CHAR_FOLDS)

import yaml
from langchain_core.messages import AIMessage, HumanMessage

from core.schemas import (
    AssertionRule,
    BenchmarkRunSummary,
    BenchmarkScenario,
    ConversationTurn,
    TurnResult,
)

logger = logging.getLogger(__name__)


def _slice_token_usage(token_usage: dict[str, Any]) -> dict[str, int]:
    """Slice raw token usage metadata into plan §4.2 accounting buckets.

    Providers expose (optionally, hence all the ``or`` guards):
    - ``input_tokens`` / ``output_tokens`` / ``total_tokens``
    - ``input_token_details.cache_read``  → prefix-cache hit tokens
    - ``output_token_details.reasoning`` → hidden chain-of-thought tokens

    ``visible_output_tokens`` is derived as the billed output minus hidden
    reasoning, floored at zero. Missing fields default to 0 so L0/L1 and
    providers without detail blocks keep working unchanged.
    """
    input_details = token_usage.get("input_token_details") or {}
    output_details = token_usage.get("output_token_details") or {}
    prompt_tokens = int(
        token_usage.get("input_tokens") or token_usage.get("prompt_tokens") or 0
    )
    completion_tokens = int(
        token_usage.get("output_tokens") or token_usage.get("completion_tokens") or 0
    )
    reasoning_tokens = int(output_details.get("reasoning") or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": int(
            token_usage.get("total_tokens") or (prompt_tokens + completion_tokens)
        ),
        "reasoning_tokens": reasoning_tokens,
        "visible_output_tokens": max(0, completion_tokens - reasoning_tokens),
        "cached_prompt_tokens": int(input_details.get("cache_read") or 0),
    }


def _read_l2_telemetry(result: Any) -> dict[str, Any]:
    """Read plan §4.2 L2 telemetry channels from a graph invoke result.

    Guarded read: L0/L1 graphs never write these channels, so the same
    runner code serves every ablation level — absent channels simply
    default to zeros (and non-dict results, e.g. from custom graphs, to a
    fully-zeroed telemetry block).
    """
    if not isinstance(result, dict):
        result = {}
    return {
        "retrieved_context_chars": int(result.get("retrieved_context_chars") or 0),
        "retrieved_context_tokens": int(result.get("retrieved_context_tokens") or 0),
        "retrieval_latency_ms": float(result.get("retrieval_latency_ms") or 0.0),
        "append_latency_ms": float(result.get("append_latency_ms") or 0.0),
        "embedding_latency_ms": float(result.get("embedding_latency_ms") or 0.0),
        "embedding_tokens": int(result.get("embedding_tokens") or 0),
        "retrieved_memories": list(result.get("retrieved_memories") or []),
        "appended_chunk_ids": list(result.get("appended_chunk_ids") or []),
    }


# ── Default paths ────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_DIR = _PROJECT_ROOT / "configs"
_DEFAULT_DATASET_DIR = _PROJECT_ROOT / "datasets" / "conversations"
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "outputs" / "runs"

class BenchmarkRunner:
    """Orchestrates benchmark execution across the full experimental matrix.

    Loads scenarios, prompts, and model configurations, then executes every
    combination as a multi-turn conversation against the provided LangGraph
    agent graph.
    """

    @staticmethod
    def _extract_text_content(content: Any) -> str:
        """Extract clean text from message content (handles str, list of dicts/blocks, etc.)."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict):
                    if block.get("type") == "text" and "text" in block:
                        text_parts.append(str(block["text"]))
                    elif "text" in block:
                        text_parts.append(str(block["text"]))
                    elif "content" in block and isinstance(block["content"], str):
                        text_parts.append(block["content"])
                elif hasattr(block, "text"):
                    text_parts.append(str(getattr(block, "text")))
            return "\n".join(text_parts).strip() if text_parts else str(content)
        return str(content)

    def __init__(
        self,
        config_dir: Path | str = _DEFAULT_CONFIG_DIR,
        dataset_dir: Path | str = _DEFAULT_DATASET_DIR,
        output_dir: Path | str = _DEFAULT_OUTPUT_DIR,
    ) -> None:
        self.config_dir = Path(config_dir)
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        #: Shared output directory for the current invocation, pre-created
        #: when a memory backend is attached so ``memory/<run_id>/`` snapshot
        #: exports land next to the final results export (plan §4.3).
        #: Consumed by ``_export_results``; ``None`` = create a fresh
        #: timestamped directory at export time.
        self._current_run_dir: Path | None = None

    def ensure_shared_run_dir(self) -> Path:
        """Create (once per invocation) the shared output directory.

        Eagerly created when a memory backend is attached so per-run
        ``memory/<run_id>/`` snapshot exports land next to the final
        results export (plan §4.3) instead of in a directory created
        moments later with a different timestamp. Idempotent: the first
        caller wins and later levels of the same CLI invocation (e.g.
        ``--level 2,0``) reuse the same directory; ``_export_results``
        consumes (clears) it at export time.

        Returns:
            The shared run directory for this invocation.
        """
        if self._current_run_dir is None:
            self._current_run_dir = self.output_dir / datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            self._current_run_dir.mkdir(parents=True, exist_ok=True)
        return self._current_run_dir

    # ── Loaders ──────────────────────────────────────────────────────────

    def load_scenarios(
        self, filter_ids: list[str] | None = None
    ) -> list[BenchmarkScenario]:
        """Load benchmark scenarios from YAML files in the dataset directory.

        Scenario files are organized in a nested theme structure
        (``datasets/conversations/<theme>/<test_category>/test_*.yaml``),
        so discovery is recursive. Files may also live flat at the top level.

        Args:
            filter_ids: Optional list of scenario IDs to include. If ``None``,
                        all scenarios are loaded.

        Returns:
            List of validated ``BenchmarkScenario`` instances.
        """
        scenarios: list[BenchmarkScenario] = []

        for yaml_path in sorted(self.dataset_dir.rglob("*.yaml")):
            with open(yaml_path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)

            try:
                scenario = BenchmarkScenario(**raw)
            except Exception as exc:
                logger.warning("Skipping invalid scenario %s: %s", yaml_path.name, exc)
                continue

            if filter_ids is None or scenario.id in filter_ids:
                scenarios.append(scenario)
                logger.info("Loaded scenario: %s (%s)", scenario.id, scenario.name)

        if not scenarios:
            logger.warning("No scenarios loaded from %s", self.dataset_dir)

        return scenarios

    def load_prompts(self) -> dict[str, dict[str, str]]:
        """Load system prompts from ``configs/prompts.yaml``.

        Returns:
            Dictionary mapping prompt IDs to their config dicts, each
            containing at least ``id``, ``description``, and ``content``.
        """
        prompts_path = self.config_dir / "prompts.yaml"
        if not prompts_path.exists():
            raise FileNotFoundError(f"Prompts config not found: {prompts_path}")

        with open(prompts_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        return raw.get("prompts", {})

    def load_model_ids(self) -> list[str]:
        """Load model IDs from ``configs/models.yaml``.

        Returns:
            List of model ID strings registered in the config.
        """
        models_path = self.config_dir / "models.yaml"
        if not models_path.exists():
            raise FileNotFoundError(f"Models config not found: {models_path}")

        with open(models_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        return list(raw.get("models", {}).keys())

    # ── Assertion Evaluation ─────────────────────────────────────────────

    @staticmethod
    def evaluate_assertions(
        response: str, rules: list[AssertionRule]
    ) -> list[dict[str, Any]]:
        """Evaluate a list of assertion rules against an agent response.

        Args:
            response: The agent's response text.
            rules: List of ``AssertionRule`` instances to check.

        Returns:
            List of dicts, each with keys: ``rule_type``, ``value``,
            ``passed``, ``actual_found``.
        """
        results: list[dict[str, Any]] = []

        for rule in rules:
            # Normalize cosmetic Unicode variance (typographic hyphens,
            # spaces, quotes, fullwidth forms) on BOTH sides so e.g. a
            # response containing "DELTA\u20117749" (non-breaking hyphen)
            # still satisfies ``must_contain: "DELTA-7749"``.
            search_in = _normalize_for_matching(response)
            search_for = _normalize_for_matching(rule.value)

            if not rule.case_sensitive:
                search_in = search_in.casefold()
                search_for = search_for.casefold()

            found = search_for in search_in

            if rule.type == "must_contain":
                passed = found
            else:  # must_not_contain
                passed = not found

            results.append(
                {
                    "rule_type": rule.type,
                    "value": rule.value,
                    "passed": passed,
                    "found_in_response": found,
                }
            )

        return results

    # ── Single Run Execution ─────────────────────────────────────────────

    def run_single(
        self,
        scenario: BenchmarkScenario,
        prompt_id: str,
        prompt_content: str,
        model_id: str,
        agent_graph: Any,
        agent_level: str,
        memory_backend: Any = None,
        memory_backend_config: dict[str, Any] | None = None,
        snapshot_dir: Path | None = None,
    ) -> BenchmarkRunSummary:
        """Execute a single benchmark tuple: ⟨Scenario, Prompt, Model, Level⟩.

        Runs the full multi-turn conversation, evaluates assertions per turn,
        and returns a structured summary.

        Args:
            scenario: The benchmark scenario to execute.
            prompt_id: Identifier for the system prompt.
            prompt_content: The actual system prompt text.
            model_id: Model identifier (registry key or provider:name).
            agent_graph: Compiled LangGraph ``CompiledStateGraph``.
            agent_level: Agent persistence level string.
            memory_backend: Optional ``EpisodicMemoryBackend`` (L2+). When
                        attached it is injected into
                        ``config["configurable"]["memory_backend"]`` for the
                        graph to consume, its snapshot is exported under
                        ``snapshot_dir/<run_id>/`` at the end of the run —
                        BEFORE any caller-side ``reset()`` (plan §4.3) — and
                        the summary records backend identity. ``None`` for
                        L0/L1 keeps behavior identical to before.
            memory_backend_config: Optional identity dict for the summary
                        (plan §4.3): ``backend``, ``embedding_model_id``,
                        ``top_k``, ``chunk_size``, ``chunk_overlap``.
            snapshot_dir: Directory to place ``memory/<run_id>/`` snapshots
                        under. ``None`` skips the export (e.g. backend
                        attached but caller manages snapshots itself).

        Returns:
            A ``BenchmarkRunSummary`` with all turn results and metrics.
        """
        run_id = f"{scenario.id}_{model_id}_{prompt_id}_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now(timezone.utc).isoformat()
        turn_results: list[TurnResult] = []

        logger.info(
            "Starting run: scenario=%s, model=%s, prompt=%s",
            scenario.id,
            model_id,
            prompt_id,
        )

        # Build runtime config (Dynamic Injection — Zero Hardcoding).
        # thread_id scopes conversation state for checkpointer-enabled agents
        # (Level 1+). For Level 0 (no checkpointer) it is harmless, keeping
        # the runner code identical across all ablation levels.
        #
        # Per-SESSION thread_id (plan §4.2): multi-session scenarios (e.g.
        # cross-thread probes in s2 after facts learned in s1) MUST give each
        # session its own thread so a checkpointer can't leak history across
        # session boundaries — the very capability L2 is being tested on.
        # The schema-default "default" session keeps legacy single-session
        # scenarios on the exact pre-L2 thread form (backward compatible).

        def _thread_id_for(session: str) -> str:
            if session == "default":
                return f"{run_id}_{scenario.id}"
            return f"{run_id}_{scenario.id}_{session}"

        # The first interactive turn's session seeds the initial configurable
        # (avoids a spurious rotation on turn 1); "default" when every turn
        # is a system_event.
        initial_session = next(
            (t.session_id for t in scenario.turns if t.role != "system_event"),
            "default",
        )

        # Memory-backend identity (plan §4.3 reproducibility): L0/L1 callers
        # attach no backend and keep the "none" default; L2+ callers pass an
        # explicit identity dict (backend, embedding_model_id, top_k, ...).
        backend_cfg = memory_backend_config or {}

        config = {
            "configurable": {
                "model_id": model_id,
                "system_prompt": prompt_content,
                "thread_id": _thread_id_for(initial_session),
                # Session identity injected for L2+ agents (retrieve_node
                # exclude filter / append_node provenance). Harmlessly
                # ignored by L0/L1 graphs.
                "session_id": initial_session,
            }
        }

        # Backend wiring (Milestone 5, plan §4.3): the L2 graph reads the
        # backend instance and the retrieval cut-off from ``configurable``
        # (Zero Hardcoding — nothing is bound at graph build time). Injected
        # ONLY when a backend is attached, so L0/L1 keeps the exact pre-L2
        # configurable and never sees these keys.
        if memory_backend is not None:
            config["configurable"]["memory_backend"] = memory_backend
            configured_top_k = backend_cfg.get("top_k")
            if configured_top_k is not None:
                config["configurable"]["top_k"] = configured_top_k

        for turn in scenario.turns:
            if turn.role == "system_event":
                # System events are injected but not sent to the model
                logger.info("  System event at turn %d: %s", turn.turn_number, turn.content[:80])
                continue

            # Session boundary (plan §4.2): a turn whose session_id differs
            # from the active one rotates the thread_id and updates the
            # configurable — giving each session a fresh checkpointer thread.
            # The Pydantic schema guarantees session_id is always present
            # (defaults to "default" for legacy single-session scenarios).
            session_id = turn.session_id
            if session_id != config["configurable"]["session_id"]:
                config["configurable"]["session_id"] = session_id
                config["configurable"]["thread_id"] = _thread_id_for(session_id)
                logger.info(
                    "  Session boundary at turn %d → session=%s (new thread)",
                    turn.turn_number,
                    session_id,
                )

            # Build user message — send only the current turn delta.
            # History retention across turns is the agent's responsibility
            # (via its checkpointer), not the runner's.
            #
            # The stable message id (plan §4.2) dedups the SAME logical turn
            # re-sent across nodes within one invoke (e.g. L2's
            # retrieve_node → model_node chain re-passing state) AND lets
            # the L2 graph mark its injected SystemMessages with ids like
            # f"{turn_number}_memory" without ever colliding with user ids.
            user_msg = HumanMessage(
                content=turn.content,
                id=f"{turn.turn_number}_user",
            )

            # Invoke the agent graph
            t_start = time.perf_counter()
            # Zeroed until the graph returns — error turns keep zeroed
            # telemetry (the exception path never reaches the assignment).
            l2_tel = _read_l2_telemetry({})
            sliced_usage = _slice_token_usage({})
            try:
                result = agent_graph.invoke(
                    {"messages": [user_msg]},
                    config=config,
                )
                t_end = time.perf_counter()
                latency_ms = (t_end - t_start) * 1000.0

                # Plan §4.2: guarded read of L2 telemetry channels. L0/L1
                # graphs never write them, so the block stays zeroed and the
                # same runner code serves every ablation level.
                l2_tel = _read_l2_telemetry(result)

                # Extract the agent's response (last AI message)
                response_messages = result.get("messages", [])
                ai_response = ""
                token_usage: dict[str, Any] = {}
                finish_reason: str | None = None

                for msg in reversed(response_messages):
                    if isinstance(msg, AIMessage):
                        ai_response = self._extract_text_content(msg.content)
                        # Try to extract token usage from metadata
                        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                            token_usage = dict(msg.usage_metadata)
                        elif hasattr(msg, "response_metadata"):
                            token_usage = msg.response_metadata.get("usage", {})
                        # Capture the provider's stop reason — "length" means
                        # the answer was cut off at max_tokens (truncation).
                        if hasattr(msg, "response_metadata"):
                            finish_reason = msg.response_metadata.get(
                                "finish_reason"
                            )
                        break

                # Plan §4.2 slicing: bucket raw usage into accounting channels
                # (prompt/completion/total/reasoning/visible/cached).
                sliced_usage = _slice_token_usage(token_usage)

            except Exception as exc:
                t_end = time.perf_counter()
                latency_ms = (t_end - t_start) * 1000.0
                logger.error(
                    "  Error at turn %d: %s", turn.turn_number, exc
                )
                # Record the infrastructure failure for observability, but
                # never evaluate assertions against it — error strings must
                # not contaminate pass-rate or token metrics (audit §3.3).
                turn_results.append(
                    TurnResult(
                        turn_number=turn.turn_number,
                        user_input=turn.content,
                        agent_response="",
                        status="error",
                        error=f"{type(exc).__name__}: {exc}",
                        latency_ms=latency_ms,
                        session_id=session_id,
                        role=turn.role,
                    )
                )
                continue

            # Guard against blank-but-"successful" responses (observed in the
            # wild: a gateway occasionally strips reasoning-format output,
            # billing hundreds of output tokens while returning empty
            # content). A blank response is unscorable — it is an
            # infrastructure anomaly, not model behavior — so it is flagged
            # as an error turn and never reaches evaluate_assertions
            # (same contamination policy as the except-block above, §3.3).
            if not ai_response.strip():
                billed = (
                    token_usage.get("output_tokens")
                    or token_usage.get("completion_tokens")
                    or 0
                )
                logger.warning(
                    "  Turn %d: blank agent response (%d output tokens "
                    "billed) — flagged as error turn",
                    turn.turn_number,
                    billed,
                )
                turn_results.append(
                    TurnResult(
                        turn_number=turn.turn_number,
                        user_input=turn.content,
                        agent_response=ai_response,
                        status="error",
                        error=(
                            "EmptyResponse: agent returned blank content "
                            f"({billed} output tokens billed)"
                        ),
                        latency_ms=latency_ms,
                        token_usage=token_usage,
                        session_id=session_id,
                        role=turn.role,
                        **sliced_usage,
                        **l2_tel,
                    )
                )
                continue

            # Guard against truncated responses (finish_reason == "length"):
            # the model hit max_tokens before finishing its answer. A partial
            # answer is unscorable — assertions would measure the token cap,
            # not memory — so the turn is flagged as an error turn and never
            # reaches evaluate_assertions (same contamination policy as the
            # blank-response guard above, audit §3.3). Observed in the wild:
            # glm-5.3-flash cut off mid-answer at max_tokens=300 on
            # explicit_forget_01 turn 2, and the partial text was scored as
            # a memory failure (2026-10-09).
            if finish_reason == "length":
                billed = (
                    token_usage.get("output_tokens")
                    or token_usage.get("completion_tokens")
                    or 0
                )
                logger.warning(
                    "  Turn %d: response truncated at max_tokens "
                    "(finish_reason='length', %d output tokens billed) "
                    "— flagged as error turn",
                    turn.turn_number,
                    billed,
                )
                turn_results.append(
                    TurnResult(
                        turn_number=turn.turn_number,
                        user_input=turn.content,
                        agent_response=ai_response,
                        status="error",
                        error=(
                            "TruncatedResponse: output cut off at max_tokens "
                            f"(finish_reason='length', {billed} output tokens "
                            "billed)"
                        ),
                        latency_ms=latency_ms,
                        token_usage=token_usage,
                        session_id=session_id,
                        role=turn.role,
                        **sliced_usage,
                        **l2_tel,
                    )
                )
                continue

            # Evaluate assertions (only for successfully completed turns)
            assertion_details = self.evaluate_assertions(
                ai_response, turn.assertions
            )
            passed = sum(1 for a in assertion_details if a["passed"])
            failed = sum(1 for a in assertion_details if not a["passed"])

            turn_result = TurnResult(
                turn_number=turn.turn_number,
                user_input=turn.content,
                agent_response=ai_response,
                status="ok",
                assertions_passed=passed,
                assertions_failed=failed,
                assertion_details=assertion_details,
                latency_ms=latency_ms,
                token_usage=token_usage,
                session_id=session_id,
                role=turn.role,
                **sliced_usage,
                **l2_tel,
            )
            turn_results.append(turn_result)

            logger.info(
                "  Turn %d: latency=%.1fms assertions=%d/%d",
                turn.turn_number,
                latency_ms,
                passed,
                passed + failed,
            )

        # Compute summary statistics — error turns are kept in the record for
        # observability but excluded from all scientific metrics (pass rate,
        # latency, tokens) so infrastructure failures never contaminate
        # benchmark results (audit §3.3).
        num_turns = len(turn_results)
        ok_turns = [tr for tr in turn_results if tr.status == "ok"]
        error_count = num_turns - len(ok_turns)

        total_assertions = sum(tr.assertions_passed + tr.assertions_failed for tr in ok_turns)
        total_passed = sum(tr.assertions_passed for tr in ok_turns)
        total_failed = sum(tr.assertions_failed for tr in ok_turns)
        total_latency = sum(tr.latency_ms for tr in ok_turns)

        # Aggregate token usage across successfully completed turns
        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        for tr in ok_turns:
            usage = tr.token_usage or {}
            p_tok = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
            c_tok = usage.get("output_tokens") or usage.get("completion_tokens") or 0
            t_tok = usage.get("total_tokens") or (p_tok + c_tok)
            prompt_tokens += p_tok
            completion_tokens += c_tok
            total_tokens += t_tok

        # ── Plan §4.3 aggregation ──────────────────────────────────────────
        # Token-cost slices (reasoning/visible/cached) follow the same ok-turn
        # policy as the scientific metrics above: an errored turn's partial
        # billing must not contaminate cost figures. L2 telemetry channels
        # (retrieved context, embedding tokens, retrieve/append/embedding
        # latency) instead sum over ALL turns — they are infrastructure
        # costs that were actually incurred regardless of whether the
        # response turned out scorable.
        reasoning_tokens = sum(tr.reasoning_tokens for tr in ok_turns)
        visible_output_tokens = sum(tr.visible_output_tokens for tr in ok_turns)
        cached_prompt_tokens = sum(tr.cached_prompt_tokens for tr in ok_turns)
        retrieved_context_tokens = sum(
            tr.retrieved_context_tokens for tr in turn_results
        )
        embedding_tokens_total = sum(tr.embedding_tokens for tr in turn_results)
        total_retrieval_latency = sum(tr.retrieval_latency_ms for tr in turn_results)
        total_append_latency = sum(tr.append_latency_ms for tr in turn_results)
        total_embedding_latency = sum(tr.embedding_latency_ms for tr in turn_results)
        # ingestion_latency_ms: reserved for system_event document-chunk
        # ingestion, deferred with document ingestion itself — kept at 0.0
        # so the export column exists from day one.
        ingestion_latency = 0.0

        # backend_cfg was already resolved at config-build time (above).
        backend_name = (
            "none" if memory_backend is None else backend_cfg.get("backend", "unknown")
        )

        # Snapshot export happens HERE — after the last turn, before the
        # summary is returned and long before any caller-side reset() —
        # so the exported episodes always reflect this run's final state
        # (plan §4.3). The recorded path is relative to the run folder
        # ("memory/<run_id>"), matching the export layout.
        memory_snapshot_path: str | None = None
        if memory_backend is not None and snapshot_dir is not None:
            snapshot_dir = Path(snapshot_dir)
            try:
                memory_backend.export_snapshot(
                    snapshot_dir / run_id,
                    run_metadata={
                        "run_id": run_id,
                        "scenario_id": scenario.id,
                        "model_id": model_id,
                        "prompt_id": prompt_id,
                        "agent_level": agent_level,
                    },
                )
                memory_snapshot_path = f"{snapshot_dir.name}/{run_id}"
            except Exception as exc:  # noqa: BLE001 — export must not kill the run
                logger.error("Memory snapshot export failed for %s: %s", run_id, exc)

        summary = BenchmarkRunSummary(
            run_id=run_id,
            scenario_id=scenario.id,
            model_id=model_id,
            prompt_id=prompt_id,
            agent_level=agent_level,
            total_turns=num_turns,
            total_assertions=total_assertions,
            passed_assertions=total_passed,
            failed_assertions=total_failed,
            pass_rate=total_passed / total_assertions if total_assertions > 0 else 0.0,
            error_turns=error_count,
            error_rate=error_count / num_turns if num_turns > 0 else 0.0,
            total_latency_ms=total_latency,
            avg_latency_per_turn_ms=total_latency / len(ok_turns) if ok_turns else 0.0,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            turn_results=turn_results,
            timestamp=timestamp,
            # ── Plan §4.3 memory configuration identity ──────────────────
            memory_backend=backend_name,
            embedding_model_id=backend_cfg.get("embedding_model_id"),
            top_k=backend_cfg.get("top_k"),
            chunk_size=backend_cfg.get("chunk_size"),
            chunk_overlap=backend_cfg.get("chunk_overlap"),
            memory_snapshot_path=memory_snapshot_path,
            # ── Plan §4.3 aggregated token metrics ────────────────────────
            reasoning_tokens=reasoning_tokens,
            visible_output_tokens=visible_output_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
            retrieved_context_tokens=retrieved_context_tokens,
            embedding_tokens=embedding_tokens_total,
            # ── Plan §4.3 aggregated latency metrics ──────────────────────
            total_retrieval_latency_ms=total_retrieval_latency,
            total_append_latency_ms=total_append_latency,
            total_embedding_latency_ms=total_embedding_latency,
            ingestion_latency_ms=ingestion_latency,
        )

        logger.info(
            "Run complete: %s — pass_rate=%.2f%% latency=%.1fms",
            run_id,
            summary.pass_rate * 100,
            total_latency,
        )

        return summary

    # ── Matrix Execution ─────────────────────────────────────────────────

    def run_matrix(
        self,
        agent_graph: Any,
        agent_level: str,
        scenario_ids: list[str] | None = None,
        prompt_ids: list[str] | None = None,
        model_ids: list[str] | None = None,
        export: bool = True,
        memory_backend: Any = None,
        memory_backend_config: dict[str, Any] | None = None,
    ) -> list[BenchmarkRunSummary]:
        """Execute the full Cartesian product: Scenarios × Prompts × Models.

        Args:
            agent_graph: Compiled LangGraph ``CompiledStateGraph``.
            agent_level: Agent persistence level string.
            scenario_ids: Optional filter for specific scenarios.
            prompt_ids: Optional filter for specific prompts.
            model_ids: Optional filter for specific models.
            export: When False, skip the automatic export — used by the CLI
                when looping over multiple levels so one invocation writes a
                single shared results directory (and back-to-back exports
                can never collide on the same timestamp).
            memory_backend: Optional ``EpisodicMemoryBackend`` (L2+) passed
                through to every ``run_single`` call and **reset after each
                tuple** for isolation (plan §4.3). Ordering is guaranteed:
                ``run_single`` exports the tuple's snapshot internally,
                before returning, so on-disk artifacts always reflect the
                run's final state and are never affected by the reset.
                ``None`` keeps L0/L1 behavior identical.
            memory_backend_config: Optional identity dict (plan §4.3) for
                run summaries and snapshot manifests.

        Returns:
            List of ``BenchmarkRunSummary`` objects for all executed tuples.
        """
        # Load components
        scenarios = self.load_scenarios(filter_ids=scenario_ids)
        prompts = self.load_prompts()
        all_model_ids = self.load_model_ids()

        # Apply filters
        if prompt_ids:
            prompts = {k: v for k, v in prompts.items() if k in prompt_ids}
        if model_ids:
            all_model_ids = [m for m in all_model_ids if m in model_ids]

        # Compute Cartesian product
        matrix = list(itertools.product(scenarios, prompts.items(), all_model_ids))
        total_runs = len(matrix)

        logger.info(
            "Benchmark matrix: %d scenarios × %d prompts × %d models = %d runs",
            len(scenarios),
            len(prompts),
            len(all_model_ids),
            total_runs,
        )

        # Shared output directory for this invocation — created EAGERLY when a
        # memory backend is attached so per-run memory snapshots
        # (memory/<run_id>/) land in the same directory as the final results
        # export, instead of in a directory created moments later with a
        # different timestamp. Recorded on the instance; ``_export_results``
        # consumes it (the CLI batches levels with export=False and writes a
        # single shared export at the end of the invocation).
        snapshot_dir: Path | None = None
        if memory_backend is not None:
            snapshot_dir = self.ensure_shared_run_dir() / "memory"
            logger.info(
                "Memory backend attached — shared run directory: %s",
                self._current_run_dir,
            )
        # No backend attached: leave any existing shared directory intact.
        # The CLI batches levels into ONE results directory per invocation,
        # so a backend-less pass (e.g. L0 after L2 in --level 2,0) must not
        # clobber the directory that holds L2's memory snapshots.

        # Execute
        results: list[BenchmarkRunSummary] = []
        failed_tuples: list[str] = []
        for idx, (scenario, (prompt_id, prompt_cfg), model_id) in enumerate(matrix, 1):
            logger.info("─── Run %d/%d ───", idx, total_runs)

            try:
                summary = self.run_single(
                    scenario=scenario,
                    prompt_id=prompt_id,
                    prompt_content=prompt_cfg["content"],
                    model_id=model_id,
                    agent_graph=agent_graph,
                    agent_level=agent_level,
                    memory_backend=memory_backend,
                    memory_backend_config=memory_backend_config,
                    snapshot_dir=snapshot_dir,
                )
                results.append(summary)
            except Exception as exc:
                failed_tuples.append(
                    f"scenario={scenario.id}, prompt={prompt_id}, model={model_id}: {exc}"
                )
                logger.error(
                    "Run %d/%d FAILED (scenario=%s, model=%s): %s",
                    idx,
                    total_runs,
                    scenario.id,
                    model_id,
                    exc,
                )
            finally:
                # Tuple isolation (plan §4.3): reset the backend after EVERY
                # tuple — even failed ones — so the next run starts from a
                # clean store. Ordering is guaranteed: run_single exported
                # this tuple's snapshot internally BEFORE returning, so the
                # on-disk artifacts keep the run's final state while the
                # live store is recycled. A reset failure propagates loudly
                # — scoring the next tuple against contaminated memory
                # would silently invalidate the whole matrix.
                if memory_backend is not None:
                    memory_backend.reset()

        # Failed tuples must never vanish silently — report them loudly so
        # the exported matrix size can be reconciled against the plan.
        if failed_tuples:
            logger.error(
                "⚠ %d of %d tuples FAILED and were excluded from exports:\n  - %s",
                len(failed_tuples),
                total_runs,
                "\n  - ".join(failed_tuples),
            )

        # Export results (suppressible so the CLI can batch multiple levels
        # into a single shared results directory per invocation)
        if results and export:
            self._export_results(results)

        return results

    # ── Export Functions ──────────────────────────────────────────────────

    def _export_results(
        self,
        results: list[BenchmarkRunSummary],
        run_dir: Path | None = None,
    ) -> Path:
        """Export results to timestamped JSONL, CSV and Markdown files.

        Args:
            results: List of run summaries to export.
            run_dir: Optional pre-created output directory — used when a
                memory backend was attached so results land next to the
                ``memory/`` snapshot tree. Defaults to the runner's shared
                directory from the current invocation (``_current_run_dir``);
                when that is also unset a fresh timestamped directory is
                created (L0/L1 behavior, unchanged).

        Returns:
            Path to the output directory.
        """
        if run_dir is None:
            run_dir = self._current_run_dir
        if run_dir is None:
            run_dir = self.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        # Consume the shared directory so a subsequent invocation (e.g. the
        # next benchmark sweep in the same process) always writes fresh.
        self._current_run_dir = None
        run_dir.mkdir(parents=True, exist_ok=True)

        # ── JSON (indented, human-readable) ──────────────────────────
        json_path = run_dir / "results.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(
                [summary.model_dump() for summary in results],
                fh,
                indent=2,
                ensure_ascii=False,
            )

        logger.info("Exported JSON: %s", json_path)

        # ── CSV (flattened summary, Excel compatible) ────────────────
        csv_path = run_dir / "summary.csv"
        fieldnames = [
            "run_id",
            "scenario_id",
            "model_id",
            "prompt_id",
            "agent_level",
            # ── Plan §4.3 / Milestone 5: merge identity — L2-vector vs a
            # future L2-markdown run must never overwrite each other.
            "memory_backend",
            "embedding_model_id",
            "total_turns",
            "total_assertions",
            "passed_assertions",
            "failed_assertions",
            "pass_rate",
            "error_turns",
            "error_rate",
            "total_latency_ms",
            "avg_latency_per_turn_ms",
            "total_tokens",
            "prompt_tokens",
            "completion_tokens",
            "timestamp",
        ]

        with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for summary in results:
                row = summary.model_dump(include=set(fieldnames))
                writer.writerow(row)

        logger.info("Exported CSV: %s", csv_path)

        # ── Markdown Table (immediate human-friendly preview) ─────────
        md_path = run_dir / "summary.md"
        headers = ["Scenario", "Model", "Prompt", "Level", "Turns", "Pass Rate", "Avg Latency", "Total Tokens", "Errors"]
        table_rows = []
        for summary in results:
            pass_pct = f"{summary.pass_rate * 100:.1f}%"
            latency = f"{summary.avg_latency_per_turn_ms:.0f} ms"
            tokens = f"{summary.total_tokens:,}" if summary.total_tokens > 0 else "N/A"
            table_rows.append([
                f"`{summary.scenario_id}`",
                f"`{summary.model_id}`",
                f"`{summary.prompt_id}`",
                f"`{summary.agent_level}`",
                str(summary.total_turns),
                f"**{pass_pct}**",
                latency,
                tokens,
                str(summary.error_turns),
            ])

        col_widths = [len(h) for h in headers]
        for row in table_rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))

        # Format rows with padding
        header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
        sep_parts = []
        for i, w in enumerate(col_widths):
            # Right-align numeric columns (turns, pass rate, latency, tokens), left-align others
            if i >= 4:
                sep_parts.append("-" * (w - 1) + ":")
            else:
                sep_parts.append(":" + "-" * (w - 1))
        sep_line = "| " + " | ".join(sep_parts) + " |"

        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(f"# Benchmark Execution Summary\n\n")
            fh.write(f"- **Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            fh.write(f"- **Total Runs:** {len(results)}\n\n")
            fh.write(header_line + "\n")
            fh.write(sep_line + "\n")
            for row in table_rows:
                row_cells = []
                for i, cell in enumerate(row):
                    if i >= 4:
                        row_cells.append(cell.rjust(col_widths[i]))
                    else:
                        row_cells.append(cell.ljust(col_widths[i]))
                fh.write("| " + " | ".join(row_cells) + " |\n")

        logger.info("Exported Markdown: %s", md_path)
        logger.info("Results directory: %s", run_dir)

        return run_dir

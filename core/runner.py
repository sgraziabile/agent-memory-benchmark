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
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

    # ── Loaders ──────────────────────────────────────────────────────────

    def load_scenarios(
        self, filter_ids: list[str] | None = None
    ) -> list[BenchmarkScenario]:
        """Load benchmark scenarios from YAML files in the dataset directory.

        Args:
            filter_ids: Optional list of scenario IDs to include. If ``None``,
                        all scenarios are loaded.

        Returns:
            List of validated ``BenchmarkScenario`` instances.
        """
        scenarios: list[BenchmarkScenario] = []

        for yaml_path in sorted(self.dataset_dir.glob("*.yaml")):
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
            search_in = response if rule.case_sensitive else response.lower()
            search_for = rule.value if rule.case_sensitive else rule.value.lower()

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
        config = {
            "configurable": {
                "model_id": model_id,
                "system_prompt": prompt_content,
                "thread_id": f"{scenario.id}_{run_id}",
            }
        }

        for turn in scenario.turns:
            if turn.role == "system_event":
                # System events are injected but not sent to the model
                logger.info("  System event at turn %d: %s", turn.turn_number, turn.content[:80])
                continue

            # Build user message — send only the current turn delta.
            # History retention across turns is the agent's responsibility
            # (via its checkpointer), not the runner's.
            user_msg = HumanMessage(content=turn.content)

            # Invoke the agent graph
            t_start = time.perf_counter()
            try:
                result = agent_graph.invoke(
                    {"messages": [user_msg]},
                    config=config,
                )
                t_end = time.perf_counter()
                latency_ms = (t_end - t_start) * 1000.0

                # Extract the agent's response (last AI message)
                response_messages = result.get("messages", [])
                ai_response = ""
                token_usage: dict[str, Any] = {}

                for msg in reversed(response_messages):
                    if isinstance(msg, AIMessage):
                        ai_response = self._extract_text_content(msg.content)
                        # Try to extract token usage from metadata
                        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                            token_usage = dict(msg.usage_metadata)
                        elif hasattr(msg, "response_metadata"):
                            token_usage = msg.response_metadata.get("usage", {})
                        break

            except Exception as exc:
                t_end = time.perf_counter()
                latency_ms = (t_end - t_start) * 1000.0
                ai_response = f"[ERROR] {type(exc).__name__}: {exc}"
                token_usage = {}
                logger.error(
                    "  Error at turn %d: %s", turn.turn_number, exc
                )

            # Evaluate assertions
            assertion_details = self.evaluate_assertions(
                ai_response, turn.assertions
            )
            passed = sum(1 for a in assertion_details if a["passed"])
            failed = sum(1 for a in assertion_details if not a["passed"])

            turn_result = TurnResult(
                turn_number=turn.turn_number,
                user_input=turn.content,
                agent_response=ai_response,
                assertions_passed=passed,
                assertions_failed=failed,
                assertion_details=assertion_details,
                latency_ms=latency_ms,
                token_usage=token_usage,
            )
            turn_results.append(turn_result)

            logger.info(
                "  Turn %d: latency=%.1fms assertions=%d/%d",
                turn.turn_number,
                latency_ms,
                passed,
                passed + failed,
            )

        # Compute summary statistics
        total_assertions = sum(tr.assertions_passed + tr.assertions_failed for tr in turn_results)
        total_passed = sum(tr.assertions_passed for tr in turn_results)
        total_failed = sum(tr.assertions_failed for tr in turn_results)
        total_latency = sum(tr.latency_ms for tr in turn_results)
        num_turns = len(turn_results)

        # Aggregate token usage across all turns
        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        for tr in turn_results:
            usage = tr.token_usage or {}
            p_tok = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
            c_tok = usage.get("output_tokens") or usage.get("completion_tokens") or 0
            t_tok = usage.get("total_tokens") or (p_tok + c_tok)
            prompt_tokens += p_tok
            completion_tokens += c_tok
            total_tokens += t_tok

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
            total_latency_ms=total_latency,
            avg_latency_per_turn_ms=total_latency / num_turns if num_turns > 0 else 0.0,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            turn_results=turn_results,
            timestamp=timestamp,
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
    ) -> list[BenchmarkRunSummary]:
        """Execute the full Cartesian product: Scenarios × Prompts × Models.

        Args:
            agent_graph: Compiled LangGraph ``CompiledStateGraph``.
            agent_level: Agent persistence level string.
            scenario_ids: Optional filter for specific scenarios.
            prompt_ids: Optional filter for specific prompts.
            model_ids: Optional filter for specific models.

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

        # Execute
        results: list[BenchmarkRunSummary] = []
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
                )
                results.append(summary)
            except Exception as exc:
                logger.error(
                    "Run %d/%d FAILED (scenario=%s, model=%s): %s",
                    idx,
                    total_runs,
                    scenario.id,
                    model_id,
                    exc,
                )

        # Export results
        if results:
            self._export_results(results)

        return results

    # ── Export Functions ──────────────────────────────────────────────────

    def _export_results(self, results: list[BenchmarkRunSummary]) -> Path:
        """Export results to timestamped JSONL, CSV and Markdown files.

        Args:
            results: List of run summaries to export.

        Returns:
            Path to the created output directory.
        """
        run_dir = self.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
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
            "total_turns",
            "total_assertions",
            "passed_assertions",
            "failed_assertions",
            "pass_rate",
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
        headers = ["Scenario", "Model", "Prompt", "Level", "Turns", "Pass Rate", "Avg Latency", "Total Tokens"]
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

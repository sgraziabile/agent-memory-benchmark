"""run.py — CLI entrypoint for agent-memory-benchmark.

Allows running benchmark scenarios against selected models and prompts
from the command line.

Examples:
    # Run a single scenario with Gemini:
    python run.py --model gemini-3.8-flash --scenario belief_revision_01

    # Run with a custom provider-qualified model string:
    python run.py --model google:gemini-2.0-flash --scenario needle_haystack_01

    # Run with strict epistemic prompt:
    python run.py --model gemini-3.8-flash --scenario belief_revision_01 --prompt strict_epistemic

    # List available components:
    python run.py --list-models
    python run.py --list-scenarios
    python run.py --list-prompts
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Load .env if present
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from agents.level_0_reactive.graph import Level0ReactiveAgent
from core.runner import BenchmarkRunner


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="agent-memory-benchmark: Experimental harness runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gemini-3.8-flash",
        help="Model ID from configs/models.yaml or 'provider:model_name'",
    )
    parser.add_argument(
        "--scenario",
        "-s",
        type=str,
        default="belief_revision_01",
        help="Scenario ID from datasets/conversations/ or 'all'",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default="baseline_react",
        help="Prompt ID from configs/prompts.yaml",
    )
    parser.add_argument(
        "--level",
        "-l",
        type=str,
        default="level_0_reactive",
        choices=["level_0_reactive"],
        help="Agent persistence level",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all registered models in configs/models.yaml and exit",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List all available benchmark scenarios and exit",
    )
    parser.add_argument(
        "--list-prompts",
        action="store_true",
        help="List all available system prompts and exit",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    runner = BenchmarkRunner()

    # ── Inspection helpers ───────────────────────────────────────────
    if args.list_models:
        models = runner.load_model_ids()
        print("\nRegistered Models (configs/models.yaml):")
        for mid in models:
            print(f"  - {mid}")
        return 0

    if args.list_scenarios:
        scenarios = runner.load_scenarios()
        print("\nAvailable Scenarios (datasets/conversations/):")
        for sc in scenarios:
            print(f"  - {sc.id:22s} ({sc.category}) — {len(sc.turns)} turns: {sc.name}")
        return 0

    if args.list_prompts:
        prompts = runner.load_prompts()
        print("\nAvailable Prompts (configs/prompts.yaml):")
        for pid, pdata in prompts.items():
            print(f"  - {pid:18s} — {pdata.get('description', '')}")
        return 0

    # ── Agent Selection ──────────────────────────────────────────────
    if args.level == "level_0_reactive":
        agent = Level0ReactiveAgent()
    else:
        print(f"Error: Level '{args.level}' is not yet implemented.")
        return 1

    graph = agent.build_graph()

    # ── Filter setup ─────────────────────────────────────────────────
    scenario_ids = None if args.scenario == "all" else [args.scenario]
    prompt_ids = [args.prompt]
    model_ids = [args.model]

    print("\n" + "=" * 60)
    print("AGENT MEMORY BENCHMARK — EXECUTION RUN")
    print("=" * 60)
    print(f"  Level:    {agent.level}")
    print(f"  Model:    {args.model}")
    print(f"  Prompt:   {args.prompt}")
    print(f"  Scenario: {args.scenario}")
    print("=" * 60 + "\n")

    # Validate model
    if ":" not in args.model:
        registered_models = runner.load_model_ids()
        if args.model not in registered_models:
            print(f"Error: Model '{args.model}' not found in configs/models.yaml.")
            print(f"Available registered models: {', '.join(registered_models)}")
            print("Tip: If you want to use an unregistered model, use the 'provider:model_name' format.")
            print(f"     Example: --model google:{args.model}")
            return 1

    # If args.model contains ':', handle it as an inline provider:model string
    if ":" in args.model:
        # Load scenario and prompt directly
        scenarios = runner.load_scenarios(filter_ids=scenario_ids)
        if not scenarios:
            print(f"Error: Scenario '{args.scenario}' not found.")
            return 1

        prompts = runner.load_prompts()
        if args.prompt not in prompts:
            print(f"Error: Prompt '{args.prompt}' not found.")
            return 1

        prompt_content = prompts[args.prompt]["content"]
        results = []
        for sc in scenarios:
            summary = runner.run_single(
                scenario=sc,
                prompt_id=args.prompt,
                prompt_content=prompt_content,
                model_id=args.model,
                agent_graph=graph,
                agent_level=agent.level,
            )
            results.append(summary)

        if results:
            run_dir = runner._export_results(results)
            _print_run_report(results, run_dir)
        return 0

    # Otherwise run via registry matrix
    results = runner.run_matrix(
        agent_graph=graph,
        agent_level=agent.level,
        scenario_ids=scenario_ids,
        prompt_ids=prompt_ids,
        model_ids=model_ids,
    )

    if not results:
        print("No runs executed. Check your scenario, model, or prompt IDs.")
        return 1

    # Output directory was logged by runner._export_results
    _print_run_report(results)
    return 0


def _print_run_report(results: list, run_dir: Path | None = None) -> None:
    print("\n" + "=" * 60)
    print("EXECUTION SUMMARY")
    print("=" * 60)
    for res in results:
        pass_pct = res.pass_rate * 100
        print(f"Scenario:     {res.scenario_id}")
        print(f"Model:        {res.model_id}")
        print(f"Prompt:       {res.prompt_id}")
        print(f"Total Turns:  {res.total_turns}")
        print(f"Assertions:   {res.passed_assertions}/{res.total_assertions} passed ({pass_pct:.1f}%)")
        print(f"Avg Latency:  {res.avg_latency_per_turn_ms:.1f} ms/turn")
        print(f"Total Time:   {res.total_latency_ms:.1f} ms")
        if res.total_tokens > 0:
            print(f"Total Tokens: {res.total_tokens:,} (prompt: {res.prompt_tokens:,}, completion: {res.completion_tokens:,})")
        print("-" * 60)
        print("Detailed Turn Breakdown:")
        for tr in res.turn_results:
            status = "PASS" if tr.assertions_failed == 0 else "FAIL"
            tok_info = ""
            if tr.token_usage:
                t_count = tr.token_usage.get("total_tokens") or (
                    (tr.token_usage.get("input_tokens", 0) or tr.token_usage.get("prompt_tokens", 0)) +
                    (tr.token_usage.get("output_tokens", 0) or tr.token_usage.get("completion_tokens", 0))
                )
                if t_count:
                    tok_info = f" tokens={t_count}"
            print(f"  [Turn {tr.turn_number}] ({status}) latency={tr.latency_ms:.1f}ms{tok_info}")
            print(f"    User:     {tr.user_input[:80]}...")
            print(f"    Agent:    {tr.agent_response[:100]}...")
            if tr.assertion_details:
                for a in tr.assertion_details:
                    mark = "[PASS]" if a["passed"] else "[FAIL]"
                    print(f"      {mark} {a['rule_type']}: '{a['value']}'")
        print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())

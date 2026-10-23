"""run.py — CLI entrypoint for agent-memory-benchmark.

Allows running benchmark scenarios against selected models and prompts
from the command line.

Examples:
    # Run a single scenario (NanoGPT subscription model):
    python run.py --model qwen3.6-27b --scenario belief_revision_01

    # Run with a custom provider-qualified model string:
    python run.py --model google:gemini-2.0-flash --scenario needle_haystack_01

    # Run with strict epistemic prompt:
    python run.py --model qwen3.6-27b --scenario belief_revision_01 --prompt strict_epistemic

    # Select agent level(s) by number — full registry names also accepted:
    python run.py --level 1 --model glm-5.3-flash --scenario belief_revision_01
    python run.py --level 0,1 --model qwen3.6-27b --scenario belief_revision_01

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

from dotenv import load_dotenv

# Load .env if present (python-dotenv is a declared dependency — a missing
# install must fail loudly, never silently skip the user's API keys).
load_dotenv()

from agents import AGENT_REGISTRY, resolve_level_ids
from agents.level_2_episodic_rag.backends.vector import VectorMemoryBackend
from agents.level_2_episodic_rag.graph import DEFAULT_TOP_K
from core.model_factory import create_embeddings, get_embedding_dims
from core.runner import BenchmarkRunner


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _resolve_backend_choice(raw: str) -> str:
    """Validate the ``--backend`` argument; return the single backend name.

    Comma-separated values (e.g. ``vector,markdown``) are reserved for a
    future multi-backend milestone — they are parsed here and rejected
    loudly so the flag's syntax stays stable when that lands.

    Raises:
        ValueError: on empty input, multiple backends, or an unknown name.
    """
    backends = [b.strip() for b in raw.split(",") if b.strip()]
    if not backends:
        raise ValueError("No memory backend specified — use --backend vector.")
    if len(backends) > 1:
        raise ValueError(
            f"Multi-backend runs ({raw!r}) are not supported yet — run one "
            "backend per invocation."
        )
    if backends[0] != "vector":
        raise ValueError(
            f"Memory backend {backends[0]!r} is not implemented yet. "
            "Available backends: vector"
        )
    return backends[0]


def _build_memory_backend(
    backend_name: str, embedding_model_id: str
) -> VectorMemoryBackend:
    """Construct the requested ``EpisodicMemoryBackend`` (L2+ levels only).

    ``vector`` → :class:`VectorMemoryBackend` backed by the registry
    embedding client (:func:`create_embeddings`) with its declared
    dimensionality (:func:`get_embedding_dims`) so the ``InMemoryStore``
    index is never built with a hardcoded vector size.
    """
    embeddings = create_embeddings(embedding_model_id)
    dims = get_embedding_dims(embedding_model_id)
    return VectorMemoryBackend(embeddings, dims, embedding_id=embedding_model_id)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser (extracted for testability)."""
    parser = argparse.ArgumentParser(
        description="agent-memory-benchmark: Experimental harness runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="qwen3.6-27b",
        help="Model ID from configs/models.yaml or 'provider:model_name'",
    )
    parser.add_argument(
        "--scenario",
        "-s",
        type=str,
        default="all",
        help="Scenario ID from datasets/conversations/, 'all' (default), or a comma-separated list",
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
        default="0",
        help="Agent level(s): number (0-4), full registry name, 'all', or comma-separated list",
    )
    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        default="vector",
        help="Memory backend for levels that require one (L2+): 'vector' (default). Comma-separated lists are reserved for future multi-backend runs",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="qwen3-embedding-8b",
        help="Embedding model ID from the 'embeddings:' section of configs/models.yaml (vector backend only)",
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
    return parser


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = build_parser().parse_args()
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

    # ── Level resolution ('0', '1', '0,1', 'all', or full names) ──────
    try:
        level_ids = resolve_level_ids(args.level)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    # ── Memory backend resolution (validated once, before any run) ──
    # Only levels with requires_memory_backend=True actually construct one;
    # L0/L1 never need an embedding API key (full backward compatibility).
    try:
        backend_name = _resolve_backend_choice(args.backend)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    # ── Filter setup ─────────────────────────────────────────────────
    scenario_ids = (
        None
        if args.scenario.strip().lower() == "all"
        else [s.strip() for s in args.scenario.split(",") if s.strip()]
    )
    prompt_ids = [args.prompt]
    model_ids = [args.model]

    # ── Model validation (once, before the per-level loop) ───────────
    # Registry IDs run via run_matrix; 'provider:model_name' strings run
    # inline — scenario/prompt are validated up front so the loop never
    # repeats that work.
    inline_model = ":" in args.model
    scenarios = None
    prompt_content = None
    if not inline_model:
        registered_models = runner.load_model_ids()
        if args.model not in registered_models:
            print(f"Error: Model '{args.model}' not found in configs/models.yaml.")
            print(f"Available registered models: {', '.join(registered_models)}")
            print("Tip: If you want to use an unregistered model, use the 'provider:model_name' format.")
            print(f"     Example: --model google:{args.model}")
            return 1
    else:
        scenarios = runner.load_scenarios(filter_ids=scenario_ids)
        if not scenarios:
            print(f"Error: Scenario '{args.scenario}' not found.")
            return 1

        prompts = runner.load_prompts()
        if args.prompt not in prompts:
            print(f"Error: Prompt '{args.prompt}' not found.")
            return 1

        prompt_content = prompts[args.prompt]["content"]

    print("\n" + "=" * 60)
    print("AGENT MEMORY BENCHMARK — EXECUTION RUN")
    print("=" * 60)
    print(f"  Level:    {', '.join(level_ids)}")
    print(f"  Model:    {args.model}")
    print(f"  Prompt:   {args.prompt}")
    print(f"  Scenario: {args.scenario}")
    if any(
        getattr(AGENT_REGISTRY[lid], "requires_memory_backend", False)
        for lid in level_ids
    ):
        print(f"  Backend:  {backend_name} (embedding model: {args.embedding_model})")
    print("=" * 60 + "\n")

    # ── Execute one pass per level ────────────────────────────────────
    # Per-pass exports are suppressed (run_matrix(export=False), deferred
    # for the inline path) so ONE invocation writes ONE results directory
    # containing every requested level.
    all_results: list = []
    for level_id in level_ids:
        if len(level_ids) > 1:
            print(f"\n─── Level: {level_id} ────────────────────────")

        agent = AGENT_REGISTRY[level_id]()
        graph = agent.build_graph()

        # ── Memory backend lifecycle (plan §4.3, Milestone 5) ─────────
        # Constructed ONLY for levels that require one, so L0/L1 never
        # touches the embedding API (backward compatible, no keys needed).
        # The backend + identity dict flow into run_matrix / run_single,
        # which inject them via config["configurable"] for the graph,
        # export the snapshot before returning, and reset after each
        # tuple for isolation.
        memory_backend = None
        memory_backend_config: dict | None = None
        if getattr(agent, "requires_memory_backend", False):
            try:
                memory_backend = _build_memory_backend(
                    backend_name, args.embedding_model
                )
            except Exception as exc:
                print(
                    f"Error: Could not construct {backend_name!r} memory "
                    f"backend with embedding model {args.embedding_model!r}: {exc}"
                )
                return 1
            # top_k defaults to the L2 graph's DEFAULT_TOP_K so the summary
            # identity records the retrieval cut-off actually in effect.
            memory_backend_config = {
                "backend": backend_name,
                "embedding_model_id": args.embedding_model,
                "top_k": DEFAULT_TOP_K,
            }
            print(
                f"  Memory backend: {backend_name} "
                f"(embedding: {args.embedding_model}, top_k={DEFAULT_TOP_K})"
            )

        if inline_model:
            # Inline path mirrors run_matrix: snapshots land under the
            # shared run dir's memory/ tree, and the backend is reset
            # after every scenario (the snapshot was already exported
            # inside run_single, before it returned).
            snapshot_dir = None
            if memory_backend is not None:
                snapshot_dir = runner.ensure_shared_run_dir() / "memory"
            for sc in scenarios:
                summary = runner.run_single(
                    scenario=sc,
                    prompt_id=args.prompt,
                    prompt_content=prompt_content,
                    model_id=args.model,
                    agent_graph=graph,
                    agent_level=agent.level,
                    memory_backend=memory_backend,
                    memory_backend_config=memory_backend_config,
                    snapshot_dir=snapshot_dir,
                )
                if memory_backend is not None:
                    memory_backend.reset()
                all_results.append(summary)
        else:
            all_results.extend(
                runner.run_matrix(
                    agent_graph=graph,
                    agent_level=agent.level,
                    scenario_ids=scenario_ids,
                    prompt_ids=prompt_ids,
                    model_ids=model_ids,
                    export=False,
                    memory_backend=memory_backend,
                    memory_backend_config=memory_backend_config,
                )
            )

    if not all_results:
        print("No runs executed. Check your scenario, model, or prompt IDs.")
        return 1

    # Single export per invocation — the directory covers all levels.
    run_dir = runner._export_results(all_results)
    _print_run_report(all_results, run_dir)
    return _error_exit_code(all_results)


def _error_exit_code(results: list) -> int:
    """Return 2 if any turn errored (e.g. provider rate limits), else 0.

    Distinct from 1 (invalid IDs/configuration) so shell wrappers can retry
    only the combinations that hit infrastructure failures. A re-run gets a
    fresh thread_id, so the retried conversation starts clean.
    """
    error_turns = sum(res.error_turns for res in results)
    if error_turns:
        print(
            f"\n⚠  {error_turns} error turn(s) detected (e.g. API rate limits)."
            " Exit code 2 — re-run this combination when quotas reset."
        )
        return 2
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
        if res.error_turns:
            print(f"Error Turns:  {res.error_turns} ({res.error_rate * 100:.1f}%) — excluded from metrics")
        print(f"Assertions:   {res.passed_assertions}/{res.total_assertions} passed ({pass_pct:.1f}%)")
        print(f"Avg Latency:  {res.avg_latency_per_turn_ms:.1f} ms/turn")
        print(f"Total Time:   {res.total_latency_ms:.1f} ms")
        if res.total_tokens > 0:
            print(f"Total Tokens: {res.total_tokens:,} (prompt: {res.prompt_tokens:,}, completion: {res.completion_tokens:,})")
        print("-" * 60)
        print("Detailed Turn Breakdown:")
        for tr in res.turn_results:
            if tr.status == "error":
                print(f"  [Turn {tr.turn_number}] (ERROR) latency={tr.latency_ms:.1f}ms — excluded from metrics")
                print(f"    User:     {tr.user_input[:80]}...")
                print(f"    Error:    {tr.error}")
                continue
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

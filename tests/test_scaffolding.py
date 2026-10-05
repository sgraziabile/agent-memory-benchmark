"""Smoke tests for the benchmark harness scaffolding."""

from pathlib import Path

import yaml
import pytest
from langchain_core.messages import AIMessage

from core.schemas import (
    AgentState,
    AssertionRule,
    BenchmarkRunSummary,
    BenchmarkScenario,
    ConversationTurn,
    TurnResult,
)
from core.model_factory import (
    clear_model_cache,
    create_model,
    load_model_registry,
    PROVIDER_MAP,
)
from core.runner import BenchmarkRunner
from agents.base import BaseBenchmarkAgent
from agents.level_0_reactive.graph import Level0ReactiveAgent


# ── Schema Tests ─────────────────────────────────────────────────────────────


class TestSchemas:
    """Verify Pydantic schemas accept and validate data correctly."""

    def test_assertion_rule_must_contain(self):
        rule = AssertionRule(type="must_contain", value="hello")
        assert rule.type == "must_contain"
        assert rule.case_sensitive is False

    def test_assertion_rule_must_not_contain(self):
        rule = AssertionRule(type="must_not_contain", value="secret", case_sensitive=True)
        assert rule.case_sensitive is True

    def test_conversation_turn_defaults(self):
        turn = ConversationTurn(turn_number=1, role="user", content="Hi")
        assert turn.assertions == []

    def test_benchmark_scenario_from_dict(self):
        data = {
            "id": "test_01",
            "name": "Test",
            "description": "A test scenario",
            "category": "unit_test",
            "turns": [
                {"turn_number": 1, "role": "user", "content": "Hello"},
            ],
        }
        scenario = BenchmarkScenario(**data)
        assert scenario.id == "test_01"
        assert len(scenario.turns) == 1

    def test_turn_result_defaults(self):
        tr = TurnResult(
            turn_number=1,
            user_input="test",
            agent_response="response",
        )
        assert tr.latency_ms == 0.0
        assert tr.token_usage == {}
        assert tr.status == "ok"
        assert tr.error is None

    def test_turn_result_error_status(self):
        tr = TurnResult(
            turn_number=1,
            user_input="test",
            agent_response="",
            status="error",
            error="RateLimitError: 429 Too Many Requests",
        )
        assert tr.status == "error"
        assert tr.error == "RateLimitError: 429 Too Many Requests"

    def test_run_summary_error_defaults(self):
        summary = BenchmarkRunSummary(
            run_id="test",
            scenario_id="s1",
            model_id="m1",
            prompt_id="p1",
            agent_level="level_0",
        )
        assert summary.error_turns == 0
        assert summary.error_rate == 0.0

    def test_run_summary_pass_rate(self):
        summary = BenchmarkRunSummary(
            run_id="test",
            scenario_id="s1",
            model_id="m1",
            prompt_id="p1",
            agent_level="level_0",
            total_assertions=10,
            passed_assertions=7,
            failed_assertions=3,
            pass_rate=0.7,
            total_tokens=1500,
            prompt_tokens=1200,
            completion_tokens=300,
        )
        assert summary.pass_rate == 0.7
        assert summary.total_tokens == 1500
        assert summary.prompt_tokens == 1200
        assert summary.completion_tokens == 300


# ── Model Factory Tests ─────────────────────────────────────────────────────


class TestModelFactory:
    """Verify model registry loading and provider dispatch."""

    def test_load_registry(self):
        registry = load_model_registry()
        assert len(registry) >= 8
        assert "gemini-3.8-flash" in registry
        assert "gpt-4.1-mini" in registry

    def test_registry_entries_have_required_keys(self):
        registry = load_model_registry()
        for model_id, cfg in registry.items():
            assert "provider" in cfg, f"{model_id} missing 'provider'"
            assert "model_name" in cfg, f"{model_id} missing 'model_name'"
            assert cfg.get("timeout") == 60.0, f"{model_id} missing or incorrect timeout"
            assert cfg.get("max_retries") == 2, f"{model_id} missing or incorrect max_retries"
            assert cfg.get("max_tokens") == 300, f"{model_id} missing or incorrect max_tokens"

    def test_all_providers_in_map(self):
        registry = load_model_registry()
        for model_id, cfg in registry.items():
            assert cfg["provider"] in PROVIDER_MAP, (
                f"{model_id} uses unknown provider: {cfg['provider']}"
            )


# ── Model Factory Caching Tests (audit §3.4) ────────────────────────────────


class _DummyChatModel:
    """Stand-in for a provider chat model — accepts any constructor kwargs.

    Deliberately a plain class (not a ``BaseChatModel`` subclass) so that
    registry YAML extras (``timeout``, ``max_retries``, ``max_tokens``, ...)
    never trip Pydantic validation in these cache tests. No network.
    """

    def __init__(self, model: str, temperature: float = 0.0, **kwargs):
        self.model = model
        self.temperature = temperature
        self.kwargs = kwargs


class TestModelCaching:
    """Audit §3.4 — model instances must be cached per configuration so
    per-turn latency reflects model inference, not YAML re-reads and
    provider client-pool construction inside the measured window."""

    @pytest.fixture(autouse=True)
    def _isolate_cache(self, monkeypatch):
        """Stub the provider resolver and reset the cache around each test."""
        clear_model_cache()
        monkeypatch.setattr(
            "core.model_factory._resolve_provider_class",
            lambda provider: _DummyChatModel,
        )
        yield
        clear_model_cache()

    def test_same_config_returns_same_instance(self):
        first = create_model("fake:test-model", temperature=0.0)
        second = create_model("fake:test-model", temperature=0.0)
        assert first is second

    def test_registry_read_once_across_calls(self, monkeypatch):
        calls = {"count": 0}
        real_loader = load_model_registry

        def counting_loader(config_path):
            calls["count"] += 1
            return real_loader(config_path)

        monkeypatch.setattr(
            "core.model_factory.load_model_registry", counting_loader
        )

        model_id = next(iter(load_model_registry()))
        create_model(model_id)
        create_model(model_id)
        # Second call must be served from cache — no re-read of models.yaml.
        assert calls["count"] == 1

    def test_different_config_returns_new_instance(self):
        base = create_model("fake:test-model", temperature=0.0)
        hotter = create_model("fake:test-model", temperature=1.0)
        tuned = create_model("fake:test-model", temperature=0.0, max_tokens=100)
        # Cache-key sensitivity: distinct configs must never share an instance.
        assert base is not hotter
        assert base is not tuned
        assert hotter is not tuned

    def test_clear_model_cache_forces_new_instance(self):
        first = create_model("fake:test-model")
        clear_model_cache()
        second = create_model("fake:test-model")
        assert first is not second


# ── Runner Assertion Tests ───────────────────────────────────────────────────


class TestRunnerAssertions:
    """Verify the assertion evaluation engine."""

    def test_must_contain_pass(self):
        rules = [AssertionRule(type="must_contain", value="hello")]
        results = BenchmarkRunner.evaluate_assertions("Hello world!", rules)
        assert results[0]["passed"] is True

    def test_must_contain_fail(self):
        rules = [AssertionRule(type="must_contain", value="goodbye")]
        results = BenchmarkRunner.evaluate_assertions("Hello world!", rules)
        assert results[0]["passed"] is False

    def test_must_not_contain_pass(self):
        rules = [AssertionRule(type="must_not_contain", value="secret")]
        results = BenchmarkRunner.evaluate_assertions("Hello world!", rules)
        assert results[0]["passed"] is True

    def test_must_not_contain_fail(self):
        rules = [AssertionRule(type="must_not_contain", value="world")]
        results = BenchmarkRunner.evaluate_assertions("Hello world!", rules)
        assert results[0]["passed"] is False

    def test_case_insensitive_default(self):
        rules = [AssertionRule(type="must_contain", value="HELLO")]
        results = BenchmarkRunner.evaluate_assertions("hello world", rules)
        assert results[0]["passed"] is True

    def test_case_sensitive_explicit(self):
        rules = [AssertionRule(type="must_contain", value="HELLO", case_sensitive=True)]
        results = BenchmarkRunner.evaluate_assertions("hello world", rules)
        assert results[0]["passed"] is False

    def test_extract_text_content_string(self):
        assert BenchmarkRunner._extract_text_content("Simple string") == "Simple string"

    def test_extract_text_content_blocks(self):
        blocks = [{"type": "text", "text": "The answer is 42.", "extras": {"signature": "sig123"}}]
        assert BenchmarkRunner._extract_text_content(blocks) == "The answer is 42."


# ── Error-Turn Handling Tests (audit §3.3) ──────────────────────────────────


class _ScriptedGraph:
    """Minimal deterministic stand-in for a compiled LangGraph.

    Fails the first ``fail_first`` invocations with an infrastructure error,
    then returns a scripted ``AIMessage``. No network, no API keys.
    """

    def __init__(self, fail_first: int = 0):
        self.calls = 0
        self.fail_first = fail_first

    def invoke(self, state, config=None):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise RuntimeError("simulated provider outage")
        return {"messages": [AIMessage(content="Miso is the cat.")]}


class TestErrorTurnHandling:
    """Verify error turns are recorded, visible, and excluded from metrics."""

    @staticmethod
    def _make_scenario() -> BenchmarkScenario:
        return BenchmarkScenario(
            id="error_handling_01",
            name="Error handling",
            description="Turn 1 fails at the provider; turn 2 succeeds.",
            category="unit_test",
            turns=[
                ConversationTurn(
                    turn_number=1,
                    role="user",
                    content="Who is Miso?",
                    # Pre-fix behavior: the error string "[ERROR] ..."
                    # would be evaluated here and silently FAIL this rule,
                    # scoring an outage as a model failure.
                    assertions=[AssertionRule(type="must_contain", value="Miso")],
                ),
                ConversationTurn(
                    turn_number=2,
                    role="user",
                    content="Again: who is Miso?",
                    assertions=[
                        AssertionRule(type="must_contain", value="Miso"),
                        AssertionRule(type="must_not_contain", value="Luna"),
                    ],
                ),
            ],
        )

    def test_error_turn_recorded_and_flagged(self):
        summary = BenchmarkRunner().run_single(
            scenario=self._make_scenario(),
            prompt_id="unit_test_prompt",
            prompt_content="You are a deterministic test agent.",
            model_id="fake-model",
            agent_graph=_ScriptedGraph(fail_first=1),
            agent_level="level_0_reactive",
        )

        # Both turns executed and recorded
        assert summary.total_turns == 2

        # The failed turn is explicitly flagged as an error
        first = summary.turn_results[0]
        assert first.status == "error"
        assert first.error is not None
        assert "RuntimeError" in first.error

        # No assertions were evaluated against the error string (§3.3 fix)
        assert first.assertions_passed == 0
        assert first.assertions_failed == 0
        assert first.assertion_details == []

        # Error accounting surfaces in the run summary
        assert summary.error_turns == 1
        assert summary.error_rate == 0.5

    def test_pass_rate_excludes_error_turns(self):
        summary = BenchmarkRunner().run_single(
            scenario=self._make_scenario(),
            prompt_id="unit_test_prompt",
            prompt_content="You are a deterministic test agent.",
            model_id="fake-model",
            agent_graph=_ScriptedGraph(fail_first=1),
            agent_level="level_0_reactive",
        )

        # Only the healthy turn's 2 assertions count: 2/2 → 1.0.
        # (Pre-fix behavior would also score the error string: 1/3 → 0.33.)
        assert summary.total_assertions == 2
        assert summary.passed_assertions == 2
        assert summary.pass_rate == 1.0

        # Latency metrics exclude the failed turn's time-to-error
        healthy = summary.turn_results[1]
        assert summary.total_latency_ms == pytest.approx(healthy.latency_ms)
        assert summary.avg_latency_per_turn_ms == pytest.approx(healthy.latency_ms)

    def test_healthy_run_has_no_errors(self):
        summary = BenchmarkRunner().run_single(
            scenario=self._make_scenario(),
            prompt_id="unit_test_prompt",
            prompt_content="You are a deterministic test agent.",
            model_id="fake-model",
            agent_graph=_ScriptedGraph(fail_first=0),
            agent_level="level_0_reactive",
        )
        assert summary.error_turns == 0
        assert summary.error_rate == 0.0
        assert summary.pass_rate == 1.0
        assert all(tr.status == "ok" for tr in summary.turn_results)


# ── Level 0 Agent Tests ─────────────────────────────────────────────────────


class TestLevel0Agent:
    """Verify Level 0 agent construction and constraints."""

    def test_agent_is_base_subclass(self):
        agent = Level0ReactiveAgent()
        assert isinstance(agent, BaseBenchmarkAgent)

    def test_level_property(self):
        agent = Level0ReactiveAgent()
        assert agent.level == "level_0_reactive"

    def test_graph_compiles(self):
        agent = Level0ReactiveAgent()
        graph = agent.build_graph()
        assert graph is not None

    def test_no_checkpointer(self):
        agent = Level0ReactiveAgent()
        graph = agent.build_graph()
        assert graph.checkpointer is None

    def test_repr(self):
        agent = Level0ReactiveAgent()
        assert "level_0_reactive" in repr(agent)

    def test_graph_compiles_with_tools(self):
        def sample_tool(query: str) -> str:
            """Sample tool."""
            return query
        agent = Level0ReactiveAgent()
        graph = agent.build_graph(tools=[sample_tool])
        assert graph is not None
        assert graph.checkpointer is None


# ── Scenario Loading Tests ───────────────────────────────────────────────────


class TestScenarioLoading:
    """Verify YAML scenarios load and validate against Pydantic schemas."""

    def test_load_all_scenarios(self):
        runner = BenchmarkRunner()
        scenarios = runner.load_scenarios()
        assert len(scenarios) >= 5

    def test_scenario_ids(self):
        runner = BenchmarkRunner()
        scenarios = runner.load_scenarios()
        ids = {s.id for s in scenarios}
        assert "belief_revision_01" in ids
        assert "attrition_01" in ids
        assert "needle_haystack_01" in ids
        assert "temporal_multi_hop_01" in ids
        assert "explicit_forget_01" in ids

    def test_scenario_filter(self):
        runner = BenchmarkRunner()
        scenarios = runner.load_scenarios(filter_ids=["belief_revision_01"])
        assert len(scenarios) == 1
        assert scenarios[0].id == "belief_revision_01"

    def test_load_prompts(self):
        runner = BenchmarkRunner()
        prompts = runner.load_prompts()
        assert "baseline_react" in prompts
        assert "strict_epistemic" in prompts
        assert "content" in prompts["baseline_react"]

    def test_load_model_ids(self):
        runner = BenchmarkRunner()
        model_ids = runner.load_model_ids()
        assert len(model_ids) >= 8
        assert "gemini-3.8-flash" in model_ids


# ── Dependency Declaration Tests (audit §3.2) ─────────────────────────────────

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 — tomllib landed in 3.11
    tomllib = None


class TestDependencyDeclarations:
    """Audit §3.2 — runtime dependencies must be declared and imported loudly.

    ``run.py`` used to wrap ``from dotenv import load_dotenv`` in a silent
    ``except ImportError: pass`` while ``python-dotenv`` was missing from
    ``pyproject.toml``. In a fresh venv the ``.env`` file was then silently
    never loaded, so users got a confusing provider auth error despite
    following the README quickstart. These tests pin both the declaration
    and the unconditional import so the silent fallback cannot regress.
    """

    REPO_ROOT = Path(__file__).resolve().parent.parent
    PYPROJECT = REPO_ROOT / "pyproject.toml"
    RUN_PY = REPO_ROOT / "run.py"

    @pytest.mark.skipif(tomllib is None, reason="tomllib requires Python >= 3.11")
    def test_python_dotenv_declared_in_pyproject(self):
        with open(self.PYPROJECT, "rb") as fh:
            data = tomllib.load(fh)
        declared = [
            dep.split(">=")[0].split("==")[0].strip()
            for dep in data["project"]["dependencies"]
        ]
        assert "python-dotenv" in declared, (
            "python-dotenv must be a declared runtime dependency (audit §3.2): "
            "run.py loads .env unconditionally"
        )

    def test_dotenv_importable_and_unconditional(self):
        # Hard dependency now — ImportError must propagate, not be swallowed.
        import dotenv  # noqa: F401

        source = self.RUN_PY.read_text(encoding="utf-8")
        assert "from dotenv import load_dotenv" in source
        # The old silent fallback must not guard the dotenv import.
        assert "except ImportError" not in source, (
            "run.py must import dotenv unconditionally (audit §3.2): "
            "a silent except hides a missing dependency by design"
        )

"""Smoke tests for the benchmark harness scaffolding."""

import yaml
import pytest

from core.schemas import (
    AgentState,
    AssertionRule,
    BenchmarkRunSummary,
    BenchmarkScenario,
    ConversationTurn,
    TurnResult,
)
from core.model_factory import load_model_registry, PROVIDER_MAP
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
        )
        assert summary.pass_rate == 0.7


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

    def test_all_providers_in_map(self):
        registry = load_model_registry()
        for model_id, cfg in registry.items():
            assert cfg["provider"] in PROVIDER_MAP, (
                f"{model_id} uses unknown provider: {cfg['provider']}"
            )


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


# ── Scenario Loading Tests ───────────────────────────────────────────────────


class TestScenarioLoading:
    """Verify YAML scenarios load and validate against Pydantic schemas."""

    def test_load_all_scenarios(self):
        runner = BenchmarkRunner()
        scenarios = runner.load_scenarios()
        assert len(scenarios) == 3

    def test_scenario_ids(self):
        runner = BenchmarkRunner()
        scenarios = runner.load_scenarios()
        ids = {s.id for s in scenarios}
        assert "belief_revision_01" in ids
        assert "attrition_01" in ids
        assert "needle_haystack_01" in ids

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

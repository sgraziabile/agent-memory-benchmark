"""Offline tests for USD cost reporting.

Covers the registry ``pricing:`` contract and its propagation end-to-end:

- ``core.model_factory.get_model_pricing`` — unit prices resolved from YAML
  (and the graceful ``(None, None)`` path for unpriced models).
- ``pricing`` never leaks into the provider constructor kwargs.
- ``core.runner._compute_cost_usd`` — token → USD arithmetic.
- ``BenchmarkRunner.run_single`` — per-turn cost + summary aggregation.
- ``_export_results`` — CSV/Markdown/JSON carry the cost fields.

No network: the chat model is a fake graph and the provider class is stubbed.
"""

import csv
import json
from pathlib import Path

import pytest
import yaml
from langchain_core.messages import AIMessage

from core.model_factory import (
    clear_model_cache,
    create_model,
    get_model_pricing,
    load_model_registry,
)
from core.runner import BenchmarkRunner, _compute_cost_usd
from core.schemas import BenchmarkScenario, ConversationTurn


# ── Fixtures / doubles ───────────────────────────────────────────────────────


MODELS_YAML = """
models:
  costed-model:
    provider: fake
    model_name: upstream/costed-model
    tier: apex
    timeout: 60.0
    max_retries: 2
    max_tokens: 2048
    pricing:
      input_per_mtok: 2.0
      output_per_mtok: 4.0
    description: "Priced test model"

  unpriced-model:
    provider: fake
    model_name: upstream/unpriced-model
    tier: floor
    timeout: 60.0
    max_retries: 2
    max_tokens: 2048
    description: "Test model without pricing"
"""


class FakeGraph:
    """Returns scripted AI messages carrying fixed token usage."""

    def __init__(self, responses, usage=None):
        self.responses = list(responses)
        self.usage = usage

    def invoke(self, state, config=None):
        message_kwargs = {"content": self.responses.pop(0)}
        if self.usage is not None:
            message_kwargs["usage_metadata"] = dict(self.usage)
        return {"messages": [AIMessage(**message_kwargs)]}


def make_scenario(turns=2):
    return BenchmarkScenario(
        id="cost_scenario",
        name="Cost scenario",
        description="USD reporting test",
        category="unit_test",
        turns=[
            ConversationTurn(
                turn_number=i + 1, role="user", content=f"q{i + 1}"
            )
            for i in range(turns)
        ],
    )


def make_runner(config_dir: Path, output_dir: Path) -> BenchmarkRunner:
    return BenchmarkRunner(
        config_dir=config_dir,
        dataset_dir=config_dir,
        output_dir=output_dir,
    )


@pytest.fixture
def temp_config(tmp_path):
    """Write a two-model registry into a temp config dir."""
    (tmp_path / "models.yaml").write_text(MODELS_YAML, encoding="utf-8")
    return tmp_path


# ── Registry pricing contract ────────────────────────────────────────────────


class TestGetModelPricing:
    def test_reads_declared_prices(self, temp_config):
        assert get_model_pricing("costed-model", temp_config / "models.yaml") == (
            2.0,
            4.0,
        )

    def test_missing_pricing_block_returns_none(self, temp_config):
        assert get_model_pricing("unpriced-model", temp_config / "models.yaml") == (
            None,
            None,
        )

    def test_unknown_model_returns_none(self, temp_config):
        # Inline provider:model ids are not registry entries → no pricing.
        assert get_model_pricing("fake:upstream/model", temp_config / "models.yaml") == (
            None,
            None,
        )

    def test_malformed_prices_disable_cost(self, tmp_path):
        registry = {
            "models": {
                "bad-model": {
                    "provider": "fake",
                    "model_name": "upstream/bad",
                    "pricing": {"input_per_mtok": "cheap", "output_per_mtok": 4.0},
                }
            }
        }
        path = tmp_path / "models.yaml"
        path.write_text(yaml.safe_dump(registry), encoding="utf-8")
        assert get_model_pricing("bad-model", path) == (None, None)

    def test_rebuilt_registry_prices_every_model(self):
        """Every shipped chat model must declare USD unit prices."""
        registry = load_model_registry()
        assert len(registry) == 10
        for model_id, cfg in registry.items():
            pricing = cfg.get("pricing")
            assert isinstance(pricing, dict), f"{model_id} missing pricing block"
            for key in ("input_per_mtok", "output_per_mtok"):
                value = pricing.get(key)
                assert isinstance(value, (int, float)) and value > 0, (
                    f"{model_id} has invalid pricing.{key}={value!r}"
                )


class _CapturingModel:
    """Plain class so registry extras never trip Pydantic validation."""

    def __init__(self, model: str, temperature: float = 0.0, **kwargs):
        self.model = model
        self.temperature = temperature
        self.kwargs = kwargs


class TestPricingIsHarnessMetadata:
    def test_pricing_never_reaches_provider_kwargs(self, temp_config, monkeypatch):
        clear_model_cache()
        monkeypatch.setattr(
            "core.model_factory._resolve_provider_class",
            lambda provider: _CapturingModel,
        )
        try:
            built = create_model("costed-model", config_path=temp_config / "models.yaml")
        finally:
            clear_model_cache()

        assert "pricing" not in built.kwargs
        # Ordinary YAML knobs must still be forwarded.
        assert built.kwargs["timeout"] == 60.0
        assert built.kwargs["max_tokens"] == 2048


# ── Token → USD arithmetic ───────────────────────────────────────────────────


class TestComputeCostUsd:
    def test_million_token_units(self):
        cost = _compute_cost_usd(1_000_000, 1_000_000, 2.0, 4.0)
        assert cost == {
            "input_cost_usd": 2.0,
            "output_cost_usd": 4.0,
            "total_cost_usd": 6.0,
        }

    def test_fractional_turn(self):
        cost = _compute_cost_usd(1_500, 500, 0.10, 0.60)
        assert cost["input_cost_usd"] == pytest.approx(0.00015)
        assert cost["output_cost_usd"] == pytest.approx(0.0003)
        assert cost["total_cost_usd"] == pytest.approx(0.00045)

    def test_unpriced_model_yields_zero(self):
        assert _compute_cost_usd(1_000_000, 1_000_000, None, None) == {
            "input_cost_usd": 0.0,
            "output_cost_usd": 0.0,
            "total_cost_usd": 0.0,
        }


# ── Runner integration ───────────────────────────────────────────────────────


USAGE = {
    "input_tokens": 1_000_000,
    "output_tokens": 1_000_000,
    "total_tokens": 2_000_000,
}


class TestRunSingleCost:
    def test_turn_and_summary_costs(self, temp_config, tmp_path):
        runner = make_runner(temp_config, tmp_path)
        graph = FakeGraph(["a1", "a2"], usage=USAGE)
        summary = runner.run_single(
            make_scenario(2), "baseline_react", "sys", "costed-model", graph, "0"
        )

        assert summary.input_cost_per_mtok == 2.0
        assert summary.output_cost_per_mtok == 4.0
        assert len(summary.turn_results) == 2
        for tr in summary.turn_results:
            assert tr.input_cost_usd == pytest.approx(2.0)
            assert tr.output_cost_usd == pytest.approx(4.0)
            assert tr.total_cost_usd == pytest.approx(6.0)

        assert summary.input_cost_usd == pytest.approx(4.0)
        assert summary.output_cost_usd == pytest.approx(8.0)
        assert summary.total_cost_usd == pytest.approx(12.0)

    def test_unpriced_model_reports_no_unit_prices(self, temp_config, tmp_path):
        runner = make_runner(temp_config, tmp_path)
        graph = FakeGraph(["a1"], usage=USAGE)
        summary = runner.run_single(
            make_scenario(1), "baseline_react", "sys", "unpriced-model", graph, "0"
        )

        assert summary.input_cost_per_mtok is None
        assert summary.output_cost_per_mtok is None
        assert summary.total_cost_usd == 0.0
        assert summary.turn_results[0].total_cost_usd == 0.0

    def test_model_outside_registry_has_no_cost(self, temp_config, tmp_path):
        runner = make_runner(temp_config, tmp_path)
        graph = FakeGraph(["a1"], usage=USAGE)
        summary = runner.run_single(
            make_scenario(1), "baseline_react", "sys", "fake:inline-model", graph, "0"
        )
        assert summary.total_cost_usd == 0.0
        assert summary.input_cost_per_mtok is None


# ── Exports ──────────────────────────────────────────────────────────────────


class TestCostExports:
    def _summary(self, temp_config, tmp_path):
        runner = make_runner(temp_config, tmp_path)
        graph = FakeGraph(["a1", "a2"], usage=USAGE)
        return runner, runner.run_single(
            make_scenario(2), "baseline_react", "sys", "costed-model", graph, "0"
        )

    def test_csv_has_cost_columns(self, temp_config, tmp_path):
        runner, summary = self._summary(temp_config, tmp_path)
        run_dir = runner._export_results([summary], run_dir=tmp_path / "out")

        with open(run_dir / "summary.csv", encoding="utf-8-sig", newline="") as fh:
            row = next(csv.DictReader(fh))

        assert float(row["input_cost_usd"]) == pytest.approx(4.0)
        assert float(row["output_cost_usd"]) == pytest.approx(8.0)
        assert float(row["total_cost_usd"]) == pytest.approx(12.0)
        assert float(row["input_cost_per_mtok"]) == pytest.approx(2.0)
        assert float(row["output_cost_per_mtok"]) == pytest.approx(4.0)

    def test_markdown_has_cost_column(self, temp_config, tmp_path):
        runner, summary = self._summary(temp_config, tmp_path)
        run_dir = runner._export_results([summary], run_dir=tmp_path / "out")

        markdown = (run_dir / "summary.md").read_text(encoding="utf-8")
        assert "Cost (USD)" in markdown
        assert "$12.000000" in markdown

    def test_json_carries_cost(self, temp_config, tmp_path):
        runner, summary = self._summary(temp_config, tmp_path)
        run_dir = runner._export_results([summary], run_dir=tmp_path / "out")

        payload = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
        assert payload[0]["total_cost_usd"] == pytest.approx(12.0)
        assert payload[0]["turn_results"][0]["total_cost_usd"] == pytest.approx(6.0)


"""Per-model token usage report: reasoning vs visible content tokens.

Scans every results.json under outputs/ (any layout: runs/ or by_theme/)
and aggregates the token_usage recorded per turn by the benchmark runner.

Key fields (LangChain usage_metadata, as persisted by core.runner):
  input_tokens                    - prompt tokens billed
  output_tokens                   - ALL output tokens (reasoning + content)
  output_token_details.reasoning  - hidden chain-of-thought tokens
  input_token_details.cache_read  - prompt-cache hits

content tokens = output_tokens - reasoning.

Usage:  python scripts/token_report.py
"""
import glob
import json
import os

agg: dict[str, dict] = {}


def bucket(model_id: str) -> dict:
    return agg.setdefault(
        model_id,
        {"turns": 0, "in": 0, "out": 0, "reasoning": 0, "cached_in": 0},
    )


files = glob.glob(os.path.join("outputs", "**", "results.json"), recursive=True)
for path in sorted(files):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    for run in data if isinstance(data, list) else [data]:
        for turn in run["turn_results"]:
            u = turn.get("token_usage") or {}
            if not u:
                continue
            b = bucket(run["model_id"])
            b["turns"] += 1
            b["in"] += u.get("input_tokens", u.get("prompt_tokens", 0))
            b["out"] += u.get("output_tokens", u.get("completion_tokens", 0))
            out_det = u.get("output_token_details") or {}
            b["reasoning"] += out_det.get("reasoning") or 0
            in_det = u.get("input_token_details") or {}
            b["cached_in"] += in_det.get("cache_read") or 0

print(f"scanned {len(files)} results.json file(s)\n")
hdr = (
    f"{'model':<18} {'turns':>5} {'input':>8} {'output':>8} "
    f"{'reasoning':>9} {'content':>8} {'%reason':>8}"
)
print(hdr)
print("-" * len(hdr))
for model, b in sorted(agg.items()):
    content = b["out"] - b["reasoning"]
    pct = (b["reasoning"] / b["out"] * 100) if b["out"] else 0.0
    print(
        f"{model:<18} {b['turns']:>5} {b['in']:>8} {b['out']:>8} "
        f"{b['reasoning']:>9} {content:>8} {pct:>7.1f}%"
    )

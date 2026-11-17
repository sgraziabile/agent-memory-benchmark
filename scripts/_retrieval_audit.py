#!/usr/bin/env python3
"""Temporary retrieval audit: was the CURRENT-value chunk retrieved? (safe to delete)."""

import json
from pathlib import Path

RUNS = [
    ("deepseek", 2, "20261106_111310"),
    ("deepseek", 5, "20261106_105814"),
    ("deepseek", 8, "20261106_105326"),
    ("mistral", 2, "20261106_111528"),
    ("mistral", 5, "20261106_105711"),
    ("mistral", 8, "20261106_105452"),
    ("nemotron", 5, "20261106_111004"),
    ("nemotron", 8, "20261106_111137"),
]

QZX = "QZX-7P4K-1180"   # current value (chat-only)
YKX = "YKX-HMD0-7872"   # stale value (document + old chat echo)
BQR = "BQR-485R-6500"   # LOOP-969 document value


def audit(mems, token):
    ids = [m.get("chunk_id") for m in mems]
    has_token = any(token in (m.get("content") or "") for m in mems)
    return ids, has_token


for model, k, name in RUNS:
    run = json.loads(
        (Path("outputs/runs") / name / "results.json").read_text(encoding="utf-8")
    )[0]
    turns = {t["turn_number"]: t for t in run["turn_results"]}
    print("=" * 100)
    print(f"{name} {model} k={k}  pass={run['passed_assertions']}/{run['total_assertions']}")

    for tn in (7, 8, 10):
        tr = turns[tn]
        mems = tr.get("retrieved_memories") or []
        ids = [m.get("chunk_id") for m in mems]
        ans = tr.get("agent_response") or ""
        if tn == 8:
            token, label = BQR, "BQR(LOOP-969 doc value)"
        else:
            token, label = QZX, "QZX(current LOOP-420)"
        has_tok = any(token in (m.get("content") or "") for m in mems)
        has_old = any(YKX in (m.get("content") or "") for m in mems) if tn != 8 else None
        print(
            f"  T{tn}: ep_s1_t2={'ep_s1_t2' in ids}  "
            f"{label}_in_window={has_tok}  "
            + (f"old(YKX)_in_window={has_old}  " if has_old is not None else "")
            + f"answer_ok={token in ans}"
        )
        print(f"        ids={ids}")

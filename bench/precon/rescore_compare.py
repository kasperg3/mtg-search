"""Before/after table for the review-m3 rescore: every paired comparison the write-ups report.

    uv run python -m bench.precon.rescore_compare --out bench/precon/runs/rescore_m3_compare.json

Uses each episode's ``metrics_before_rescore`` (old scorer) and ``metrics`` (current scorer), so no
scoring is repeated. Diff = arm - base, paired over precons (repeats averaged), percentile bootstrap
CI and two-sided sign test, as in ``run.compare``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from bench.precon import scorer  # noqa: E402
from bench.precon.run import RUNS  # noqa: E402

P = "20261014_"
PAIRS = [  # (experiment, base run, arm run)
    ("EXP-007", "cc-sonnet55_none_test", "cc-sonnet55_mcp_test"),
    ("EXP-007", "cc-sonnet55_none_test", "cc-sonnet55_mcp-hint_test"),
    ("EXP-009", "cc-sonnet55_shipped-v1_test", "cc-sonnet55_shipped-v2_test"),
    ("EXP-009", "cc-sonnet55_shipped-v1_test", "cc-sonnet55_shipped-v3_test"),
    ("EXP-009", "cc-sonnet55_shipped-v1_test", "cc-sonnet55_shipped-v3al-deferred_test"),
    ("EXP-009", "cc-sonnet55_shipped-v1_test", "cc-sonnet55_shipped-v2-sphint_test"),
    ("EXP-010", "smoke-add_shipped", "smoke-add_e1desc"),
    ("EXP-010", "smoke-add_shipped", "smoke-add_e6desc_slim"),
    ("EXP-010", "cc-sonnet55_mcp-hint_test", "smoke-add_lexical-hint"),
    ("EXP-011", "exp011-test_v1", "exp011-test_shipped"),
    ("EXP-011 dev", "exp011-dev_shipped", "exp011-dev_e6desc_slim"),
    ("EXP-011 dev", "exp011-dev_shipped", "exp011-dev_e1desc"),
]
METRICS = ["cut.ndcg@10", "add.ndcg@10", "add.ndcg@20", "add.p@20", "swap.cut.ndcg@10", "swap.add.ndcg@10"]


def load(rid: str) -> list[dict]:
    return [json.loads(x) for x in (RUNS / f"{P}{rid}" / "episodes.jsonl").read_text().splitlines()]


def pair(base: list[dict], arm: list[dict], metric: str, key: str) -> dict | None:
    task = metric.split(".")[0]
    pick = lambda eps: scorer.per_precon([{**e, "metrics": e[key]} for e in eps if e["task"] == task], metric)  # noqa: E731
    a, b = pick(arm), pick(base)
    if not a or not b:
        return None
    r = scorer.paired_bootstrap(a, b, 10_000)
    return {"n": r["n"], "mean": r["mean_diff"], "ci": r["ci95"], "sign_test_p": r["sign_test_p"]} if r.get("n") else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = []
    for exp, base_id, arm_id in PAIRS:
        base, arm = load(base_id), load(arm_id)
        for m in METRICS:
            old = pair(base, arm, m, "metrics_before_rescore")
            new = pair(base, arm, m, "metrics")
            if not old or not new or not old.get("n"):
                continue
            rows.append({"experiment": exp, "base": base_id, "arm": arm_id, "metric": m, "old": old, "new": new})
    for r in rows:
        o, n = r["old"], r["new"]
        flip = (o["ci"][0] > 0) != (n["ci"][0] > 0) or (o["ci"][1] < 0) != (n["ci"][1] < 0) or \
            ((o["sign_test_p"] < 0.05) != (n["sign_test_p"] < 0.05))
        print(f"{r['experiment']:11s} {r['arm'][:30]:30s} - {r['base'][:26]:26s} {r['metric']:17s} "
              f"old {o['mean']:+.3f} [{o['ci'][0]:+.3f},{o['ci'][1]:+.3f}] p={o['sign_test_p']:.3f} | "
              f"new {n['mean']:+.3f} [{n['ci'][0]:+.3f},{n['ci'][1]:+.3f}] p={n['sign_test_p']:.3f}{'  <-- CHANGED' if flip else ''}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()

"""EXP-013 analysis: search infrastructure (A) vs grep only (G) vs everything in context (I).

    uv run python eval/agentic/analyze_cc.py --runs cc-main --runs cc-main-I --out eval/agentic/results/cc_main.json

Reuses analyze.py's metric definitions (original task labels). Everything is reported per label
stratum, because how the labels were made favours one arm or another (infra_vs_none.md):
deck/deckneeds (tags or regex), deck/paraphrase-need (regex only: favours grep), attr (tag
closure), known (25, chosen where single-shot BM25 missed), known-easy (8, BM25 top 3).
Tokens: total = input + cache writes + cache reads + output, per episode, from Claude Code usage.
Paired comparisons over tasks: mean difference, 10k bootstrap CI, two-sided sign test.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analyze import episode_metrics, label_sets  # noqa: E402

TASK_FILES = ("tasks.yaml", "tasks_known_easy.yaml")
STRATA = {  # name -> predicate on task
    "deck/deckneeds (tags|regex)": lambda t: t["kind"] == "deck" and t["source"] == "deckneeds",
    "deck/paraphrase (regex only)": lambda t: t["kind"] == "deck" and t["source"] == "paraphrase-need",
    "attr (tag closure)": lambda t: t["kind"] == "attr",
    "known (25)": lambda t: t["kind"] == "known" and not t["source"].startswith("known-easy"),
    "known-easy (8)": lambda t: t["source"].startswith("known-easy"),
    "known, all 33": lambda t: t["kind"] == "known",
    "all tasks": lambda t: True,
}
QUALITY = {"deck": ["ndcg@10", "p@10", "recall@20_final"], "attr": ["ndcg@10", "p@10", "recall@20_final"],
           "known": ["found_in_final", "rr"]}
TOKENS = ["total_tokens", "tok_input", "tok_cache_write", "tok_cache_read", "tok_output", "tok_thinking", "turns", "wall_s",
          "tool_calls", "notional_usd"]
PAIRS = (("A", "G"), ("A", "I"), ("G", "I"))


def sign_test(d: np.ndarray) -> float:
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def paired(rows: list[dict], metric: str, a: str, b: str, pred, tasks, n: int = 10000, seed: int = 0) -> dict | None:
    va = {r["task_id"]: r[metric] for r in rows if r["cond"] == a and pred(tasks[r["task_id"]]) and _ok(r.get(metric))}
    vb = {r["task_id"]: r[metric] for r in rows if r["cond"] == b and pred(tasks[r["task_id"]]) and _ok(r.get(metric))}
    common = sorted(set(va) & set(vb))
    if not common:
        return None
    d = np.array([vb[t] - va[t] for t in common], dtype=float)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    ra = np.array([va[t] for t in common], dtype=float)
    rb = np.array([vb[t] for t in common], dtype=float)
    return {"n": len(common), "mean_a": float(ra.mean()), "mean_b": float(rb.mean()), "diff": float(d.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "b_better": int((d > 0).sum()), "a_better": int((d < 0).sum()), "p_sign": sign_test(d),
            "median_ratio_b_over_a": float(np.median(rb / np.where(ra == 0, np.nan, ra))) if metric in TOKENS else None}


def _ok(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", action="append", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tasks = {}
    for f in TASK_FILES:
        for t in yaml.safe_load((HERE / f).read_text()):
            tasks[t["id"]] = t
    eps = []
    for r in args.runs:
        for line in (HERE / "runs" / r / "episodes.jsonl").read_text().splitlines():
            e = json.loads(line)
            if e["task_id"] in tasks:
                eps.append(e)
    labels, _, _ = label_sets(tasks)
    import polars as pl

    c = pl.read_parquet(HERE.parents[1] / "data" / "cards.parquet", columns=["oracle_id", "tags"])
    tagged = {o for o, t in zip(c["oracle_id"].to_list(), c["tags"].to_list(), strict=True) if t}
    rows = []
    for e in eps:
        t = tasks[e["task_id"]]
        m = episode_metrics(e, t, labels[e["task_id"]], tagged)
        if e["cond"] == "I":  # no tool calls: retrieval-at-t and "unretrieved" are undefined
            for k in list(m):
                if k.startswith(("recall@", "success@")) and k != "recall@20_final" or k in ("final_unretrieved", "novel_per_iter"):
                    m[k] = float("nan")
        u = e.get("usage_detail") or {}
        m.update(total_tokens=e.get("total_tokens"), tok_input=u.get("input"), tok_cache_write=u.get("cache_write"),
                 tok_cache_read=u.get("cache_read"), tok_output=u.get("output"), tok_thinking=u.get("thinking"),
                 turns=e.get("llm_calls"), wall_s=e.get("wall_s"), tool_calls=len(e["searches"]),
                 notional_usd=e.get("notional_usd"), stop=e.get("stop"), final_len=len(e["final"]))
        rows.append({"task_id": e["task_id"], "cond": e["cond"], "kind": t["kind"], **m})
    conds = [c for c in ("A", "G", "I") if any(r["cond"] == c for r in rows)]
    res = {"runs": args.runs, "n_episodes": len(rows), "conds": conds, "strata": {}}
    for sname, pred in STRATA.items():
        kinds = {tasks[r["task_id"]]["kind"] for r in rows if pred(tasks[r["task_id"]])}
        qmets = sorted({m for k in kinds for m in QUALITY[k]}) if sname != "all tasks" else []
        s = {"summary": {}, "paired": {}}
        for cnd in conds:
            rr = [r for r in rows if r["cond"] == cnd and pred(tasks[r["task_id"]])]
            if not rr:
                continue
            s["summary"][cnd] = {"n": len(rr), **{m: float(np.nanmean([r[m] for r in rr if _ok(r.get(m))])) if any(_ok(r.get(m)) for r in rr) else None
                                                   for m in qmets + TOKENS},
                                 "median_total_tokens": float(np.median([r["total_tokens"] for r in rr]))}
        for a, b in PAIRS:
            for m in qmets + ["total_tokens", "turns", "wall_s"]:
                p = paired(rows, m, a, b, pred, tasks)
                if p:
                    s["paired"][f"{b}-{a}:{m}"] = p  # holds b - a
        res["strata"][sname] = s
    res["episodes"] = rows
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1, default=float) + "\n")
    for sname, s in res["strata"].items():
        print(f"\n## {sname}")
        for cnd, v in s["summary"].items():
            q = " ".join(f"{m}={v[m]:.3f}" for m in v if m in ("ndcg@10", "p@10", "recall@20_final", "found_in_final", "rr") and v[m] is not None)
            print(f"  {cnd} n={v['n']:3d} {q} | tokens mean={v['total_tokens']:,.0f} median={v['median_total_tokens']:,.0f} "
                  f"turns={v['turns']:.2f} calls={v['tool_calls']:.2f} wall={v['wall_s']:.1f}s out={v['tok_output']:,.0f}")
        for k, p in s["paired"].items():
            extra = f" ratio(b/a) median={p['median_ratio_b_over_a']:.2f}" if p.get("median_ratio_b_over_a") is not None else ""
            print(f"  {k:28s} n={p['n']:3d} {p['mean_a']:.3f} -> {p['mean_b']:.3f} diff {p['diff']:+.3f} "
                  f"[{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] b>a {p['b_better']} a>b {p['a_better']} p_sign={p['p_sign']:.3g}{extra}")


if __name__ == "__main__":
    main()

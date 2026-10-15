"""Analyse the pooled-judging smoke (judge_cc.py): self-consistency, agreement with the original labels,
missing labels, effect on the A vs G comparison, and cost.

    uv run python eval/agentic/judge_smoke_analyze.py --run judge-smoke --pool-runs cc-main
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cc_agent import load_tasks  # noqa: E402


def kappa(a: list[int], b: list[int], cats=(0, 1, 2), weighted: bool = False) -> float:
    n = len(a)
    w = (lambda i, j: 1 - abs(i - j) / (len(cats) - 1)) if weighted else (lambda i, j: float(i == j))
    po = sum(w(x, y) for x, y in zip(a, b, strict=True)) / n
    pa = {c: a.count(c) / n for c in cats}
    pb = {c: b.count(c) / n for c in cats}
    pe = sum(pa[i] * pb[j] * w(i, j) for i in cats for j in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="judge-smoke")
    ap.add_argument("--pool-runs", default="cc-main")
    args = ap.parse_args()
    d = HERE / "runs" / args.run
    rows = [json.loads(x) for x in (d / "judgments.jsonl").read_text().splitlines()]
    calls = [json.loads(x) for x in (d / "calls.jsonl").read_text().splitlines()]
    tasks = {t["id"]: t for t in load_tasks()}
    g = defaultdict(dict)  # (task, card) -> pass -> grade
    lab = {}
    for r in rows:
        g[(r["task_id"], r["oracle_id"])][r["pass"]] = r["grade"]
        lab[(r["task_id"], r["oracle_id"])] = r["labelled"]
    both = [k for k, v in g.items() if 0 in v and 1 in v]
    p0, p1 = [g[k][0] for k in both], [g[k][1] for k in both]
    cons = {"n_cards": len(both), "exact_agreement": sum(a == b for a, b in zip(p0, p1, strict=True)) / len(both),
            "binary_agreement": sum((a >= 1) == (b >= 1) for a, b in zip(p0, p1, strict=True)) / len(both),
            "kappa": kappa(p0, p1), "weighted_kappa": kappa(p0, p1, weighted=True)}
    # consensus: relevant if both passes grade >= 1; grade = min of the two passes (conservative)
    cg = {k: min(g[k][0], g[k][1]) for k in both}
    labelled = [k for k in both if lab[k]]
    unlabelled = [k for k in both if not lab[k]]
    agree = {
        "labelled_pooled": len(labelled), "unlabelled_pooled": len(unlabelled),
        "labelled_judged_relevant": sum(cg[k] >= 1 for k in labelled) / len(labelled),
        "unlabelled_judged_relevant": sum(cg[k] >= 1 for k in unlabelled) / len(unlabelled),
        "unlabelled_judged_grade2": sum(cg[k] == 2 for k in unlabelled) / len(unlabelled),
        "kappa_label_vs_judge": kappa([int(lab[k]) for k in both], [int(cg[k] >= 1) for k in both], cats=(0, 1)),
    }
    per_task = {}
    for tid in sorted({k[0] for k in both}):
        ks = [k for k in both if k[0] == tid]
        per_task[tid] = {"source": tasks[tid]["source"], "pooled": len(ks), "labelled": sum(lab[k] for k in ks),
                         "labelled_judged_rel": sum(lab[k] and cg[k] >= 1 for k in ks),
                         "unlabelled_judged_rel": sum((not lab[k]) and cg[k] >= 1 for k in ks)}
    # effect on A vs G: P@10 of the final lists with original vs pooled labels (original OR judged relevant)
    eff = defaultdict(lambda: defaultdict(list))
    for r_name in args.pool_runs.split(","):
        for line in (HERE / "runs" / r_name / "episodes.jsonl").read_text().splitlines():
            e = json.loads(line)
            if e["task_id"] not in per_task:
                continue
            rel = set(tasks[e["task_id"]]["relevant"])
            top = [o for o in e["final_oids"] if o][:10]
            pooled_rel = rel | {o for (t, o) in both if t == e["task_id"] and cg[(t, o)] >= 1}
            eff[e["cond"]]["p10_orig"].append(sum(o in rel for o in top) / 10)
            eff[e["cond"]]["p10_pooled"].append(sum(o in pooled_rel for o in top) / 10)
    effect = {c: {k: round(st.mean(v), 3) for k, v in m.items()} for c, m in eff.items()}
    tok = [sum(c["usage"][k] for k in ("input", "cache_write", "cache_read", "output")) for c in calls]
    cost = {"calls": len(calls), "tokens_per_call_mean": round(st.mean(tok)), "tokens_total": sum(tok),
            "output_tokens_total": sum(c["usage"]["output"] for c in calls),
            "wall_s_per_call_mean": round(st.mean(c["wall_s"] for c in calls), 1),
            "complete_parses": sum(c["n_graded"] == c["n_cards"] for c in calls),
            "notional_usd": round(sum(c["usage"].get("notional_usd", 0) for c in calls), 3)}
    out = {"consistency": cons, "agreement_with_labels": agree, "per_task": per_task, "a_vs_g_p10": effect, "cost": cost}
    dpath = HERE / "runs" / f"{args.run}-decoys" / "judgments.jsonl"
    if dpath.exists():  # blind negative controls mixed into the batches (judge_cc.py --decoys-*)
        by = defaultdict(list)
        for x in dpath.read_text().splitlines():
            r = json.loads(x)
            by[r.get("kind", "pool")].append(r["grade"])
        out["controls"] = {k: {"n": len(v), "judged_relevant": round(sum(x >= 1 for x in v) / len(v), 3),
                               "grade2": round(sum(x == 2 for x in v) / len(v), 3)} for k, v in by.items()}
    (d / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

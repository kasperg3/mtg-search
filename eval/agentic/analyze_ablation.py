"""EXP-018 analysis: prompt ablation of EXP-013's search (A) vs grep (G) (prompt_ablation.md).

    uv run python eval/agentic/analyze_ablation.py --run cc-ablation --judge-runs judge-full,judge-ablation

A0 / G0 are A / G with identical minimal prompts and mechanics-only tool descriptions (G0 matched truly
line by line); A / G are the EXP-013 episodes (``runs/cc-main``) on the same 30 tasks.

Relevance (deck and attr tasks): a card in the deduplicated top 10 counts if it is on the task's original
list or a judge run graded it >= 1 (EXP-016 rule); judge runs later in ``--judge-runs`` override earlier ones.
Find-this-card tasks: target in the final list, and its reciprocal rank. Paired over tasks: mean difference,
10k bootstrap 95% CI (seed 0), two-sided sign test. Tokens: per-task ratio, median and sign test.
Output: ``runs/<run>/summary.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analyze import dedupe, label_sets  # noqa: E402
from analyze_cc import sign_test  # noqa: E402
from cc_agent import load_tasks  # noqa: E402

SEED = 18
PER_KIND = 10
LAYOUT = re.compile(r"\^|\$|\\n|\\s\\S|Keywords:|\\\{")


def sample_ids(tasks: list[dict]) -> list[str]:
    """The pre-registered sample: 10 tasks per kind, ``random.Random(18)`` over sorted ids."""
    rng = random.Random(SEED)
    return [i for k in ("deck", "attr", "known") for i in rng.sample(sorted(t["id"] for t in tasks if t["kind"] == k), PER_KIND)]


def paired(va: dict, vb: dict, n: int = 10000, seed: int = 0) -> dict | None:
    """b - a over common tasks."""
    common = sorted(t for t in set(va) & set(vb) if not (math.isnan(va[t]) or math.isnan(vb[t])))
    if not common:
        return None
    a, b = (np.array([v[t] for t in common], dtype=float) for v in (va, vb))
    d = b - a
    boots = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(axis=1)
    return {"n": len(common), "mean_a": float(a.mean()), "mean_b": float(b.mean()), "diff": float(d.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "b_better": int((d > 0).sum()), "a_better": int((d < 0).sum()), "ties": int((d == 0).sum()), "p_sign": sign_test(d)}


def ratio(va: dict, vb: dict) -> dict:
    """Per-task b / a token ratio: median, share of tasks where b used more, sign test."""
    common = sorted(set(va) & set(vb))
    r = [vb[t] / va[t] for t in common]
    d = np.array([vb[t] - va[t] for t in common], dtype=float)
    return {"n": len(common), "median_ratio": st.median(r), "mean_a": st.mean(va[t] for t in common),
            "mean_b": st.mean(vb[t] for t in common), "b_more": int((d > 0).sum()), "p_sign": sign_test(d)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="cc-ablation")
    ap.add_argument("--base-run", default="cc-main")
    ap.add_argument("--judge-runs", default="judge-full,judge-ablation")
    args = ap.parse_args()
    all_tasks = load_tasks()
    tasks = {t["id"]: t for t in all_tasks}
    ids = sample_ids(all_tasks)
    orig, _, _ = label_sets(tasks)
    grade = {}
    for r in args.judge_runs.split(","):
        p = HERE / "runs" / r / "judgments.jsonl"
        if p.exists():
            for j in map(json.loads, p.read_text().splitlines()):
                if j.get("kind", "pool") == "pool":
                    grade[(j["task_id"], j["oracle_id"])] = j["grade"]
    eps = {}
    for run, conds in ((args.base_run, ("A", "G")), (args.run, ("A0", "G0"))):
        for e in map(json.loads, (HERE / "runs" / run / "episodes.jsonl").read_text().splitlines()):
            if e["task_id"] in ids and e["cond"] in conds and e["rep"] == 0:
                eps[(e["cond"], e["task_id"])] = e
    conds = ("A", "G", "A0", "G0")
    missing = [(c, t) for c in conds for t in ids if (c, t) not in eps]
    m: dict[str, dict[str, dict[str, float]]] = {c: {} for c in conds}
    unjudged = []
    for (c, t), e in eps.items():
        final = dedupe(e["final_oids"])[:20]
        row = {"tokens": e["total_tokens"], "turns": e["llm_calls"], "tool_calls": len(e["searches"]), "wall_s": e["wall_s"],
               "final_len": len(final)}
        if tasks[t]["kind"] == "known":
            tgt = tasks[t]["relevant"][0]
            row.update(found=float(tgt in final), rr=1 / (final.index(tgt) + 1) if tgt in final else 0.0)
        else:
            top = final[:10]
            unjudged += [(c, t, o) for o in top if o not in orig[t] and (t, o) not in grade]
            row["p@10"] = sum(o in orig[t] or grade.get((t, o), 0) >= 1 for o in top) / 10
        for k, v in row.items():
            m[c].setdefault(k, {})[t] = v
    ret = [t for t in ids if tasks[t]["kind"] != "known"]
    known = [t for t in ids if tasks[t]["kind"] == "known"]

    def sub(c, k, keep):
        return {t: v for t, v in m[c].get(k, {}).items() if t in keep}

    out = {"experiment": "EXP-018", "run": args.run, "tasks": ids, "missing_episodes": missing, "unjudged_top10": len(unjudged),
           "means": {c: {k: st.mean(v.values()) for k, v in m[c].items()} for c in conds},
           "primary_minimal": {"tokens_G0_over_A0": ratio(m["A0"]["tokens"], m["G0"]["tokens"]),
                               "p@10_G0_minus_A0": paired(sub("A0", "p@10", ret), sub("G0", "p@10", ret)),
                               "found_G0_minus_A0": paired(sub("A0", "found", known), sub("G0", "found", known))},
           "exp013_same_tasks": {"tokens_G_over_A": ratio(m["A"]["tokens"], m["G"]["tokens"]),
                                 "p@10_G_minus_A": paired(sub("A", "p@10", ret), sub("G", "p@10", ret)),
                                 "found_G_minus_A": paired(sub("A", "found", known), sub("G", "found", known))},
           "prompt_effect": {}}
    for a, b in (("A", "A0"), ("G", "G0")):
        out["prompt_effect"][f"{b}_vs_{a}"] = {
            "tokens": ratio(m[a]["tokens"], m[b]["tokens"]), "turns": paired(m[a]["turns"], m[b]["turns"]),
            "tool_calls": paired(m[a]["tool_calls"], m[b]["tool_calls"]),
            "p@10": paired(sub(a, "p@10", ret), sub(b, "p@10", ret)), "found": paired(sub(a, "found", known), sub(b, "found", known))}
    # does the grep - search gap change? per task (G0 - A0) - (G - A); tokens on the log ratio
    gap = {}
    for k, keep in (("p@10", ret), ("found", known)):
        d0 = {t: m["G0"][k][t] - m["A0"][k][t] for t in keep if t in m["G0"].get(k, {}) and t in m["A0"].get(k, {})}
        d1 = {t: m["G"][k][t] - m["A"][k][t] for t in keep if t in m["G"].get(k, {}) and t in m["A"].get(k, {})}
        gap[k] = paired(d1, d0)
    lt = {c: {t: math.log(v) for t, v in m[c]["tokens"].items()} for c in conds}
    gap["log_token_ratio"] = paired({t: lt["G"][t] - lt["A"][t] for t in ids if t in lt["G"] and t in lt["A"]},
                                    {t: lt["G0"][t] - lt["A0"][t] for t in ids if t in lt["G0"] and t in lt["A0"]})
    out["gap_change_minimal_minus_exp013"] = gap
    diag = {}
    for c in ("G", "G0"):
        pats = [s for (cc, t), e in eps.items() if cc == c for s in e["searches"]]
        diag[c] = {"calls": len(pats), "layout_share": sum(bool(LAYOUT.search(s["query"])) for s in pats) / len(pats),
                   "newline_share": sum("\\n" in s["query"] for s in pats) / len(pats),
                   "zero_match_share": sum((s.get("n_match") or 0) == 0 for s in pats) / len(pats),
                   "offset_share": sum((s.get("offset") or 0) > 0 for s in pats) / len(pats),
                   "error_share": sum("error" in s for s in pats) / len(pats),
                   "alternation_share": sum("|" in s["query"] for s in pats) / len(pats)}
    out["grep_diagnostics"] = diag
    (HERE / "runs" / args.run / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("missing_episodes", "unjudged_top10", "means")}, indent=1))
    for sec in ("primary_minimal", "exp013_same_tasks", "prompt_effect", "gap_change_minimal_minus_exp013", "grep_diagnostics"):
        print(f"== {sec}\n{json.dumps(out[sec], indent=1)}")


if __name__ == "__main__":
    main()

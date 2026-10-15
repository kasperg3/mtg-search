"""EXP-016 analysis: EXP-013's search (A) vs grep (G) under pooled labels (pooled_judging.md).

    uv run python eval/agentic/analyze_pooled.py --judge-runs judge-full [--pool-run cc-main]

Labels, fixed in the pre-registration:
  original      : analyze.label_sets (relevant -> gain 1, or the task's ``graded`` value);
  pooled        : original relevant OR judged grade >= 1 (a judged 0 never removes an original label);
  pooled graded : original grade if labelled, else the judged grade (nDCG).
Random decoys (``kind = decoy_random``) are judge diagnostics only, never labels. ``--judge-runs`` takes
several comma-separated runs (a resume), later runs overriding earlier ones per (task, card).

Metrics per episode on the deduplicated final list: P@10 and nDCG@10 (original and pooled), and the judged
share of the top 10 (of the top-10 cards with a judgment, the share graded >= 1). Final R@20 is not reported:
under pooling its denominator depends on what the agents found. Paired over tasks, G - A, 10k bootstrap
95% CI (seed 0) and two-sided sign test, for all 80 tasks and per stratum. Output: runs/<last judge run>/summary.json.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analyze import dedupe, label_sets, ndcg  # noqa: E402
from analyze_cc import sign_test  # noqa: E402
from cc_agent import load_tasks  # noqa: E402

STRATA = {
    "all 80": lambda t: t["kind"] in ("deck", "attr"),
    "deck/deckneeds": lambda t: t["kind"] == "deck" and t["source"] == "deckneeds",
    "deck/paraphrase": lambda t: t["kind"] == "deck" and t["source"] == "paraphrase-need",
    "attr": lambda t: t["kind"] == "attr",
}
METRICS = ["p@10_pooled", "ndcg@10_pooled", "judged_share@10", "p@10_orig", "ndcg@10_orig",
           "p@10_pooled_strict"]  # strict = exploratory, not pre-registered: new cards need judged grade 2
TOK = ("input", "cache_write", "cache_read", "output")


def paired(va: dict, vb: dict, n: int = 10000, seed: int = 0) -> dict | None:
    common = sorted(t for t in set(va) & set(vb) if not (math.isnan(va[t]) or math.isnan(vb[t])))
    if not common:
        return None
    a = np.array([va[t] for t in common], dtype=float)
    b = np.array([vb[t] for t in common], dtype=float)
    d = b - a
    boots = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(axis=1)
    return {"n": len(common), "mean_A": float(a.mean()), "mean_G": float(b.mean()), "diff_G_minus_A": float(d.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "G_better": int((d > 0).sum()), "A_better": int((d < 0).sum()), "ties": int((d == 0).sum()), "p_sign": sign_test(d)}


def rate(xs: list[int], cond) -> float | None:
    return round(sum(map(cond, xs)) / len(xs), 4) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-runs", default="judge-full")
    ap.add_argument("--pool-run", default="cc-main")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    judge_runs = args.judge_runs.split(",")
    tasks = {t["id"]: t for t in load_tasks()}
    ids = [t for t in tasks if STRATA["all 80"](tasks[t])]
    stratum = {t: next(s for s in list(STRATA)[1:] if STRATA[s](tasks[t])) for t in ids}
    orig, _, _ = label_sets(tasks)

    # judgments (one pass; a later run overrides an earlier one for the same task and card)
    grade, kind = {}, {}
    for r in judge_runs:
        for line in (HERE / "runs" / r / "judgments.jsonl").read_text().splitlines():
            j = json.loads(line)
            grade[(j["task_id"], j["oracle_id"])] = j["grade"]
            kind[(j["task_id"], j["oracle_id"])] = j.get("kind", "pool")
    pool_grade = {k: g for k, g in grade.items() if kind[k] == "pool"}
    decoy = {k: g for k, g in grade.items() if kind[k] == "decoy_random"}

    # episodes and the pool (union of A and G final top-20, as judge_cc.build_pool)
    eps = [json.loads(x) for x in (HERE / "runs" / args.pool_run / "episodes.jsonl").read_text().splitlines()]
    eps = [e for e in eps if e["task_id"] in tasks and e["task_id"] in set(ids) and e["cond"] in ("A", "G")]
    picked = defaultdict(lambda: defaultdict(set))  # task -> card -> {conds}
    for e in eps:
        for o in e["final_oids"][:20]:
            if o:
                picked[e["task_id"]][o].add(e["cond"])
    pool = {t: set(picked[t]) for t in ids}
    missing = sorted((t, o) for t in ids for o in pool[t] if (t, o) not in pool_grade)

    # pooled labels
    pooled, strict = {}, {}
    for t in ids:
        p, s = dict(orig[t]), dict(orig[t])
        for o in pool[t]:
            g = pool_grade.get((t, o))
            if g is not None and o not in p and g >= 1:
                p[o] = g
            if g == 2 and o not in s:
                s[o] = 2
        pooled[t], strict[t] = p, s

    # per-episode metrics
    rows = []
    for e in eps:
        t = e["task_id"]
        final = dedupe(e["final_oids"])[:20]
        top = final[:10]
        jg = [pool_grade[(t, o)] for o in top if (t, o) in pool_grade]
        rows.append({"task_id": t, "cond": e["cond"], "stratum": stratum[t], "final_len": len(final),
                     "p@10_orig": sum(o in orig[t] for o in top) / 10, "p@10_pooled": sum(o in pooled[t] for o in top) / 10,
                     "p@10_pooled_strict": sum(o in strict[t] for o in top) / 10,
                     "ndcg@10_orig": ndcg(final, orig[t], 10), "ndcg@10_pooled": ndcg(final, pooled[t], 10),
                     "judged_share@10": sum(g >= 1 for g in jg) / len(jg) if jg else float("nan"),
                     "top10_judged": len(jg), "top10_new_relevant": sum(o in pooled[t] and o not in orig[t] for o in top)})
    by = {(r["cond"], r["task_id"]): r for r in rows}

    comparisons = {}
    for sname, pred in STRATA.items():
        sids = [t for t in ids if pred(tasks[t])]
        comparisons[sname] = {}
        for m in METRICS:
            va = {t: by[("A", t)][m] for t in sids if ("A", t) in by}
            vb = {t: by[("G", t)][m] for t in sids if ("G", t) in by}
            comparisons[sname][m] = paired(va, vb)

    # judge diagnostics
    names = dict(pl.read_parquet(HERE.parents[1] / "data" / "cards.parquet", columns=["oracle_id", "name"]).iter_rows())
    lab = [g for (t, o), g in pool_grade.items() if o in orig[t]]
    unl = [g for (t, o), g in pool_grade.items() if o not in orig[t]]
    diag = {"pool_cards": sum(map(len, pool.values())), "judged_pool_cards": len(pool_grade), "missing_grades": len(missing),
            "labelled": {"n": len(lab), "judged_ge1": rate(lab, lambda g: g >= 1), "judged_2": rate(lab, lambda g: g == 2)},
            "unlabelled": {"n": len(unl), "judged_ge1": rate(unl, lambda g: g >= 1), "judged_2": rate(unl, lambda g: g == 2)},
            "decoy_random": {"n": len(decoy), "judged_ge1": rate(list(decoy.values()), lambda g: g >= 1),
                             "judged_2": rate(list(decoy.values()), lambda g: g == 2),
                             "accepted": sorted(f"{t} | {names.get(o)} | {g}" for (t, o), g in decoy.items() if g >= 1)},
            "per_stratum": {}}
    for sname in list(STRATA)[1:]:
        sl = [g for (t, o), g in pool_grade.items() if stratum[t] == sname and o in orig[t]]
        su = [g for (t, o), g in pool_grade.items() if stratum[t] == sname and o not in orig[t]]
        sd = [g for (t, o), g in decoy.items() if stratum[t] == sname]
        diag["per_stratum"][sname] = {"labelled_n": len(sl), "labelled_judged_ge1": rate(sl, lambda g: g >= 1),
                                      "unlabelled_n": len(su), "unlabelled_judged_ge1": rate(su, lambda g: g >= 1),
                                      "unlabelled_judged_2": rate(su, lambda g: g == 2),
                                      "decoys_n": len(sd), "decoys_judged_ge1": rate(sd, lambda g: g >= 1)}

    per_task = {}
    for t in ids:
        new = [o for o in pool[t] if o not in orig[t] and pool_grade.get((t, o), 0) >= 1]
        who = Counter("both" if len(picked[t][o]) == 2 else f"{next(iter(picked[t][o]))} only" for o in new)
        per_task[t] = {"stratum": stratum[t], "pooled": len(pool[t]), "labelled": sum(o in orig[t] for o in pool[t]),
                       "labelled_judged_0": sum(o in orig[t] and pool_grade.get((t, o)) == 0 for o in pool[t]),
                       "new_relevant": len(new), "new_grade2": sum(pool_grade[(t, o)] == 2 for o in new),
                       "new_A_only": who["A only"], "new_G_only": who["G only"], "new_both": who["both"],
                       **{f"{c}_{m}": round(by[(c, t)][m], 3) for c in ("A", "G") if (c, t) in by
                          for m in ("p@10_orig", "p@10_pooled", "ndcg@10_pooled")}}
    new_tot = Counter()
    for v in per_task.values():
        for k in ("new_relevant", "new_A_only", "new_G_only", "new_both"):
            new_tot[k] += v[k]

    # judge cost
    calls = []
    for r in judge_runs:
        calls += [json.loads(x) for x in (HERE / "runs" / r / "calls.jsonl").read_text().splitlines()]
    tok = [sum(c["usage"][k] for k in TOK) for c in calls]
    cost = {"calls": len(calls), "complete_parses": sum(c["n_graded"] == c["n_cards"] for c in calls),
            "cards_sent": sum(c["n_cards"] for c in calls), "cards_graded": sum(c["n_graded"] for c in calls),
            **{f"tokens_{k}": sum(c["usage"][k] for c in calls) for k in TOK},
            "tokens_total": sum(tok), "tokens_per_call_mean": round(st.mean(tok)) if tok else None,
            "wall_s_per_call_mean": round(st.mean(c["wall_s"] for c in calls), 2) if calls else None,
            "wall_s_sum": round(sum(c["wall_s"] for c in calls), 1),
            "notional_usd": round(sum(c["usage"].get("notional_usd", 0) for c in calls), 3)}
    tdir = HERE / "runs" / judge_runs[-1] / "transcripts"
    if tdir.exists() and len(judge_runs) == 1:  # elapsed time of the run, from transcript write times
        mt = sorted(p.stat().st_mtime for p in tdir.glob("*.jsonl"))
        cost["elapsed_s_approx"] = round(mt[-1] - mt[0] + st.mean(c["wall_s"] for c in calls), 0) if mt else None

    diag["labelled_judged_0_examples"] = sorted(f"{t} | {names.get(o)}" for (t, o), g in pool_grade.items()
                                                if o in orig[t] and g == 0)[:60]

    out = {"judge_runs": judge_runs, "pool_run": args.pool_run, "n_tasks": len(ids),
           "strata_n": dict(Counter(stratum.values())), "comparisons": comparisons, "judge": diag,
           "new_relevant_totals": dict(new_tot), "cost": cost, "per_task": per_task, "episodes": rows}
    path = Path(args.out) if args.out else HERE / "runs" / judge_runs[-1] / "summary.json"
    path.write_text(json.dumps(out, indent=1, default=float) + "\n")

    print(f"tasks {len(ids)} {dict(Counter(stratum.values()))}; pool {diag['pool_cards']}, judged {len(pool_grade)}, missing {len(missing)}")
    for sname, cm in comparisons.items():
        print(f"\n## {sname}")
        for m, p in cm.items():
            if p:
                print(f"  {m:20s} n={p['n']:2d} A={p['mean_A']:.3f} G={p['mean_G']:.3f} G-A={p['diff_G_minus_A']:+.3f} "
                      f"[{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] G>A {p['G_better']} A>G {p['A_better']} ties {p['ties']} p={p['p_sign']:.3g}")
    print("\njudge:", json.dumps({k: v for k, v in diag.items() if k not in ("labelled_judged_0_examples",)}, indent=1))
    print("new relevant:", dict(new_tot))
    print("cost:", json.dumps(cost, indent=1))


if __name__ == "__main__":
    main()

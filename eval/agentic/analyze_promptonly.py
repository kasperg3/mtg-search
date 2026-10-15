"""EXP-019 analysis: prompt-only ablation on all 113 EXP-013 tasks (prompt_only.md).

    uv run python eval/agentic/analyze_promptonly.py --run cc-promptonly --judge-runs judge-full,judge-ablation,judge-promptonly

Arms: A and G are EXP-013 (``runs/cc-main``); A0 (search) and G1 (EXP-013's grep tool unchanged) use the same
minimal prompt (``runs/<run>``). G1 differs from G only in the prompt. Relevance as in EXP-016/018: a card in the
deduplicated top 10 counts if it is on the task's list or graded >= 1 (later judge runs override earlier ones).
Find-this-card: target in the final list. Paired over tasks: mean difference, 10k bootstrap 95% CI (seed 0),
two-sided sign test; tokens as per-task ratios. Output: ``runs/<run>/summary.json``.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analyze import dedupe, label_sets  # noqa: E402
from analyze_ablation import LAYOUT, paired, ratio  # noqa: E402
from cc_agent import load_tasks  # noqa: E402

ARMS = {"A": "base", "G": "base", "A0": "run", "G1": "run"}
PAIRS = [("G", "G1"), ("A", "A0"), ("A0", "G1"), ("A", "G")]  # (a, b): b - a
NO_TUNING_MARGIN = -0.05  # search needs no prompt tuning if the A0 - A CI lower bound for P@10 and found is above this


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="cc-promptonly")
    ap.add_argument("--base-run", default="cc-main")
    ap.add_argument("--judge-runs", default="judge-full,judge-ablation,judge-promptonly")
    args = ap.parse_args()
    tasks = {t["id"]: t for t in load_tasks()}
    orig, _, _ = label_sets(tasks)
    grade = {}
    for r in args.judge_runs.split(","):
        p = HERE / "runs" / r / "judgments.jsonl"
        if p.exists():
            for j in map(json.loads, p.read_text().splitlines()):
                if j.get("kind", "pool") == "pool":
                    grade[(j["task_id"], j["oracle_id"])] = j["grade"]
    eps = {}
    for c, src in ARMS.items():
        run = args.base_run if src == "base" else args.run
        for e in map(json.loads, (HERE / "runs" / run / "episodes.jsonl").read_text().splitlines()):
            if e["cond"] == c and e["rep"] == 0:
                eps[(c, e["task_id"])] = e
    ids = sorted(tasks)
    missing = [(c, t) for c in ARMS for t in ids if (c, t) not in eps]
    m = {c: {} for c in ARMS}
    unjudged = 0
    for (c, t), e in eps.items():
        final = dedupe(e["final_oids"])[:20]
        row = {"tokens": e["total_tokens"], "turns": e["llm_calls"], "tool_calls": len(e["searches"]), "wall_s": e["wall_s"]}
        if tasks[t]["kind"] == "known":
            row["found"] = float(tasks[t]["relevant"][0] in final)
        else:
            top = final[:10]
            unjudged += sum(o not in orig[t] and (t, o) not in grade for o in top)
            row["p@10"] = sum(o in orig[t] or grade.get((t, o), 0) >= 1 for o in top) / 10
            row["top10_len"] = len(top)
        for k, v in row.items():
            m[c].setdefault(k, {})[t] = v
    groups = {"all": ids, "deck": [t for t in ids if tasks[t]["kind"] == "deck"],
              "attr": [t for t in ids if tasks[t]["kind"] == "attr"], "known": [t for t in ids if tasks[t]["kind"] == "known"]}

    def sub(c, k, keep):
        return {t: v for t, v in m[c].get(k, {}).items() if t in keep}

    means = {c: {g: {k: st.mean(v[t] for t in groups[g] if t in v) for k, v in m[c].items() if any(t in v for t in groups[g])}
                 for g in groups} for c in ARMS}
    comp = {}
    for a, b in PAIRS:
        key = f"{b}_vs_{a}"
        comp[key] = {"tokens": ratio(m[a]["tokens"], m[b]["tokens"]), "turns": paired(m[a]["turns"], m[b]["turns"]),
                     "tool_calls": paired(m[a]["tool_calls"], m[b]["tool_calls"]),
                     "found": paired(sub(a, "found", groups["known"]), sub(b, "found", groups["known"]))}
        for g in ("deck", "attr"):
            comp[key][f"p@10_{g}"] = paired(sub(a, "p@10", groups[g]), sub(b, "p@10", groups[g]))
        comp[key]["p@10"] = paired(sub(a, "p@10", groups["deck"] + groups["attr"]), sub(b, "p@10", groups["deck"] + groups["attr"]))
    s = comp["A0_vs_A"]
    no_tuning = s["p@10"]["ci95"][0] > NO_TUNING_MARGIN and s["found"]["ci95"][0] > NO_TUNING_MARGIN
    diag = {}
    for c in ("G", "G1"):
        pats = [x for (cc, _), e in eps.items() if cc == c for x in e["searches"]]
        diag[c] = {"calls": len(pats), "layout_share": sum(bool(LAYOUT.search(x["query"])) for x in pats) / len(pats),
                   "newline_share": sum("\\n" in x["query"] for x in pats) / len(pats),
                   "zero_match_share": sum((x.get("n_match") or 0) == 0 for x in pats) / len(pats),
                   "offset_share": sum((x.get("offset") or 0) > 0 for x in pats) / len(pats)}
    out = {"experiment": "EXP-019", "run": args.run, "n_tasks": len(ids), "missing_episodes": missing, "unjudged_top10": unjudged,
           "means": means, "comparisons": comp, "search_needs_no_prompt_tuning": no_tuning,
           "no_tuning_rule": f"A0 - A 95% CI lower bound above {NO_TUNING_MARGIN} for P@10 (80 tasks) and found (33 tasks)",
           "grep_diagnostics": diag}
    (HERE / "runs" / args.run / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("missing_episodes", "unjudged_top10", "search_needs_no_prompt_tuning")}))
    for c in ARMS:
        a = means[c]
        print(f"{c:3} tokens {a['all']['tokens']:8.0f}  turns {a['all']['turns']:.2f}  calls {a['all']['tool_calls']:.2f}  "
              f"wall {a['all']['wall_s']:.1f}  P@10 deck {a['deck']['p@10']:.3f} attr {a['attr']['p@10']:.3f}  found {a['known']['found']:.3f}")
    for key, v in comp.items():
        t, p = v["tokens"], v["p@10"]
        print(f"{key:10} tokens ratio {t['median_ratio']:.2f} (b more on {t['b_more']}/{t['n']}, p={t['p_sign']:.2g})  "
              f"P@10 {p['diff']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] p={p['p_sign']:.2g}  "
              f"deck {v['p@10_deck']['diff']:+.3f} attr {v['p@10_attr']['diff']:+.3f}  found {v['found']['diff']:+.3f}")
    print(json.dumps(diag, indent=1))


if __name__ == "__main__":
    main()

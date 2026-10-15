"""EXP-006 metrics, paired bootstrap and plots.

    uv run python eval/agentic/analyze.py --runs main [--runs pilot] [--out eval/agentic/results/main.json]
    uv run --with matplotlib python eval/agentic/analyze.py --runs main --plot eval/experiments/agentic_recall_at_t.png

Metrics use the ORIGINAL task labels (tag/regex unions, known-item target), which are incomplete.
Pooled LLM judging (judge.py) is deferred / future work; if judgments.jsonl exists, a second
"pooled" label set is also reported.
Condensed-list nDCG (robustness check, no LLM): cards whose relevance is unknown are removed
from the ranking before scoring. A card counts as unjudged when it is not labelled relevant and
carries no Scryfall Tagger tags at all (the tag source never assessed it); for pooled labels,
GLM-judged cards count as judged too.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
T_MAX = 5
STOP = set("a an the and or of to in for with that this my me i is are it its on at by from as be can any all their them they you your some more cards card deck decks want need needs".split())


def load(run_names):
    eps = []
    for r in run_names:
        for line in (HERE / "runs" / r / "episodes.jsonl").read_text().splitlines():
            e = json.loads(line)
            e["run"] = r
            eps.append(e)
    return eps


def label_sets(tasks):
    from judge import load_judgments

    j = load_judgments()  # empty unless pooled judging has been run
    orig, pooled, judged = {}, {}, {}
    for tid, t in tasks.items():
        g = {o: 1 for o in t["relevant"]}
        g.update({o: int(v) for o, v in (t.get("graded") or {}).items()})
        orig[tid] = g
        p = dict(g)
        jt = {o: v for (tt, o), v in j.items() if tt == tid}
        for o, v in jt.items():
            if o not in p and v > 0:
                p[o] = v
        pooled[tid] = p
        judged[tid] = jt
    return orig, pooled, judged


def dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg(ranked, rel, k):
    ideal = sorted(rel.values(), reverse=True)[:k]
    return dcg([rel.get(o, 0) for o in ranked[:k]]) / dcg(ideal) if ideal else 0.0


def dedupe(seq):
    seen, out = set(), []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def words(q):
    return {w for w in re.findall(r"[a-z][a-z'-]+", q.lower()) if len(w) > 2 and w not in STOP}


def same(a, b):
    return a == b or (len(a) >= 4 and len(b) >= 4 and a[:5] == b[:5] and abs(len(a) - len(b)) <= 3)


def episode_metrics(e, task, labels, judged):
    m = {}
    rel = labels
    final = dedupe(e["final_oids"])[:20]
    union, prev_sets = set(), []
    novel = []
    for t in range(1, T_MAX + 1):
        if t <= len(e["searches"]):
            s = set(e["searches"][t - 1]["oids"])
            novel.append(len(s - union))
            union |= s
            prev_sets.append(s)
        m[f"recall@{t}"] = len(union & rel.keys()) / len(rel) if rel else 0.0
        if task["kind"] == "known":
            m[f"success@{t}"] = float(bool(union & rel.keys()))
    m["retrieved_relevant"] = len(union & rel.keys())
    m["n_unique_retrieved"] = len(union)
    m["ndcg@10"] = ndcg(final, rel, 10)
    m["ndcg@20"] = ndcg(final, rel, 20)
    m["p@10"] = sum(o in rel for o in final[:10]) / 10
    m["recall@20_final"] = sum(o in rel for o in final[:20]) / len(rel) if rel else 0.0
    if task["kind"] == "known":
        tgt = task["relevant"][0]
        m["rr"] = 1 / (final.index(tgt) + 1) if tgt in final else 0.0
        m["found_in_final"] = float(tgt in final)
    m["condensed_ndcg@10"] = ndcg([o for o in final if o in rel or o in judged], rel, 10)
    m["final_unretrieved"] = sum(o not in union for o in final) / len(final) if final else float("nan")
    m["final_len"] = len(final)
    m["iterations"] = len(e["searches"])
    m["novel_per_iter"] = float(np.mean(novel)) if novel else 0.0
    m["result_jaccard"] = float(np.mean([len(a & b) / len(a | b) for a, b in combinations(prev_sets, 2) if a | b])) if len(prev_sets) > 1 else float("nan")
    qs = [words(s["query"]) for s in e["searches"]]
    m["query_jaccard"] = float(np.mean([len(a & b) / len(a | b) for a, b in combinations(qs, 2) if a | b])) if len(qs) > 1 else float("nan")
    allw = [w for s in e["searches"] for w in re.findall(r"[a-z][a-z'-]+", s["query"].lower())]
    m["query_type_token"] = len(set(allw)) / len(allw) if allw else float("nan")
    m["cost_usd"] = e["cost_usd"]
    m["prompt_tokens"] = e["usage"]["prompt"]
    m["completion_tokens"] = e["usage"]["completion"]
    # term uptake: share of follow-up queries that introduce a new word matching a term in the
    # previous search's aggregate line or a starred per-result term (computed for every condition;
    # shown only in C/D, so A/B give the chance-overlap baseline)
    used, used_agg, n_follow = 0, 0, 0
    seen_words = words(task["prompt"])
    for i, s in enumerate(e["searches"]):
        w = words(s["query"])
        if i > 0 and e["searches"][i - 1].get("tool", "search") != "grep" and e["searches"][i - 1].get("agg_terms") is not None:
            prev = e["searches"][i - 1]
            agg = [t for t, _ in prev.get("agg_terms") or []]
            star = [t for terms in prev.get("per_terms") or [] for t, x in terms if x]
            new = [x for x in w if not any(same(x, y) for y in seen_words)]
            n_follow += 1
            hit_agg = any(same(x, t) for x in new for t in agg)
            hit_any = hit_agg or any(same(x, t) for x in new for t in star)
            used_agg += hit_agg
            used += hit_any
        seen_words |= w
    m["followups"] = n_follow
    m["term_uptake_agg"] = used_agg / n_follow if n_follow else float("nan")
    m["term_uptake_any"] = used / n_follow if n_follow else float("nan")
    return m


def per_task_means(rows, metric, cond, kinds=None):
    """task_id -> mean over repeats of metric for one condition."""
    acc = defaultdict(list)
    for r in rows:
        if r["cond"] == cond and (kinds is None or r["kind"] in kinds or r.get("kind2") in kinds) and metric in r and not (isinstance(r[metric], float) and math.isnan(r[metric])):
            acc[r["task_id"]].append(r[metric])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def paired_bootstrap(rows, metric, a, b, kinds=None, n=10000, seed=0):
    pa, pb = per_task_means(rows, metric, a, kinds), per_task_means(rows, metric, b, kinds)
    common = sorted(set(pa) & set(pb))
    if not common:
        return None
    d = np.array([pb[t] - pa[t] for t in common])
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"a": a, "b": b, "metric": metric, "n_tasks": len(common), "mean_a": float(np.mean([pa[t] for t in common])),
            "mean_b": float(np.mean([pb[t] for t in common])), "diff": float(d.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "p": float(min(1.0, p))}


def summarize(rows, conds, kinds, metrics):
    out = {}
    for c in conds:
        out[c] = {}
        for mt in metrics:
            pt = per_task_means(rows, mt, c, kinds)
            if pt:
                out[c][mt] = float(np.mean(list(pt.values())))
    return out


def rep_spread(rows, metric, cond, kinds):
    """Mean over tasks per repeat -> list (variance across repeats)."""
    by = defaultdict(list)
    for r in rows:
        if r["cond"] == cond and (r["kind"] in kinds or r.get("kind2") in kinds) and metric in r:
            by[r["rep"]].append(r[metric])
    return {int(k): float(np.mean(v)) for k, v in sorted(by.items())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", action="append", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()
    tasks = {t["id"]: t for t in yaml.safe_load((HERE / "tasks.yaml").read_text())}
    eps = [e for e in load(args.runs) if e["task_id"] in tasks]
    orig, pooled, judged = label_sets(tasks)
    import polars as pl

    c = pl.read_parquet(HERE.parents[1] / "data" / "cards.parquet", columns=["oracle_id", "tags"])
    tagged = {o for o, t in zip(c["oracle_id"].to_list(), c["tags"].to_list(), strict=True) if t}
    have_judgments = any(judged.values())
    conds = sorted({e["cond"] for e in eps})
    result = {"runs": args.runs, "n_episodes": len(eps), "conds": conds, "labels": {}}
    label_list = (("orig", orig), ("pooled", pooled)) if have_judgments else (("orig", orig),)
    for lname, labels in label_list:
        jd = {tid: (tagged | set(judged[tid]) if lname == "pooled" else tagged) for tid in tasks}
        rows = []
        for e in eps:
            t = tasks[e["task_id"]]
            kind2 = ("known-popular" if t.get("popular") else "known-obscure") if t["kind"] == "known" else None
            rows.append({"task_id": e["task_id"], "cond": e["cond"], "rep": e["rep"], "kind": t["kind"], "kind2": kind2,
                         **episode_metrics(e, t, labels[e["task_id"]], jd[e["task_id"]])})
        res = {}
        groups = {"all-nonknown": {"deck", "attr"}, "deck": {"deck"}, "attr": {"attr"}, "known": {"known"},
                  "known-popular": {"known-popular"}, "known-obscure": {"known-obscure"}}
        ret_metrics = [f"recall@{t}" for t in range(1, T_MAX + 1)] + ["ndcg@10", "ndcg@20", "condensed_ndcg@10", "p@10", "recall@20_final", "retrieved_relevant"]
        known_metrics = [f"success@{t}" for t in range(1, T_MAX + 1)] + ["rr", "found_in_final"]
        behav = ["final_len", "final_unretrieved", "iterations", "novel_per_iter", "n_unique_retrieved", "result_jaccard", "query_jaccard", "query_type_token",
                 "term_uptake_agg", "term_uptake_any", "cost_usd", "prompt_tokens", "completion_tokens"]
        for g, kinds in groups.items():
            isk = g.startswith("known")
            mets = known_metrics + behav if isk else ret_metrics + behav
            res[g] = {"summary": summarize(rows, conds, kinds, mets), "bootstrap": {}, "repeats": {}}
            key = "found_in_final" if isk else "ndcg@10"
            for c in conds:
                res[g]["repeats"][c] = {key: rep_spread(rows, key, c, kinds), "success@5" if isk else "recall@5":
                                        rep_spread(rows, "success@5" if isk else "recall@5", c, kinds)}
            for a, b in (("A", "C"), ("B", "C"), ("A", "D"), ("A", "B"), ("F", "C"), ("F", "A")):
                if a in conds and b in conds:
                    for mt in (known_metrics if isk else ["recall@5", "ndcg@10", "ndcg@20", "condensed_ndcg@10", "p@10", "recall@20_final"]) + ["novel_per_iter", "result_jaccard", "term_uptake_agg"]:
                        bs = paired_bootstrap(rows, mt, a, b, kinds)
                        if bs:
                            # key "<b>-<a>:<metric>" holds b - a (review m10: the old "<a>vs<b>" read backwards)
                            res[g]["bootstrap"][f"{b}-{a}:{mt}"] = bs
        result["labels"][lname] = res
        if lname == "orig":
            result["episodes"] = rows
    result["cost_total_usd"] = float(sum(e["cost_usd"] for e in eps))
    result["pool"] = {"judged_cards": sum(len(v) for v in judged.values()),
                      "judged_relevant": sum(sum(1 for x in v.values() if x > 0) for v in judged.values())}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    # console summary
    for lname, _ in label_list:
        for g in ("deck", "attr", "all-nonknown", "known", "known-popular", "known-obscure"):
            s = result["labels"][lname][g]["summary"]
            keys = ["recall@1", "recall@3", "recall@5", "ndcg@10", "ndcg@20", "condensed_ndcg@10", "p@10", "novel_per_iter", "result_jaccard", "term_uptake_agg", "iterations", "cost_usd"] if not g.startswith("known") else \
                ["success@1", "success@3", "success@5", "found_in_final", "rr", "iterations", "cost_usd"]
            print(f"\n[{lname}/{g}] " + " ".join(f"{k:>10s}" for k in keys))
            for c in conds:
                print(f"  {c:3s}" + " ".join(f"{s[c].get(k, float('nan')):10.3f}" for k in keys))
    print(f"\ntotal cost ${result['cost_total_usd']:.3f} over {len(eps)} episodes")
    if args.plot:
        plot(result, args.plot)


def plot(result, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    colors = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c", "D": "#9467bd", "F": "#7f7f7f"}
    panels = [("orig", "deck", "recall@{}", "Deck needs: Recall@t (cumulative)"), ("orig", "attr", "recall@{}", "Attribute: Recall@t (cumulative)"),
              ("orig", "known", "success@{}", "Known-item: success@t")]
    for ax, (ln, g, pat, title) in zip(axes, panels, strict=True):
        s = result["labels"][ln][g]["summary"]
        for c in result["conds"]:
            ys = [s[c].get(pat.format(t), np.nan) for t in range(1, T_MAX + 1)]
            ax.plot(range(1, T_MAX + 1), ys, marker="o", label=c, color=colors.get(c))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("search iteration t")
        ax.set_xticks(range(1, T_MAX + 1))
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main()

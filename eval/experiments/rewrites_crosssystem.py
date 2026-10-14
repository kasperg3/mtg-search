"""EXP-008: does the sparse vs late vs fusion ranking hold under agent rewrites?

    OMP_NUM_THREADS=2 uv run --extra models python eval/experiments/rewrites_crosssystem.py

Reuses the EXP-003 rewrites (rewrites.yaml + rewrites_blind.yaml; no new LLM calls) and
the EXP-003 per-text retrieval cache layout (eval/experiments/.cache/<system>[__<rewrites>].json,
key "<set>/<id>/<field>", top-100 under the query's filter mask). late-mLateOn comes from
the EXP-003 cache; sparseup and bm25 are run here (torch limited to 2 threads).

Systems: sparseup, bm25, late-mLateOn, rrf(sparseup+mLateOn) (1/1 and 0.5/1),
rrf(bm25+mLateOn). A fusion of a multi-text variant is one flat RRF over all component
lists of both systems (as rrf(dense+mLateOn) in EXP-003), with per-system weights.
Variants: orig, oracle, hyde, multi (RRF over 3), all (RRF over orig+oracle+hyde+3 multi).

Paired cluster-bootstrap CIs are vs late-mLateOn on the same variant and rewrites file.
Also: handwritten/tags reference from eval/.cache (original queries), and the
oracle-vs-origin lexical-overlap control (content-word Jaccard, eval/lexical_overlap.py).

Output: printed tables + eval/experiments/runs/<ts>_rewrites-crosssystem.json
"""

from __future__ import annotations

import datetime as dt
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from eval.lexical_overlap import content_tokens  # noqa: E402
from eval.queries import SETS as QFILES  # noqa: E402
from eval.queries import load_queries  # noqa: E402
from eval.run import (  # noqa: E402
    BASE, CARDS, DEPTH, INDEX_DIR, N_BOOT, SEED, Masks, clusters_for, paired_stats, per_query_metrics, query_key,
)
from mtg_search.fusion import rrf  # noqa: E402

SETS = ["paraphrase", "deckneeds"]
CACHE = EXP / ".cache"
RUNS = EXP / "runs"
REWRITES = {"main": EXP / "rewrites.yaml", "blind": EXP / "rewrites_blind.yaml"}
CACHE_TAG = {"main": "", "blind": "__rewrites_blind"}
FIELDS = ["orig", "oracle", "hyde", "multi0", "multi1", "multi2"]
VARIANTS = {
    "orig": ["orig"],
    "oracle": ["oracle"],
    "hyde": ["hyde"],
    "multi": ["multi0", "multi1", "multi2"],
    "all": FIELDS,
}
BASES = ["sparseup", "bm25", "late-mLateOn"]
FUSIONS = {
    "rrf(sparseup+mLateOn)": {"sparseup": 1.0, "late-mLateOn": 1.0},
    "rrf(sparseup=0.5+mLateOn)": {"sparseup": 0.5, "late-mLateOn": 1.0},
    "rrf(bm25+mLateOn)": {"bm25": 1.0, "late-mLateOn": 1.0},
}
SYSTEMS = BASES + list(FUSIONS)
REF = "late-mLateOn"
METRICS = ["ndcg10", "r50", "r100"]
JACCARD_HI = 0.5


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def texts_for(q: dict, rw: dict) -> dict[str, str]:
    r = rw[q["id"]]
    return {"orig": q["query"], "oracle": r["oracle"], "hyde": r["hyde"], **{f"multi{i}": m for i, m in enumerate(r["multi"])}}


def text_key(text: str, filters: dict) -> str:
    return hashlib.sha1(json.dumps([text, filters]).encode()).hexdigest()[:16]


def load_base(system: str):
    if system == "sparseup":
        import torch

        torch.set_num_threads(2)
        from mtg_search.retrievers.sparseup import SparseUpRetriever

        return SparseUpRetriever.from_dir(INDEX_DIR, BASE["sparseup"][1])
    from mtg_search.retrievers import load_retriever

    return load_retriever("bm25", None, index_dir=INDEX_DIR, cards_parquet=CARDS)


def run_system(system: str, rwname: str, queries, rw, masks) -> dict:
    p = CACHE / f"{system}{CACHE_TAG[rwname]}.json"
    cache = json.loads(p.read_text()) if p.exists() else {}
    todo = []
    for st in SETS:
        for q in queries[st]:
            for f, text in texts_for(q, rw).items():
                key, h = f"{st}/{q['id']}/{f}", text_key(text, q["filters"])
                if cache.get(key, {}).get("key") != h:
                    todo.append((key, text, q["filters"], h))
    if not todo:
        print(f"[{system}/{rwname}] cached")
        return cache
    if system == REF:
        sys.exit(f"{REF} cache {p} is missing {len(todo)} texts; run eval/experiments/query_rewriting.py first")
    r = load_base(system)
    for w in ["destroy target creature", "draw a card whenever"]:
        r.search(w, DEPTH, masks.get({}))
    print(f"[{system}/{rwname}] {len(todo)} texts")
    for i, (key, text, filters, h) in enumerate(todo):
        t = time.perf_counter()
        hits = r.search(text, DEPTH, masks.get(filters))
        cache[key] = {"key": h, "ms": (time.perf_counter() - t) * 1000, "hits": [[int(a), round(float(b), 5)] for a, b in hits]}
        if (i + 1) % 300 == 0:
            print(f"  {i + 1}/{len(todo)}")
    CACHE.mkdir(exist_ok=True)
    p.write_text(json.dumps(cache))
    del r
    gc.collect()
    return cache


def ranked_for(system: str, variant: str, key_prefix: str, caches: dict) -> list:
    comps = VARIANTS[variant]
    if system in FUSIONS:
        lists, weights = [], []
        for base, w in FUSIONS[system].items():
            for c in comps:
                lists.append(caches[base][f"{key_prefix}/{c}"]["hits"])
                weights.append(w)
        return rrf(lists, weights=weights, limit=DEPTH)
    lists = [caches[system][f"{key_prefix}/{c}"]["hits"] for c in comps]
    return lists[0] if len(lists) == 1 else rrf(lists, limit=DEPTH)


def jaccard(a: str, b: str) -> float:
    x, y = set(content_tokens(a)), set(content_tokens(b))
    return len(x & y) / len(x | y) if x | y else 0.0


def main() -> None:
    cards = pl.read_parquet(CARDS)
    oids = cards["oracle_id"].to_list()
    queries = {s: load_queries(s) for s in SETS}
    masks = Masks(cards)
    origin_text = {q["id"]: q["query"] for s in ("tags", "handwritten") for q in load_queries(s)}
    origin_set = {q["id"]: s for s in ("tags", "handwritten") for q in load_queries(s)}

    keys = [(st, q) for st in SETS for q in queries[st]]
    cl_all = [clusters_for(st, q) for st, q in keys]
    rng = np.random.default_rng(SEED)
    out: dict = {"rewrites": {}, "overlap": {}}

    for rwname, rwpath in REWRITES.items():
        rw = {r["id"]: r for r in yaml.safe_load(rwpath.read_text())}
        caches = {s: run_system(s, rwname, queries, rw, masks) for s in BASES}

        # per-query metrics M[system][variant] = list over keys
        M = {sy: {v: [per_query_metrics([oids[int(i)] for i, _ in ranked_for(sy, v, f"{st}/{q['id']}", caches)], q)
                      for st, q in keys] for v in VARIANTS} for sy in SYSTEMS}
        lat = {sy: {v: [sum(caches[b][f"{st}/{q['id']}/{c}"]["ms"] for b in (FUSIONS[sy] if sy in FUSIONS else [sy])
                            for c in VARIANTS[v]) for st, q in keys] for v in VARIANTS} for sy in SYSTEMS}

        # overlap: oracle rewrite vs origin query text (paraphrase only; deckneeds has no origin)
        ov = []
        for st, q in keys:
            r = rw[q["id"]]
            if st == "paraphrase":
                o = origin_text[q["origin"]]
                ov.append({"id": q["id"], "origin": q["origin"], "origin_set": origin_set[q["origin"]],
                           "jac_oracle": jaccard(r["oracle"], o),
                           "jac_orig_query": jaccard(q["query"], o),
                           "jac_max_any": max(jaccard(t, o) for t in [r["oracle"], r["hyde"], *r["multi"]])})
            else:
                ov.append(None)
        para_ov = [x for x in ov if x]
        jo = np.array([x["jac_oracle"] for x in para_ov])
        overlap_summary = {
            "n_paraphrase": len(jo),
            "jac_oracle_mean": float(jo.mean()), "jac_oracle_median": float(np.median(jo)),
            "share_jac_oracle_gt_0.5": float((jo > 0.5).mean()),
            "share_jac_oracle_gt_0.3": float((jo > 0.3).mean()),
            "share_jac_oracle_eq_1": float((jo >= 0.999).mean()),
            "share_jac_maxany_gt_0.5": float(np.mean([x["jac_max_any"] > 0.5 for x in para_ov])),
            "jac_paraphrase_query_vs_origin_mean": float(np.mean([x["jac_orig_query"] for x in para_ov])),
            "by_origin_set": {s: {"n": int(sum(x["origin_set"] == s for x in para_ov)),
                                  "share_gt_0.5": float(np.mean([x["jac_oracle"] > 0.5 for x in para_ov if x["origin_set"] == s]))}
                              for s in ("tags", "handwritten")},
            "high_overlap_examples": sorted(
                [{"id": x["id"], "jac": round(x["jac_oracle"], 2), "oracle": rw[x["id"]]["oracle"], "origin": origin_text[x["origin"]]}
                 for x in para_ov if x["jac_oracle"] > 0.5], key=lambda d: -d["jac"])[:8],
        }
        out["overlap"][rwname] = overlap_summary

        subsets = {
            "para+deck": [True] * len(keys),
            "paraphrase": [st == "paraphrase" for st, _ in keys],
            "deckneeds": [st == "deckneeds" for st, _ in keys],
            "paraphrase_lowoverlap": [x is not None and x["jac_oracle"] <= JACCARD_HI for x in ov],
            "paraphrase_highoverlap": [x is not None and x["jac_oracle"] > JACCARD_HI for x in ov],
            "paraphrase_lowoverlap_strict": [x is not None and x["jac_max_any"] <= JACCARD_HI for x in ov],
        }
        rows = []
        for sub, mask in subsets.items():
            idx = np.flatnonzero(mask)
            cl = [cl_all[i] for i in idx]
            for sy in SYSTEMS:
                for v in VARIANTS:
                    row = {"subset": sub, "n": int(len(idx)), "system": sy, "variant": v}
                    for m in METRICS:
                        a = np.array([M[sy][v][i][m] for i in idx])
                        row[m] = float(a.mean())
                        if sy != REF:
                            b = np.array([M[REF][v][i][m] for i in idx])
                            ps = paired_stats(a, b, cl, rng)
                            row[f"{m}_vs_ref"] = {k: float(ps[k]) for k in ("diff", "lo", "hi", "p")}
                    if sub == "para+deck":
                        row["latency_ms_p50"] = float(np.percentile(lat[sy][v], 50))
                    rows.append(row)
        out["rewrites"][rwname] = {"file": str(rwpath.relative_to(ROOT)), "sha256_16": sha(rwpath), "metrics": rows}

        print(f"\n=== rewrites: {rwname} ===")
        print("overlap:", json.dumps({k: v for k, v in overlap_summary.items() if k != "high_overlap_examples"}))
        for sub in subsets:
            print(f"\n## {sub} (n={sum(subsets[sub])})  nDCG@10 / R@100  [Δ nDCG@10 vs {REF}]")
            for sy in SYSTEMS:
                cells = []
                for v in VARIANTS:
                    r = next(x for x in rows if x["subset"] == sub and x["system"] == sy and x["variant"] == v)
                    d = r.get("ndcg10_vs_ref")
                    ci = f" [{100 * d['diff']:+.1f} ({100 * d['lo']:+.1f},{100 * d['hi']:+.1f})]" if d else ""
                    cells.append(f"{v}: {100 * r['ndcg10']:.1f}/{100 * r['r100']:.1f}{ci}")
                print(f"{sy:28s} " + " | ".join(cells))

    # handwritten / tags reference: original queries from eval/.cache (EXP-001/004 rankings)
    ref_rows = []
    hw_caches = {s: json.loads((ROOT / "eval" / ".cache" / f"{s}.json").read_text()) for s in BASES}
    for st in ("handwritten", "tags"):
        qs = load_queries(st)
        for s in BASES:
            for q in qs:
                assert hw_caches[s][f"{st}/{q['id']}"]["key"] == query_key(q), f"stale eval/.cache for {s} {st}/{q['id']}"
        cl = [clusters_for(st, q) for q in qs]
        MM = {}
        for sy in SYSTEMS:
            comps = FUSIONS.get(sy, {sy: 1.0})
            MM[sy] = []
            for q in qs:
                lists = [hw_caches[c][f"{st}/{q['id']}"]["hits"] for c in comps]
                ranked = lists[0] if len(lists) == 1 else rrf(lists, weights=list(comps.values()), limit=DEPTH)
                MM[sy].append(per_query_metrics([oids[int(i)] for i, _ in ranked], q))
        for sy in SYSTEMS:
            row = {"set": st, "n": len(qs), "system": sy}
            for m in METRICS:
                a = np.array([x[m] for x in MM[sy]])
                row[m] = float(a.mean())
                if sy != REF:
                    ps = paired_stats(a, np.array([x[m] for x in MM[REF]]), cl, rng)
                    row[f"{m}_vs_ref"] = {k: float(ps[k]) for k in ("diff", "lo", "hi", "p")}
            ref_rows.append(row)
    out["reference_original_queries"] = ref_rows
    print("\n## reference (original queries)")
    for r in ref_rows:
        d = r.get("ndcg10_vs_ref")
        ci = f" [{100 * d['diff']:+.1f} ({100 * d['lo']:+.1f},{100 * d['hi']:+.1f})]" if d else ""
        print(f"{r['set']:12s} {r['system']:28s} nDCG@10 {100 * r['ndcg10']:.1f}{ci}  R@50 {100 * r['r50']:.1f}  R@100 {100 * r['r100']:.1f}")

    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    ts = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    RUNS.mkdir(exist_ok=True)
    p = RUNS / f"{ts}_rewrites-crosssystem.json"
    p.write_text(json.dumps({
        "experiment": "EXP-008", "label": "rewrites-crosssystem", "created_at": ts,
        "git_sha": git("rev-parse", "HEAD"), "git_dirty_src": bool(git("status", "--porcelain", "--", "src", "eval/run.py")),
        "argv": sys.argv, "torch_threads": 2, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "models": {s: {"kind": BASE[s][0], "model": BASE[s][1]} for s in BASES},
        "fusions": FUSIONS, "variants": VARIANTS, "fusion_mode": "flat RRF over all component lists of both systems, per-system weights",
        "depth": DEPTH, "rrf_k": 60, "bootstrap": {"n": N_BOOT, "seed": SEED, "cluster": "paraphrase origin pairs", "paired_vs": REF},
        "overlap_def": {"tokens": "eval/lexical_overlap.content_tokens", "metric": "set Jaccard", "high": f"> {JACCARD_HI}"},
        "inputs_sha256_16": {**{f"queries_{s}.yaml": sha(QFILES[s]) for s in ("tags", "handwritten", *SETS)}},
        **out,
    }, indent=1))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()

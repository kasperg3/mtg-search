"""EXP-003: does agent-side query rewriting help on paraphrase + deckneeds?

    uv run --extra models python eval/experiments/query_rewriting.py            # retrieve (cached) + report
    uv run --extra models python eval/experiments/query_rewriting.py --refresh  # ignore cache

Rewrites come from eval/experiments/rewrites.yaml (written blind to labels; see
dump_queries.py). For each retriever (late-mLateOn via LateInMemory, dense
embeddinggemma-2) every query text (original, oracle, hyde, multi[0..2]) is run once with
the query's filter mask (same as eval/run.py), top-100 cached in
eval/experiments/.cache/<system>.json. Variants are then built from the cache:

    orig, oracle, hyde, multi = RRF(multi[0..2]), orig+multi = RRF(orig, multi[0..2]),
    plus orig+oracle, orig+hyde, all = RRF(orig, oracle, hyde, multi[0..2]).

A sensitivity system `late-mLateOn@32` caps the query at 32 tokens (PyLate's classic
default), to price truncation; mLateOn's own config uses query_length 8192, no expansion.

Outputs: printed tables, eval/experiments/runs/<ts>_query-rewriting.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
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

from eval.queries import SETS as QFILES  # noqa: E402
from eval.queries import load_queries  # noqa: E402
from eval.run import CARDS, DEPTH, INDEX_DIR, N_BOOT, SEED, Masks, clusters_for, paired_stats, per_query_metrics  # noqa: E402
from mtg_search.fusion import rrf  # noqa: E402

SETS = ["paraphrase", "deckneeds"]
REWRITES = EXP / "rewrites.yaml"
CACHE = EXP / ".cache"
RUNS = EXP / "runs"
SYSTEMS = {
    "late-mLateOn": ("late", "lightonai/mLateOn", None),
    "dense": ("dense", "google/embeddinggemma-2", None),
    "late-mLateOn@32": ("late", "lightonai/mLateOn", 32),
}
FIELDS = ["orig", "oracle", "hyde", "multi0", "multi1", "multi2"]
VARIANTS = {
    "orig": ["orig"],
    "oracle": ["oracle"],
    "hyde": ["hyde"],
    "multi": ["multi0", "multi1", "multi2"],
    "orig+multi": ["orig", "multi0", "multi1", "multi2"],
    "orig+oracle": ["orig", "oracle"],
    "orig+hyde": ["orig", "hyde"],
    "all": ["orig", "oracle", "hyde", "multi0", "multi1", "multi2"],
}
METRICS = ["ndcg10", "r50", "r100", "mrr10"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def texts_for(q: dict, rw: dict) -> dict[str, str]:
    r = rw[q["id"]]
    return {"orig": q["query"], "oracle": r["oracle"], "hyde": r["hyde"], **{f"multi{i}": m for i, m in enumerate(r["multi"])}}


def run_system(name, queries, rw, masks, refresh, tag="") -> dict:
    p = CACHE / f"{name}{tag}.json"
    cache = {} if refresh or not p.exists() else json.loads(p.read_text())
    todo = []
    for st in SETS:
        for q in queries[st]:
            for f, text in texts_for(q, rw).items():
                key = f"{st}/{q['id']}/{f}"
                h = hashlib.sha1(json.dumps([text, q["filters"]]).encode()).hexdigest()[:16]
                if cache.get(key, {}).get("key") != h:
                    todo.append((key, text, q["filters"], h))
    if not todo:
        print(f"[{name}] cached")
        return cache
    kind, model_id, qlen = SYSTEMS[name]
    from mtg_search.retrievers import load_retriever

    r = load_retriever(kind, model_id, index_dir=INDEX_DIR, cards_parquet=CARDS)
    if qlen is not None:
        r.model[0].query_length = qlen
        assert r.encode_query("x " * 100).shape[0] <= qlen, "query cap not applied"
    for w in ["destroy target creature", "draw a card whenever"]:
        r.search(w, DEPTH, masks.get({}))
    print(f"[{name}] {len(todo)} texts")
    for i, (key, text, filters, h) in enumerate(todo):
        t = time.perf_counter()
        hits = r.search(text, DEPTH, masks.get(filters))
        cache[key] = {"key": h, "ms": (time.perf_counter() - t) * 1000, "hits": [[int(a), round(float(b), 5)] for a, b in hits]}
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(todo)}")
    CACHE.mkdir(exist_ok=True)
    p.write_text(json.dumps(cache))
    del r
    gc.collect()
    return cache


def token_stats(queries, rw) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("lightonai/mLateOn")
    out = {}
    for f in ["orig", "oracle", "hyde", "multi"]:
        lens = []
        for st in SETS:
            for q in queries[st]:
                t = texts_for(q, rw)
                for k in ([f] if f != "multi" else ["multi0", "multi1", "multi2"]):
                    lens.append(len(tok("[Q] " + t[k])["input_ids"]))
        a = np.array(lens)
        out[f] = {"n": len(a), "p50": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": int(a.max()),
                  "frac_gt32": float((a > 32).mean()), "frac_gt64": float((a > 64).mean())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--rewrites", type=Path, default=REWRITES, help="rewrites YAML (default rewrites.yaml)")
    ap.add_argument("--systems", nargs="*", default=list(SYSTEMS))
    args = ap.parse_args()
    tag = "" if args.rewrites.resolve() == REWRITES else f"__{args.rewrites.stem}"

    cards = pl.read_parquet(CARDS)
    oids = cards["oracle_id"].to_list()
    queries = {s: load_queries(s) for s in SETS}
    rw = {r["id"]: r for r in yaml.safe_load(args.rewrites.read_text())}
    masks = Masks(cards)
    caches = {s: run_system(s, queries, rw, masks, args.refresh, tag) for s in args.systems}

    # per-query metrics M[system][variant][set] = list of metric dicts (query order)
    M: dict = {}
    lat: dict = {}
    for sy, cache in caches.items():
        M[sy], lat[sy] = {}, {}
        for v, comps in VARIANTS.items():
            M[sy][v] = {st: [] for st in SETS}
            lat[sy][v] = []
            for st in SETS:
                for q in queries[st]:
                    lists = [cache[f"{st}/{q['id']}/{c}"]["hits"] for c in comps]
                    ranked = lists[0] if len(lists) == 1 else rrf(lists, limit=DEPTH)
                    M[sy][v][st].append(per_query_metrics([oids[int(i)] for i, _ in ranked], q))
                    lat[sy][v].append(sum(cache[f"{st}/{q['id']}/{c}"]["ms"] for c in comps))
    # cross-model: RRF(dense + mLateOn) over orig vs over orig+multi (8 lists)
    for v in ("orig", "multi", "orig+multi") if {"dense", "late-mLateOn"} <= set(caches) else ():
        M.setdefault("rrf(dense+mLateOn)", {})[v] = {st: [] for st in SETS}
        for st in SETS:
            for q in queries[st]:
                lists = [caches[s][f"{st}/{q['id']}/{c}"]["hits"] for s in ("dense", "late-mLateOn") for c in VARIANTS[v]]
                ranked = rrf(lists, limit=DEPTH)
                M["rrf(dense+mLateOn)"][v][st].append(per_query_metrics([oids[int(i)] for i, _ in ranked], q))

    rng = np.random.default_rng(SEED)
    cl = {st: [clusters_for(st, q) for q in queries[st]] for st in SETS}
    cl["pooled"] = cl["paraphrase"] + cl["deckneeds"]

    def vec(sy, v, st, m):
        if st == "pooled":
            return np.array([x[m] for s in SETS for x in M[sy][v][s]])
        return np.array([x[m] for x in M[sy][v][st]])

    rows = []
    for sy in M:
        for v in M[sy]:
            for st in SETS + ["pooled"]:
                row = {"system": sy, "variant": v, "set": st}
                for m in METRICS:
                    a = vec(sy, v, st, m)
                    row[m] = float(a.mean())
                    if v != "orig":
                        ps = paired_stats(a, vec(sy, "orig", st, m), cl[st], rng)
                        row[f"{m}_diff"] = {k: float(ps[k]) for k in ("diff", "lo", "hi", "p")}
                rows.append(row)
    # also: each mLateOn variant vs the best original-query system in results.md, rrf(dense+mLateOn)
    vs_best = []
    for v in VARIANTS if "rrf(dense+mLateOn)" in M else ():
        for st in SETS + ["pooled"]:
            ps = paired_stats(vec("late-mLateOn", v, st, "ndcg10"), vec("rrf(dense+mLateOn)", "orig", st, "ndcg10"), cl[st], rng)
            vs_best.append({"variant": v, "set": st, **{k: float(ps[k]) for k in ("diff", "lo", "hi", "p")}})

    toks = token_stats(queries, rw)
    latency = {sy: {v: {"p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95))} for v, x in d.items()} for sy, d in lat.items()}

    def fmt(r, m):
        if f"{m}_diff" not in r:
            return f"{100 * r[m]:.1f}"
        d = r[f"{m}_diff"]
        return f"{100 * r[m]:.1f} ({100 * d['diff']:+.1f} [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}])"

    for st in SETS + ["pooled"]:
        print(f"\n## {st}\n| system | variant | nDCG@10 | Recall@50 | Recall@100 | MRR@10 |\n|---|---|---|---|---|---|")
        for r in rows:
            if r["set"] == st:
                print(f"| {r['system']} | {r['variant']} | " + " | ".join(fmt(r, m) for m in METRICS) + " |")
    print("\n## mLateOn variants vs rrf(dense+mLateOn) on original queries (nDCG@10)")
    for x in vs_best:
        print(f"{x['variant']:12s} {x['set']:10s} {100 * x['diff']:+.1f} [{100 * x['lo']:+.1f}, {100 * x['hi']:+.1f}] p={x['p']:.4f}")
    print("\n## tokens (mLateOn tokenizer incl. [Q] prefix)\n", json.dumps(toks, indent=1))
    print("\n## latency ms (sum of component searches)\n", json.dumps(latency, indent=1))

    sha_git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    RUNS.mkdir(exist_ok=True)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RUNS / f"{ts}_query-rewriting{tag.replace('__', '_')}.json"
    out.write_text(json.dumps({
        "experiment": "EXP-003",
        "label": "query-rewriting",
        "created_at": ts,
        "git_sha": sha_git,
        "git_dirty": bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True).stdout.strip()),
        "argv": sys.argv,
        "models": {k: {"kind": v[0], "model": v[1], "query_length_cap": v[2]} for k, v in SYSTEMS.items() if k in args.systems},
        "rewrites_file": str(args.rewrites.relative_to(ROOT)) if args.rewrites.is_absolute() else str(args.rewrites),
        "depth": DEPTH, "rrf_k": 60, "bootstrap": {"n": N_BOOT, "seed": SEED, "cluster": "paraphrase origin pairs"},
        "inputs_sha256_16": {**{f"queries_{s}.yaml": sha(QFILES[s]) for s in SETS}, args.rewrites.name: sha(args.rewrites)},
        "variants": VARIANTS,
        "metrics": rows,
        "late_variants_vs_rrf_dense_mLateOn_orig": vs_best,
        "query_tokens": toks,
        "latency_ms": latency,
    }, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

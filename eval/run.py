"""Retrieval evaluation: base retrievers + RRF fusions over the four query sets.

    uv run --extra models python eval/run.py            # retrieve (cached) + write eval/results.md
    uv run --extra models python eval/run.py --only bm25 dense   # (re)run some base systems
    uv run --extra models python eval/run.py --refresh dense     # ignore cache for these
    uv run --extra models python eval/run.py --report-only       # metrics from cache only
    uv run --extra models python eval/run.py --label sparseup    # tag the run record

Every invocation also writes a machine-readable run record to
``eval/experiments/runs/<UTC timestamp>_<label>.json`` (git sha + dirty flag, systems,
per-set metrics and latency, model ids/revisions from the index meta files, sha256 of each
query-set YAML). Reference it from ``eval/experiments/REGISTRY.md``.

Base systems are run one at a time (load -> warm up -> every query of every set -> free),
and their top-100 rankings plus per-query latency (query encoding + scoring + top-k, CPU,
warm) are cached in ``eval/.cache/<system>.json``. A cache entry is reused only if the
query text and filters are unchanged. Fusions are computed from the cache with
``mtg_search.fusion.rrf``; their latency is the sum of the component latencies
(sequential execution) plus the measured fusion time.

Every query is restricted to the cards passing ``filter_expr(**q["filters"])``, which
also drops funny cards (include_funny=False), just like the MCP server.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from eval.queries import gains, load_queries  # noqa: E402
from mtg_search.filters import filter_expr  # noqa: E402
from mtg_search.fusion import rrf  # noqa: E402

CARDS = ROOT / "data" / "cards.parquet"
INDEX_DIR = ROOT / "index" / "embeddings"
CACHE_DIR = EVAL_DIR / ".cache"
RESULTS_MD = EVAL_DIR / "results.md"
RUNS_DIR = EVAL_DIR / "experiments" / "runs"

SETS = ["tags", "handwritten", "paraphrase", "deckneeds"]
DEPTH = 100  # retrieved per component
SEED = 0
N_BOOT = 10000

# name -> (kind, model_id)
BASE = {
    "bm25": ("bm25", None),
    "bm25+tags": ("bm25+tags", None),
    "sparse": ("sparse", "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte"),
    "dense": ("dense", "google/embeddinggemma-2"),
    "late-mLateOn": ("late", "lightonai/mLateOn"),
    "late-answerai": ("late", "answerdotai/answerai-colbert-small-v1"),
    # Learned sparse with a transformer query encoder (EXP-004, eval/experiments/sparseup.md).
    "sparseup": ("sparseup", "Linkup-Platform/linkup-sparseup-embed-v1"),
}
LATE_COMPONENTS = {"late-mLateOn", "late-answerai"}

# name -> {component: weight}
FUSIONS: dict[str, dict[str, float]] = {
    "rrf(bm25+mLateOn)": {"bm25": 1, "late-mLateOn": 1},
    "rrf(bm25+sparse+dense)": {"bm25": 1, "sparse": 1, "dense": 1},
    "rrf(sparse+dense)": {"sparse": 1, "dense": 1},
    "rrf(dense+mLateOn)": {"dense": 1, "late-mLateOn": 1},
    "rrf(bm25+dense+mLateOn)": {"bm25": 1, "dense": 1, "late-mLateOn": 1},
    "rrf(all)": {"bm25": 1, "sparse": 1, "dense": 1, "late-mLateOn": 1},
    # Weightings, chosen after a first unweighted run (see results.md: tuned on the test sets).
    "rrf(all; bm25=0.5)": {"bm25": 0.5, "sparse": 1, "dense": 1, "late-mLateOn": 1},
    "rrf(bm25=0.5+dense+mLateOn)": {"bm25": 0.5, "dense": 1, "late-mLateOn": 1},
    "rrf(bm25=0.5+sparse+dense)": {"bm25": 0.5, "sparse": 1, "dense": 1},
    "rrf(sparse+dense=2)": {"sparse": 1, "dense": 2},
    "rrf(dense=2+mLateOn)": {"dense": 2, "late-mLateOn": 1},
    "rrf(dense+mLateOn=2)": {"dense": 1, "late-mLateOn": 2},
    "rrf(sparseup+mLateOn)": {"sparseup": 1, "late-mLateOn": 1},
    "rrf(sparseup+dense)": {"sparseup": 1, "dense": 1},
}

# Extra paired comparisons reported (and stored in the run record): (A, B).
EXTRA_PAIRS = [
    ("sparseup", "sparse"),
    ("sparseup", "dense"),
    ("sparseup", "late-mLateOn"),
    ("rrf(sparseup+mLateOn)", "late-mLateOn"),
    ("rrf(sparseup+mLateOn)", "rrf(dense+mLateOn)"),
    ("rrf(sparseup+dense)", "dense"),
]


def uses_late(system: str) -> bool:
    if system in BASE:
        return system in LATE_COMPONENTS
    return any(c in LATE_COMPONENTS and w > 0 for c, w in FUSIONS[system].items())


# --------------------------------------------------------------------------- data


def load_all_queries() -> dict[str, list[dict]]:
    return {s: load_queries(s) for s in SETS}


def query_key(q: dict) -> str:
    raw = json.dumps([q["query"], q["filters"]], sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


class Masks:
    def __init__(self, cards: pl.DataFrame):
        self.cards = cards
        self._cache: dict[str, np.ndarray] = {}

    def get(self, filters: dict) -> np.ndarray:
        key = json.dumps(filters, sort_keys=True)
        if key not in self._cache:
            self._cache[key] = self.cards.select(filter_expr(self.cards, **filters)).to_series().to_numpy().astype(bool)
        return self._cache[key]


# --------------------------------------------------------------------------- retrieval


def cache_path(system: str) -> Path:
    return CACHE_DIR / f"{system}.json"


def load_cache(system: str) -> dict:
    p = cache_path(system)
    return json.loads(p.read_text()) if p.exists() else {}


def run_base(system: str, queries: dict[str, list[dict]], masks: Masks, refresh: bool) -> None:
    cache = {} if refresh else load_cache(system)
    todo = [(s, q) for s in SETS for q in queries[s] if cache.get(f"{s}/{q['id']}", {}).get("key") != query_key(q)]
    if not todo:
        print(f"[{system}] cached ({len(cache)} queries)")
        return
    kind, model_id = BASE[system]
    t0 = time.perf_counter()
    if kind == "sparseup":
        from mtg_search.retrievers.sparseup import SparseUpRetriever

        r = SparseUpRetriever.from_dir(INDEX_DIR, model_id)
    else:
        r = __import__("mtg_search.retrievers", fromlist=["load_retriever"]).load_retriever(
            kind, model_id, index_dir=INDEX_DIR, cards_parquet=CARDS
        )
    print(f"[{system}] loaded in {time.perf_counter() - t0:.1f}s; {len(todo)} queries to run")
    # Warm-up: first calls pay for lazy init / allocator growth.
    for w in ["destroy target creature", "draw a card whenever", "cheap removal for my deck"]:
        r.search(w, DEPTH, masks.get({}))
    for i, (s, q) in enumerate(todo):
        allowed = masks.get(q["filters"])
        t = time.perf_counter()
        hits = r.search(q["query"], DEPTH, allowed)
        ms = (time.perf_counter() - t) * 1000
        cache[f"{s}/{q['id']}"] = {"key": query_key(q), "ms": ms, "hits": [[int(a), round(float(b), 5)] for a, b in hits]}
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(todo)}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path(system).write_text(json.dumps(cache))
    del r
    gc.collect()


# --------------------------------------------------------------------------- metrics


def ndcg_at(ranked: list[str], g: dict[str, int], k: int = 10) -> float:
    dcg = sum(g.get(o, 0) / math.log2(i + 2) for i, o in enumerate(ranked[:k]))
    ideal = sorted(g.values(), reverse=True)[:k]
    idcg = sum(v / math.log2(i + 2) for i, v in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall_at(ranked: list[str], rel: set[str], k: int) -> float:
    return len(rel & set(ranked[:k])) / len(rel)


def mrr_at(ranked: list[str], rel: set[str], k: int = 10) -> float:
    for i, o in enumerate(ranked[:k]):
        if o in rel:
            return 1 / (i + 1)
    return 0.0


def per_query_metrics(ranked: list[str], q: dict) -> dict[str, float]:
    g = gains(q)
    rel = set(g)
    return {
        "ndcg10": ndcg_at(ranked, g),
        "r50": recall_at(ranked, rel, 50),
        "r100": recall_at(ranked, rel, 100),
        "mrr10": mrr_at(ranked, rel),
    }


# --------------------------------------------------------------------------- stats


def clusters_for(set_name: str, q: dict) -> str:
    # Paraphrase queries come in pairs (-para/-need) sharing labels: resample by origin.
    return q.get("origin", q["id"]) if set_name == "paraphrase" else f"{set_name}/{q['id']}"


def paired_stats(a: np.ndarray, b: np.ndarray, cl: list[str], rng: np.random.Generator) -> dict:
    """Mean diff a-b, cluster-bootstrap 95% CI, and paired sign-flip permutation p (cluster level)."""
    d = a - b
    labels, inv = np.unique(np.asarray(cl), return_inverse=True)
    m = len(labels)
    csum = np.bincount(inv, weights=d, minlength=m)
    ccnt = np.bincount(inv, minlength=m).astype(float)
    idx = rng.integers(0, m, size=(N_BOOT, m))
    boot = csum[idx].sum(1) / ccnt[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    signs = rng.choice([-1.0, 1.0], size=(N_BOOT, m))
    perm = (signs * csum).sum(1) / ccnt.sum()
    obs = d.mean()
    p = (np.sum(np.abs(perm) >= abs(obs) - 1e-12) + 1) / (N_BOOT + 1)
    return {"diff": obs, "lo": lo, "hi": hi, "p": p, "n": len(d), "clusters": m}


# --------------------------------------------------------------------------- run record


def _git(*a: str) -> str:
    try:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def model_meta(kind: str, model_id: str | None) -> dict:
    """Model id / revision / encode provenance from the index meta json (None for bm25)."""
    if model_id is None:
        return {"kind": kind}
    slug = model_id.replace("/", "__")
    sub = {"sparseup": "sparse"}.get(kind, kind)
    cands = {"dense": [INDEX_DIR / "dense" / f"{slug}.json"], "sparse": [INDEX_DIR / "sparse" / f"{slug}.json"],
             "late": [INDEX_DIR / "late" / slug / "meta.json"]}.get(sub, [])
    for c in cands:
        if c.exists():
            m = json.loads(c.read_text())
            keep = ("model", "revision", "kind", "card_count", "oracle_ids_sha256", "scryfall_updated_at",
                    "query_prompt", "doc_prompt", "versions", "device", "created_at", "nnz_per_doc", "query_encoder")
            return {"kind": kind, **{k: m[k] for k in keep if k in m}}
    return {"kind": kind, "model": model_id, "meta": "missing"}


def write_run_record(label: str, args_ns, systems, M, lat, pooled, extra_stats, cards_n) -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe = "".join(c if c.isalnum() or c in "-_." else "-" for c in (label or "run"))
    sets = {}
    for st in SETS:
        p = EVAL_DIR / f"queries_{st}.yaml"
        sets[st] = {"file": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    rec = {
        "label": label,
        "timestamp": ts,
        "git": {"sha": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain", "--", "src", "eval/run.py", "jobs"))},
        "argv": sys.argv[1:],
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "corpus": {"file": "data/cards.parquet", "cards": cards_n,
                   "sha256": hashlib.sha256(CARDS.read_bytes()).hexdigest()},
        "query_sets": sets,
        "protocol": {"depth": DEPTH, "rrf_k": 60, "n_boot": N_BOOT, "seed": SEED,
                     "latency": "CPU, warm, sequential; fusion = sum of components + RRF"},
        "models": {name: model_meta(kind, mid) for name, (kind, mid) in BASE.items()},
        "fusions": FUSIONS,
        "systems": {},
        "paired": extra_stats,
    }
    for sy in systems:
        ls = np.array(list(lat[sy].values()))
        rec["systems"][sy] = {
            "para+deck_ndcg10": round(pooled[sy], 5),
            "latency_ms": {"p50": round(float(np.percentile(ls, 50)), 2), "p95": round(float(np.percentile(ls, 95)), 2)},
            "sets": {st: {m: round(float(np.mean([v[m] for v in M[sy][st].values()])), 5) for m in ("ndcg10", "r50", "r100", "mrr10")}
                     for st in SETS},
        }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / f"{ts}_{safe}.json"
    out.write_text(json.dumps(rec, indent=1) + "\n")
    return out


# --------------------------------------------------------------------------- report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="base systems to (re)run; default all")
    ap.add_argument("--refresh", nargs="*", default=[], help="ignore cache for these base systems")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--label", default="run", help="short name for the run record in eval/experiments/runs/")
    args = ap.parse_args()

    cards = pl.read_parquet(CARDS)
    oids = cards["oracle_id"].to_list()
    names = cards["name"].to_list()
    queries = load_all_queries()
    masks = Masks(cards)

    if not args.report_only:
        for system in args.only or list(BASE):
            run_base(system, queries, masks, refresh=system in args.refresh)

    caches = {s: load_cache(s) for s in BASE}
    missing = [s for s in BASE if not caches[s]]
    if missing:
        sys.exit(f"missing caches for {missing}; run without --report-only")

    # rankings[system][set/qid] = list of row idx; lat[system][set/qid] = ms
    rankings: dict[str, dict[str, list[int]]] = {}
    lat: dict[str, dict[str, float]] = {}
    for s in BASE:
        rankings[s] = {k: [h[0] for h in v["hits"]] for k, v in caches[s].items()}
        lat[s] = {k: v["ms"] for k, v in caches[s].items()}
    for f, comps in FUSIONS.items():
        rankings[f], lat[f] = {}, {}
        for key in rankings["bm25"]:
            lists = [caches[c][key]["hits"] for c in comps]
            t = time.perf_counter()
            fused = rrf(lists, weights=list(comps.values()), limit=DEPTH)
            fuse_ms = (time.perf_counter() - t) * 1000
            rankings[f][key] = [int(i) for i, _ in fused]
            lat[f][key] = sum(caches[c][key]["ms"] for c in comps) + fuse_ms
    systems = list(BASE) + list(FUSIONS)

    # per-query metrics
    M: dict[str, dict[str, dict[str, dict[str, float]]]] = {sy: {} for sy in systems}
    for sy in systems:
        for st in SETS:
            M[sy][st] = {q["id"]: per_query_metrics([oids[i] for i in rankings[sy][f"{st}/{q['id']}"]], q) for q in queries[st]}

    def mean(sy, st, metric, ids=None):
        vals = [m[metric] for qid, m in M[sy][st].items() if ids is None or qid in ids]
        return float(np.mean(vals))

    out: list[str] = []
    w = out.append
    w("# Retrieval evaluation results\n")
    w("Generated by `uv run --extra models python eval/run.py` (cached rankings in `eval/.cache/`).\n")
    w(f"- Corpus: {len(cards):,} cards (`data/cards.parquet`). Every query is restricted to its `filters`; funny cards are always excluded.")
    w(f"- Each component retrieves top-{DEPTH}; fusions are RRF (k=60) over those lists, truncated to {DEPTH}.")
    w("- nDCG@10 uses graded gains (2 for `graded` cards, 1 otherwise, linear gain). Recall@50/100 and MRR@10 are binary.")
    w("- Latency: CPU (6 cores), warm, one query at a time, includes query encoding + scoring + top-k with the filter mask. "
      "Fusion latency = sum of component latencies (sequential) + RRF time. Other processes were running on the machine, so treat p95 as noisy.")
    w("- Weighted fusions (`=0.5`, `=2`) were picked after looking at unweighted results on these same sets, so their numbers are optimistic.\n")

    # Summary
    w("## Summary: nDCG@10 per set\n")
    pd_ids = None
    pooled = {}
    for sy in systems:
        vals = [m["ndcg10"] for st in ("paraphrase", "deckneeds") for m in M[sy][st].values()]
        pooled[sy] = float(np.mean(vals))
    w("`para+deck` = pooled mean over the 252 paraphrase + deck-needs queries (the decision metric); "
      "`macro` = mean of the four set means. Latency in ms.\n")
    w("| system | late? | tags | handwritten | paraphrase | deckneeds | para+deck | macro | p50 ms | p95 ms |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    order = sorted(systems, key=lambda s: -pooled[s])
    for sy in order:
        sm = [mean(sy, st, "ndcg10") for st in SETS]
        ls = np.array(list(lat[sy].values()))
        w(f"| {sy} | {'yes' if uses_late(sy) else ''} | " + " | ".join(f"{100 * v:.1f}" for v in sm)
          + f" | **{100 * pooled[sy]:.1f}** | {100 * np.mean(sm):.1f} | {np.percentile(ls, 50):.0f} | {np.percentile(ls, 95):.0f} |")
    w("")

    # Per-set tables
    for st in SETS:
        w(f"## {st} ({len(queries[st])} queries)\n")
        w("| system | nDCG@10 | Recall@50 | Recall@100 | MRR@10 |")
        w("|---|---|---|---|---|")
        for sy in sorted(systems, key=lambda s: -mean(s, st, "ndcg10")):
            w(f"| {sy} | " + " | ".join(f"{100 * mean(sy, st, m):.1f}" for m in ("ndcg10", "r50", "r100", "mrr10")) + " |")
        w("")

    # Paraphrase drop
    w("## Paraphrase drop vs. originals (nDCG@10 points)\n")
    w("Mean over origin pairs of nDCG(original) − nDCG(reformulation); positive = worse on the reformulation. "
      "`orig` is the mean over the 111 originals (tags + handwritten).\n")
    orig = {}
    for st in ("tags", "handwritten"):
        for sy in systems:
            for qid, m in M[sy][st].items():
                orig.setdefault(sy, {})[qid] = m["ndcg10"]
    w("| system | orig | -para | drop -para | -need | drop -need |")
    w("|---|---|---|---|---|---|")
    for sy in sorted(systems, key=lambda s: -mean(s, "paraphrase", "ndcg10")):
        row = [100 * np.mean(list(orig[sy].values()))]
        for suf in ("-para", "-need"):
            pq = [q for q in queries["paraphrase"] if q["id"].endswith(suf)]
            v = np.array([M[sy]["paraphrase"][q["id"]]["ndcg10"] for q in pq])
            o = np.array([orig[sy][q["origin"]] for q in pq])
            row += [100 * v.mean(), 100 * (o - v).mean()]
        w(f"| {sy} | " + " | ".join(f"{x:.1f}" for x in row) + " |")
    w("")

    # Paired comparisons
    rng = np.random.default_rng(SEED)
    pd_keys = [(st, q) for st in ("paraphrase", "deckneeds") for q in queries[st]]
    cl = [clusters_for(st, q) for st, q in pd_keys]

    def vec(sy, keys=pd_keys):
        return np.array([M[sy][st][q["id"]]["ndcg10"] for st, q in keys])

    def fmt(r):
        return f"{100 * r['diff']:+.1f} [{100 * r['lo']:+.1f}, {100 * r['hi']:+.1f}] | {r['p']:.4f}"

    w("## Paired comparisons on paraphrase + deckneeds (nDCG@10 points)\n")
    w(f"Differences are row − column. 95% CI from a cluster bootstrap ({N_BOOT:,} resamples; paraphrase pairs that share an origin "
      "form one cluster, so 111 + 30 = 141 clusters), p from a paired sign-flip permutation test at the cluster level.\n")
    top3 = order[:3]
    w("| A | B | A − B [95% CI] | p |")
    w("|---|---|---|---|")
    for i in range(len(top3)):
        for j in range(i + 1, len(top3)):
            w(f"| {top3[i]} | {top3[j]} | " + fmt(paired_stats(vec(top3[i]), vec(top3[j]), cl, rng)) + " |")
    for ref in ("dense", "bm25"):
        if ref not in top3:
            w(f"| {top3[0]} | {ref} | " + fmt(paired_stats(vec(top3[0]), vec(ref), cl, rng)) + " |")
    w("")

    # Decision input
    best_late = max((s for s in systems if uses_late(s)), key=lambda s: pooled[s])
    best_nolate = max((s for s in systems if not uses_late(s)), key=lambda s: pooled[s])
    best_late_unw = max((s for s in systems if uses_late(s) and "=" not in s), key=lambda s: pooled[s])
    best_nolate_unw = max((s for s in systems if not uses_late(s) and "=" not in s), key=lambda s: pooled[s])
    hw_keys = [("handwritten", q) for q in queries["handwritten"]]
    hw_cl = [q["id"] for q in queries["handwritten"]]
    decision_lines = []
    w("## Decision input: late interaction vs. best in-process stack without it\n")
    w("Best system selected by pooled para+deck nDCG@10 within each group (selection on the test set favours both groups equally). "
      "The plan's rule: NextPlaid only if late-interaction beats the best non-late stack by ≥ 3 points on para+deck, and does no worse on handwritten.\n")
    w("| comparison | para+deck: late − no-late [95% CI] | p | handwritten: late − no-late [95% CI] | p |")
    w("|---|---|---|---|---|")
    best_nolate_hw = max((s for s in systems if not uses_late(s) and "=" not in s), key=lambda s: mean(s, "handwritten", "ndcg10"))
    comparisons = (
        ("all systems incl. weighted", best_late, best_nolate),
        ("unweighted only", best_late_unw, best_nolate_unw),
        ("single models", "late-mLateOn", "dense"),
        ("vs best non-late on handwritten", best_late_unw, best_nolate_hw),
    )
    for label, a, b in comparisons:
        r1 = paired_stats(vec(a), vec(b), cl, rng)
        r2 = paired_stats(vec(a, hw_keys), vec(b, hw_keys), hw_cl, rng)
        w(f"| {label}: **{a}** vs **{b}** | {fmt(r1)} | {fmt(r2)} |")
        decision_lines.append((label, a, b, r1, r2))
    # macro variant (equal weight to the two sets)
    w("")
    for label, a, b, *_ in decision_lines:
        mac = lambda s: (mean(s, "paraphrase", "ndcg10") + mean(s, "deckneeds", "ndcg10")) / 2  # noqa: E731
        w(f"- {label}: macro (paraphrase, deckneeds equally weighted) {a} {100 * mac(a):.1f} vs {b} {100 * mac(b):.1f} "
          f"→ {100 * (mac(a) - mac(b)):+.1f} points. Latency p50 {np.percentile(list(lat[a].values()), 50):.0f} ms vs "
          f"{np.percentile(list(lat[b].values()), 50):.0f} ms.")
    w("")

    # Extra paired comparisons (EXTRA_PAIRS), on para+deck and on handwritten.
    extra_stats = []
    w("## Additional paired comparisons (nDCG@10 points)\n")
    w("Row − column, same cluster bootstrap / sign-flip test as above. See `eval/experiments/` for the experiment write-ups.\n")
    w("| A | B | para+deck: A − B [95% CI] | p | handwritten: A − B [95% CI] | p |")
    w("|---|---|---|---|---|---|")
    for a, b in EXTRA_PAIRS:
        if a not in M or b not in M:
            continue
        r1 = paired_stats(vec(a), vec(b), cl, rng)
        r2 = paired_stats(vec(a, hw_keys), vec(b, hw_keys), hw_cl, rng)
        w(f"| {a} | {b} | {fmt(r1)} | {fmt(r2)} |")
        extra_stats.append({"a": a, "b": b, "metric": "ndcg10",
                            "para+deck": {k: float(v) for k, v in r1.items()},
                            "handwritten": {k: float(v) for k, v in r2.items()}})
    w("")

    # Observations
    w("## Observations\n")
    for suf in ("-para", "-need"):
        sub = [q for q in queries["paraphrase"] if q["id"].endswith(suf)]
        parts = []
        for sy in (order[0], "late-mLateOn", "dense", "sparse"):
            z = sum(1 for q in sub if M[sy]["paraphrase"][q["id"]]["mrr10"] == 0)
            parts.append(f"{sy} {z}/{len(sub)}")
        w(f"- Paraphrase `{suf}` queries with no relevant card in the top 10: " + ", ".join(parts)
          + ". Most paraphrase pairs are ties at 0, so the paraphrase set separates systems on a minority of queries.")
    w(f"- Late interaction here runs **in-process** (`LateInMemory`, brute-force MaxSim over fp32 token matrices): "
      f"late-mLateOn p50 {np.percentile(list(lat['late-mLateOn'].values()), 50):.0f} ms incl. query encoding, "
      f"dense p50 {np.percentile(list(lat['dense'].values()), 50):.0f} ms. So these quality numbers do not require NextPlaid; "
      "they price the model choice, not the runtime.")
    w("")

    # Qualitative
    w("## Qualitative: where late interaction (mLateOn) wins and loses\n")
    ref = best_nolate_unw
    w(f"Per-query nDCG@10 of `late-mLateOn` vs `{ref}` (best unweighted system without late interaction), "
      "over paraphrase + deckneeds. Top-5 card names per system; ✓ = relevant (✓✓ = grade 2).\n")
    diffs = []
    for st, q in pd_keys:
        diffs.append((M["late-mLateOn"][st][q["id"]]["ndcg10"] - M[ref][st][q["id"]]["ndcg10"], st, q))
    diffs.sort(key=lambda x: x[0])
    show = [("late-mLateOn", "late-mLateOn"), (ref, ref), ("bm25", "bm25")]

    def top5(sy, st, q):
        g = gains(q)
        parts = []
        for i in rankings[sy][f"{st}/{q['id']}"][:5]:
            mark = {2: " ✓✓", 1: " ✓"}.get(g.get(oids[i], 0), "")
            parts.append(names[i] + mark)
        return "; ".join(parts) or "(none)"

    for title, picks in (("Wins", diffs[::-1][:3]), ("Losses", diffs[:3])):
        w(f"### {title}\n")
        for d, st, q in picks:
            filt = f" — filters `{q['filters']}`" if q["filters"] else ""
            w(f"**{q['id']}** ({st}): “{q['query']}”{filt}  ")
            w(f"nDCG@10: mLateOn {100 * M['late-mLateOn'][st][q['id']]['ndcg10']:.1f}, {ref} {100 * M[ref][st][q['id']]['ndcg10']:.1f} (Δ {100 * d:+.1f})\n")
            for label, sy in show:
                w(f"- {label}: {top5(sy, st, q)}")
            w("")
    w("### Win/loss counts (late-mLateOn vs " + ref + ", |Δ nDCG@10| > 0.1)\n")
    for st in SETS:
        d = np.array([M["late-mLateOn"][st][q["id"]]["ndcg10"] - M[ref][st][q["id"]]["ndcg10"] for q in queries[st]])
        w(f"- {st}: {int((d > 0.1).sum())} wins, {int((d < -0.1).sum())} losses, {int((np.abs(d) <= 0.1).sum())} ties")
    w("")

    RESULTS_MD.write_text("\n".join(out))
    print(f"wrote {RESULTS_MD}")
    rec = write_run_record(args.label, args, systems, M, lat, pooled, extra_stats, len(cards))
    print(f"wrote {rec}")
    # console summary
    for sy in order:
        print(f"{sy:32s} " + " ".join(f"{100 * mean(sy, st, 'ndcg10'):5.1f}" for st in SETS) + f"  pd={100 * pooled[sy]:.1f}")


if __name__ == "__main__":
    main()

"""Retrieval evaluation of mLateOn (the project's only retriever) over the four query sets.

    uv run python eval/run.py                 # retrieve (cached) + print metrics + write a run record
    uv run python eval/run.py --refresh       # ignore the cache
    uv run python eval/run.py --label exact   # tag the run record

Every query is restricted to the cards passing ``filter_expr(**q["filters"])`` (funny cards excluded), as
in the MCP server, and ranked by ``mtg_search.mlateon.MLateOn`` (the server's own search). The top-100 per
query and its latency are cached in ``eval/.cache/mLateOn.json`` (reused while query text and filters are
unchanged). A run record goes to ``eval/experiments/runs/<UTC timestamp>_<label>.json``.

``eval/results.md`` is the EXP-001 comparison of mLateOn with BM25, dense and sparse retrievers and their
fusions; those retrievers were removed after mLateOn was chosen (code up to commit 61071da), so this script
no longer regenerates that report.
"""

from __future__ import annotations

import argparse
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

CARDS = ROOT / "data" / "cards.parquet"
CACHE_DIR = EVAL_DIR / ".cache"
RUNS_DIR = EVAL_DIR / "experiments" / "runs"
SYSTEM = "mLateOn"
SETS = ["tags", "handwritten", "paraphrase", "deckneeds"]
DEPTH = 100
SEED = 0
N_BOOT = 10000


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


def cache_path(system: str = SYSTEM) -> Path:
    return CACHE_DIR / f"{system}.json"


def load_cache(system: str = SYSTEM) -> dict:
    """Cached rankings; ``late-mLateOn`` is the EXP-001..012 reference cache (PyTorch ranker, identical ranks)."""
    p = cache_path(system)
    return json.loads(p.read_text()) if p.exists() else {}


def card_oracle_ids() -> list[str]:
    return pl.read_parquet(CARDS, columns=["oracle_id"])["oracle_id"].to_list()


def search_engine():
    from mtg_search.mlateon import MLateOn

    return MLateOn.local(card_oracle_ids())


def run(queries: dict[str, list[dict]], masks: Masks, refresh: bool) -> dict:
    cache = {} if refresh else load_cache()
    todo = [(s, q) for s in SETS for q in queries[s] if cache.get(f"{s}/{q['id']}", {}).get("key") != query_key(q)]
    if todo:
        r = search_engine()
        for w in ["destroy target creature", "draw a card whenever"]:  # warm-up
            r.search(w, DEPTH, masks.get({}))
        print(f"[{SYSTEM}] {len(todo)} queries to run")
        for s, q in todo:
            t = time.perf_counter()
            hits = r.search(q["query"], DEPTH, masks.get(q["filters"]))
            cache[f"{s}/{q['id']}"] = {"key": query_key(q), "ms": (time.perf_counter() - t) * 1000,
                                       "hits": [[int(a), round(float(b), 5)] for a, b in hits]}
        r.close()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path().write_text(json.dumps(cache))
    return cache


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


def _git(*a: str) -> str:
    try:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="ignore the ranking cache")
    ap.add_argument("--label", default="mlateon", help="short name for the run record")
    args = ap.parse_args()
    cards = pl.read_parquet(CARDS)
    oids = cards["oracle_id"].to_list()
    queries = load_all_queries()
    cache = run(queries, Masks(cards), args.refresh)
    sets = {}
    for st in SETS:
        m = [per_query_metrics([oids[h[0]] for h in cache[f"{st}/{q['id']}"]["hits"]], q) for q in queries[st]]
        sets[st] = {k: round(float(np.mean([x[k] for x in m])), 5) for k in ("ndcg10", "r50", "r100", "mrr10")} | {"n": len(m)}
        print(f"{st:12s} n={len(m):3d} " + " ".join(f"{k} {100 * v:5.1f}" for k, v in sets[st].items() if k != "n"))
    ms = np.array([v["ms"] for v in cache.values()])
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    rec = {"label": args.label, "timestamp": ts, "system": SYSTEM,
           "git": {"sha": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain", "--", "src", "eval/run.py"))},
           "host": {"platform": platform.platform(), "python": platform.python_version()},
           "corpus": {"file": "data/cards.parquet", "cards": cards.height, "sha256": hashlib.sha256(CARDS.read_bytes()).hexdigest()},
           "query_sets": {st: hashlib.sha256((EVAL_DIR / f"queries_{st}.yaml").read_bytes()).hexdigest() for st in SETS},
           "depth": DEPTH, "sets": sets,
           "latency_ms": {"p50": round(float(np.percentile(ms, 50)), 1), "p95": round(float(np.percentile(ms, 95)), 1)}}
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / f"{ts}_{args.label}.json"
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

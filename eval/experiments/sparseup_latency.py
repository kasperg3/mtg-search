"""EXP-004 latency re-measurement: sparseup vs sparse / dense / mLateOn under identical conditions.

    uv run --extra models python eval/experiments/sparseup_latency.py [--threads 2 6] [--n 120] [--json out.json]

The main eval measured each system at a different time, and while sparseup ran the machine had
load ~20 on 6 cores. That oversubscribed torch's 6 intra-op threads, and a 129 ms forward pass
became >1 s. This script runs the systems back to back on the same para+deck queries with a
fixed torch thread count. It repeats the measurement for each thread count and reports the
p50/p95 of search() (query encoding + scoring + top-100, unfiltered).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from eval.queries import load_queries  # noqa: E402

INDEX = ROOT / "index" / "embeddings"
SYSTEMS = {
    "sparseup": ("sparseup", "Linkup-Platform/linkup-sparseup-embed-v1"),
    "sparse": ("sparse", "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte"),
    "dense": ("dense", "google/embeddinggemma-2"),
    "late-mLateOn": ("late", "lightonai/mLateOn"),
}


def load(kind, mid):
    if kind == "sparseup":
        from mtg_search.retrievers.sparseup import SparseUpRetriever

        return SparseUpRetriever.from_dir(INDEX, mid)
    from mtg_search.retrievers import load_retriever

    return load_retriever(kind, mid, index_dir=INDEX, cards_parquet=ROOT / "data" / "cards.parquet")


def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, nargs="+", default=[2, 6])
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    qs = [q["query"] for s in ("paraphrase", "deckneeds") for q in load_queries(s)]
    qs = qs[:: max(1, len(qs) // args.n)][: args.n]
    res = {"loadavg_start": os.getloadavg(), "n_queries": len(qs), "results": {}}
    for name in args.systems:
        r = load(*SYSTEMS[name])
        for nt in args.threads:
            torch.set_num_threads(nt)
            for q in qs[:5]:
                r.search(q, 100)
            ms = []
            for q in qs:
                t = time.perf_counter()
                r.search(q, 100)
                ms.append((time.perf_counter() - t) * 1000)
            p50, p95 = np.percentile(ms, [50, 95])
            res["results"][f"{name}@{nt}t"] = {"p50": round(float(p50), 1), "p95": round(float(p95), 1), "loadavg": os.getloadavg()[0]}
            print(f"{name:14s} threads={nt}  p50 {p50:7.1f} ms  p95 {p95:7.1f} ms  (load {os.getloadavg()[0]:.1f})", flush=True)
        del r
        gc.collect()
    res["loadavg_end"] = os.getloadavg()
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()

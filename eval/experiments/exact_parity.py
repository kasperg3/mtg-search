"""EXP-012: parity of the shipped exact backend (fp32 ONNX + numpy MaxSim) with the benchmark ranker.

    OMP_NUM_THREADS=4 uv run python eval/experiments/exact_parity.py

Reference: the cached top-100 rankings of ``late-mLateOn`` (``LateInMemory``, PyTorch fp32, the ranker
behind every retrieval number in EXP-001..011, since removed) for all 363 eval queries under their filters
(``eval/.cache/late-mLateOn.json``). Candidate: ``mtg_search.mlateon.MLateOn`` over the same
cards and embeddings, as loaded by the Docker image. Reports top-k agreement, score differences,
per-set nDCG@10 for both, search latency and peak RSS.

Output: printed summary + eval/experiments/runs/<ts>_exact-parity.json
"""

from __future__ import annotations

import datetime as dt
import json
import resource
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from eval.run import CARDS, Masks, load_all_queries, load_cache, per_query_metrics  # noqa: E402
from mtg_search.mlateon import MLateOn  # noqa: E402

MODEL_DIR = ROOT / "index" / "models" / "lightonai__mLateOn-fp32"
EMB_DIR = ROOT / "index" / "embeddings" / "late" / "lightonai__mLateOn"
DEPTH = 100


def main() -> None:
    cards = pl.read_parquet(CARDS)
    oids = cards["oracle_id"].to_list()
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    be = MLateOn(oids, embeddings_dir=EMB_DIR, model_dir=MODEL_DIR)
    load_s = time.perf_counter() - t0
    rss_loaded = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ref = load_cache("late-mLateOn")
    queries = load_all_queries()
    masks = Masks(cards)
    be.search("warm up", 10)
    rows, lat = [], []
    per_set = {}
    for s, qs in queries.items():
        for q in qs:
            r = ref[f"{s}/{q['id']}"]["hits"]
            allowed = np.flatnonzero(masks.get(q["filters"]))
            t = time.perf_counter()
            hits = be.search(q["query"], DEPTH, allowed)
            lat.append((time.perf_counter() - t) * 1000)
            a = [h for h, _ in hits]
            b = [h for h, _ in r]
            ref_score = {h: sc for h, sc in r}
            common = [h for h in a if h in ref_score]
            rows.append({
                "set": s, "id": q["id"], "n_allowed": int(allowed.size),
                "top1": a[:1] == b[:1],
                **{f"overlap@{k}": len(set(a[:k]) & set(b[:k])) / max(1, min(k, len(b))) for k in (10, 50, 100)},
                "max_abs_score_diff": max((abs(sc - ref_score[h]) for h, sc in hits if h in ref_score), default=0.0),
                "n_common": len(common),
            })
            m_new = per_query_metrics([oids[h] for h in a], q)
            m_ref = per_query_metrics([oids[h] for h in b], q)
            per_set.setdefault(s, {"new": [], "ref": []})
            per_set[s]["new"].append(m_new["ndcg10"])
            per_set[s]["ref"].append(m_ref["ndcg10"])
    rss_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    agg = {k: float(np.mean([r[k] for r in rows])) for k in ("top1", "overlap@10", "overlap@50", "overlap@100")}
    agg["max_abs_score_diff"] = float(max(r["max_abs_score_diff"] for r in rows))
    agg["min_overlap@10"] = float(min(r["overlap@10"] for r in rows))
    ndcg = {s: {"exact_onnx": 100 * float(np.mean(v["new"])), "reference": 100 * float(np.mean(v["ref"]))}
            for s, v in per_set.items()}
    lat_s = sorted(lat)
    perf = {"load_s": round(load_s, 1), "search_ms_p50": round(st.median(lat), 1),
            "search_ms_p95": round(lat_s[int(0.95 * len(lat_s)) - 1], 1),
            "rss_mb_before": rss0 // 1024, "rss_mb_loaded": rss_loaded // 1024, "rss_mb_peak": rss_peak // 1024}
    print(f"{len(rows)} queries: " + ", ".join(f"{k} {v:.4f}" for k, v in agg.items()))
    for s, v in ndcg.items():
        print(f"  {s:12s} nDCG@10 exact-onnx {v['exact_onnx']:.2f}  reference {v['reference']:.2f}")
    print("perf:", perf)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    out = EXP / "runs" / f"{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%SZ}_exact-parity.json"
    out.write_text(json.dumps({"experiment": "EXP-012", "git": sha, "model_dir": str(MODEL_DIR.relative_to(ROOT)),
                               "reference": "eval/.cache/late-mLateOn.json (LateInMemory, PyTorch fp32)",
                               "n_queries": len(rows), "agreement": agg, "ndcg@10": ndcg, "perf": perf,
                               "threads": be.session.get_session_options().intra_op_num_threads,
                               "per_query": rows}, indent=1) + "\n")
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()

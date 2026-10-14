"""Eval smoke test for a refreshed index: late-mLateOn on the handwritten + paraphrase sets.

    uv run --extra models python scripts/eval_smoke.py [--tolerance 3.0]

Runs the in-process late retriever over ``index/embeddings`` and ``data/cards.parquet``
(no eval cache is read or written, ``eval/results.md`` is untouched) and compares
nDCG@10 per set with the ``late-mLateOn`` row of the summary table in
``eval/results.md``. Exits 1 if any set drops more than ``--tolerance`` points.

Query labels are oracle_ids from the snapshot the eval sets were built on; new cards
are unlabelled, so small drifts are expected (rebuild labels with
``scripts/refresh.py --rebuild-labels`` when they matter).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

SYSTEM = "late-mLateOn"
SETS = ("handwritten", "paraphrase")
SUMMARY_COLS = ["tags", "handwritten", "paraphrase", "deckneeds"]


def baseline_from_results(text: str, system: str = SYSTEM) -> dict[str, float]:
    """Parse nDCG@10 per set from the summary row ``| system | late? | tags | handwritten | ...``."""
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == system and len(cells) >= 6:
            return {s: float(re.sub(r"[^\d.]", "", cells[2 + i])) for i, s in enumerate(SUMMARY_COLS)}
    raise ValueError(f"no summary row for {system} in results.md")


def regressions(scores: dict[str, float], baseline: dict[str, float], tolerance: float) -> list[str]:
    return [f"{s}: {scores[s]:.1f} < baseline {baseline[s]:.1f} - {tolerance}" for s in scores
            if scores[s] < baseline[s] - tolerance]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tolerance", type=float, default=3.0, help="max allowed nDCG@10 drop in points")
    args = ap.parse_args(argv)

    import numpy as np
    import polars as pl

    from eval.queries import load_queries
    from eval.run import DEPTH, Masks, per_query_metrics
    from mtg_search import config
    from mtg_search.retrievers import load_retriever

    baseline = baseline_from_results((ROOT / "eval" / "results.md").read_text())
    cards = pl.read_parquet(config.CARDS_PARQUET)
    oids = cards["oracle_id"].to_list()
    masks = Masks(cards)
    r = load_retriever("late", config.LATE_MODEL, index_dir=config.INDEX_DIR / "embeddings",
                       cards_parquet=config.CARDS_PARQUET)
    scores: dict[str, float] = {}
    for s in SETS:
        qs = load_queries(s)
        t0 = time.perf_counter()
        vals = []
        for q in qs:
            hits = r.search(q["query"], DEPTH, masks.get(q.get("filters") or {}))
            vals.append(per_query_metrics([oids[i] for i, _ in hits], q)["ndcg10"])
        scores[s] = 100 * float(np.mean(vals))
        print(f"{SYSTEM} {s}: nDCG@10 {scores[s]:.1f} (baseline {baseline[s]:.1f}; {len(qs)} queries, "
              f"{time.perf_counter() - t0:.0f}s)")
    bad = regressions(scores, baseline, args.tolerance)
    if bad:
        print("eval smoke FAILED:\n  " + "\n  ".join(bad))
        return 1
    print("eval smoke passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

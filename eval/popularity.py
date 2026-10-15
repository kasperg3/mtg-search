"""Popularity (EDHREC rank) for offline evaluation only.

The shipped card table (``data/cards.parquet``) has no popularity column; ingest writes it to
``data/eval/popularity.parquet`` instead, and the Docker image never contains that file (the build
context is an allow-list, and on the HF dataset it lives under ``eval/``, outside ``data/*``).
Only evaluation code reads it, through this module:

* precon benchmark: the popularity-oracle reference baseline and the popularity-drift metric;
* eval-set builders: the order relevant cards are listed in (no effect on scores);
* agentic known-item tasks: the popular / obscure split.

Missing locally? ``python -m mtg_search.artifacts download --parts eval --dest data``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mtg_search import config  # noqa: E402


def load_popularity(path: Path | None = None) -> pl.DataFrame:
    """``oracle_id, edhrec_rank`` (rank may be null)."""
    p = Path(path or config.POPULARITY_PARQUET)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run `python -m mtg_search.ingest` or "
                                "`python -m mtg_search.artifacts download --parts eval --dest data`")
    return pl.read_parquet(p)


def with_popularity(cards: pl.DataFrame, path: Path | None = None) -> pl.DataFrame:
    """``cards`` plus ``edhrec_rank`` (left join on oracle_id; row order kept)."""
    if "edhrec_rank" in cards.columns:
        raise ValueError("card table already has edhrec_rank: it must come only from the eval-only popularity file")
    return cards.join(load_popularity(path), on="oracle_id", how="left", maintain_order="left")

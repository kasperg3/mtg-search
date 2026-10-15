"""Retrievers over the card table; all return ``(row_index, score)`` lists."""

from __future__ import annotations

from pathlib import Path

from .base import Retriever, model_slug, top_k

__all__ = ["Retriever", "load_retriever", "model_slug", "top_k"]


def load_retriever(kind: str, model_id: str | None = None, *, index_dir: str | Path, cards_parquet: str | Path | None = None,
                   oracle_ids=None):
    """Construct a retriever. ``index_dir`` contains ``dense/``, ``sparse/``, ``late/``. With ``oracle_ids``
    (the card table in row order) embedding-based retrievers verify their row alignment first."""
    if kind in ("bm25", "bm25+tags"):
        from .bm25 import BM25Retriever

        return BM25Retriever.from_parquet(cards_parquet, with_tags=kind == "bm25+tags")
    if kind == "dense":
        from .dense import DenseRetriever

        return DenseRetriever.from_dir(index_dir, model_id, oracle_ids=oracle_ids)
    if kind == "sparse":
        from .sparse import SparseRetriever

        return SparseRetriever.from_dir(index_dir, model_id, oracle_ids=oracle_ids)
    if kind == "late":
        from .late import LateInMemory

        return LateInMemory.from_dir(index_dir, model_id, oracle_ids=oracle_ids)
    raise ValueError(f"unknown retriever kind {kind!r}")

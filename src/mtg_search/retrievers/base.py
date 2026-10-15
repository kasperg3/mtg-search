"""Common retriever protocol and helpers.

Every retriever indexes the rows of ``data/cards.parquet`` in file order and returns
row indices into that table (not oracle ids), so callers can map back cheaply.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int, allowed: np.ndarray | None = None) -> list[tuple[int, float]]:
        """Return up to ``k`` ``(row_index, score)`` pairs, best first.

        ``allowed`` is an optional boolean mask (length = number of cards); rows where it
        is False are never returned.
        """
        ...


def model_slug(model_id: str) -> str:
    """Filesystem-safe name for a HF model id (``org/name`` -> ``org__name``)."""
    return model_id.replace("/", "__")


def oracle_ids_digest(oracle_ids) -> str:
    """Order-sensitive digest of the card table, stored in embedding meta files."""
    h = hashlib.sha256()
    for oid in oracle_ids:
        h.update(oid.encode())
        h.update(b"\n")
    return h.hexdigest()


def load_meta(path: Path, oracle_ids=None) -> dict:
    """Read an embedding meta file; with ``oracle_ids`` (the loaded card table's ids, in row order) also
    verify that the embeddings were built for exactly that table (review m6)."""
    meta = json.loads(Path(path).read_text())
    if oracle_ids is not None:
        check_alignment(meta, oracle_ids, str(path))
    return meta


def check_alignment(meta: dict, oracle_ids, where: str = "embeddings") -> None:
    """Embedding rows are matched to cards.parquet rows by position, so a refreshed card table with stale
    embeddings would silently misalign every result. Refuse that instead."""
    ids = list(oracle_ids)
    want = meta.get("oracle_ids_sha256")
    if want is None:
        raise RuntimeError(f"{where}: no oracle_ids_sha256 in the meta, cannot verify row alignment with the card table")
    if meta.get("card_count", len(ids)) != len(ids) or want != oracle_ids_digest(ids):
        raise RuntimeError(f"{where}: embeddings were built for a different card table ({meta.get('card_count')} cards, "
                           f"digest {want[:12]}) than the one loaded ({len(ids)} cards, digest {oracle_ids_digest(ids)[:12]}); "
                           "re-encode or use the matching cards.parquet")


def top_k(scores: np.ndarray, k: int, allowed: np.ndarray | None = None) -> list[tuple[int, float]]:
    """Top-k over a dense score vector, honouring an optional boolean mask."""
    scores = np.asarray(scores, dtype=np.float32)
    if allowed is not None:
        allowed = np.asarray(allowed, dtype=bool)
        if allowed.shape != scores.shape:
            raise ValueError(f"allowed mask shape {allowed.shape} != scores shape {scores.shape}")
        idx = np.flatnonzero(allowed)
        sub = scores[idx]
    else:
        idx = None
        sub = scores
    k = min(k, sub.shape[0])
    if k <= 0:
        return []
    part = np.argpartition(-sub, k - 1)[:k]
    part = part[np.argsort(-sub[part], kind="stable")]
    rows = part if idx is None else idx[part]
    return [(int(r), float(s)) for r, s in zip(rows, sub[part], strict=True)]

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


def load_meta(path: Path) -> dict:
    return json.loads(Path(path).read_text())


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

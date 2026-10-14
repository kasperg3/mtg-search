"""Late interaction (ColBERT) retrieval with brute-force in-memory MaxSim.

Query encoding uses ``sentence_transformers.MultiVectorEncoder`` (sentence-transformers
>= 6), which loads PyLate-format checkpoints (lightonai/mLateOn) and Stanford ColBERT
checkpoints (answerai-colbert-small-v1) and reproduces PyLate's embeddings exactly
(verified bit-identical against pylate 1.6). PyLate itself is not a dependency because it
pins sentence-transformers==5.3 / transformers<=5.3, which cannot load embeddinggemma-2.

Artifacts (``jobs/encode.py --kind late``):
  ``late/<slug>/tokens.npy``   float16 (total_tokens, dim), all docs' token vectors concatenated
  ``late/<slug>/offsets.npy``  int64 (n_cards + 1); doc i owns tokens[offsets[i]:offsets[i+1]]
  ``late/<slug>/meta.json``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import load_meta, model_slug, top_k


def maxsim_scores(
    tokens,
    offsets: np.ndarray,
    q: np.ndarray,
    *,
    docs_per_chunk: int = 4096,
) -> np.ndarray:
    """Sum over query tokens of max over doc tokens of <q_i, d_j>, for every doc.

    ``tokens`` is a torch tensor (any float dtype) of shape (T, dim); ``q`` is (Lq, dim).
    Work is done in float32 chunks aligned to document boundaries.
    """
    import torch

    n = offsets.shape[0] - 1
    qt = torch.from_numpy(np.ascontiguousarray(q, dtype=np.float32)).T.contiguous()  # (dim, Lq)
    out = np.empty(n, dtype=np.float32)
    lengths_all = torch.from_numpy(np.diff(offsets))
    for d0 in range(0, n, docs_per_chunk):
        d1 = min(n, d0 + docs_per_chunk)
        t0, t1 = int(offsets[d0]), int(offsets[d1])
        sim = tokens[t0:t1].float() @ qt  # (chunk_tokens, Lq)
        lens = lengths_all[d0:d1]
        if bool((lens > 0).all()):
            mx = torch.segment_reduce(sim, "max", lengths=lens, axis=0)
        else:  # empty docs (shouldn't happen) score 0
            mx = torch.zeros((d1 - d0, sim.shape[1]))
            nz = lens > 0
            mx[nz] = torch.segment_reduce(sim, "max", lengths=lens[nz], axis=0)
        out[d0:d1] = mx.sum(dim=1).numpy()
    return out


def build_buckets(tokens, offsets: np.ndarray, *, step: int = 16, dtype=None):
    """Group docs into length buckets padded to a multiple of ``step`` tokens.

    Returns a list of ``(rows, block)`` where ``block`` is (n_docs_in_bucket, L, dim). Padding
    repeats the doc's first token, which cannot change a max. Contiguous padded blocks make
    MaxSim a plain batched matmul + amax: ~70 ms for mLateOn's 2.2M tokens on 6 CPU cores
    vs ~110 ms for the segment-reduce path and ~270 ms when converting from fp16 per query.
    """
    import torch

    lens = np.diff(offsets)
    padded = np.maximum(step, ((lens + step - 1) // step) * step)
    out = []
    for length in np.unique(padded):
        rows = np.flatnonzero(padded == length)
        ar = np.arange(length)[None, :]
        idx = offsets[rows][:, None] + np.where(ar < lens[rows][:, None], ar, 0)
        block = tokens[torch.from_numpy(idx.ravel())]
        if dtype is not None:
            block = block.to(dtype)
        out.append((rows, block.view(len(rows), int(length), -1)))
    return out


class LateInMemory:
    def __init__(
        self,
        model_id: str,
        tokens: np.ndarray,
        offsets: np.ndarray,
        meta: dict | None = None,
        *,
        device: str = "cpu",
        dtype: str = "float32",
    ):
        """``dtype`` is the in-RAM compute dtype; float32 (default) is fastest on CPUs without
        native fp16/bf16 matmul (mLateOn: ~1.25 GB). float16 halves RAM but is far slower."""
        import torch
        from sentence_transformers import MultiVectorEncoder

        self.meta = meta or {}
        self.model_id = model_id
        self.name = f"late:{model_id}"
        self.model = MultiVectorEncoder(model_id, device=device, revision=self.meta.get("revision"))
        self.model.eval()
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.n = self.offsets.shape[0] - 1
        if self.offsets[-1] != tokens.shape[0]:
            raise ValueError(f"offsets end {self.offsets[-1]} != token rows {tokens.shape[0]}")
        self.dtype = getattr(torch, dtype)
        tok = torch.from_numpy(np.ascontiguousarray(tokens))
        self.buckets = build_buckets(tok, self.offsets, dtype=self.dtype)

    @classmethod
    def from_dir(cls, root: str | Path, model_id: str, **kw) -> LateInMemory:
        d = Path(root) / "late" / model_slug(model_id)
        meta = load_meta(d / "meta.json")
        return cls(model_id, np.load(d / "tokens.npy", mmap_mode="r"), np.load(d / "offsets.npy"), meta, **kw)

    def encode_query(self, query: str) -> np.ndarray:
        q = self.model.encode_query([query], convert_to_numpy=True)
        return np.asarray(q[0], dtype=np.float32)

    def scores(self, query: str, allowed: np.ndarray | None = None) -> np.ndarray:
        """MaxSim score per card; disallowed cards get -inf (and are not computed)."""
        import torch

        qt = torch.from_numpy(self.encode_query(query)).to(self.dtype).T.contiguous()
        out = np.full(self.n, -np.inf, dtype=np.float32)
        with torch.inference_mode():
            for rows, block in self.buckets:
                if allowed is not None:
                    sel = allowed[rows]
                    if not sel.any():
                        continue
                    if sel.mean() < 0.5:
                        rows = rows[sel]
                        block = block[torch.from_numpy(sel)]
                n_b, length, dim = block.shape
                sims = (block.reshape(n_b * length, dim) @ qt).view(n_b, length, -1)
                out[rows] = sims.amax(dim=1).sum(dim=1).float().numpy()
        if allowed is not None:
            out[~allowed] = -np.inf
        return out

    def search(self, query: str, k: int, allowed: np.ndarray | None = None) -> list[tuple[int, float]]:
        if allowed is not None:
            allowed = np.asarray(allowed, dtype=bool)
        return top_k(self.scores(query, allowed), k, allowed)

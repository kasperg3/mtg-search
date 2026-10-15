"""Dense single-vector retrieval: SentenceTransformer query encoder + precomputed doc matrix.

Artifacts (written by ``jobs/encode.py --kind dense``):
  ``dense/<slug>.npy``        float16 (n_cards, dim), L2-normalised rows
  ``dense/<slug>.json``       meta: model, revision, query_prompt, doc_prompt, load kwargs, ...
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .base import load_meta, model_slug, top_k

# Per-model loading quirks. Keep in sync with jobs/encode.py MODEL_LOAD.
MODEL_LOAD: dict[str, dict] = {
    # Multimodal (740M); text-only loading drops the vision/audio towers (~270M left).
    # Must not run in float16 (activations overflow) -> float32 on CPU.
    "google/embeddinggemma-2": {"config_kwargs": {"vision_config": None, "audio_config": None}},
}


class DenseRetriever:
    def __init__(self, model_id: str, doc_matrix: np.ndarray, meta: dict | None = None, *, device: str = "cpu"):
        import torch
        from sentence_transformers import SentenceTransformer

        self.meta = meta or {}
        self.model_id = model_id
        self.name = f"dense:{model_id}"
        kwargs = dict(MODEL_LOAD.get(model_id, {}))
        revision = self.meta.get("revision")
        self.model = SentenceTransformer(
            model_id,
            device=device,
            revision=revision,
            model_kwargs={"dtype": torch.float32},
            **kwargs,
        )
        # Stored as float16; held as float32 (34k x 768 x 4B ~ 100 MB) for fast BLAS matvec.
        self.docs = np.asarray(doc_matrix, dtype=np.float32)
        self.n, self.dim = self.docs.shape
        if "query_prompt" in self.meta:
            self.query_prompt = self.meta["query_prompt"]
        else:
            self.query_prompt = (self.model.prompts or {}).get("query", "")
        self.truncate_dim = self.meta.get("truncate_dim")

    @classmethod
    def from_dir(cls, root: str | Path, model_id: str, *, oracle_ids=None, **kw) -> DenseRetriever:
        root = Path(root) / "dense"
        slug = model_slug(model_id)
        meta = load_meta(root / f"{slug}.json", oracle_ids)
        mat = np.load(root / f"{slug}.npy", mmap_mode="r")
        return cls(model_id, mat, meta, **kw)

    def encode_query(self, query: str) -> np.ndarray:
        q = self.model.encode(
            [query],
            prompt=self.query_prompt or None,
            normalize_embeddings=True,
            convert_to_numpy=True,
            truncate_dim=self.truncate_dim,
        )[0]
        return q.astype(np.float32)

    def scores(self, query: str) -> np.ndarray:
        q = self.encode_query(query)
        return self.docs @ q

    def search(self, query: str, k: int, allowed: np.ndarray | None = None) -> list[tuple[int, float]]:
        return top_k(self.scores(query), k, allowed)


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import sys

    r = DenseRetriever.from_dir(sys.argv[1], sys.argv[2])
    t = time.perf_counter()
    print(r.search(sys.argv[3], 5), time.perf_counter() - t)

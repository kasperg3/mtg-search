"""Learned sparse retrieval with an inference-free query side.

``opensearch-neural-sparse-encoding-doc-v3-gte`` encodes documents with a transformer
(done offline by ``jobs/encode.py --kind sparse``) but encodes queries with only its
tokenizer and a static per-token weight table (the model's ``query_0_SparseStaticEmbedding``
module). So no transformer runs at query time: score = sum_t w_q[t] * D[doc, t].

Artifacts:
  ``sparse/<slug>.npz``   scipy CSR float32 (n_cards, vocab_size)
  ``sparse/<slug>.json``  meta
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import load_meta, model_slug, top_k

QUERY_SUBFOLDER = "query_0_SparseStaticEmbedding"


class SparseQueryEncoder:
    """Tokenizer + idf-like weight lookup; mirrors SparseStaticEmbedding.forward."""

    def __init__(self, model_id: str, revision: str | None = None):
        from huggingface_hub import hf_hub_download
        from safetensors.numpy import load_file
        from tokenizers import Tokenizer

        tok_path = hf_hub_download(model_id, f"{QUERY_SUBFOLDER}/tokenizer.json", revision=revision)
        w_path = hf_hub_download(model_id, f"{QUERY_SUBFOLDER}/model.safetensors", revision=revision)
        self.tokenizer = Tokenizer.from_file(tok_path)
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(max_length=512)
        self.weight = load_file(w_path)["weight"].astype(np.float32)

    def encode(self, query: str) -> tuple[np.ndarray, np.ndarray]:
        # Special tokens ([CLS]/[SEP]) are excluded, as SparseEncoder.encode_query does.
        ids = np.unique(np.asarray(self.tokenizer.encode(query, add_special_tokens=False).ids, dtype=np.int64))
        w = self.weight[ids]
        keep = w != 0
        return ids[keep], w[keep]


class SparseRetriever:
    def __init__(self, model_id: str, doc_matrix, meta: dict | None = None):
        import scipy.sparse as sp

        self.meta = meta or {}
        self.model_id = model_id
        self.name = f"sparse:{model_id}"
        self.encoder = SparseQueryEncoder(model_id, self.meta.get("revision"))
        # Column slicing is what queries need -> CSC.
        self.docs = sp.csc_matrix(doc_matrix, dtype=np.float32)
        self.n = self.docs.shape[0]

    @classmethod
    def from_dir(cls, root: str | Path, model_id: str) -> SparseRetriever:
        import scipy.sparse as sp

        root = Path(root) / "sparse"
        slug = model_slug(model_id)
        meta = load_meta(root / f"{slug}.json")
        return cls(model_id, sp.load_npz(root / f"{slug}.npz"), meta)

    def scores(self, query: str) -> np.ndarray:
        ids, w = self.encoder.encode(query)
        if ids.size == 0:
            return np.zeros(self.n, dtype=np.float32)
        return np.asarray(self.docs[:, ids] @ w, dtype=np.float32).ravel()

    def search(self, query: str, k: int, allowed: np.ndarray | None = None) -> list[tuple[int, float]]:
        hits = top_k(self.scores(query), k, allowed)
        return [(i, s) for i, s in hits if s > 0]

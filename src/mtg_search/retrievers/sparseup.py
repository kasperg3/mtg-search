"""SPARSEUP learned sparse retrieval (``Linkup-Platform/linkup-sparseup-embed-v1``).

Unlike the inference-free ``SparseRetriever``, SPARSEUP encodes *queries* with the full
ModernBERT (149M) SPLADE head too: ``"[Q] " + query`` (prefix attended, not pooled, max
128 tokens) -> MLM logits -> ``log1p(relu(x - 15))`` -> top-12 vocab dims per position ->
max-pool -> case/space vocab fold. Documents were encoded offline the same way with
``"[D] "`` (``jobs/encode.py --kind sparse``). Score = dot product.

Because both sides are bags of (folded) vocabulary terms, every score decomposes into
per-term contributions ``q[t] * d[t]``; ``explain`` returns them, flagging expansion
terms that are not literally in the card text (or the query).

Artifacts: ``sparse/<slug>.npz`` (CSR float32, n_cards x 50370) + ``sparse/<slug>.json``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import load_meta, model_slug, top_k

MODEL_ID = "Linkup-Platform/linkup-sparseup-embed-v1"


class SparseUpRetriever:
    def __init__(self, model_id: str, doc_matrix, meta: dict | None = None, *, device: str = "cpu", doc_texts=None):
        import scipy.sparse as sp
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel

        self.meta = meta or {}
        self.model_id = model_id
        self.name = f"sparseup:{model_id}"
        # The remote code ignores `revision` on some paths, so load a pinned local snapshot.
        path = snapshot_download(model_id, revision=self.meta.get("revision"))
        self.model = AutoModel.from_pretrained(path, trust_remote_code=True, dtype=torch.float32).to(device).eval()
        self.device = device
        cfg = self.model.config
        self.query_prefix, self.query_max_length = cfg.query_prefix, cfg.query_max_length
        self.doc_max_length = cfg.doc_max_length
        self.tokenizer = self.model._get_tokenizer()
        self.fold = self.model.vocab_fold_index.cpu().numpy()
        self.csr = sp.csr_matrix(doc_matrix, dtype=np.float32)
        self.docs = self.csr.tocsc()  # column slicing for scoring
        self.n = self.docs.shape[0]
        self.doc_texts = doc_texts

    @classmethod
    def from_dir(cls, root: str | Path, model_id: str = MODEL_ID, **kw) -> SparseUpRetriever:
        import scipy.sparse as sp

        root = Path(root) / "sparse"
        slug = model_slug(model_id)
        return cls(model_id, sp.load_npz(root / f"{slug}.npz"), load_meta(root / f"{slug}.json"), **kw)

    # ------------------------------------------------------------------ encoding

    def encode_query(self, query: str) -> tuple[np.ndarray, np.ndarray]:
        """-> (vocab ids, weights) of the query's non-zero (folded) dims."""
        import torch

        ids, attn, pool, _ = self.model._tokenize([query], self.query_prefix, self.query_max_length)
        with torch.inference_mode():
            v = self.model(ids.to(self.device), attn.to(self.device), pool.to(self.device))[0].float().cpu().numpy()
        nz = np.flatnonzero(v)
        return nz.astype(np.int64), v[nz]

    def scores(self, query: str) -> np.ndarray:
        ids, w = self.encode_query(query)
        if ids.size == 0:
            return np.zeros(self.n, dtype=np.float32)
        return np.asarray(self.docs[:, ids] @ w, dtype=np.float32).ravel()

    def search(self, query: str, k: int, allowed: np.ndarray | None = None) -> list[tuple[int, float]]:
        hits = top_k(self.scores(query), k, allowed)
        return [(i, s) for i, s in hits if s > 0]

    # ------------------------------------------------------------------ explainability

    def _literal_dims(self, text: str, max_length: int) -> set[int]:
        """Folded vocab dims of the text's own subtokens (what an exact-match index would hold)."""
        ids = self.tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
        return {int(self.fold[i]) for i in ids}

    @staticmethod
    def display(token: str) -> str:
        """``Ġword`` -> ``word``; word-internal pieces get a leading ``·`` (``·ify``)."""
        return token[1:] if token.startswith("Ġ") else "·" + token

    def explain(self, query: str, row_idx: int, top_n: int = 8, doc_text: str | None = None) -> list[tuple[str, float, bool]]:
        """Stable API: top ``top_n`` ``(term, contribution, is_expansion)`` for score(query, card).

        ``term`` is the display form (``Ġ`` stripped; word-internal pieces as ``·x``),
        ``contribution`` = q[t] * d[t] (they sum to the score over all shared terms), and
        ``is_expansion`` is True when the term does not occur literally in the card text
        (the encoder added it). Needs the card text: pass ``doc_text`` or construct the
        retriever with ``doc_texts``. See ``explain_detail`` for weights and query-side flags.
        """
        ex = self.explain_detail(query, row_idx, top_n, doc_text)
        return [(t["term"], t["contrib"], t["in_card"] is False) for t in ex["terms"]]

    def explain_detail(self, query: str, row_idx: int, top_n: int = 8, doc_text: str | None = None) -> dict:
        """Decompose score(query, card ``row_idx``) into per-term contributions.

        Returns ``{"score", "n_shared", "coverage", "terms": [...]}`` where each term is
        ``{"term", "token", "q", "d", "contrib", "share", "subword", "in_query", "in_card"}``:
        ``in_card`` False marks a document-side expansion (the card never says the word,
        the encoder added it); ``in_query`` False marks a query-side expansion. ``coverage``
        is the share of the score explained by the returned ``top_n`` terms.
        """
        ids, w = self.encode_query(query)
        row = self.csr.getrow(row_idx)
        d = dict(zip(row.indices.tolist(), row.data.tolist(), strict=True))
        contrib = [(int(t), float(qw), d[int(t)]) for t, qw in zip(ids, w, strict=True) if int(t) in d]
        total = float(sum(qw * dw for _, qw, dw in contrib))
        contrib.sort(key=lambda x: -x[1] * x[2])
        if doc_text is None and self.doc_texts is not None:
            doc_text = self.doc_texts[row_idx]
        card_lit = self._literal_dims(doc_text, self.doc_max_length) if doc_text is not None else None
        q_lit = self._literal_dims(query, self.query_max_length)
        terms = []
        for t, qw, dw in contrib[:top_n]:
            tok = self.tokenizer.convert_ids_to_tokens(t)
            terms.append(
                {
                    "term": self.display(tok),
                    "token": tok,
                    "q": round(qw, 3),
                    "d": round(dw, 3),
                    "contrib": round(qw * dw, 3),
                    "share": round(qw * dw / total, 3) if total else 0.0,
                    "subword": not tok.startswith("Ġ"),
                    "in_query": t in q_lit,
                    "in_card": None if card_lit is None else t in card_lit,
                }
            )
        shown = sum(x["contrib"] for x in terms)
        return {"score": round(total, 3), "n_shared": len(contrib), "coverage": round(shown / total, 3) if total else 0.0, "terms": terms}


def format_explanation(ex: dict) -> str:
    """One-line human-readable form: ``term(q*d=c)`` with ``[+q]``/``[+d]`` expansion marks."""
    parts = []
    for t in ex["terms"]:
        marks = ("" if t["in_query"] else "+q") + ("" if t["in_card"] in (True, None) else "+d")
        parts.append(f"{t['term']}{'[' + marks + ']' if marks else ''} {t['contrib']:.1f}")
    return f"score {ex['score']:.1f} ({ex['n_shared']} shared terms, top {len(ex['terms'])} = {100 * ex['coverage']:.0f}%): " + ", ".join(parts)

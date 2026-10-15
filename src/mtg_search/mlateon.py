"""The search engine: exact late interaction with lightonai/mLateOn (the project's only retriever).

* Queries: ``tokenizers`` + ``onnxruntime`` running mLateOn's published fp32 ``model.onnx`` (``[Q] `` prefix,
  ``<bos> [Q] ... <eos>``, no expansion, per-token L2 normalisation); per-token cosine 1.00000 against
  sentence-transformers on PyTorch.
* Documents: the precomputed token vectors (``embeddings/late/lightonai__mLateOn/tokens.npy``, fp16 on
  disk, fp32 in RAM), encoded on GPU by ``jobs/encode.py``.
* Score: for every card, the sum over query tokens of the max dot product with the card's tokens
  (MaxSim), computed exactly over all ~35k cards. There is no approximate index; at this size exact
  search is one matrix product (~100 ms). The server, the evaluations and the precon benchmark all use
  this class, so every reported number describes what the server returns (EXP-012).

Rows are aligned with ``cards.parquet`` by order; the constructor refuses embeddings built for a
different card table (``meta.json`` ``oracle_ids_sha256``) or a different model revision.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import config

log = logging.getLogger(__name__)

QUERY_PREFIX = "[Q] "
SLUG = config.LATE_SLUG
EMBEDDINGS_SUBDIR = config.EMBEDDINGS_SUBDIR
MODEL_SUBDIR = config.MODEL_SUBDIR


def oracle_ids_digest(oracle_ids) -> str:
    """Order-sensitive digest of the card table, stored in the embeddings' meta.json."""
    h = hashlib.sha256()
    for oid in oracle_ids:
        h.update(oid.encode())
        h.update(b"\n")
    return h.hexdigest()


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


class MLateOn:
    def __init__(self, oracle_ids: list[str], *, embeddings_dir: str | Path, model_dir: str | Path,
                 threads: int | None = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        t0 = time.perf_counter()
        embeddings_dir, model_dir = Path(embeddings_dir), Path(model_dir)
        meta = json.loads((embeddings_dir / "meta.json").read_text())
        check_alignment(meta, oracle_ids, str(embeddings_dir))
        src = model_dir / "source.json"
        if src.exists() and meta.get("revision") and json.loads(src.read_text()).get("revision") != meta["revision"]:
            raise RuntimeError(f"query encoder revision {json.loads(src.read_text()).get('revision')} != "
                               f"embeddings revision {meta['revision']} (queries and docs must come from one checkpoint)")
        cfg = json.loads((model_dir / "onnx_config.json").read_text())
        if cfg.get("query_prefix") != QUERY_PREFIX or cfg.get("do_query_expansion"):
            raise RuntimeError(f"unexpected query config in {model_dir}/onnx_config.json: {cfg}")
        self.offsets = np.load(embeddings_dir / "offsets.npy").astype(np.int64)
        if np.any(np.diff(self.offsets) <= 0):
            raise RuntimeError("every card must have at least one token vector")
        self.tokens = np.load(embeddings_dir / "tokens.npy").astype(np.float32)  # (T, dim)
        self.n = len(self.offsets) - 1
        threads = threads or int(os.environ.get("MTG_SEARCH_THREADS", "0")) or min(8, os.cpu_count() or 4)
        if hasattr(ort, "disable_telemetry_events"):  # no telemetry from the server (also quiets --read-only)
            ort.disable_telemetry_events()
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_dir / "model.onnx"), so, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        self._init_scoring(threads)
        log.info("mLateOn: %d cards, %d token vectors, loaded in %.1fs", self.n, len(self.tokens), time.perf_counter() - t0)

    @classmethod
    def local(cls, oracle_ids: list[str], index_dir: str | Path | None = None, **kw) -> MLateOn:
        """From a checkout: ``index/embeddings/late/lightonai__mLateOn`` and ``index/models/lightonai__mLateOn-fp32``
        (``python -m mtg_search.artifacts download --parts embeddings,model --dest index``)."""
        root = Path(index_dir or config.INDEX_DIR)
        return cls(oracle_ids, embeddings_dir=root / EMBEDDINGS_SUBDIR, model_dir=root / MODEL_SUBDIR, **kw)

    def _init_scoring(self, threads: int) -> None:
        # MaxSim is memory-bandwidth bound: score card-aligned chunks in parallel (numpy/BLAS release the GIL);
        # run with OPENBLAS_NUM_THREADS=1 so the pool, not BLAS, owns the cores.
        # ~4k tokens per chunk keeps the (chunk x query-tokens) intermediate in cache: 1 query ~100 ms,
        # a 4-query search ~330 ms on 6 cores (EXP-012); bigger chunks make batches slower
        n_chunks = max(threads * 4, len(self.tokens) // 4096)
        bounds = np.unique(np.concatenate([[0], np.searchsorted(self.offsets, np.linspace(0, self.offsets[-1], n_chunks + 1)[1:-1]), [self.n]]))
        self._chunks = [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:], strict=True) if b > a]
        self._pool = ThreadPoolExecutor(threads, thread_name_prefix="maxsim")
        self.threads = threads

    def encode_query(self, query: str) -> np.ndarray:
        """(n_tokens, dim) float32, L2-normalised per token."""
        enc = self.tokenizer.encode(QUERY_PREFIX + query)
        ids = np.asarray([enc.ids], dtype=np.int64)
        out = self.session.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0][0]
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    def scores_many(self, queries: list[str]) -> np.ndarray:
        """(len(queries), n) MaxSim scores. All queries share one pass over the doc tokens (the cost is
        reading ~1.1 GB of fp32 vectors, not the arithmetic), so 4 queries cost little more than 1."""
        qs = [self.encode_query(q) for q in queries]
        q_all = np.concatenate(qs).T.copy()  # (dim, sum Lq)
        cuts = np.cumsum([0] + [len(q) for q in qs])
        out = np.empty((len(qs), self.n), dtype=np.float32)

        def part(ab):
            a, b = ab
            t0, t1 = self.offsets[a], self.offsets[b]
            mx = np.maximum.reduceat(self.tokens[t0:t1] @ q_all, self.offsets[a:b] - t0, axis=0)  # (b-a, sum Lq)
            return a, b, np.add.reduceat(mx, cuts[:-1], axis=1)  # per query: sum over its tokens

        for a, b, v in self._pool.map(part, self._chunks):
            out[:, a:b] = v.T
        return out

    def search(self, query: str, k: int, allowed_rows: np.ndarray | None = None) -> list[tuple[int, float]]:
        return self.search_many([query], k, allowed_rows)[0]

    def search_many(self, queries: list[str], k: int, allowed_rows: np.ndarray | None = None) -> list[list[tuple[int, float]]]:
        """Top-k (row, score) per query among ``allowed_rows`` (all rows if None)."""
        return [self._top(s, k, allowed_rows) for s in self.scores_many(queries)]

    def _top(self, s: np.ndarray, k: int, allowed_rows: np.ndarray | None) -> list[tuple[int, float]]:
        if allowed_rows is not None:
            allowed_rows = np.asarray(allowed_rows)
            if allowed_rows.dtype == bool:  # a boolean mask over all rows
                allowed_rows = np.flatnonzero(allowed_rows)
            allowed_rows = allowed_rows.astype(np.int64)
            if allowed_rows.size == 0:
                return []
            cand, sc = allowed_rows, s[allowed_rows]
        else:
            cand, sc = np.arange(self.n), s
        k = min(k, len(cand))
        if k <= 0:
            return []
        kth = sc[np.argpartition(-sc, k - 1)[k - 1]]  # k-th largest score
        above = np.flatnonzero(sc > kth)
        ties = np.flatnonzero(sc == kth)
        top = np.concatenate([above, ties[np.argsort(cand[ties], kind="stable")][: k - len(above)]])
        top = top[np.lexsort((cand[top], -sc[top]))]  # score desc, then row asc (deterministic, also at the cut)
        return [(int(cand[i]), float(sc[i])) for i in top]

    def close(self) -> None:
        self._pool.shutdown(wait=False)

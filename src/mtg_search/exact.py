"""Exact late-interaction backend for the shipped server: fp32 ONNX query encoder + brute-force MaxSim.

Replaces NextPlaid in the image (EXP-012). At 34.6k cards (~2.2M doc token vectors) exact MaxSim over
every card is one matrix product, so the approximate PLAID index (4-bit residuals; top-10 agreement
with exact 72-85%, docs/review_2026-10-14.md M1) buys nothing. This backend ranks exactly like the
in-process ``LateInMemory`` reference that every benchmark used, without PyTorch:

* queries: ``tokenizers`` + ``onnxruntime`` running mLateOn's published fp32 ``model.onnx``
  (``[Q] `` prefix -> ``<bos> [Q] ... <eos>``, no expansion, token-level L2 normalisation),
  verified per-token cosine 1.00000 against sentence-transformers on PyTorch;
* documents: the precomputed fp16 token vectors (``embeddings/late/<slug>/tokens.npy``), held as
  fp32 in RAM; score(card) = sum over query tokens of the max dot product over the card's tokens.

Rows are aligned with ``cards.parquet`` by order, checked against the embeddings' ``meta.json``
(``oracle_ids_sha256``) at load time.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

QUERY_PREFIX = "[Q] "


class ExactLateBackend:
    def __init__(self, oracle_ids: list[str], *, embeddings_dir: str | Path, model_dir: str | Path,
                 threads: int | None = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        from .artifacts import oracle_ids_digest

        t0 = time.perf_counter()
        embeddings_dir, model_dir = Path(embeddings_dir), Path(model_dir)
        meta = json.loads((embeddings_dir / "meta.json").read_text())
        if meta.get("card_count") != len(oracle_ids) or meta.get("oracle_ids_sha256") != oracle_ids_digest(oracle_ids):
            raise RuntimeError(
                f"embeddings in {embeddings_dir} do not match cards.parquet "
                f"({meta.get('card_count')} vs {len(oracle_ids)} cards, or a different card order)")
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
        self.ready = True
        log.info("exact backend: %d cards, %d token vectors, model %s, loaded in %.1fs",
                 self.n, len(self.tokens), model_dir.name, time.perf_counter() - t0)

    def _init_scoring(self, threads: int) -> None:
        # MaxSim is memory-bandwidth bound: score card-aligned chunks in parallel (numpy/BLAS release the GIL);
        # run with OPENBLAS_NUM_THREADS=1 so the pool, not BLAS, owns the cores.
        # ~4k tokens per chunk keeps the (chunk x query-tokens) intermediate in cache: 1 query ~100 ms,
        # a 4-query search_cards call ~330 ms on 6 cores (EXP-012); bigger chunks make batches slower
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
        reading ~1.1 GB of fp32 vectors, not the arithmetic), so 4 queries cost about as much as 1."""
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

    def scores(self, query: str) -> np.ndarray:
        return self.scores_many([query])[0]

    def search(self, query: str, k: int, allowed_rows: np.ndarray | None = None) -> list[tuple[int, float]]:
        return self.search_many([query], k, allowed_rows)[0]

    def search_many(self, queries: list[str], k: int, allowed_rows: np.ndarray | None = None) -> list[list[tuple[int, float]]]:
        return [self._top(s, k, allowed_rows) for s in self.scores_many(queries)]

    def _top(self, s: np.ndarray, k: int, allowed_rows: np.ndarray | None) -> list[tuple[int, float]]:
        if allowed_rows is not None:
            allowed_rows = np.asarray(allowed_rows, dtype=np.int64)
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

    def start_background(self) -> None:  # interface parity with NextPlaidBackend
        pass

    def close(self) -> None:
        self._pool.shutdown(wait=False)

"""NextPlaid backend for the MCP server: late-interaction (mLateOn) search over the card index.

In the Docker image the server launches ``next-plaid-api`` itself on 127.0.0.1 (all of its
output goes to *stderr*; stdout belongs to MCP). For development, set ``NEXTPLAID_URL`` to an
already running server (e.g. the image started with ``--entrypoint next-plaid-api``).

Filtering: NextPlaid's search accepts ``subset`` (a list of document ids) and restricts the
IVF candidate generation to centroids holding those docs, scaling ``n_ivf_probe`` up for small
subsets. So filters are evaluated in Python by ``mtg_search.filters`` (single source of truth)
and only the allowed doc ids are sent; selective filters return the best *allowed* cards
rather than a post-filtered top-k that may come back empty.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx
import numpy as np

from .artifacts import DOC_ROWS

log = logging.getLogger("mtg_search.nextplaid")

INDEX_NAME = "cards"
WARMUP_QUERY = "draw a card"


class BackendNotReady(RuntimeError):
    pass


class NextPlaidBackend:
    """Search ``(query, k, allowed rows)`` -> ``[(row, score)]`` against a NextPlaid index.

    ``oracle_ids`` are the card table's ids in row order; the doc id -> row mapping comes from
    ``<index_dir>/doc_rows.npy`` (written by docker/build_index.py) or, for an external server,
    from the index's metadata store (matched by oracle_id).
    """

    def __init__(
        self,
        oracle_ids: list[str],
        *,
        url: str | None = None,
        index_dir: str | Path | None = None,
        model_dir: str | Path | None = None,
        port: int = 8080,
        int8: bool = True,
        query_length: int = 512,
        binary: str = "next-plaid-api",
    ):
        self.oracle_ids = oracle_ids
        self.n_rows = len(oracle_ids)
        self.url = url
        self.index_dir = Path(index_dir) if index_dir else None
        self.model_dir = Path(model_dir) if model_dir else None
        self.port = port
        self.int8 = int8
        self.query_length = query_length
        self.binary = binary
        self.proc: subprocess.Popen | None = None
        self.client: httpx.Client | None = None
        self.doc_to_row: np.ndarray | None = None
        self.row_to_doc: np.ndarray | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None

    # ----------------------------------------------------------------- lifecycle
    def start_background(self) -> None:
        threading.Thread(target=self._start_safe, name="nextplaid-start", daemon=True).start()

    def _start_safe(self) -> None:
        try:
            self.start()
        except BaseException as e:  # surfaced to tool calls via wait_ready()
            log.exception("NextPlaid backend failed to start")
            self._error = e
            self._ready.set()

    def start(self) -> None:
        t0 = time.time()
        if self.url is None:
            self._spawn()
        self.client = httpx.Client(base_url=self.url, timeout=60)
        self._wait_health()
        self._load_mapping()
        self.warmup()
        log.info("NextPlaid ready in %.1fs (%s)", time.time() - t0, self.url)
        self._ready.set()

    def _writable_index_dir(self) -> Path:
        assert self.index_dir is not None
        probe = self.index_dir / INDEX_NAME
        if os.access(probe, os.W_OK) and os.access(self.index_dir, os.W_OK):
            return self.index_dir
        # next-plaid opens lock files inside the index dir; with `docker run --read-only`
        # (plus a tmpfs on /tmp) we serve a writable copy instead.
        tmp = Path(tempfile.mkdtemp(prefix="mtg-index-"))
        log.info("index dir %s is read-only; copying to %s", self.index_dir, tmp)
        shutil.copytree(self.index_dir, tmp, dirs_exist_ok=True)
        return tmp

    def _spawn(self) -> None:
        if self.index_dir is None or self.model_dir is None:
            raise RuntimeError("index_dir and model_dir are required when NEXTPLAID_URL is not set")
        idx = self._writable_index_dir()
        self.index_dir = idx
        cmd = [self.binary, "--host", "127.0.0.1", "--port", str(self.port), "--index-dir", str(idx),
               "--model", str(self.model_dir), "--query-length", str(self.query_length)]
        if self.int8:
            cmd.append("--int8")
        log.info("starting %s", " ".join(cmd))
        # stdout of the child must never reach our stdout (the MCP stdio channel)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=sys.stderr, stderr=sys.stderr)
        self.url = f"http://127.0.0.1:{self.port}"

    def _wait_health(self, timeout: float = 180) -> None:
        assert self.client is not None
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                if self.client.get("/health", timeout=2).status_code == 200:
                    return
            except httpx.TransportError:
                pass
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(f"next-plaid-api exited with code {self.proc.returncode}")
            time.sleep(0.2)
        raise RuntimeError(f"NextPlaid at {self.url} not healthy after {timeout:.0f}s")

    def _load_mapping(self) -> None:
        path = self.index_dir / DOC_ROWS if self.index_dir else None
        if path is not None and path.exists():
            d2r = np.load(path).astype(np.int64)
        else:
            assert self.client is not None
            md = self.client.get(f"/indices/{INDEX_NAME}/metadata", timeout=120).json()
            md = md.get("metadata", md) if isinstance(md, dict) else md
            pos = {o: i for i, o in enumerate(self.oracle_ids)}
            n_docs = max(int(m["_subset_"]) for m in md) + 1
            d2r = np.full(n_docs, -1, dtype=np.int64)
            for m in md:
                d2r[int(m["_subset_"])] = pos.get(m.get("oracle_id"), -1)
            missing = int((d2r < 0).sum())
            if missing:
                log.warning("%d indexed docs are not in the local card table; they are skipped", missing)
        r2d = np.full(self.n_rows, -1, dtype=np.int64)
        ok = d2r >= 0
        r2d[d2r[ok]] = np.flatnonzero(ok)
        self.doc_to_row, self.row_to_doc = d2r, r2d

    def warmup(self) -> None:
        # The index is loaded lazily on first search (~2 s); pay that at startup.
        self._post({"queries": [WARMUP_QUERY], "params": {"top_k": 5}})

    def wait_ready(self, timeout: float = 180) -> None:
        if not self._ready.wait(timeout):
            raise BackendNotReady("search backend is still starting; retry in a few seconds")
        if self._error is not None:
            raise BackendNotReady(f"search backend failed to start: {self._error}")

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    # ----------------------------------------------------------------- search
    def _post(self, body: dict) -> dict:
        assert self.client is not None
        r = self.client.post(f"/indices/{INDEX_NAME}/search_with_encoding", json=body)
        if r.status_code != 200:
            raise RuntimeError(f"NextPlaid search failed: {r.status_code} {r.text[:300]}")
        return r.json()

    def search(self, query: str, k: int, allowed_rows: np.ndarray | None = None) -> list[tuple[int, float]]:
        """Top-``k`` ``(row, score)`` for ``query`` among ``allowed_rows`` (None = all cards)."""
        self.wait_ready()
        assert self.row_to_doc is not None and self.doc_to_row is not None
        body: dict = {"queries": [query], "params": {"top_k": int(k)}}
        if allowed_rows is not None:
            docs = self.row_to_doc[np.asarray(allowed_rows, dtype=np.int64)]
            docs = docs[docs >= 0]
            if docs.size == 0:
                return []
            if docs.size < self.doc_to_row.size:
                body["subset"] = docs.tolist()
        res = self._post(body)["results"][0]
        out = []
        for d, s in zip(res["document_ids"], res["scores"], strict=True):
            row = int(self.doc_to_row[d]) if 0 <= d < self.doc_to_row.size else -1
            if row >= 0:
                out.append((row, float(s)))
        return out

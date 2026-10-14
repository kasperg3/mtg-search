"""Build the NextPlaid index from precomputed (GPU, PyLate) mLateOn token embeddings.

    python docker/build_index.py --artifacts DIR --out INDEX_DIR [--url http://127.0.0.1:8080]

``DIR`` must contain ``data/cards.parquet`` and ``embeddings/late/lightonai__mLateOn/``
(``tokens.npy`` fp16 [T, 128], ``offsets.npy`` int64 [N+1], ``meta.json``) -- the layout of
the HF dataset ``kasper-cbx/mtg-search-index`` (see ``mtg_search.artifacts``).

Without ``--url`` an embeddings-only ``next-plaid-api`` (no model) is started on localhost
for the duration of the build (Docker builder stage). Documents are pushed via
``POST /indices/cards/update`` in batches of <= 500 docs (base64 f32; the API has a 100 MB
body limit). Doc ids are assigned by NextPlaid; the mapping NextPlaid doc id -> row of
``cards.parquet`` is read back from the metadata store and written to ``doc_rows.npy`` so
the server never relies on insertion order. ``mtg_search_index.json`` records what was
indexed (card digest, model revision) and is checked by the server at startup.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import polars as pl

INDEX_NAME = "cards"
MODEL_SLUG = "lightonai__mLateOn"
BATCH = 500


def log(*a) -> None:
    print("[build_index]", *a, file=sys.stderr, flush=True)


def digest(oracle_ids) -> str:  # same as mtg_search.retrievers.base.oracle_ids_digest
    h = hashlib.sha256()
    for oid in oracle_ids:
        h.update(oid.encode())
        h.update(b"\n")
    return h.hexdigest()


def wait_health(c: httpx.Client, proc: subprocess.Popen | None, timeout: float = 120) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if c.get("/health", timeout=2).status_code == 200:
                return
        except httpx.TransportError:
            pass
        if proc is not None and proc.poll() is not None:
            raise SystemExit(f"next-plaid-api exited with {proc.returncode}")
        time.sleep(0.3)
    raise SystemExit("next-plaid-api did not become healthy")


def updates_running(c: httpx.Client) -> bool:
    ups = c.get("/health").json().get("updates", []) or []
    return any(u.get("status") in ("running", "queued", "pending") for u in ups)


def build(c: httpx.Client, art: Path, out: Path) -> dict:
    emb_dir = art / "embeddings" / "late" / MODEL_SLUG
    meta = json.loads((emb_dir / "meta.json").read_text())
    cards = pl.read_parquet(art / "data" / "cards.parquet", columns=["oracle_id", "name"])
    oids = cards["oracle_id"].to_list()
    dig = digest(oids)
    if meta.get("oracle_ids_sha256") not in (None, dig):
        raise SystemExit("embeddings were computed for a different cards.parquet (oracle_ids_sha256 mismatch)")
    tokens = np.load(emb_dir / "tokens.npy", mmap_mode="r")
    offsets = np.load(emb_dir / "offsets.npy").astype(np.int64)
    n = len(oids)
    if offsets.shape[0] != n + 1 or offsets[-1] != tokens.shape[0]:
        raise SystemExit(f"offsets/tokens do not match {n} cards")
    dim = int(tokens.shape[1])

    r = c.post("/indices", json={"name": INDEX_NAME, "config": {"nbits": 4}})
    if r.status_code not in (200, 201, 202):
        raise SystemExit(f"create index failed: {r.status_code} {r.text}")
    names = cards["name"].to_list()
    t0 = time.time()
    for i in range(0, n, BATCH):
        j = min(n, i + BATCH)
        docs = []
        for k in range(i, j):
            e = np.ascontiguousarray(tokens[offsets[k] : offsets[k + 1]], dtype="<f4")
            docs.append({"embeddings_b64": base64.b64encode(e.tobytes()).decode(), "shape": [e.shape[0], dim]})
        md = [{"row": k, "oracle_id": oids[k], "name": names[k]} for k in range(i, j)]
        while True:
            r = c.post(f"/indices/{INDEX_NAME}/update", json={"documents": docs, "metadata": md}, timeout=600)
            if r.status_code == 503 or r.status_code == 429:
                time.sleep(1)
                continue
            if r.status_code not in (200, 202):
                raise SystemExit(f"update failed: {r.status_code} {r.text[:500]}")
            break
        # Wait for this batch before sending the next: keeps memory flat and ordering simple.
        while c.get(f"/indices/{INDEX_NAME}").json().get("num_documents", 0) < j:
            time.sleep(0.5)
        log(f"{j}/{n} docs ({time.time() - t0:.0f}s)")
    while updates_running(c):
        time.sleep(1)
    info = c.get(f"/indices/{INDEX_NAME}").json()
    # read back NextPlaid doc id -> parquet row (and wait for the metadata DB to catch up)
    for _ in range(600):
        allmd = c.get(f"/indices/{INDEX_NAME}/metadata", timeout=120).json()
        rows_md = allmd.get("metadata", allmd) if isinstance(allmd, dict) else allmd
        if len(rows_md) >= n:
            break
        time.sleep(1)
    doc_rows = np.full(n, -1, dtype=np.int32)
    for m in rows_md:
        doc_rows[int(m["_subset_"])] = int(m["row"])
    if (doc_rows < 0).any() or len(set(doc_rows.tolist())) != n:
        raise SystemExit("doc id -> row mapping incomplete")
    np.save(out / "doc_rows.npy", doc_rows)
    manifest = {
        "index_name": INDEX_NAME,
        "model": meta.get("model"),
        "model_revision": meta.get("revision"),
        "card_count": n,
        "oracle_ids_sha256": dig,
        "scryfall_updated_at": meta.get("scryfall_updated_at"),
        "num_documents": info.get("num_documents"),
        "num_partitions": info.get("num_partitions"),
        "built_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "build_seconds": round(time.time() - t0, 1),
        "identity_mapping": bool((doc_rows == np.arange(n)).all()),
    }
    (out / "mtg_search_index.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="NextPlaid --index-dir (must be fresh)")
    ap.add_argument("--url", help="use an already running embeddings-only next-plaid (its --index-dir must be --out)")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / INDEX_NAME).exists():
        raise SystemExit(f"{a.out / INDEX_NAME} already exists; build into a fresh directory")
    proc = None
    url = a.url
    if url is None:
        url = f"http://127.0.0.1:{a.port}"
        proc = subprocess.Popen(
            ["next-plaid-api", "--host", "127.0.0.1", "--port", str(a.port), "--index-dir", str(a.out)],
            stdout=sys.stderr,
            stderr=sys.stderr,
        )
    try:
        with httpx.Client(base_url=url, timeout=60) as c:
            wait_health(c, proc)
            log(json.dumps(build(c, a.artifacts, a.out)))
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait(60)


if __name__ == "__main__":
    main()

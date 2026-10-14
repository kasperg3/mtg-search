"""Artifacts the MCP server image needs, from the private HF dataset ``kasper-cbx/mtg-search-index``.

Layout (same as the dataset repo)::

    data/cards.parquet, data/tags.parquet, data/rulings.parquet, data/manifest.json
    embeddings/late/lightonai__mLateOn/{tokens.npy, offsets.npy, meta.json}   # doc tokens (GPU)
    models/lightonai__mLateOn-int8rr/{model_int8.onnx, tokenizer.json,
                                      onnx_config.json, config_sentence_transformers.json}

The query encoder is mLateOn re-quantized to int8 with ``--reduce-range`` (the published
``model_int8.onnx`` is broken on CPUs without VNNI; see spike/nextplaid.md and
``models/.../quantization.json``).

CLI (used by the Docker builder stage; token from ``HF_TOKEN``)::

    python -m mtg_search.artifacts download --dest /artifacts [--revision main]
    python -m mtg_search.artifacts check --index-dir /app/index --cards /app/data/cards.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config

LATE_SLUG = "lightonai__mLateOn"
MODEL_DIR = "models/lightonai__mLateOn-int8rr"
INDEX_MANIFEST = "mtg_search_index.json"
DOC_ROWS = "doc_rows.npy"

PATTERNS = {
    "data": ["data/*.parquet", "data/manifest.json"],
    "embeddings": [f"embeddings/late/{LATE_SLUG}/*"],
    "model": [f"{MODEL_DIR}/*"],
}


def download(
    dest: str | Path,
    *,
    revision: str = "main",
    parts: tuple[str, ...] = ("data", "embeddings", "model"),
    token: str | None = None,
) -> Path:
    """Fetch the given parts of the dataset repo into ``dest`` (returns ``dest``)."""
    from huggingface_hub import snapshot_download

    dest = Path(dest)
    allow = [p for part in parts for p in PATTERNS[part]]
    snapshot_download(
        repo_id=config.HF_REPO,
        repo_type=config.HF_REPO_TYPE,
        revision=revision,
        allow_patterns=allow,
        local_dir=dest,
        token=token,
    )
    return dest


def oracle_ids_digest(oracle_ids) -> str:
    from .retrievers.base import oracle_ids_digest as _d

    return _d(oracle_ids)


def read_index_manifest(index_dir: str | Path) -> dict:
    p = Path(index_dir) / INDEX_MANIFEST
    return json.loads(p.read_text()) if p.exists() else {}


def check_index(index_dir: str | Path, oracle_ids) -> dict:
    """Raise if the baked index was not built for this exact card table (row order matters)."""
    m = read_index_manifest(index_dir)
    if not m:
        raise RuntimeError(f"{index_dir}/{INDEX_MANIFEST} missing; was the index built by docker/build_index.py?")
    oids = list(oracle_ids)
    if m.get("card_count") != len(oids) or m.get("oracle_ids_sha256") != oracle_ids_digest(oids):
        raise RuntimeError(
            f"index/card table mismatch: index has {m.get('card_count')} cards "
            f"(sha {str(m.get('oracle_ids_sha256'))[:12]}), cards.parquet has {len(oids)}"
        )
    return m


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m mtg_search.artifacts")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--dest", required=True)
    d.add_argument("--revision", default="main")
    d.add_argument("--parts", default="data,embeddings,model")
    c = sub.add_parser("check")
    c.add_argument("--index-dir", required=True)
    c.add_argument("--cards", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "download":
        out = download(a.dest, revision=a.revision, parts=tuple(a.parts.split(",")))
        print(f"downloaded {config.HF_REPO}@{a.revision} -> {out}", file=sys.stderr)
    else:
        import polars as pl

        oids = pl.read_parquet(a.cards, columns=["oracle_id"])["oracle_id"].to_list()
        print(json.dumps(check_index(a.index_dir, oids)), file=sys.stderr)


if __name__ == "__main__":
    main()

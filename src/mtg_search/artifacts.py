"""Artifacts from the HF dataset ``kasper-cbx/mtg-search-index``.

Layout (same as the dataset repo)::

    data/cards.parquet, data/tags.parquet, data/rulings.parquet, data/manifest.json   # part "data"
    embeddings/late/lightonai__mLateOn/{tokens.npy, offsets.npy, meta.json}            # part "embeddings"
    eval/popularity.parquet       # part "eval": offline evaluation only, never requested by the image build

The query encoder (part "model", EXP-012) is mLateOn's published fp32 ``model.onnx`` from the public model
repo ``lightonai/mLateOn`` at the revision the doc embeddings were encoded with, written to
``models/lightonai__mLateOn-fp32/`` with a ``source.json`` (repo, revision, sha256). ``mlateon.MLateOn``
refuses to start if that revision differs from the embeddings' ``meta.json``.

CLI (used by the Docker builder stage; token from ``HF_TOKEN``)::

    python -m mtg_search.artifacts download --dest /artifacts [--revision main] [--parts data,embeddings,model]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config
from .config import EMBEDDINGS_SUBDIR, MODEL_SUBDIR

FP32_MODEL_DIR = MODEL_SUBDIR
QUERY_MODEL_REPO = "lightonai/mLateOn"
QUERY_MODEL_REVISION = "bbd883ae1fadb563056ff73b1061ae8172a3fcda"  # = embeddings meta.json revision
QUERY_MODEL_FILES = {  # file -> sha256 (None = small config, not pinned by hash)
    "model.onnx": "8ef72047e438bdb17993fd75fe68944bb91d6fd9208881c4ca638f0c337a4ca5",
    "tokenizer.json": "33b6ce1724e107a3298219a322b4c963fcfcb9c8661bd1f30189bdd0d36c4669",
    "onnx_config.json": None,
    "config_sentence_transformers.json": None,
}

PATTERNS = {
    "data": ["data/*.parquet", "data/manifest.json"],
    "embeddings": [f"{EMBEDDINGS_SUBDIR}/*"],
    # evaluation-only popularity table; download with --dest data -> data/eval/popularity.parquet
    "eval": ["eval/popularity.parquet"],
}


def download_query_model(dest: str | Path) -> Path:
    """Fetch the fp32 ONNX query encoder (public repo, pinned revision), verify hashes, write source.json."""
    import hashlib

    from huggingface_hub import hf_hub_download

    out = Path(dest) / FP32_MODEL_DIR
    out.mkdir(parents=True, exist_ok=True)
    for f, sha in QUERY_MODEL_FILES.items():
        # straight into ``out`` (no second copy in the HF cache: the encoder alone is 1.28 GB)
        p = hf_hub_download(QUERY_MODEL_REPO, f, revision=QUERY_MODEL_REVISION, token=False, local_dir=out)
        if sha:
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 24), b""):
                    h.update(chunk)
            if h.hexdigest() != sha:
                raise RuntimeError(f"{QUERY_MODEL_REPO}@{QUERY_MODEL_REVISION}/{f}: sha256 {h.hexdigest()} != {sha}")
    (out / "source.json").write_text(json.dumps({"repo": QUERY_MODEL_REPO, "revision": QUERY_MODEL_REVISION,
                                                  "sha256": {f: s for f, s in QUERY_MODEL_FILES.items() if s}},
                                                 indent=1) + "\n")
    return out


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
    if "model" in parts:
        download_query_model(dest)
    allow = [p for part in parts if part != "model" for p in PATTERNS[part]]
    if not allow:
        return dest
    snapshot_download(
        repo_id=config.HF_REPO,
        repo_type=config.HF_REPO_TYPE,
        revision=revision,
        allow_patterns=allow,
        local_dir=dest,
        token=token,
    )
    return dest


COMPLETE = ".complete"  # written last, so an interrupted download is resumed rather than used


def _stderr(msg: str) -> None:  # stdout belongs to MCP's JSON-RPC
    print(msg, file=sys.stderr, flush=True)


def _model_ready(base: Path) -> bool:
    src = base / FP32_MODEL_DIR / "source.json"
    return src.exists() and json.loads(src.read_text()).get("revision") == QUERY_MODEL_REVISION


def _tidy(*dirs: Path) -> None:
    import shutil

    for d in dirs:  # huggingface_hub's local_dir bookkeeping, not needed once a part is complete
        shutil.rmtree(d / ".cache", ignore_errors=True)


def complete_revisions(base: Path | None = None) -> list[str]:
    """Snapshots fully downloaded under ``base`` whose card tables this code can read, oldest first."""
    base = Path(base) if base else config.index_base()
    out = []
    for d in sorted(base.glob("scryfall-*")) if base.is_dir() else []:
        try:
            ok = (d / COMPLETE).exists() and json.loads((d / "data" / "manifest.json").read_text()).get(
                "schema_version") == config.SCHEMA_VERSION
        except (OSError, ValueError):
            ok = False
        if ok:
            out.append(d.name)
    return out


def ensure_index(revision: str | None = None, home: str | Path | None = None, *, log=_stderr) -> tuple[Path, Path]:
    """The local copy of what the server needs, downloading whatever is missing: (snapshot root with ``data/``
    and ``embeddings/…``, query-encoder dir). ``home`` is the base folder (default ``config.index_base()``)."""
    revision = revision or config.INDEX_REVISION
    base = Path(home) if home else config.index_base()
    root = base / revision
    if not _model_ready(base):
        log(f"mtg-search: downloading the query encoder {QUERY_MODEL_REPO} (1.28 GB, once) to {base}")
        download_query_model(base)
        _tidy(base / FP32_MODEL_DIR)
    if not (root / COMPLETE).exists():
        log(f"mtg-search: downloading the card index {config.HF_REPO}@{revision} (about 0.6 GB) to {root}")
        download(root, revision=revision, parts=("data", "embeddings"))
        _tidy(root)
        (root / COMPLETE).write_text(json.dumps({"repo": config.HF_REPO, "revision": revision}) + "\n")
    return root, base / FP32_MODEL_DIR


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m mtg_search.artifacts")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--dest", required=True)
    d.add_argument("--revision", default="main")
    d.add_argument("--parts", default="data,embeddings,model")
    a = ap.parse_args(argv)
    out = download(a.dest, revision=a.revision, parts=tuple(a.parts.split(",")))
    print(f"downloaded {config.HF_REPO}@{a.revision} -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()

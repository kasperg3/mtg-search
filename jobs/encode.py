# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "torch>=2.6",
#     "sentence-transformers>=6.1",
#     "transformers>=5.5",
#     "huggingface_hub>=1.3",
#     "numpy>=1.26",
#     "polars>=1.0",
# ]
# ///
"""Encode every card's ``doc_text`` with lightonai/mLateOn (token vectors for late interaction) and upload them.

Run on HF Jobs (GPU):
    hf jobs uv run --flavor t4-small --secrets HF_TOKEN --timeout 1h -d jobs/encode.py

Run locally (CPU, no upload; a standalone uv script with its own dependencies):
    uv run jobs/encode.py --cards data/cards.parquet --limit 200 --out /tmp/enc --no-upload

Outputs (relative to --out, uploaded under ``embeddings/`` of the dataset repo):
    late/lightonai__mLateOn/tokens.npy     float16 (total_tokens, dim), docs concatenated in card order
    late/lightonai__mLateOn/offsets.npy    int64 (n + 1)
    late/lightonai__mLateOn/meta.json

The MCP server encodes queries with the same checkpoint's fp32 ONNX export (``mtg_search.mlateon``), and refuses
to start unless ``meta.json`` matches the card table and the query encoder's revision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

HF_REPO = "kasper-cbx/mtg-search-index"
HF_REPO_TYPE = "dataset"
MODEL = "lightonai/mLateOn"


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def oracle_ids_digest(oracle_ids) -> str:
    # Must match mtg_search.mlateon.oracle_ids_digest
    h = hashlib.sha256()
    for oid in oracle_ids:
        h.update(oid.encode())
        h.update(b"\n")
    return h.hexdigest()


def log(*a):
    print(f"[{datetime.now().strftime('%H:%M:%S')}]", *a, flush=True)


# --------------------------------------------------------------------------- data


def load_cards(cards: str | None, limit: int | None):
    import polars as pl

    manifest = {}
    if cards is None:
        from huggingface_hub import hf_hub_download

        token = os.environ.get("HF_TOKEN")
        cards = hf_hub_download(HF_REPO, "data/cards.parquet", repo_type=HF_REPO_TYPE, token=token)
        try:
            mpath = hf_hub_download(HF_REPO, "data/manifest.json", repo_type=HF_REPO_TYPE, token=token)
            manifest = json.loads(Path(mpath).read_text())
        except Exception as e:  # noqa: BLE001
            log("no manifest.json in repo:", e)
    else:
        mpath = Path(cards).with_name("manifest.json")
        if mpath.exists():
            manifest = json.loads(mpath.read_text())
    df = pl.read_parquet(cards, columns=["oracle_id", "doc_text"])
    if limit:
        df = df.head(limit)
    return df["oracle_id"].to_list(), df["doc_text"].to_list(), manifest


# --------------------------------------------------------------------------- models


def load_model(model_id: str, device: str, revision: str | None = None):
    import torch
    from sentence_transformers import MultiVectorEncoder

    model = MultiVectorEncoder(model_id, device=device, revision=revision, model_kwargs={"dtype": torch.float32})
    model.eval()
    return model


def doc_max_length(model) -> int:
    """Effective document token limit of the first (Transformer) module."""
    first = model[0]
    for attr in ("document_length", "max_seq_length"):
        v = getattr(first, attr, None)
        if isinstance(v, int) and 0 < v < 1_000_000:
            return v
    cfg = getattr(first, "config", None) or {}
    if isinstance(cfg, dict) and cfg.get("document_length"):
        return int(cfg["document_length"])
    v = getattr(model, "max_seq_length", None)
    if isinstance(v, int) and 0 < v < 1_000_000:
        return v
    tok_max = getattr(model.tokenizer, "model_max_length", None)
    return int(tok_max) if isinstance(tok_max, int) and tok_max < 1_000_000 else -1


def token_lengths(model, texts: list[str], prompt: str = "") -> np.ndarray:
    tok = model.tokenizer
    out = []
    for i in range(0, len(texts), 4096):
        enc = tok([prompt + t for t in texts[i : i + 4096]], add_special_tokens=True)["input_ids"]
        out.extend(len(x) for x in enc)
    return np.asarray(out)


# --------------------------------------------------------------------------- encoders


def encode_late(model, texts, batch_size):
    chunks, lens = [], []
    step = 4096
    for i in range(0, len(texts), step):
        embs = model.encode_document(
            texts[i : i + step], batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False
        )
        for e in embs:
            e = np.asarray(e, dtype=np.float16)
            chunks.append(e)
            lens.append(e.shape[0])
        log(f"late: {min(i + step, len(texts))}/{len(texts)} docs")
    tokens = np.concatenate(chunks, axis=0)
    offsets = np.zeros(len(lens) + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    return tokens, offsets


# --------------------------------------------------------------------------- main


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=None, help="model revision (default: resolved main sha)")
    ap.add_argument("--cards", default=None, help="local cards.parquet (default: download from HF repo)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None, help="cuda/cpu (default: cuda if available)")
    ap.add_argument("--require-gpu", action="store_true", help="fail instead of falling back to CPU")
    ap.add_argument("--out", default="out")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--repo", default=HF_REPO)
    ap.add_argument("--path-in-repo", default="embeddings")
    ap.add_argument("--repo-revision", default="main", help="branch to upload to")
    args = ap.parse_args(argv)

    import sentence_transformers
    import torch
    import transformers
    from huggingface_hub import HfApi

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log(f"python {platform.python_version()} torch {torch.__version__} cuda_available={torch.cuda.is_available()} device={device}")
    if device == "cuda":
        log("gpu:", torch.cuda.get_device_name(0))
    elif args.require_gpu:
        sys.exit("--require-gpu set but CUDA is not available")

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    revision = args.revision or api.model_info(args.model).sha
    log(f"model {args.model} @ {revision}")

    oracle_ids, texts, manifest = load_cards(args.cards, args.limit)
    log(f"{len(texts)} cards; scryfall_updated_at={manifest.get('scryfall_updated_at')}")

    t_load = time.perf_counter()
    model = load_model(args.model, device, revision)
    batch_size = args.batch_size
    log(f"loaded in {time.perf_counter() - t_load:.1f}s: {model}")

    prompts = dict(getattr(model, "prompts", None) or {})
    query_prompt = prompts.get("query", "")
    doc_prompt = prompts.get("document", "")
    max_len = doc_max_length(model)
    lengths = token_lengths(model, texts, doc_prompt)
    truncated = int((lengths > max_len).sum())
    log(f"doc tokens: p50={np.median(lengths):.0f} p99={np.percentile(lengths, 99):.0f} max={lengths.max()} limit={max_len} truncated={truncated}")

    out = Path(args.out)
    slug = model_slug(args.model)
    meta = {
        "model": args.model,
        "revision": revision,
        "kind": "late",
        "card_count": len(texts),
        "oracle_ids_sha256": oracle_ids_digest(oracle_ids),
        "scryfall_updated_at": manifest.get("scryfall_updated_at"),
        "query_prompt": query_prompt,
        "doc_prompt": doc_prompt,
        "doc_max_length": max_len,
        "doc_tokens_p50": float(np.median(lengths)),
        "doc_tokens_max": int(lengths.max()),
        "truncated_docs": truncated,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sentence_transformers": sentence_transformers.__version__,
        },
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }

    t0 = time.perf_counter()
    with torch.inference_mode():
        tokens, offsets = encode_late(model, texts, batch_size)
    d = out / "late" / slug
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "tokens.npy", tokens)
    np.save(d / "offsets.npy", offsets)
    meta.update(dim=int(tokens.shape[1]), n_tokens=int(tokens.shape[0]), dtype="float16")
    meta_path = d / "meta.json"
    files = [d / "tokens.npy", d / "offsets.npy", meta_path]
    meta["encode_seconds"] = round(time.perf_counter() - t0, 1)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    for f in files:
        log(f"wrote {f} ({f.stat().st_size / 1e6:.1f} MB)")
    log(f"encode took {meta['encode_seconds']}s")

    if args.no_upload:
        log("--no-upload: done")
        return
    rel = [str(f.relative_to(out)) for f in files]
    for attempt in range(5):
        try:
            info = api.upload_folder(
                repo_id=args.repo,
                repo_type=HF_REPO_TYPE,
                folder_path=str(out),
                path_in_repo=args.path_in_repo,
                allow_patterns=rel,
                revision=args.repo_revision,
                commit_message=f"embeddings: {args.model} ({len(texts)} cards)",
            )
            log("uploaded:", info)
            return
        except Exception as e:  # noqa: BLE001 - concurrent commits from parallel jobs can race
            log(f"upload attempt {attempt + 1} failed: {e!r}")
            time.sleep(10 * (attempt + 1))
    sys.exit("upload failed")


if __name__ == "__main__":
    main()

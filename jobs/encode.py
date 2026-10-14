# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "torch>=2.6",
#     "torchvision>=0.21",
#     "pillow>=10",
#     "sentence-transformers>=6.1",
#     "transformers>=5.5",
#     "huggingface_hub>=1.3",
#     "safetensors>=0.4",
#     "numpy>=1.26",
#     "scipy>=1.13",
#     "polars>=1.0",
# ]
# ///
"""Encode every card's ``doc_text`` with one model and upload the artifacts.

Run on HF Jobs (GPU):
    hf jobs uv run --flavor t4-small --secrets HF_TOKEN --timeout 1h -d \\
        jobs/encode.py --model lightonai/mLateOn --kind late

Run locally (CPU, no upload), from the project venv:
    uv run --extra models python jobs/encode.py --model ... --kind late \\
        --cards data/cards.parquet --limit 200 --out /tmp/enc --no-upload

Outputs (relative to --out, uploaded under ``embeddings/`` of the dataset repo):
    late/<slug>/tokens.npy     float16 (total_tokens, dim), docs concatenated in card order
    late/<slug>/offsets.npy    int64 (n + 1)
    late/<slug>/meta.json
    dense/<slug>.npy           float16 (n, dim), L2-normalised
    dense/<slug>.json
    sparse/<slug>.npz          scipy CSR float32 (n, vocab)
    sparse/<slug>.json
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
GTE_CODE_REVISION = "40ced75c3017eb27626c9d4ea981bde21a2662f4"

# Per-model loading quirks (keep dense entries in sync with mtg_search.retrievers.dense).
MODEL_LOAD: dict[str, dict] = {
    # Multimodal 740M; text-only drops vision+audio towers. Never float16 (overflows);
    # T4 has no native bf16, so float32 throughout.
    "google/embeddinggemma-2": {"config_kwargs": {"vision_config": None, "audio_config": None}},
    "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte": {
        "trust_remote_code": True,
        "code_revision": GTE_CODE_REVISION,
        "repair_gte": True,
    },
    # SPLADE-style, NOT inference-free: queries run through the encoder too. Remote
    # code (custom_st.SpladeSTModule) applies the "[Q] "/"[D] " prompts itself (attended,
    # not pooled) and truncates to query 128 / doc 512 tokens. Its load() ignores
    # `revision`, so we pin by loading from a local snapshot of that revision.
    # Verified under transformers 5.19 / sentence-transformers 6.1: the model-card
    # example scores [[113.2993, 0], [7.0030, 99.1291]] reproduce exactly, no repair needed.
    # Logits are [B, L, 50370] fp32 (several copies) -> keep batches small on a T4.
    "Linkup-Platform/linkup-sparseup-embed-v1": {
        "trust_remote_code": True,
        "local_snapshot": True,
        "doc_max_length": 512,
        "batch_size": 16,
        "query_encoder": "transformer: [Q] prefix, max 128 tokens, ModernBERT MLM -> log1p(relu(x-15)) -> top12/pos -> maxpool -> vocab fold",
    },
}

ENCODE_ONLY_KEYS = ("repair_gte", "local_snapshot", "doc_max_length", "batch_size", "query_encoder")


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def oracle_ids_digest(oracle_ids) -> str:
    # Must match mtg_search.retrievers.base.oracle_ids_digest
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


def _repair_gte(model, model_id: str, revision: str | None) -> None:
    """Make Alibaba-NLP/new-impl remote code (written for transformers 4.x) correct under 5.x.

    transformers 5 initialises models on the meta device, so (1) non-persistent buffers
    (position_ids, rotary inv_freq/cos/sin caches) come back as garbage, (2) the tied
    lm_head.decoder.weight is reported MISSING and the whole decoder Linear -- including
    the checkpoint's decoder.bias -- gets re-initialised, and (3)
    PreTrainedModel.get_extended_attention_mask no longer exists. Fix all three, then the
    output matches transformers 4.57 exactly (model-card example scores 12.5747).
    """
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    def ext_mask(self, attention_mask, input_shape, *args, **kwargs):
        m = attention_mask[:, None, None, :] if attention_mask.dim() == 2 else attention_mask[:, None, :, :]
        m = m.to(dtype=self.dtype)
        return (1.0 - m) * torch.finfo(self.dtype).min

    for mod in model.modules():
        name = type(mod).__name__
        if name == "NewModel" and not hasattr(mod, "get_extended_attention_mask"):
            type(mod).get_extended_attention_mask = ext_mask
        elif name == "NewForMaskedLM":
            sd = load_file(hf_hub_download(model_id, "model.safetensors", revision=revision))
            dev = next(mod.parameters()).device
            res = mod.load_state_dict({k: v.to(dev) for k, v in sd.items()}, strict=False)
            log("gte repair: reloaded checkpoint tensors; missing:", res.missing_keys)
            mod.lm_head.decoder.weight = mod.new.embeddings.word_embeddings.weight
        elif name == "NewEmbeddings":
            mod.position_ids = torch.arange(mod.position_ids.numel(), device=mod.position_ids.device)
        elif name in ("RotaryEmbedding", "NTKScalingRotaryEmbedding"):
            dev = mod.inv_freq.device
            mod.inv_freq = 1.0 / (mod.base ** (torch.arange(0, mod.dim, 2, device=dev).float() / mod.dim))
            mod._set_cos_sin_cache(mod.max_position_embeddings, dev, torch.float32)
            if hasattr(mod, "scaling_factor"):
                mod._set_cos_sin_cache(int(mod.max_position_embeddings * mod.scaling_factor), dev, torch.float32)


def load_model(kind: str, model_id: str, device: str, revision: str | None = None):
    import torch

    cfg = dict(MODEL_LOAD.get(model_id, {}))
    repair = cfg.pop("repair_gte", False)
    local_snapshot = cfg.pop("local_snapshot", False)
    fixed_doc_len = cfg.pop("doc_max_length", None)
    for k in ENCODE_ONLY_KEYS:
        cfg.pop(k, None)
    code_revision = cfg.pop("code_revision", None)
    model_kwargs = {"dtype": torch.float32}
    if code_revision:
        model_kwargs["code_revision"] = code_revision
    path = model_id
    if local_snapshot:
        from huggingface_hub import snapshot_download

        path = snapshot_download(model_id, revision=revision)
        log(f"loading {model_id} from local snapshot {path}")
        revision = None
    common = dict(device=device, revision=revision, model_kwargs=model_kwargs, **cfg)
    model_id_or_path = path
    if kind == "late":
        from sentence_transformers import MultiVectorEncoder

        model = MultiVectorEncoder(model_id_or_path, **common)
    elif kind == "dense":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_id_or_path, **common)
    elif kind == "sparse":
        from sentence_transformers import SparseEncoder

        model = SparseEncoder(model_id_or_path, **common)
    else:
        raise ValueError(kind)
    if repair:
        _repair_gte(model, model_id, revision)
    if fixed_doc_len:
        model._fixed_doc_max_length = int(fixed_doc_len)
        model.eval()
        return model
    msl = getattr(model, "max_seq_length", None)
    if not (isinstance(msl, int) and 0 < msl < 1_000_000):
        # e.g. embeddinggemma-2 reports no limit; its context window is 8192 tokens.
        model.max_seq_length = 8192
    model.eval()
    return model


def doc_max_length(model) -> int:
    """Effective document token limit of the first (Transformer) module."""
    if getattr(model, "_fixed_doc_max_length", None):
        return model._fixed_doc_max_length
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


def encode_dense(model, texts, batch_size, prompt):
    embs = model.encode(
        texts,
        prompt=prompt or None,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(embs, dtype=np.float32).astype(np.float16)


def encode_sparse(model, texts, batch_size):
    import scipy.sparse as sp

    parts = []
    step = 2048
    for i in range(0, len(texts), step):
        emb = model.encode_document(
            texts[i : i + step], batch_size=batch_size, convert_to_sparse_tensor=True, show_progress_bar=False
        )
        emb = emb.coalesce().cpu()
        idx = emb.indices().numpy()
        vals = emb.values().float().numpy()
        parts.append(sp.csr_matrix((vals, (idx[0], idx[1])), shape=tuple(emb.shape), dtype=np.float32))
        log(f"sparse: {min(i + step, len(texts))}/{len(texts)} docs")
    return sp.vstack(parts, format="csr")


# --------------------------------------------------------------------------- main


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--kind", required=True, choices=["late", "dense", "sparse"])
    ap.add_argument("--revision", default=None, help="model revision (default: resolved main sha)")
    ap.add_argument("--cards", default=None, help="local cards.parquet (default: download from HF repo)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None, help="default 64, or the model's MODEL_LOAD batch_size")
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
    log(f"model {args.model} @ {revision} kind={args.kind}")

    oracle_ids, texts, manifest = load_cards(args.cards, args.limit)
    log(f"{len(texts)} cards; scryfall_updated_at={manifest.get('scryfall_updated_at')}")

    t_load = time.perf_counter()
    model = load_model(args.kind, args.model, device, revision)
    batch_size = args.batch_size or MODEL_LOAD.get(args.model, {}).get("batch_size", 64)
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
        "kind": args.kind,
        "card_count": len(texts),
        "oracle_ids_sha256": oracle_ids_digest(oracle_ids),
        "scryfall_updated_at": manifest.get("scryfall_updated_at"),
        "query_prompt": query_prompt,
        "doc_prompt": doc_prompt,
        "doc_max_length": max_len,
        "doc_tokens_p50": float(np.median(lengths)),
        "doc_tokens_max": int(lengths.max()),
        "truncated_docs": truncated,
        "load_kwargs": {k: v for k, v in MODEL_LOAD.get(args.model, {}).items() if k not in ENCODE_ONLY_KEYS},
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
        if args.kind == "late":
            tokens, offsets = encode_late(model, texts, batch_size)
            d = out / "late" / slug
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / "tokens.npy", tokens)
            np.save(d / "offsets.npy", offsets)
            meta.update(dim=int(tokens.shape[1]), n_tokens=int(tokens.shape[0]), dtype="float16")
            files = [d / "tokens.npy", d / "offsets.npy", d / "meta.json"]
            meta_path = d / "meta.json"
        elif args.kind == "dense":
            mat = encode_dense(model, texts, batch_size, doc_prompt)
            if not np.isfinite(mat).all():
                sys.exit("non-finite values in dense embeddings")
            d = out / "dense"
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / f"{slug}.npy", mat)
            meta.update(dim=int(mat.shape[1]), dtype="float16", normalized=True)
            meta_path = d / f"{slug}.json"
            files = [d / f"{slug}.npy", meta_path]
        else:
            import scipy.sparse as sp

            mat = encode_sparse(model, texts, batch_size)
            d = out / "sparse"
            d.mkdir(parents=True, exist_ok=True)
            sp.save_npz(d / f"{slug}.npz", mat, compressed=True)
            meta.update(
                dim=int(mat.shape[1]),
                nnz=int(mat.nnz),
                nnz_per_doc=float(mat.nnz / max(1, mat.shape[0])),
                dtype="float32",
                query_encoder=MODEL_LOAD.get(args.model, {}).get(
                    "query_encoder", "inference-free: query_0_SparseStaticEmbedding (tokenizer + weight lookup)"
                ),
            )
            meta_path = d / f"{slug}.json"
            files = [d / f"{slug}.npz", meta_path]
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
                commit_message=f"embeddings: {args.kind} {args.model} ({len(texts)} cards)",
            )
            log("uploaded:", info)
            return
        except Exception as e:  # noqa: BLE001 - concurrent commits from parallel jobs can race
            log(f"upload attempt {attempt + 1} failed: {e!r}")
            time.sleep(10 * (attempt + 1))
    sys.exit("upload failed")


if __name__ == "__main__":
    main()

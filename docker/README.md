# mtg-search Docker image

One image runs the whole MCP server over stdio:

```
docker run -i --rm mtg-search
```

Inside it, the Python MCP server (`python -m mtg_search.server`) starts `next-plaid-api`
on `127.0.0.1:8080`. NextPlaid serves the late-interaction index (`lightonai/mLateOn`,
34.6k cards) that was baked in at build time. stdout carries only MCP JSON-RPC, and all
logs go to stderr.

## Register with Claude Code

Build the image first, then:

```
claude mcp add mtg-search -- docker run -i --rm mtg-search
```

With a read-only root filesystem, the server copies the index to `/tmp` at startup:

```
claude mcp add mtg-search -- docker run -i --rm --read-only --tmpfs /tmp mtg-search
```

## Build

```
docker/build.sh                    # from HF dataset kasper-cbx/mtg-search-index@main
docker/build.sh --revision <tag>   # a pinned dataset revision (e.g. scryfall-<stamp>)
docker/build.sh --local            # dev: from ./data + ./index/embeddings (+ ./index/models)
```

- **HF builds** read the token from `HF_TOKEN`, falling back to `hf auth token`. It is passed as
  a BuildKit secret (`--secret id=hf_token,env=HF_TOKEN`) and never stored in a layer.
  `build.sh` resolves the revision to a commit sha so Docker's cache follows the dataset.
- **Local builds** download the query-encoder model into `./index/models/` if it is missing.
  This needs the token once.

Stages (see `Dockerfile`):

1. **base**: `python:3.12-slim-bookworm` plus `next-plaid-api` and ONNX Runtime, copied from
   `ghcr.io/lightonai/next-plaid:cpu-1.8.5`, plus the pinned `requirements.txt`.
2. **fetch**: `python -m mtg_search.artifacts download` fetches:
   - `data/*.parquet`
   - `embeddings/late/lightonai__mLateOn/` (doc token embeddings, encoded on GPU with PyLate)
   - `models/lightonai__mLateOn-int8rr/`
3. **indexer**: `build_index.py` starts an embeddings-only NextPlaid. It pushes the precomputed
   tokens through `POST /indices/cards/update` in batches of 500 docs (base64 f32, under the
   100 MB body limit), about 3.5 min on 6 cores. It then writes:
   - `doc_rows.npy`: NextPlaid doc id to `cards.parquet` row
   - `mtg_search_index.json`: card digest and model revision, checked at server startup
4. **final**: model, data, index and `src/mtg_search`, running as uid 1000.

Image size is about 2.0 GB on disk (728 MB compressed):

| Component | Size |
|---|---|
| index | 363 MB |
| int8 model + tokenizer | 347 MB |
| Python deps (polars, mcp, ...) | 298 MB |
| next-plaid + ORT | 64 MB |
| card data | 15 MB |

## Query encoder

The query encoder is mLateOn int8, re-quantized with `--reduce-range`. The `model_int8.onnx`
published in `lightonai/mLateOn` collapses embeddings on x86 CPUs without VNNI (see
`spike/nextplaid.md`). The fp32 `model.onnx` would add ~0.9 GB to the image and 1.2 GB of RAM.

The re-quantized model was validated on all 363 eval queries. Each run went through NextPlaid
with the same index and was compared with the exact in-process MaxSim reference
(`LateInMemory`, `eval/.cache/late-mLateOn.json`):

| query encoder | nDCG@10 (all) | vs reference | top-10 overlap | top-50 overlap | RAM |
|---|---|---|---|---|---|
| reference (exact, PyTorch fp32) | 28.8 | — | — | — | — |
| fp32 `model.onnx` | 29.2 | +0.4 | 82.5% | 84.9% | 2.3 GB |
| **int8 reduce-range** (shipped) | 28.3 | −0.5 | 72.3% | 75.8% | 1.0 GB |

Per set, int8-rr against the reference:

| set | nDCG@10 change |
|---|---|
| tags | −0.4 |
| handwritten | +0.2 |
| paraphrase | −0.8 |
| deckneeds | −0.0 |

The overlap below 100% for fp32 comes from PLAID's 4-bit residual compression. Raising
`n_ivf_probe` or `n_full_scores` does not change it.

## Filtering

`mtg_search.filters` is the single source of truth. The server computes the allowed rows with
`filters.allowed_indices(cards, **filters)`. It sends them as NextPlaid's `subset` (doc ids),
and NextPlaid restricts IVF candidate generation to centroids that contain those docs, scaling
`n_ivf_probe` for small subsets. This means:

- selective filters (`color_identity="G", format="pauper", cmc_max=2`) return the best *allowed*
  cards, never an empty post-filtered top-k;
- new filters in `filters.py` (with a `FILTER_PARAMS` entry) appear in the MCP tool schemas
  automatically. No metadata columns are mirrored into NextPlaid.

## Dev without rebuilding the server layer

Run only NextPlaid from the image, then the server from the checkout. It needs the same
`data/cards.parquet`; docs are matched by oracle_id:

```
docker run -d --rm --name mtgnp -p 8080:8080 --entrypoint next-plaid-api mtg-search \
    --host 0.0.0.0 --port 8080 --index-dir /app/index --model /app/model --int8 --query-length 512
NEXTPLAID_URL=http://127.0.0.1:8080 uv run mtg-search-mcp
```

## Tests

```
uv run pytest                    # unit tests (fake backend)
uv run pytest -m docker          # runs the image over stdio (initialize, tools/list, tools/call)
python docker/mcp_smoke.py       # sample calls with output and latency
```

Warm latency on an i5-9600K is 70–170 ms per `search_cards` (short query plus `subset`) and
about 150 ms for `find_similar` (the card's full text is the query). The first query after
start takes about 2–3 s while the index loads; the server warms it up in the background
after start.

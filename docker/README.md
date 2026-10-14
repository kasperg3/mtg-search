# mtg-search Docker image

One image runs the whole MCP server over stdio:

```
docker run -i --rm mtg-search
```

Inside it, the Python MCP server (`python -m mtg_search.server`) searches the 34.6k cards with
**exact late interaction** (`lightonai/mLateOn`): it encodes queries with the model's published
**fp32** ONNX export (onnxruntime) and computes exact MaxSim against every card's precomputed token
vectors (numpy). Nothing is approximated or quantized, so what users get ranks identically to the
reference used by every benchmark (EXP-012). stdout carries only MCP JSON-RPC, and all logs go to
stderr.

Tool output carries only deterministic rules and print facts: name, cost, mana value,
colours, colour identity, type line, rarity, oracle text per face, P/T / loyalty / defense,
keywords, produced mana, legality, filtered functional oracle tags, and the Legendary,
Commander-eligible / partner, Game Changer (WotC's official bracket list), Reserved and layout
markers. `get_card` adds the official rulings and the Scryfall link. Results are ordered by text
relevance only. Popularity, price and other crowd- or market-derived metrics (EDHREC rank,
prices, salt) are never loaded, shown or used for ordering. The server reads only an
allow-list of `cards.parquet` columns (`SERVER_COLUMNS` in `src/mtg_search/server.py`).

## Register with Claude Code

Build the image first, then:

```
claude mcp add mtg-search -- docker run -i --rm mtg-search
```

The image needs no writable filesystem, so `--read-only` works as is:

```
claude mcp add mtg-search -- docker run -i --rm --read-only mtg-search
```

## Build

```
docker/build.sh                    # from HF dataset kasper-cbx/mtg-search-index@main
docker/build.sh --revision <tag>   # a pinned dataset revision (e.g. scryfall-<stamp>)
docker/build.sh --local            # dev: from ./data + ./index/embeddings + ./index/models
```

- **HF builds** read the token from `HF_TOKEN`, falling back to `hf auth token`. It is passed as
  a BuildKit secret (`--secret id=hf_token,env=HF_TOKEN`) and never stored in a layer.
  `build.sh` resolves the revision to a commit sha so Docker's cache follows the dataset.
- **Local builds** download the query encoder into `./index/models/lightonai__mLateOn-fp32/` if it
  is missing. The encoder comes from the public `lightonai/mLateOn` repo, so it needs no token.

Stages (see `Dockerfile`):

1. **base**: `python:3.12-slim-bookworm` plus the pinned `requirements.txt` (mcp, polars, numpy,
   onnxruntime 1.30.0, tokenizers 0.23.2, ...).
2. **fetch**: `python -m mtg_search.artifacts download --parts data,embeddings,model` fetches:
   - `data/*.parquet` (private dataset);
   - `embeddings/late/lightonai__mLateOn/` (doc token vectors, fp16, encoded on GPU; private
     dataset);
   - `models/lightonai__mLateOn-fp32/` (`model.onnx`, `tokenizer.json`, configs). These come from
     the public `lightonai/mLateOn` at the same revision as the embeddings (`bbd883a`), are
     sha256-verified, and are recorded in `source.json`.
3. **final**: model, embeddings, data and `src/mtg_search`, running as uid 1000.

At startup the server checks:
- that the embeddings belong to this exact card table (`meta.json` `oracle_ids_sha256`, row order
  matters);
- that the encoder and the embeddings come from one model revision.

Size:

| component | size |
|---|---|
| fp32 query encoder (`model.onnx` + tokenizer) | 1.28 GB |
| doc token vectors (2.18M × 128, fp16) | 558 MB |
| Python deps (polars, onnxruntime, mcp, ...) | 397 MB |
| base image | ~130 MB |
| card data | 15 MB |
| **total content** | **≈ 2.4 GB** |

`docker images` reports 4.2 GB, because it counts compressed and unpacked layers.

## Search

- **Query:** `[Q] ` + text → tokenizer (`<bos> [Q] … <eos>`, no truncation, no query expansion) →
  fp32 ONNX → 128-dim token vectors, L2-normalised per token. This is identical to
  sentence-transformers on PyTorch: per-token cosine 1.00000 on test queries.
- **Score:** for every card, the sum over query tokens of the max dot product with the card's
  token vectors (MaxSim). The vectors are held as fp32 in RAM. The work is split into ~530
  card-aligned chunks of ~4k tokens, scored by a thread pool (`MTG_SEARCH_THREADS`, default
  `min(8, cpu count)`; BLAS stays single-threaded). The search is bound by memory bandwidth.
- **Multi-query `search_cards`:** the 2–4 queries share one pass over the vectors, then their
  rankings are fused with RRF (k=60).
- **Filters:** `mtg_search.filters` computes the allowed rows. Only those compete for the top-k,
  so selective filters return the best *allowed* cards, never an empty post-filtered list. Ties
  break by row, deterministically.

Parity with the benchmark reference (`eval/experiments/exact_parity.py`, all 363 eval queries
under their filters, vs `LateInMemory` on PyTorch fp32):

| | value |
|---|---|
| top-1 agreement | 100% |
| top-10 / top-50 / top-100 overlap | 100% / 99.99% / 100% |
| max score difference | < 1e-5 |
| nDCG@10 per set | identical (tags 60.55, handwritten 70.72, paraphrase 11.90, deckneeds 21.14) |

Performance (i5-9600K, 6 cores):

| | exact (this image) | previous NextPlaid + int8 image |
|---|---|---|
| top-10 agreement with the benchmark ranker | **100%** | 72–85% |
| warm `search_cards`, 1 query | 111–148 ms | 70–170 ms |
| `search_cards`, 4 queries | ~375 ms | — |
| `find_similar` (a card's full text as query) | ~250 ms | ~150 ms |
| startup | ~7 s | ~2–3 s (plus background index warm-up) |
| RAM after load | 2.68 GiB | 1.04 GiB |

Holding the vectors as fp16 and converting per chunk would save 0.53 GB of RAM with bit-identical
scores, but it doubles single-query latency (215 vs 106 ms), so the image keeps fp32.

### Legacy NextPlaid path

`src/mtg_search/nextplaid.py` and `docker/build_index.py` (approximate PLAID index plus int8
re-quantized encoder) remain in the repo. The server uses them with `MTG_SEARCH_BACKEND=nextplaid`,
but this image does not build them. History of that path: `spike/nextplaid.md`, review M1 in
`docs/review_2026-10-14.md`, EXP-012.

## Dev without Docker

From the checkout, with `./data`, `./index/embeddings` and `./index/models/lightonai__mLateOn-fp32`:

```
MTG_SEARCH_EMBEDDINGS_DIR=index/embeddings/late/lightonai__mLateOn \
MTG_SEARCH_MODEL_DIR=index/models/lightonai__mLateOn-fp32 OPENBLAS_NUM_THREADS=1 uv run mtg-search-mcp
```

## Tests

```
uv run pytest                    # unit tests (fake/stub backends, incl. tests/test_exact.py)
uv run pytest -m docker          # runs the image over stdio (initialize, tools/list, tools/call)
python docker/mcp_smoke.py       # sample calls with output, latency, instructions/alwaysLoad check
```

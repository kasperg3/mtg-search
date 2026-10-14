# syntax=docker/dockerfile:1.10
# mtg-search: one image = stdio MCP server (Python) + NextPlaid (Rust, localhost subprocess)
# + mLateOn int8 (reduce-range) query encoder + the card index baked at build time.
#
# Build with docker/build.sh (see docker/README.md). Directly:
#   HF_TOKEN=... docker build -f docker/Dockerfile --secret id=hf_token,env=HF_TOKEN \
#       --build-arg HF_REVISION=<commit|branch|tag> -t mtg-search .
#   docker build -f docker/Dockerfile --build-arg SOURCE=local -t mtg-search .   # ./data + ./index
# Run:
#   docker run -i --rm mtg-search

ARG NEXTPLAID_IMAGE=ghcr.io/lightonai/next-plaid:cpu-1.8.5
# hf = fetch artifacts from the private HF dataset; local = take ./data and ./index from the context
ARG SOURCE=hf

FROM ${NEXTPLAID_IMAGE} AS nextplaid

# ---- runtime base: python 3.12 on the same Debian (bookworm) as next-plaid -------------------
FROM python:3.12-slim-bookworm AS base
COPY --from=nextplaid /usr/local/bin/next-plaid-api /usr/local/bin/next-plaid-api
COPY --from=nextplaid /opt/ort_cpu /opt/ort_cpu
ENV ORT_DYLIB_PATH=/opt/ort_cpu/libonnxruntime.so.1.23.0 \
    LD_LIBRARY_PATH=/opt/ort_cpu \
    OPENBLAS_NUM_THREADS=1 \
    RUST_LOG=warn \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src
# uv is mounted only while installing, so it does not end up in the image
RUN --mount=from=ghcr.io/astral-sh/uv:0.11.0,source=/uv,target=/usr/local/bin/uv \
    --mount=type=bind,source=docker/requirements.txt,target=/tmp/requirements.txt \
    uv pip install --system --no-cache -r /tmp/requirements.txt

# ---- artifacts from the private HF dataset (token via BuildKit secret, never in a layer) ------
FROM base AS fetch-hf
RUN --mount=from=ghcr.io/astral-sh/uv:0.11.0,source=/uv,target=/usr/local/bin/uv \
    uv pip install --system --no-cache huggingface_hub==1.33.0
# only the downloader (so server code edits do not invalidate the download/index layers)
COPY src/mtg_search/__init__.py src/mtg_search/config.py src/mtg_search/artifacts.py /app/src/mtg_search/
# build.sh passes the resolved commit sha so the layer cache follows the dataset
ARG HF_REVISION=main
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN \
    test -n "$HF_TOKEN" || { echo "HF_TOKEN secret missing: --secret id=hf_token,env=HF_TOKEN" >&2; exit 1; }; \
    python -m mtg_search.artifacts download --dest /artifacts --revision "${HF_REVISION}" \
    && rm -rf /artifacts/.cache

# ---- artifacts from the build context (dev): ./data, ./index/embeddings, ./index/models -------
FROM scratch AS fetch-local
COPY data/cards.parquet data/tags.parquet data/rulings.parquet data/manifest.json /artifacts/data/
COPY index/embeddings/late/lightonai__mLateOn/ /artifacts/embeddings/late/lightonai__mLateOn/
COPY index/models/lightonai__mLateOn-int8rr/ /artifacts/models/lightonai__mLateOn-int8rr/

FROM fetch-${SOURCE} AS fetch

# ---- indexer: push precomputed doc token embeddings into an embeddings-only NextPlaid ---------
FROM base AS indexer
COPY docker/build_index.py /build/build_index.py
RUN --mount=from=fetch,source=/artifacts,target=/artifacts \
    python /build/build_index.py --artifacts /artifacts --out /index

# ---- final ------------------------------------------------------------------------------------
FROM base
RUN useradd -m -u 1000 app
COPY --from=fetch /artifacts/models/lightonai__mLateOn-int8rr/ /app/model/
COPY --from=fetch /artifacts/data/cards.parquet /artifacts/data/tags.parquet \
     /artifacts/data/rulings.parquet /artifacts/data/manifest.json /app/data/
# the index must stay writable (next-plaid opens lock files); with --read-only the server
# copies it to $TMPDIR (add --tmpfs /tmp)
COPY --from=indexer --chown=app /index /app/index
COPY src/mtg_search /app/src/mtg_search
ENV MTG_SEARCH_DATA_DIR=/app/data \
    MTG_SEARCH_NEXTPLAID_INDEX=/app/index \
    MTG_SEARCH_MODEL_DIR=/app/model \
    MTG_SEARCH_LOG=WARNING
USER app
WORKDIR /app
ENTRYPOINT ["python", "-m", "mtg_search.server"]

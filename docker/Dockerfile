# syntax=docker/dockerfile:1.10
# mtg-search: one image = stdio MCP server (Python) with exact late-interaction search:
# mLateOn fp32 ONNX query encoder (onnxruntime) + exact MaxSim over the precomputed card token vectors
# (numpy). No index service: at ~35k cards exact search is one matrix product (~100 ms) and ranks
# identically to the benchmarks' reference (EXP-012). The earlier NextPlaid image is in git history
# (up to commit 61071da).
#
# Build with docker/build.sh (see docker/README.md). Directly:
#   HF_TOKEN=... docker build -f docker/Dockerfile --secret id=hf_token,env=HF_TOKEN \
#       --build-arg HF_REVISION=<commit|branch|tag> -t mtg-search .
#   docker build -f docker/Dockerfile --build-arg SOURCE=local -t mtg-search .   # ./data + ./index
# Run:
#   docker run -i --rm mtg-search

# hf = fetch artifacts from the private HF dataset (+ the public query model); local = ./data and ./index
ARG SOURCE=hf

# ---- runtime base ------------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS base
# the MaxSim thread pool owns the cores; keep BLAS single-threaded inside each worker
ENV OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src
# uv is mounted only while installing, so it does not end up in the image
RUN --mount=from=ghcr.io/astral-sh/uv:0.11.0,source=/uv,target=/usr/local/bin/uv \
    --mount=type=bind,source=docker/requirements.txt,target=/tmp/requirements.txt \
    uv pip install --system --no-cache -r /tmp/requirements.txt

# ---- artifacts from the private HF dataset (token via BuildKit secret, never in a layer) ------
FROM base AS fetch-hf
# only the downloader (so server code edits do not invalidate the download layer)
COPY src/mtg_search/__init__.py src/mtg_search/config.py src/mtg_search/artifacts.py /app/src/mtg_search/
# build.sh passes the resolved commit sha so the layer cache follows the dataset
ARG HF_REVISION=main
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN \
    test -n "$HF_TOKEN" || { echo "HF_TOKEN secret missing: --secret id=hf_token,env=HF_TOKEN" >&2; exit 1; }; \
    python -m mtg_search.artifacts download --dest /artifacts --revision "${HF_REVISION}" \
        --parts data,embeddings,model \
    && rm -rf /artifacts/.cache /root/.cache

# ---- artifacts from the build context (dev): ./data, ./index/embeddings, ./index/models -------
FROM scratch AS fetch-local
COPY data/cards.parquet data/tags.parquet data/rulings.parquet data/manifest.json /artifacts/data/
COPY index/embeddings/late/lightonai__mLateOn/ /artifacts/embeddings/late/lightonai__mLateOn/
COPY index/models/lightonai__mLateOn-fp32/ /artifacts/models/lightonai__mLateOn-fp32/

FROM fetch-${SOURCE} AS fetch

# ---- final ------------------------------------------------------------------------------------
FROM base
RUN useradd -m -u 1000 app
COPY --from=fetch /artifacts/models/lightonai__mLateOn-fp32/ /app/model/
COPY --from=fetch /artifacts/embeddings/late/lightonai__mLateOn/ /app/embeddings/
COPY --from=fetch /artifacts/data/cards.parquet /artifacts/data/tags.parquet \
     /artifacts/data/rulings.parquet /artifacts/data/manifest.json /app/data/
COPY src/mtg_search /app/src/mtg_search
ENV MTG_SEARCH_DATA_DIR=/app/data \
    MTG_SEARCH_EMBEDDINGS_DIR=/app/embeddings \
    MTG_SEARCH_MODEL_DIR=/app/model \
    MTG_SEARCH_LOG=WARNING
USER app
WORKDIR /app
ENTRYPOINT ["python", "-m", "mtg_search.server"]

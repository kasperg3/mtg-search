# syntax=docker/dockerfile:1
# PoC: one image = Python (stdio MCP-style) process + next-plaid-api as a localhost subprocess,
# with an index baked in at build time from precomputed embeddings.
#
# docker build -t mtg-search-poc \
#   --build-context model=<dir with tokenizer.json, onnx_config.json,
#                          config_sentence_transformers.json, model_int8.onnx> \
#   --build-context data=<dir with tokens.npy, offsets.npy, cards.jsonl> \
#   spike/poc
# echo "counter target spell unless its controller pays" | docker run -i --rm mtg-search-poc

ARG NEXTPLAID=ghcr.io/lightonai/next-plaid:cpu-1.8.5

FROM ${NEXTPLAID} AS nextplaid

# ---- runtime base: official python 3.12 on the same Debian (bookworm) as next-plaid -------
FROM python:3.12-slim-bookworm AS base
COPY --from=nextplaid /usr/local/bin/next-plaid-api /usr/local/bin/next-plaid-api
COPY --from=nextplaid /opt/ort_cpu /opt/ort_cpu
ENV ORT_DYLIB_PATH=/opt/ort_cpu/libonnxruntime.so.1.23.0 \
    LD_LIBRARY_PATH=/opt/ort_cpu \
    OPENBLAS_NUM_THREADS=1 \
    RUST_LOG=warn
COPY --from=ghcr.io/astral-sh/uv:0.11.0 /uv /usr/local/bin/uv
RUN uv pip install --system --no-cache requests==2.32.3 numpy==2.2.6

# ---- indexer: build the PLAID index from precomputed (GPU) embeddings, no model needed ----
FROM base AS indexer
COPY build_index.py /build/build_index.py
RUN --mount=from=data,target=/data \
    python /build/build_index.py /data /index

# ---- final ----------------------------------------------------------------------------------
FROM base
RUN useradd -m -u 1000 app
COPY --from=model --chown=app / /app/model
COPY --from=indexer --chown=app /index /app/index
COPY poc_server.py /app/poc_server.py
USER app
ENTRYPOINT ["python", "/app/poc_server.py"]

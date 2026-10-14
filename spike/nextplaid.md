# NextPlaid feasibility spike (Phase 0 / Agent B)

Date: 2026-10-14. Machine: i5-9600K (6 cores, AVX2, **no VNNI**), 15 GB RAM, Docker 29, CPU only.

## Verdict: GO (conditional)

`lightonai/mLateOn` can be served by NextPlaid on CPU, with good result quality, ~40 ms
end-to-end query latency and a ~350 MB index for 35k cards. There is one important condition:
**do not use the `model_int8.onnx` published on the HF repo.** On this CPU it produces
garbage (see Gotcha 1). Use the fp32 ONNX for queries, or re-quantize with `--reduce-range`.
Docs should be encoded with PyLate on GPU and pushed through `/update`.

## Versions
- Image `ghcr.io/lightonai/next-plaid:cpu-1.8.5`. This is the latest tag (repo tag `v1.8.5`, image built 2026-10-10, revision c6c091d). Server reports `version: 1.8.5`.
- `next-plaid-client` 1.8.5 on PyPI (not needed: the spike scripts use raw REST via `requests`). `pylate-onnx-export` 0.1.0.
- ONNX Runtime 1.23.0 inside the image (`/opt/ort_cpu`).
- Model `lightonai/mLateOn` (HF lastModified 2026-10-05). The repo already ships `model.onnx` (fp32, 1.2 GB), `model_int8.onnx` (313 MB), `onnx_config.json`, `tokenizer.json` and `export_onnx.py`, so no export step was needed. No fallback model was needed. `lightonai/LateOn` int8 also loads and ranks sensibly.

## Model loading
- The entrypoint `docker-entrypoint.sh` treats `--model org/name` as an HF id. It curls `tokenizer.json`, `config_sentence_transformers.json`, `onnx_config.json` and either `model_int8.onnx` (with `--int8`) or `model.onnx` into `/models/<name>`, then execs `next-plaid-api --model /models/<name>`. A local directory path also works.
- The loaded config for mLateOn: dim 128, `[Q] `/`[D] ` prefixes (ids 256000/256001), `do_query_expansion=false`. `query_length`/`document_length` are both **8192**: they are taken from `onnx_config.json`, so the CLI's documented 48/300 defaults do not apply. Pass `--query-length 64` to cap them explicitly if you want.

### Gotcha 1: the published mLateOn INT8 ONNX is broken on CPUs without VNNI
- **Symptom.** With `--int8`, every query returned the same 5 cards (Excavation Elephant, Virulent Plague, ...). Token embeddings had collapsed: mean pairwise cosine was 0.998 within a doc and 0.998 across docs. For comparison, fp32 gives 0.95 and LateOn int8 gives 0.93.
- **Confirmed** with the repo's own `export_onnx.py check`. The published int8 has per-token cosine vs PyLate of −0.08 (docs) and 0.47 (queries), and 10% top-1 agreement. fp32 `model.onnx` matches PyLate to max |diff| 3e-6, with 100% ranking agreement.
- **Cause.** u8s8 dynamic-quant saturation on x86 without VNNI. The script's own `--reduce-range` help text says exactly this.
- **Fix.** Re-quantizing locally fixes it: per-token cosine vs PyLate becomes 0.994 (docs) and 0.996 (queries), with 100% top-1 agreement:
  ```
  uv run --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
    export_onnx.py quantize --model-id lightonai/mLateOn --out-dir <dir with model.onnx> --reduce-range
  ```
  Then build a model dir containing `tokenizer.json`, `config_sentence_transformers.json`, `onnx_config.json` and the new `model_int8.onnx`, and run with `--model /models/mLateOn-rr --int8`. Search results were the same as fp32 by eye (top-3 identical apart from order).

## Commands used
```bash
docker run -d --name np -p 8088:8080 -v $IDX:/data/indices -v $MODELS:/models \
  ghcr.io/lightonai/next-plaid:cpu-1.8.5 --host 0.0.0.0 --port 8080 \
  --index-dir /data/indices --model lightonai/mLateOn            # fp32
#   ... --model /models/mLateOn-rr --int8                          # re-quantized int8
#   (no --model)                                                   # embeddings-only mode
python spike/make_sample.py oracle.jsonl.gz sample.jsonl 2000      # 2000 cards (41 forced)
uv run --no-project --python 3.12 --with requests,numpy python spike/run_spike.py URL sample.jsonl index|query|push
uv run ... python spike/scale_push.py URL embdir 17 [start_from_scratch]   # 34k-doc push test
```
- Sample source: Scryfall bulk `oracle_cards`. This was 34,646 eligible cards after the plan's layout and memorabilia drops; the sample was 1,959 random cards plus 41 must-include cards.
- Doc text was `name cost / type / oracle / P/T / Keywords`, with faces joined.
- Metadata per doc was `{oracle_id, name, color_identity ("BG", "C"), cmc, type_line, doc_text}`.

## API facts that matter
- **Async writes.** `POST /indices {name, config:{nbits:4}}` declares the index. `POST /indices/{n}/update_with_encoding {documents:[str], metadata:[{}]}` adds text docs and returns 202 (async). `POST /indices/{n}/update {documents:[{embeddings:[[...]]}], metadata}` adds **precomputed embeddings** without encoding. To know when a write is done, poll `GET /indices/{n}` (`num_documents`) and `/health` (`updates[].status`).
- **Search endpoints.**
  - `search_with_encoding`: text queries.
  - `search/filtered_with_encoding`: text queries plus `filter_condition` (a SQL WHERE with `?`) and `filter_parameters`.
  - Unified `/search`: takes `queries` as embeddings and/or `text_query` (FTS5 BM25 over *all metadata columns*), with `alpha` and `fusion` set to `relative_score` or `rrf`, and filter fields allowed in the body.
  - `/encode {texts, input_type: query|document}` returns token embeddings.
- **Metadata does double duty.** It is both the SQL filter store and the FTS5 keyword corpus, so put `doc_text` in metadata to get keyword/hybrid search over card text.
- **Size limit.** The request body limit is **100 MB** (axum `DefaultBodyLimit`) and anything larger gets HTTP 413. Embeddings travel as JSON floats, so cast to float64 and round to 4 decimals before `tolist()`; float32 values serialize with ~18 digits. About 1000 docs per request fits; 2000 does not.

## Results (2000-card sample)
| | fp32 model.onnx | re-quantized int8 | published int8 |
|---|---|---|---|
| Quality by eye | excellent | ≈ fp32 | **broken** |
| Indexing via `update_with_encoding` (CPU) | 6.6 docs/s (`--parallel 1`) | not measured | 17.3 docs/s (`--parallel 6 --batch-size 4`) |
| Query e2e `search_with_encoding` p50 / p95 | **39 / 47 ms** | 40 / 65 ms* | 32 / 40 ms |
| Query encode only, p50 | 33 ms | 22 ms | 32 ms (p6, 1 thread/session) |
| Search only (precomputed query emb), p50 | 7 ms | 10 ms* | 2 ms |
| Container RAM (idle, after queries) | 2.66 GB | **0.81 GB** | 3.9 GB (6 sessions) |

\* The int8 run was made while other containers were busy, so its latency numbers are noisy.

Both working models are far under the 150 ms target. Queries are 6–10 tokens (no query expansion).

**Sample results** (fp32 model, top 3–5):
- "counter target spell unless its controller pays": Mana Tithe, Rethink, Mana Leak, Force Spike, Spell Pierce.
- "whenever another creature you control dies, draw a card": Foster, Grim Haruspex, Guardian Project, Midnight Reaper.
- "sacrifice a creature: add mana": Phyrexian Altar, Birthing Pod, Thermopod, Deadly Dispute.
- "whenever a creature dies, each opponent loses 1 life": Zulaport Cutthroat, Bastion of Remembrance, Blood Artist.
- "exile target creature, its controller gains life": Swords to Plowshares, ..., Path to Exile.
- "win the game if your library is empty": Fblthp, Thassa's Oracle.
- "whenever an opponent draws a card": Underworld Dreams, Rhystic Study, Nekusar.

**Filtered search works.** I tested `color_identity IN (?,?,?,?)` and `cmc <= ? AND type_line LIKE ?`; for example, cmc≤2 instants for the counter query returned Mana Tithe, Mana Leak, Force Spike, Spell Pierce, Syncopate.

**Keyword (FTS5) and hybrid (`relative_score` and `rrf`) both work.** Keyword search for "unless its controller pays" returned Force Spike, Mana Tithe, Mana Leak, Spell Pierce, Rethink. Note that `rrf` scores display as about 0.0.

## Precomputed embeddings and scale
- **Pushed embeddings give identical rankings.** `/encode` (fp32, equal to PyLate) followed by `/update` into a fresh index matched the text-built index exactly: top-10 overlap was 10/10 on 3 queries. The 2000-doc push took 11 s. This validates the plan: GPU PyLate → `tokens.npy` + `offsets.npy` → `/update`.
- **Average 62.7 tokens per doc.** The fp16 token matrix for 35k docs is about 560 MB.
- **34k-doc scale test** (2000 real embeddings × 17 with noise, embeddings-only server, default `start_from_scratch=999`, 500 docs per request):
  - Indexed in **169 s** total.
  - 10,524 partitions.
  - **341 MB on disk (≈10 KB/doc)**. About half of that is the per-chunk `N.residuals.npy` files duplicated by `merged_residuals.npy`.
  - Search-only p50 **4.6 ms**, filtered p50 14.7 ms.
  - RAM 1.2 GB without a model.
  - Extrapolated: ~350 MB index for 35k cards, plus `metadata.db` (holding doc_text).
- **Gotcha: `start_from_scratch=40000`** (forcing full rebuilds so k-means sees all data) is quadratic. It reached only 11k docs after 10 min and stored raw `embeddings.npy` (1 GB at 34k). Do not use it. Incremental indexing after the first 999 docs keeps adding centroids (2048 → 10.5k), which looked fine. A quality comparison against a one-shot build is still open; it can be checked in the eval.
- **Gotcha: stale state after delete/recreate.** Running `DELETE /indices/x` then re-creating `x` inside the same server process can leave stale metadata state: "Index/DB sync repair failed: index has 1000 docs but DB has 34000" and every update then fails. Build into a fresh directory/process, for example the Docker builder stage below. Do not delete and recreate in a live server.

## Persistence and portability
- **Restart.** `docker restart` brings the index back with identical results. Indices are lazily loaded on first access, so the **first query after start takes about 2 s**; send a warm-up query at startup.
- **Copy.** `cp -r indices/cards` into another mount, served by a different container with a different model file (the re-quantized int8), worked unchanged. An index built by an embeddings-only server (no `--model`) is servable by a server with `--model`.
- **The index dir must be writable.** The server opens `merged_*.lock` files, so `docker run --read-only` fails with "Index load failed: Failed to open lock file ... Read-only file system". The container's writable layer is fine; a tmpfs/volume is needed only with `--read-only`.

## Single-image packaging (user requirement: `docker run -i --rm mtg-search`)
- **The next-plaid image is a normal OS image, not distroless.** It is **Debian 12 bookworm-slim** with bash, apt-get and curl, runs as user `nextplaid` (uid 1000), uses WORKDIR `/app` and ENTRYPOINT `docker-entrypoint.sh`. The binary is `/usr/local/bin/next-plaid-api` (42 MB, Rust).
- **Dependencies.** The binary only links glibc, libm and libgcc_s; SQLite is bundled. ONNX Runtime is loaded with dlopen from `ORT_DYLIB_PATH=/opt/ort_cpu/libonnxruntime.so.1.23.0` (22 MB, needs libstdc++). There is no Python in the image.
- **Two ways to add Python 3.12:**
  - (a) `FROM next-plaid`, then `USER root` and `uv python install 3.12`. Bookworm's own Python is 3.11.
  - (b) **Recommended, and verified:** a multi-stage build that copies the binary and `/opt/ort_cpu` into `python:3.12-slim-bookworm` (same Debian release), and sets `ORT_DYLIB_PATH` and `LD_LIBRARY_PATH`. It works out of the box.
- **Baking the index at build time works.** A builder stage starts `next-plaid-api` with no model on 127.0.0.1, pushes the precomputed embeddings via `/update`, waits for the update to finish, and terminates the server. The final stage then `COPY --from=indexer /index /app/index`.
- **PoC:** `spike/poc/`, consisting of `Dockerfile`, `build_index.py`, `poc_server.py` and `export_embeddings.py`. The model and embedding data come in as BuildKit named contexts, so nothing large lives in the repo.
  ```
  docker build -t mtg-search-poc --build-context model=<mLateOn-rr dir> \
      --build-context data=<dir: tokens.npy offsets.npy cards.jsonl> spike/poc
  printf 'counter target spell unless its controller pays\n' | docker run -i --rm mtg-search-poc
  ```
  - **Build:** the 300-doc index built in 5.6 s in the builder stage.
  - **Image size:** 1.06 GB, of which the model is 313 MB and python-slim about 150 MB. With fp32 instead of the re-quantized int8 it would be about 2 GB.
  - **Startup:** `poc_server.py` spawns `next-plaid-api --host 127.0.0.1 ... --model /app/model --int8` with its stdout redirected to **stderr**, which keeps stdout clean for MCP stdio. next-plaid was ready in 2.2 s.
  - **Queries:** the first query took about 2.1 s (lazy load); later ones took 50–75 ms. Results were correct (Mana Tithe/Mana Leak/Force Spike; Zulaport/Bastion/Blood Artist; Phyrexian Altar/Pitiless Plunderer/Ashnod's Altar).

## Recommendations for the production path
1. **Docs:** encode with PyLate on HF Jobs. fp32 ONNX equals PyLate to 1e-6, so tokens are interchangeable. Push via `/update` in ≤1000-doc requests to an embeddings-only next-plaid in a Docker builder stage, using the default `start_from_scratch`. Bake the resulting directory into the image.
2. **Queries:** use mLateOn **re-quantized with `--reduce-range`**: 0.8 GB RAM, ~40 ms per query. Alternatively use fp32 (2.7 GB RAM, ~40 ms). Never use the published `model_int8.onnx`. Host the re-quantized file in `kasper-cbx/mtg-search-index` (or a private model repo).
3. **Metadata:** put `doc_text`, `name`, `color_identity`, `cmc`, `type_line` and legalities in metadata, for SQL pre-filters and FTS5 hybrid search. Complex filters (format legality, types) are simple SQL.
4. **Startup:** send a warm-up query; keep the index dir writable; route next-plaid logs to stderr.

## Files
- `spike/make_sample.py`: Scryfall bulk to a 2k-card sample with doc text and metadata.
- `spike/run_spike.py`: index, query (latency, filters, keyword, hybrid) and push-precomputed tests.
- `spike/scale_push.py`: 34k-doc `/update` scale test.
- `spike/poc/*`: single-image proof of concept.

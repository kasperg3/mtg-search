# EXP-012: Ship exact late interaction (fp32, no PLAID index) instead of NextPlaid + int8

**Question.** The shipped Docker image served mLateOn through NextPlaid: an int8 re-quantized query encoder and a PLAID index with 4-bit residuals. Its top-10 agreed with the exact ranking that every benchmark measured on only 72–85% of slots (review M1). Can the image run the ~300M-parameter model at full precision with exact search instead, at acceptable latency and memory?

**Answer: yes; shipped.**
- **Parity.** The image now ranks **identically** to the benchmark reference: 363/363 top-1, 100% top-10 overlap, identical nDCG@10 on every eval set.
- **Latency.** About 110–150 ms per `search_cards` query, about 375 ms for a 4-query call, about 250 ms for `find_similar`.
- **Cost.** RAM goes from 1.0 to 2.7 GiB and image content from about 1.1 to about 2.4 GB. Startup takes about 7 s.

## Why the old path lost agreement

From `docker/README.md` (363 queries) and the review's 8-query re-measurement:

| query encoder via NextPlaid | nDCG@10 | top-10 agreement with exact |
|---|---|---|
| exact reference (PyTorch fp32, `LateInMemory`) | 28.8 | 100% |
| fp32 `model.onnx` | 29.2 | 82.5% |
| int8 reduce-range (was shipped) | 28.3 | 72.3% (review: 85% on 8 queries) |

About a third of the lost agreement came from int8; the rest came from the PLAID index, which compresses residuals to 4 bits and searches candidates through an IVF. At 34.6k cards (2.18M token vectors) an index buys nothing: exact MaxSim over every card is one matrix product. The int8 choice had been made for image size and RAM (README: fp32 "would add ~0.9 GB to the image and 1.2 GB of RAM"), not for quality.

## What shipped

- **`src/mtg_search/exact.py`, `ExactLateBackend`:**
  - Queries go through `tokenizers` + `onnxruntime` with mLateOn's published fp32 `model.onnx` (`lightonai/mLateOn@bbd883a`, sha256-pinned in `artifacts.py`). The encoding is `[Q] ` prefix → `<bos> [Q] … <eos>`, no expansion or truncation, per-token L2 normalisation.
  - Doc token vectors are the existing fp16 `tokens.npy`, held as fp32. MaxSim runs over ~530 card-aligned chunks of ~4k tokens in a thread pool; numpy and BLAS release the GIL, and BLAS stays single-threaded.
  - Multi-query `search_cards` calls share one pass over the vectors (`search_many`).
  - Allowed rows are filtered before top-k, and ties break by row, deterministically.
  - At load the backend checks that the embeddings match `cards.parquet` (`oracle_ids_sha256`; review m6) and that encoder and embeddings come from one revision (`source.json` vs `meta.json`).
  - onnxruntime telemetry is disabled.
- **Server.** `MTG_SEARCH_BACKEND=exact` is the default; `nextplaid` stays available, legacy and not built into the image.
- **Image** (`docker/Dockerfile`):
  - no NextPlaid binary and no index build;
  - onnxruntime 1.30.0 and tokenizers 0.23.2 pinned;
  - the fp32 model fetched from the public model repo;
  - `--read-only` works.
- **Tests.**
  - `tests/test_exact.py`: MaxSim vs brute force, batching, filtering, deterministic ties including at the top-k cut (a real bug in the first draft), and the mismatch guards.
  - The Docker stdio test and `docker/mcp_smoke.py` pass.

## Results

**Encoder parity.** Per-token cosine is 1.00000 and token counts are identical against sentence-transformers on PyTorch. The test queries included rules text, slang, French, symbols (`{T}: Add {G}.`) and a 64-token query.

**Ranking parity** (`eval/experiments/exact_parity.py`, run record `runs/20261014T221603Z_exact-parity.json`): all 363 eval queries under their filters, against the cached `LateInMemory` top-100.

| | value |
|---|---|
| top-1 agreement | 1.0000 |
| top-10 overlap (mean / min) | 1.0000 / 1.0000 |
| top-50 overlap | 0.9999 (order of exact ties at depth) |
| top-100 overlap | 1.0000 |
| max abs score difference | < 1e-5 |
| nDCG@10: tags / handwritten / paraphrase / deckneeds | 60.55 / 70.72 / 11.90 / 21.14, identical to the reference |

**Performance** (i5-9600K, 6 cores; MCP over stdio in the container, `docker/mcp_smoke.py`):

| | exact (shipped now) | NextPlaid + int8 (before) |
|---|---|---|
| top-10 agreement with the benchmark ranker | **100%** | 72–85% |
| `search_cards`, 1 query (warm) | 111–148 ms | 70–170 ms |
| `search_cards`, 2 / 4 queries | 180 / 375 ms | — |
| `find_similar` | 248 ms | ~150 ms |
| `get_card` / `list_tags` | 13 / 2 ms | similar |
| startup | ~7 s | ~2–3 s |
| RAM after load (`docker stats`) | 2.68 GiB | 1.04 GiB |
| image content (uncompressed) | ~2.4 GB (`docker images`: 4.24 GB) | ~1.1 GB (`docker images`: 1.96 GB) |

In-process, outside Docker: one query takes 101–106 ms; 4 batched queries take 326 ms, against 519 ms one after another. Chunk size matters. With 24 large chunks a 4-query batch took 559 ms, because the per-chunk intermediate fell out of the CPU cache. Holding the vectors as fp16 and upcasting per chunk gives bit-identical scores and saves 0.53 GB of RAM, but doubles single-query latency (215 ms), so the image keeps fp32.

## Consequences

- **Every retrieval number in EXP-001…011 now describes what the shipped server returns.** Before this change, the benchmarks measured the exact ranker while the image served an approximation. This resolves review M1.
- **Higher resource cost.** The server needs about 2.7 GiB of RAM; Claude Code starts it once per session.
- **Kept for reference:** NextPlaid code, `docker/build_index.py`, `spike/nextplaid.md`.

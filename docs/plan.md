# MTG semantic search MCP — exploration & MVP plan

_Written 2026-10-14. Facts below were checked against live sources that day unless marked "unverified". Historical: since 2026-10-15 the repo, the image and the datasets are public; see the README._

## 1. Exploration summary

### Data source → Scryfall bulk data (primary)
- Index: `GET https://api.scryfall.com/bulk-data/oracle_cards` → `updated_at`, `jsonl_download_uri`, `compressed_size`.
  - **The format has changed:** bulk files are now gzipped JSONL only, and the old `download_uri` field is gone. Old clients break on this.
  - Every API call needs a `User-Agent` naming the app and an `Accept` header.
  - Downloads from `data.scryfall.io` have no rate limit.
- Files we need:

  | File | Size | Contents |
  |---|---|---|
  | `oracle_cards` | 25 MB | One object per oracle card |
  | `oracle_tags` | 6 MB | Community Tagger tags: "removal", "ramp", "sac outlet", … |
  | `rulings` | 5 MB | Official rulings |

  All three join on `oracle_id`.
- Corpus size: the file holds 38,751 objects. About **34.6k** remain after dropping tokens, emblems, art_series, planar, scheme, vanguard and memorabilia. About 1.5k of those are Un-cards; flag them rather than drop them.
- Freshness: Scryfall rebuilds the files every 12–24 h. Card text rarely changes, so a weekly check plus a check after set releases is enough. Detect changes with `updated_at`.
- Secondary sources:
  - MTGJSON `AtomicCards.json.gz` provides `edhrecSaltiness`, with `Meta.json` for versioning.
  - The Comprehensive Rules `.txt` gives keyword definitions for a later `rules_lookup` tool.
- Hugging Face mirrors of card data are stale (newest is 2026-06). Don't use them as the source.
- **Terms:** Scryfall allows free use but forbids "simply repackaging/republishing" its data. Card text is © WotC under the Fan Content Policy. Keep any Hugging Face repo holding card text **private**, or publish only vectors and IDs and fetch the text locally at install time.

### Prior art
- **semble** (MinishLab):
  - Retrieval: static Model2Vec embeddings plus BM25, merged with RRF, then cheap heuristic re-ranking (more lexical weight when the query looks like a symbol, boosts for defining chunks).
  - Indexing: cached index with incremental updates.
  - MCP surface: two tools, `search` and `find_related`.
  - Reported results: about 1 ms per query and NDCG@10 of 0.85.
  - Lessons we copy: **hybrid lexical plus semantic retrieval, a `find_related` tool, compact output, and a prebuilt index.**
- Existing MTG MCP servers (pato/mtg-mcp, scryfall-mcp, PyPI `mtg-mcp`) all wrap Scryfall query syntax. None of them do embedding search, so this project fills that gap.

### Retrieval candidates
- **Late interaction**:
  - **`lightonai/LateOn`**: 149M params, 128 dims per token, BEIR 57.2, Apache-2.0. Ships `model_int8.onnx` (155 MB). Document limit 300 tokens, query limit 32.
  - Small fallbacks:
    - `answerai-colbert-small-v1` (33M, 96 dims, BEIR 53.8; ONNX version from lightonai)
    - `mxbai-edge-colbert-v0-17m` (48 dims, BEIR 49.0)
  - Rejected:
    - `LiquidAI/LFM2-ColBERT-350M`: twice the size, LFM license, no BEIR number.
    - Reason-ModernColBERT and jina-colbert-v2: non-commercial licenses.
  - MaxSim keeps one vector per token, which suits "target" vs "each" and "dies" vs "is exiled". No published evidence shows this on our kind of text, so we measure it.
- **Sparse**:
  - **`opensearch-neural-sparse-encoding-doc-v3-gte`**: BEIR 54.6, Apache. It is *inference-free at query time*: queries are tokenizer IDs times IDF, so a query costs no model run. It also learns document expansion, which suits MTG jargon.
  - **SparseUp** (`Linkup-Platform/linkup-sparseup-embed-v1`, Sep 2026, BEIR 56.4, Apache) is real but needs the encoder at query time.
  - SPLADE-v3 is non-commercial and gated.
- **Dense**:
  - **`google/embeddinggemma-2`**: Apache, Matryoshka 768→128, released Sep 2026. It is multimodal; load the text tower only.
  - `embeddinggemma-300m`: gated Gemma license, slightly higher MTEB.
  - `ibm-granite/granite-embedding-small-english-r2`: 48M, a fast CPU baseline.
  - Non-commercial, so excluded: jina-v5.
- **Lexical**: BM25, through SQLite FTS5 or `bm25s`.
- **Index storage** (34.6k cards × ~80 tokens ≈ 2.8M token vectors):

  | Model | Size |
  |---|---|
  | LateOn, 4-bit PQ | ~190 MB |
  | LateOn, fp16 | ~700 MB |
  | Dense, 768 dims fp16 | ~53 MB |
  | Sparse | ~10–20 MB |

  Everything is small enough to ship.

### NextPlaid
- Basics:
  - LightOn's Rust PLAID engine, Apache-2.0, Docker image `ghcr.io/lightonai/next-plaid:cpu-1.8.5`.
  - It runs **ONNX encoding inside the server** (`--model <hf id> --int8`), so it needs no separate inference server.
- Features we want:
  - Hybrid semantic plus **FTS5** keyword search.
  - **SQL metadata filters applied before scoring**, e.g. color identity ⊆ {U,B}, legal in commander, cmc ≤ 3.
  - 4-bit PQ storage, memory-mapped indexes, incremental add and delete.
- Gaps:
  - Python access is REST only (`next-plaid-client`); there are no in-process bindings.
  - There is no MCP server, and no documented format for exporting or shipping an index. The index is just a directory.
  - **Unverified:** whether `lightonai/LateOn` loads as-is. It is not on NextPlaid's "ready" list, so we may need `pylate-onnx-export`.

## 2. Decisions for the MVP

1. **Evaluation first, model second.** Build a small evaluation set and score BM25, dense, sparse and late interaction before committing. Ground truth:
   - **Scryfall oracle tags:** turn each tag into a natural-language query, e.g. `otag:sacrifice-outlet` → "free sacrifice outlet". The tagged cards are the relevant set.
   - **About 40 hand-written "minute wording" queries**, e.g. "creatures that can't be blocked by more than one creature", "instant-speed exile target permanent until end of turn", "whenever another creature you control dies, draw a card". Score these on nDCG@10 and Recall@50.
2. **Default architecture (expected winner): LateOn served by NextPlaid, plus FTS5 keyword search, merged with RRF, with SQL pre-filters.** If sparse or dense comes close on the evaluation, the fallback is a pure-Python in-process stack (sparse plus dense in numpy) with no Docker dependency.
3. **One document per `oracle_id`.** Embedded text:
   ```
   {name} {mana_cost}
   {type_line}
   {oracle_text}
   {P/T | loyalty | defense}
   Keywords: {keywords}
   ```
   - Multi-face cards: join each face with ` // `.
   - Oracle tags: index them as a separate FTS field rather than embedding them, so we can measure their effect separately.
   - Exclude flavor text.
4. **Precompute on Hugging Face, ship the artifacts.**
   - An HF Job (or a local CPU run of ~10 min) builds the index.
   - The output goes to a **private** HF dataset repo, e.g. `kasper-cbx/mtg-search-index`, tagged by Scryfall `updated_at`.
   - The MCP server pulls the latest revision on startup if it changed.

## 3. MVP plan

### Phase 0 — Spikes (½ day)
- [ ] Start NextPlaid with `--model lightonai/LateOn --int8` and confirm it loads.
  - If it doesn't, run `pylate-onnx-export lightonai/LateOn`.
  - If that fails too, fall back to `GTE-ModernColBERT-v1`, which is on the ready list.
- [ ] Check that NextPlaid keeps an index across container restarts and that the index directory can be copied to another machine.
- [ ] Time CPU query latency. Target: under 150 ms per query including encoding.

### Phase 1 — Data pipeline (`mtg_search/ingest.py`)
- [ ] Fetch `/bulk-data` with proper headers. Download `oracle_cards`, `oracle_tags` and `rulings` (`.jsonl.gz`) into `data/raw/<updated_at>/`.
- [ ] Normalize into `data/cards.parquet`:
  - `oracle_id`, `name`, `face_names`, `doc_text`
  - `mana_cost`, `cmc`, `colors`, `color_identity`, `type_line`, `supertypes/types/subtypes` (parsed)
  - `keywords`, `legalities`, `edhrec_rank`, `game_changer`
  - `is_funny`, `image_uri`, `scryfall_uri`, `tags`
- [ ] Write `manifest.json` with `scryfall_updated_at`, card count, model ids and a hash of the build config.

### Phase 2 — Evaluation harness (`eval/`)
- [ ] `eval/build_tag_queries.py`:
  - Pick about 60 functional oracle tags with 20–500 cards each.
  - Write a natural-language query for each (hand-edited once and committed).
- [ ] `eval/handwritten.yaml`: about 40 wording-sensitive queries, labelled by running Scryfall syntax searches and reviewing the results by hand.
- [ ] `eval/run.py`: scores BM25, dense, sparse, LateOn, and RRF combinations of them, on nDCG@10, Recall@50 and p50 latency. Results go to `eval/results.md`.

### Phase 3 — Index build (`mtg_search/build.py`)
- [ ] Encode the documents with the winning model or models. Run it on HF Jobs (`hf jobs uv run`) or locally.
- [ ] Late interaction path:
  - Push the precomputed embeddings to NextPlaid through `/update`, with metadata columns for SQL filtering.
  - Use `nbits=4`, and always rebuild from scratch, which is cheap at 35k cards.
- [ ] Also build the FTS5 table, plus the dense or sparse side if it is part of the fusion.
- [ ] Upload `index/`, `cards.parquet` and `manifest.json` to the private HF dataset repo.

### Phase 4 — MCP server (`mtg_search/server.py`, FastMCP, stdio, run with `uvx`)
Tools:

- **`search_cards(query, color_identity?, colors?, format?, types?, cmc_min?, cmc_max?, include_funny=False, limit=15)`**
  - Runs a semantic search and an FTS search, merges them with RRF, and applies SQL pre-filters. Quoted phrases in the query also get an exact-match boost.
  - Returns compact results: name, cost, type, oracle text, legality in the given format, and EDHREC rank.
- **`get_card(name)`**: exact or fuzzy name lookup, returning full text plus rulings.
- **`find_similar(card_name, …same filters)`**: uses the card's own document as the query, the equivalent of semble's `find_related`, for "more cards like Rhystic Study in my colors".
- **`list_tags(prefix?)`**, optional: exposes oracle tags so the agent can combine them into searches.

Startup behaviour:
- Check the HF repo revision, download if it has changed, and start the NextPlaid sidecar (Docker) or attach to a running one.
- If the sidecar isn't available, log a clear error.

### Phase 5 — Refresh automation
- [ ] A weekly GitHub Action (or HF scheduled Job):
  - Compare Scryfall `updated_at` with the manifest. If unchanged, exit.
  - If changed, run ingest, then build, then the evaluation smoke test (fail if nDCG@10 drops more than 3 points), then upload a new revision.
- [ ] Clients get the update on their next MCP start.

### Out of scope for the MVP (later)
- `rules_lookup` over the Comprehensive Rules glossary.
- Deck-aware tools: synergy with a pasted decklist, mana curve analysis.
- Fine-tuning LateOn with PyLate distillation on MTG pairs (oracle-tag positives, Scryfall "similar cards").
- Prices and printings from `default_cards`.
- Card images in results.

## 4. Open risks
- LateOn compatibility with NextPlaid (Phase 0).
- Docker as a dependency of an MCP server. The fallback is the in-process sparse/dense path, or PyLate's in-process PLAID index.
- Generic tag queries may favour dense and sparse models over late interaction. Keep the hand-written wording set separate in the report so its results stay visible.
- Redistribution: keep the HF repo with card text private (Scryfall and WotC terms).

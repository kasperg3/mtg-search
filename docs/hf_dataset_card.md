---
pretty_name: MTG Search Index (Scryfall oracle cards + retrieval artifacts)
license: other
license_name: derived-scryfall-wotc-fan-content
license_link: https://company.wizards.com/en/legal/fancontentpolicy
language:
  - en
tags:
  - magic-the-gathering
  - retrieval
  - colbert
  - late-interaction
  - sparse-retrieval
  - mcp
task_categories:
  - text-retrieval
size_categories:
  - 10K<n<100K
configs:
  - config_name: cards
    default: true
    data_files:
      - split: train
        path: data/cards.parquet
  - config_name: tags
    data_files:
      - split: train
        path: data/tags.parquet
  - config_name: rulings
    data_files:
      - split: train
        path: data/rulings.parquet
---

# MTG Search Index

This repo holds the prebuilt corpus and index artifacts for the **mtg-search** MCP server. The server gives semantic search over every Magic: The Gathering card, for use while building decks. You can ask for "creatures that sacrifice another creature for free, BG commander" or "counter target spell unless its controller pays, legal in pauper" and get back the cards whose rules text does that, filtered by colour identity, format legality, mana cost and type.

- **One document per Scryfall `oracle_id`**: each unique card appears once, whatever its printings.
- **34,646 cards**, 4,577 oracle tags and 78,973 rulings.
- **Snapshot:** Scryfall `oracle_cards` `updated_at` = **2026-10-14T09:01:55.948+00:00**. The tables were built at 2026-10-14T13:33:57Z.
- **Shipped retriever:** `lightonai/mLateOn` late interaction (ColBERT-style MaxSim), served by [NextPlaid](https://github.com/lightonai/next-plaid) inside one Docker image. The dense and sparse embeddings here were used in the evaluation; they are not shipped.

> **Private repo.** This dataset contains card names and rules text derived from Scryfall
> bulk data (© Wizards of the Coast). It is kept **private** on purpose. Please don't
> mirror or republish it. See [Data sources and terms](#data-sources-and-terms).

---

## Data sources and terms

### Scryfall bulk data

All card data comes from [Scryfall bulk data](https://scryfall.com/docs/api/bulk-data). It is fetched through `https://api.scryfall.com/bulk-data/{type}` (with a `User-Agent: mtg-search/0.1` header), then each `jsonl_download_uri` (gzipped JSONL) is streamed. The three files used, as recorded in `data/manifest.json`:

| Bulk type | Scryfall `updated_at` | File |
|---|---|---|
| `oracle_cards` | 2026-10-14T09:01:55.948+00:00 | https://data.scryfall.io/oracle-cards/oracle-cards-20261014090155.jsonl.gz |
| `oracle_tags` | 2026-10-14T09:00:35.830+00:00 | https://data.scryfall.io/oracle-tags/oracle-tags-20261014090035.jsonl.gz |
| `rulings` | 2026-10-14T09:00:34.088+00:00 | https://data.scryfall.io/rulings/rulings-20261014090034.jsonl.gz |

- **Oracle tags** are community annotations from [Scryfall Tagger](https://tagger.scryfall.com), such as `sacrifice-outlet`, `ramp` and `removal-creature`.
- **Rulings** are the official Wizards of the Coast rulings plus Scryfall's notes, as Scryfall publishes them.

### Terms, copyright and affiliation

- **Card content.** Card names, rules text, mana costs and type lines are © **Wizards of the Coast LLC**. They are used under the [Wizards of the Coast Fan Content Policy](https://company.wizards.com/en/legal/fancontentpolicy). This is unofficial Fan Content, not approved or endorsed by Wizards. Portions of the materials used are property of Wizards of the Coast. © Wizards of the Coast LLC.
- **Scryfall.** Use of Scryfall data follows the [Scryfall API terms](https://scryfall.com/docs/api). Those terms allow free use but forbid simply repackaging or republishing Scryfall's data. That is why this repo is **private** and exists only as the build input for the mtg-search tool. The `license: other` in the metadata reflects this: the dataset is derived data, not openly licensed.
- **No affiliation.** This project is **not affiliated with, endorsed by, or sponsored by Scryfall or Wizards of the Coast**.
- **Images.** Card images are not stored here. `image_uri` and `scryfall_uri` are links to Scryfall.

### Models used

Every model is used unmodified for encoding. The one exception is the re-quantized ONNX export of mLateOn described below.

| Model | Role | License | Revision used |
|---|---|---|---|
| [lightonai/mLateOn](https://huggingface.co/lightonai/mLateOn) | Late interaction (shipped retriever and query encoder) | Apache-2.0 | `bbd883ae1fadb563056ff73b1061ae8172a3fcda` |
| [answerdotai/answerai-colbert-small-v1](https://huggingface.co/answerdotai/answerai-colbert-small-v1) | Late interaction (small baseline) | Apache-2.0 | `934fa8bb4ce2284f4c2baa232d81aca4d076fa5e` |
| [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) | Dense (text tower only) | Apache-2.0 | `914f7f89142e33e77833254d9c9b90c3cef7303b` |
| [opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte) | Learned sparse, inference-free queries | Apache-2.0 | `1646fef40807937e8e130c66d327a26421c408d5` |
| [Linkup-Platform/linkup-sparseup-embed-v1](https://huggingface.co/Linkup-Platform/linkup-sparseup-embed-v1) | Learned sparse (SPLADE-style, encoder at query time) | Apache-2.0 | `08314498d4f6a3a205b930ab9f27001404ea94b8` |

---

## Repository layout

Sizes are for the current `main` (decimal MB).

| Path | Size | Description |
|---|---|---|
| `data/cards.parquet` | 8.6 MB | One row per oracle card: text, costs, types, legalities, tags, deck-building fields. Viewer config `cards`. |
| `data/tags.parquet` | 3.6 MB | One row per Scryfall Tagger oracle tag, with its direct taggings. Viewer config `tags`. |
| `data/rulings.parquet` | 2.3 MB | One row per ruling. Viewer config `rulings`. |
| `data/manifest.json` | 1.1 KB | Snapshot metadata: Scryfall `updated_at` and URIs per bulk file, row counts, build time, and the [content fingerprint](#content-fingerprint) that decides what a refresh must rebuild. |
| `embeddings/late/lightonai__mLateOn/tokens.npy` | 557.9 MB | float16 `(2,179,176, 128)`: per-token doc embeddings, concatenated in card order. |
| `embeddings/late/lightonai__mLateOn/offsets.npy` | 277 KB | int64 `(34,647,)`: doc *i* is `tokens[offsets[i]:offsets[i+1]]`. |
| `embeddings/late/lightonai__mLateOn/meta.json` | 0.7 KB | Model, revision, prompts, alignment hash, build info. |
| `embeddings/late/answerdotai__answerai-colbert-small-v1/tokens.npy` | 321.7 MB | float16 `(1,675,441, 96)`. |
| `embeddings/late/answerdotai__answerai-colbert-small-v1/offsets.npy` | 277 KB | int64 `(34,647,)`. |
| `embeddings/late/answerdotai__answerai-colbert-small-v1/meta.json` | 0.8 KB | As above. |
| `embeddings/dense/google__embeddinggemma-2.npy` | 53.2 MB | float16 `(34,646, 768)`, L2-normalised. |
| `embeddings/dense/google__embeddinggemma-2.json` | 0.9 KB | Metadata. |
| `embeddings/sparse/opensearch-project__opensearch-neural-sparse-encoding-doc-v3-gte.npz` | 36.7 MB | scipy CSR float32 `(34,646, 30,522)`, 7,579,238 nnz (~219 per doc). |
| `embeddings/sparse/opensearch-project__opensearch-neural-sparse-encoding-doc-v3-gte.json` | 1.0 KB | Metadata. |
| `embeddings/sparse/Linkup-Platform__linkup-sparseup-embed-v1.npz` | 35.5 MB | scipy CSR float32 `(34,646, 50,370)`, 6,870,821 nnz (~198 per doc). |
| `embeddings/sparse/Linkup-Platform__linkup-sparseup-embed-v1.json` | 1.0 KB | Metadata. |
| `models/lightonai__mLateOn-int8rr/model_int8.onnx` | 312.8 MB | mLateOn query encoder, ONNX Runtime dynamic int8 with `reduce_range=True` (see below). |
| `models/lightonai__mLateOn-int8rr/tokenizer.json` | 34.4 MB | Tokenizer, from the source model. |
| `models/lightonai__mLateOn-int8rr/onnx_config.json` | 0.5 KB | NextPlaid / PyLate ONNX config: dim 128, `[Q] ` / `[D] ` prefixes. |
| `models/lightonai__mLateOn-int8rr/config_sentence_transformers.json` | 0.5 KB | PyLate config, from the source model. |
| `models/lightonai__mLateOn-int8rr/quantization.json` | 0.6 KB | How and why it was re-quantized, with validation numbers. |
| `.gitattributes` | 2.6 KB | LFS rules. |

---

## Schema

### `data/cards.parquet` (34,646 rows, sorted by `name`)

| Column | Type | Meaning |
|---|---|---|
| `oracle_id` | string | Scryfall oracle id. The primary key, and the join key for every other file. |
| `name` | string | Card name. Multi-face cards use `Front // Back`. |
| `face_names` | list[string] | Name of each face; a single-faced card has one. |
| `doc_text` | string | The text that is embedded and indexed (see [below](#how-doc_text-is-built)). |
| `mana_cost` | string | e.g. `{1}{U}{U}`. Faces are joined with ` // ` when there is no top-level cost. |
| `cmc` | float64 | Mana value. |
| `pips_W` … `pips_G`, `pips_C` | int16 | Coloured-symbol counts with *devotion* semantics: hybrid `{G/W}` counts for both, and `{2/W}` and Phyrexian `{W/P}` count for W. The value is the maximum over faces. |
| `generic` | int32 | Sum of the numeric generic symbols (maximum over faces). |
| `has_x` / `has_hybrid` / `has_phyrexian` | bool | Any face has `{X}`/`{Y}`/`{Z}`, a hybrid symbol, or a Phyrexian symbol. |
| `face_costs` | list[struct] | The same cost breakdown per face (`name`, `mana_cost`, pips, `generic`, flags). |
| `colors` | list[string] | Card colours (the union over faces if needed), in WUBRG order. |
| `color_identity` | list[string] | Commander colour identity. |
| `type_line` | string | Full type line; faces joined with ` // `. |
| `supertypes` / `types` / `subtypes` | list[string] | Parsed from `type_line`, across all faces. |
| `keywords` | list[string] | Scryfall keyword abilities and actions. |
| `legalities` | struct of 23 strings | Per format: `legal` / `not_legal` / `restricted` / `banned`. The formats are alchemy, brawl, commander, competitivebrawl, duel, future, gladiator, historic, legacy, modern, oathbreaker, oldschool, pauper, paupercommander, penny, pioneer, predh, premodern, standard, standardbrawl, timeless, tlr and vintage. |
| `edhrec_rank` | int64 (nullable) | EDHREC popularity rank (null for 2,306 cards). Kept for offline analysis only: the MCP server never loads it, shows it or ranks by it. It changes often and is not part of the content fingerprint, so a rank change alone never triggers a refresh. |
| `game_changer` | bool | Card is on the Commander "Game Changers" list. |
| `is_funny` | bool | Un-set / playtest card that is **not tournament legal** (see [filtering](#filtering-rules)). |
| `layout` | string | Scryfall layout (normal, transform, modal_dfc, split, adventure, saga, …). |
| `set` / `set_type` / `released_at` | string | Set code, set type and release date of the representative printing. |
| `scryfall_uri` | string | Card page on Scryfall. |
| `image_uri` | string | Scryfall `normal` image URL (front face for multi-face cards). |
| `tags` | list[string] | Slugs of the Tagger oracle tags applied **directly** to this card (no ancestor closure). |
| `rarity` | string | Rarity of the representative printing. |
| `produced_mana` | list[string] | Mana the card can produce, card-level (e.g. `["C"]` for Sol Ring). |
| `is_permanent` | bool | Any face is an artifact, creature, enchantment, land, planeswalker or battle. |
| `is_legendary` | bool | `Legendary` supertype. |
| `can_be_commander` | bool | Legendary creature on the front face, or text saying it "can be your commander". |
| `partner_kind` | string (nullable) | How it pairs as a second commander: `partner`, `partner with`, `friends forever`, `partner—<group>`, `choose a background`, `background`, `doctor's companion`, `doctor`. |
| `power` / `toughness` / `loyalty` / `defense` | string (nullable) | Front-face stats. These are strings because values like `*`, `1+*` and `X` occur. |
| `faces` | list[struct] | Per face: `name`, `mana_cost`, `type_line`, `oracle_text`, `power`, `toughness`, `loyalty`, `defense`, `is_permanent`. |
| `reserved` | bool | On the Reserved List. |

Prices are not stored (removed in manifest `schema_version` 2): they change daily, the server does not use them, and they would make every snapshot look new.

### `data/tags.parquet` (4,577 rows)

| Column | Type | Meaning |
|---|---|---|
| `slug` | string | Tag slug, as in Scryfall's `otag:<slug>`. |
| `label` | string | Display name. |
| `description` | string | Tagger description (often empty). |
| `aliases` | list[string] | Alternative slugs. |
| `parent_slugs` / `child_slugs` | list[string] | Tag hierarchy. Relevance in the eval uses the closure over `child_slugs`. |
| `card_count` | int64 | Number of cards in this corpus tagged *directly*. 4,409 tags have at least one. |
| `oracle_ids` | list[string] | Directly tagged oracle ids, restricted to this corpus. |

### `data/rulings.parquet` (78,973 rows)

| Column | Type | Meaning |
|---|---|---|
| `oracle_id` | string | Card (join to `cards`). |
| `source` | string | `wotc` or `scryfall`. |
| `published_at` | date | Ruling date. |
| `comment` | string | Ruling text. |

Only rulings for cards that appear in `cards.parquet` are kept. They are sorted by `(oracle_id, published_at)`.

### Content fingerprint

`data/manifest.json` carries deterministic hashes of the tables (`fingerprint_version` 1, computed by `mtg_search.ingest.fingerprint`). Each hash is SHA-256 over the selected columns of every row as canonical JSON (sorted keys), one line per row, in `cards.parquet` order. Every `cards.parquet` column belongs to exactly one group.

| Field | Covers | A change means |
|---|---|---|
| `text_hash` | `oracle_id`, `doc_text` | The embeddings must be re-encoded. |
| `oracle_ids_sha256` | `oracle_id` order (same digest as the embedding `meta.json` files) | Cards were added, removed or reordered: re-encode. |
| `gameplay_hash` | `oracle_id`, `name`, `face_names`, `faces`, `mana_cost`, `face_costs`, `pips_W` … `pips_C`, `generic`, `has_x`, `has_hybrid`, `has_phyrexian`, `cmc`, `colors`, `color_identity`, `type_line`, `supertypes`, `types`, `subtypes`, `keywords`, `legalities`, `game_changer`, `reserved`, `layout`, `produced_mana`, `is_funny`, `is_permanent`, `is_legendary`, `can_be_commander`, `partner_kind`, `power`, `toughness`, `loyalty`, `defense`, `tags` | Rules facts changed (e.g. a ban or new tags): publish new tables, reuse the embeddings. |
| `aux_hash` | all of `tags.parquet` and `rulings.parquet` (row order ignored) | Publish new tables, reuse the embeddings. |
| (not hashed) | `edhrec_rank`, `image_uri`, `scryfall_uri`, and the representative printing's `set`, `set_type`, `released_at`, `rarity` | Nothing: these are volatile and never trigger a refresh on their own. |

Current values: `text_hash` `d046da6d…`, `gameplay_hash` `43058ab9…`, `aux_hash` `edbdd173…`, `oracle_ids_sha256` `1bff7b49…`.

### How `doc_text` is built

`src/mtg_search/doctext.py` renders one block per face:

```
{name} {mana_cost}
{type_line}
{oracle_text}
{power/toughness | Loyalty: n | Defense: n}
```

- **Multi-face cards** (transform, MDFC, split, adventure, flip, …) repeat the block for each face, separated by a line containing only `//`.
- **Keywords:** the card-level keywords are appended once, as `Keywords: Flying, Transform`.
- **Left out:** empty lines and **flavor text** (never included). Tags are not part of `doc_text` either. Example:

```
Delver of Secrets {U}
Creature — Human Wizard
At the beginning of your upkeep, look at the top card of your library. ...
1/1
//
Insectile Aberration
Creature — Human Insect
Flying
3/2
Keywords: Flying, Transform
```

### Filtering rules

Rules applied by `src/mtg_search/ingest.py` to the 38k-object `oracle_cards` file:

- **Excluded layouts:** `token`, `double_faced_token`, `emblem`, `art_series`, `planar`, `scheme`, `vanguard`, `front_card`.
- **Excluded set type:** `memorabilia`.
- **Kept but flagged:** Un-cards. `is_funny = True` only when `set_type == "funny"` **and** the card is not legal or restricted in vintage. Eternal-legal cards from funny sets, such as Unfinity's non-acorn cards, therefore stay `False`. This gives 1,420 cards flagged, out of 1,467 with `set_type == "funny"`. The MCP server excludes funny cards by default (`include_funny=False`).
- **Join scope:** tags and rulings are restricted to the oracle ids that remain after filtering.

---

## Embedding artifacts

All embeddings were computed by [`jobs/encode.py`](https://github.com/kasperg3/mtg-search/blob/main/jobs/encode.py), a PEP-723 uv script, on **Hugging Face Jobs** (`hf jobs uv run --flavor t4-small`, Tesla T4). The software was torch 2.14.1+cu130, transformers 5.19.0 and sentence-transformers 6.1.0. The input is `data/cards.parquet` (`oracle_id`, `doc_text`) from this repo. Each run uploads its own files with a commit titled `embeddings: <kind> <model> (<n> cards)`.

### Row alignment

Row *i* of every embedding artifact belongs to row *i* of `data/cards.parquet`, which is sorted by `name`. Every meta file records `oracle_ids_sha256`: the SHA-256 of all `oracle_id`s in that order, each followed by `\n`. **Check it before you use an artifact.** If it differs from the hash of your `cards.parquet`, the artifact belongs to another snapshot.

```python
import hashlib, polars as pl
ids = pl.read_parquet("data/cards.parquet", columns=["oracle_id"])["oracle_id"]
h = hashlib.sha256(); [h.update(i.encode() + b"\n") for i in ids]
assert h.hexdigest() == meta["oracle_ids_sha256"]
```

All five artifacts on `main` share hash `1bff7b4949a0fef7784539209f504514cde0c7f9d5c15f951e983b4988d5acf1` (Scryfall 2026-10-14T09:01:55.948+00:00).

### Per-model details

| Artifact | Format | Dim | Query prompt | Doc prompt | Doc max len | Doc tokens p50 / max | Truncated | Encode time (T4) |
|---|---|---|---|---|---|---|---|---|
| late `lightonai/mLateOn` | `tokens.npy` fp16 + `offsets.npy` int64; 2,179,176 tokens | 128 | `[Q] ` | `[D] ` | 8192 | 59 / 398 | 0 | 207 s |
| late `answerdotai/answerai-colbert-small-v1` | as above; 1,675,441 tokens | 96 | `[unused0] ` | `[unused1] ` | 300 | 59 / 380 | 1 | 50 s |
| dense `google/embeddinggemma-2` | `.npy` fp16, L2-normalised | 768 | `task: search result \| query: ` | `title: none \| text: ` | 8192 | 65 / 412 | 0 | 294 s |
| sparse `opensearch-…-doc-v3-gte` | `.npz` scipy CSR fp32 | 30,522 | none (inference-free: tokenizer + IDF weight lookup) | none | 8192 | 58 / 379 | 0 | 250 s |
| sparse `linkup-sparseup-embed-v1` | `.npz` scipy CSR fp32 | 50,370 | `[Q] ` (encoder at query time, max 128 tokens) | `[D] ` | 512 | 64 / 426 | 0 | 327 s |

Model-specific notes:

- **embeddinggemma-2** is loaded text-only (`config_kwargs: {vision_config: null, audio_config: null}`) and run in float32, because float16 overflows.
- **doc-v3-gte** uses `trust_remote_code` with remote-code revision `40ced75c3017eb27626c9d4ea981bde21a2662f4`. `jobs/encode.py` repairs the remote code so it runs correctly under transformers 5.x.
- **SparseUp** is loaded from a local snapshot of the pinned revision.

Loading:

```python
import numpy as np, scipy.sparse as sp
tokens  = np.load("embeddings/late/lightonai__mLateOn/tokens.npy", mmap_mode="r")
offsets = np.load("embeddings/late/lightonai__mLateOn/offsets.npy")
doc_i   = tokens[offsets[i]:offsets[i + 1]]          # (n_tokens_i, 128) float16

dense   = np.load("embeddings/dense/google__embeddinggemma-2.npy")   # (34646, 768)
sparse  = sp.load_npz("embeddings/sparse/opensearch-project__opensearch-neural-sparse-encoding-doc-v3-gte.npz")
```

### `models/lightonai__mLateOn-int8rr`: the re-quantized query encoder

The shipped Docker image encodes queries on CPU with an int8 ONNX export of `lightonai/mLateOn` (source revision `bbd883ae…`). It was built from the published fp32 `model.onnx` with ONNX Runtime dynamic quantization, u8s8, **`reduce_range=True`** (`export_onnx.py quantize --reduce-range`).

**Why.** The `model_int8.onnx` published in `lightonai/mLateOn` collapses token embeddings on x86 CPUs **without VNNI**, because of u8s8 saturation. On such a CPU (an i5-9600K) every query returned the same cards, and per-token cosine against PyLate was −0.08 for docs and 0.47 for queries. Reduce-range int8 matches fp32/PyLate, with per-token cosine **0.994 (docs) and 0.996 (queries)**. The fp32 model would add about 0.9 GB to the image and about 1.2 GB of RAM.

**Validation.** All 363 eval queries were run through NextPlaid (same index) and compared with an exact in-process MaxSim reference:

| Query encoder | nDCG@10 (all) | vs reference | Top-10 overlap | RAM |
|---|---|---|---|---|
| Reference (exact, PyTorch fp32) | 28.8 | — | — | — |
| fp32 `model.onnx` via NextPlaid | 29.2 | +0.4 | 82% | 2.3 GB |
| **int8 reduce-range** (shipped) | 28.3 | −0.5 | 72% | 1.0 GB |

Overlap stays below 100% even for fp32 because of PLAID's 4-bit residual compression.

---

## Evaluation summary

Four query sets were scored against the 34,646 cards, with funny cards excluded and per-query filters applied:

- **tags** (65): from Tagger labels.
- **handwritten** (46): wording-sensitive queries with regex labels.
- **paraphrase** (222): low-overlap rewordings of the queries in the first two sets.
- **deckneeds** (30): open-ended deck-building needs.

The table shows nDCG@10. `para+deck` is the pooled decision metric.

| System | tags | handwritten | paraphrase | deckneeds | para+deck | p50 ms (CPU) |
|---|---|---|---|---|---|---|
| rrf(dense+mLateOn) | 59.0 | 59.0 | 13.0 | 21.8 | 14.0 | 227 |
| **late-mLateOn** (shipped) | **60.5** | **70.7** | 11.9 | 21.1 | 13.0 | 105 |
| dense (embeddinggemma-2) | 42.8 | 36.7 | 9.1 | 17.9 | 10.1 | 120 |
| late-answerai | 48.2 | 65.1 | 8.8 | 17.0 | 9.8 | 81 |
| sparseup | 47.2 | 47.3 | 6.4 | 20.0 | 8.0 | 2128 |
| sparse (doc-v3-gte) | 40.0 | 47.3 | 4.9 | 19.6 | 6.7 | 0 |
| bm25 | 36.4 | 61.1 | 2.5 | 8.5 | 3.2 | 0 |

- **mLateOn vs dense:** mLateOn alone beats dense by +2.9 points on para+deck (95% CI [+0.4, +5.5]) and by +34.0 on handwritten.
- **Shipped choice:** mLateOn alone. Adding dense fusion gains about 1 point on para+deck but costs 7–12 points on handwritten (weighted vs unweighted fusion), and would bring torch and Gemma into the image.

Labels come from community tags and regexes, so absolute scores are lower bounds. Compare systems against each other rather than reading the numbers as absolute quality. The full results, query sets, labelling rules and experiment registry are in the (private) GitHub repo [kasperg3/mtg-search](https://github.com/kasperg3/mtg-search): see `eval/README.md`, `eval/results.md` and `eval/experiments/`.

---

## Update process

[`scripts/refresh.py`](https://github.com/kasperg3/mtg-search/blob/main/scripts/refresh.py) keeps this repo in sync with Scryfall. A GitHub Action (`.github/workflows/refresh.yml`) runs it **weekly, Mondays at 06:00 UTC**, and it can also be dispatched by hand.

Price fluctuations never trigger a corpus update; legality changes always do.

1. Compare Scryfall's `oracle_cards` `updated_at` with `data/manifest.json` on `main`. If they match, stop. (Scryfall's `updated_at` changes almost daily, mostly because of prices, so it only decides whether to look.)
2. Otherwise download the bulk files, re-ingest locally, check that the card count is within 33k–36k, and compare the [content fingerprint](#content-fingerprint) with the published manifest. The result picks one of three tiers:

| Tier | When | What happens |
|---|---|---|
| **No-op** | All hashes equal: only volatile fields changed (prices, EDHREC rank, printing metadata). | Nothing is uploaded or tagged. |
| **Metadata-only** | `gameplay_hash` or `aux_hash` changed, but `text_hash` and `oracle_ids_sha256` did not (bans and other legality changes, tags, rulings, …). | Upload `data/*.parquet` plus `manifest.json`, check that the published embeddings still match `oracle_ids_sha256`, tag. No re-encoding. |
| **Re-encode** | `text_hash` or `oracle_ids_sha256` changed (oracle text or errata, cards added or removed). | The full pipeline below. |

Full pipeline:

1. Tag the snapshot currently on `main` as `scryfall-<YYYYMMDDHHMMSS>` if it has no tag yet.
2. Upload `data/*.parquet` plus `manifest.json`.
3. Re-encode **mLateOn only**, on an HF Job (t4-small), and check that `oracle_ids_sha256` matches the new `cards.parquet`.
4. Run the eval smoke test, which fails if late-mLateOn nDCG@10 drops more than 3 points below the recorded baseline.
5. Tag the new snapshot `scryfall-<YYYYMMDDHHMMSS>`.

Both publishing tiers are followed by a Docker image rebuild: the image bakes `cards.parquet` and builds its NextPlaid index from the published embeddings.

Notes:

- **Old builds:** to reproduce one, pin `revision="scryfall-…"`.
- **Stale embeddings after a refresh:** the dense, sparse and answerai embeddings on `main` stay at the previous snapshot. Use `oracle_ids_sha256` to tell. They are only re-encoded by hand.
- **Current state:** no `scryfall-*` tag exists yet. The first refresh will tag the current snapshot (`scryfall-20261014090155`).

---

## Usage

The repo is private, so authenticate first (`hf auth login`, or set `HF_TOKEN`).

```python
from datasets import load_dataset
cards   = load_dataset("kasper-cbx/mtg-search-index", "cards",   split="train")
tags    = load_dataset("kasper-cbx/mtg-search-index", "tags",    split="train")
rulings = load_dataset("kasper-cbx/mtg-search-index", "rulings", split="train")
```

```python
import polars as pl
from huggingface_hub import hf_hub_download

path = hf_hub_download("kasper-cbx/mtg-search-index", "data/cards.parquet", repo_type="dataset")
cards = pl.read_parquet(path)
cards.filter(
    pl.col("legalities").struct.field("pauper").is_in(["legal", "restricted"]),
    pl.col("color_identity").list.eval(pl.element().is_in(["U", "B"])).list.all(),
    ~pl.col("is_funny"),
).select("name", "mana_cost", "type_line")
```

Run the MCP server with Docker. The build pulls this dataset, using `HF_TOKEN` passed as a BuildKit secret, and bakes a NextPlaid index into the image (about 2.0 GB on disk):

```bash
git clone https://github.com/kasperg3/mtg-search && cd mtg-search
docker/build.sh                                   # from kasper-cbx/mtg-search-index@main
docker/build.sh --revision scryfall-<stamp>       # a pinned snapshot
claude mcp add mtg-search -- docker run -i --rm mtg-search
```

The server's tools are `search_cards`, `get_card`, `find_similar` and `list_tags`. `search_cards` and `find_similar` take filters for colour identity, colours (with `includes` / `subset` / `exact` modes), format, types, cmc and mana pips. Results are ordered by text relevance only. Tool output carries only deterministic rules and print facts. Popularity, price and other crowd- or market-derived metrics are never loaded, shown or used for ordering (`edhrec_rank` is in the table for offline analysis only; prices are not stored). `game_changer` is shown because it is WotC's official Commander bracket designation.

---

## Limitations

- **Tags are incomplete and noisy.** Tagger is community-maintained: 204 cards have no tags at all, newer or obscure cards are often under-tagged, and some tags are loose. `tags` holds direct taggings only; take the closure over `child_slugs` yourself.
- **One printing per card.** `rarity`, `set`, `released_at` and `image_uri` come from the single representative printing that Scryfall's `oracle_cards` file picks. They are not the first printing. There are no prices.
- **English only.** All text is English Oracle text; non-English printings are not represented.
- **Un-cards are present.** They are flagged with `is_funny`, not removed. Filter them out unless you want them.
- **Oracle text only.** There is no flavor text, art, artist or printing-specific text. Reminder text is whatever Scryfall's Oracle text includes.
- **Snapshot drift.** Card text and legalities change with bans and errata. Use the snapshot in `manifest.json`, and check `oracle_ids_sha256` before mixing embeddings with tables.
- **Evaluation labels are proxies.** They come from tags and regexes, so absolute retrieval scores understate quality.
- **Private, derived data.** Not for redistribution; see [Data sources and terms](#data-sources-and-terms).

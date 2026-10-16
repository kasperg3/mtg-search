---
pretty_name: MTG Search Index (Scryfall Oracle cards + mLateOn embeddings + MTGJSON precons)
license: other
license_name: derived-scryfall-wotc-fan-content
license_link: https://company.wizards.com/en/legal/fancontentpolicy
language:
  - en
tags:
  - magic-the-gathering
  - retrieval
  - late-interaction
  - colbert
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
  - config_name: precons
    data_files:
      - split: train
        path: data/precons.parquet
  - config_name: popularity
    data_files:
      - split: train
        path: eval/popularity.parquet
---

# MTG Search Index

The card corpus and search index of **mtg-search**, an MCP server that gives an AI agent semantic search over every Magic: The Gathering card for deck building. Ask for "counter target spell unless its controller pays, legal in pauper" and it returns the cards whose rules text does that, filtered by colour identity, format, type and mana value.

- **One entry per distinct card** (Scryfall `oracle_id`), whatever its printings: **34,646 cards**, 4,579 Tagger function tags and 78,973 rulings.
- **Official precon decklists:** 1,871 preconstructed decks from [MTGJSON](https://mtgjson.com) (MIT), 197 of them Commander decks, keyed by `oracle_id`. The server's `get_deck` tool loads them by deck or commander name.
- **Snapshot:** Scryfall `oracle_cards` `updated_at` **2026-10-15T21:01:57.844+00:00**, MTGJSON 5.3.0+20261015, tagged **`scryfall-20261015210157`**.
- **Search:** [`lightonai/mLateOn`](https://huggingface.co/lightonai/mLateOn) late interaction (multi-vector, MaxSim), exact over all cards: no approximate index. Queries are encoded with mLateOn's published fp32 ONNX model at the revision the cards were encoded with. The shipped server returns exactly the same top 10 as the benchmark reference on all 363 benchmark queries.
- **Evaluation data that never reaches the server:** `eval/popularity.parquet` (EDHREC rank), used only to analyse popularity bias.

**Related datasets** (same project; both were built on the card snapshot `scryfall-20261014090155`):
- [`kasper-cbx/mtg-search-bench`](https://huggingface.co/datasets/kasper-cbx/mtg-search-bench): the search benchmark (363 queries), the 113 agent search tasks, and graded relevance labels.
- [`kasper-cbx/mtg-precon-bench`](https://huggingface.co/datasets/kasper-cbx/mtg-precon-bench): the precon deck benchmark (player add/cut rates from EDHREC, and deck-fit grades).

> This dataset contains card names and rules text derived from Scryfall bulk data
> (© Wizards of the Coast) and decklists from MTGJSON (MIT), published as the search index of the
> mtg-search MCP server.
> See [Data sources and terms](#data-sources-and-terms).

## Citation

Cite a tagged snapshot, so the cards and embeddings are fixed:

```bibtex
@misc{mtgsearch_index_2026,
  author       = {Gr{\o}ntved, Kasper},
  title        = {{MTG Search Index}: Scryfall Oracle cards with mLateOn late-interaction embeddings},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/kasper-cbx/mtg-search-index}},
  note         = {Revision scryfall-20261015210157 (Scryfall oracle\_cards 2026-10-15). Card data from Scryfall, decklists from MTGJSON, \textcopyright{} Wizards of the Coast}
}
```

---

## Data sources and terms

### Scryfall bulk data

All card data comes from [Scryfall bulk data](https://scryfall.com/docs/api/bulk-data). It is fetched through `https://api.scryfall.com/bulk-data/{type}` (with a `User-Agent: mtg-search/0.1` header), then each `jsonl_download_uri` (gzipped JSONL) is streamed. The three files used, as recorded in `data/manifest.json`:

| Bulk type | Scryfall `updated_at` | File |
|---|---|---|
| `oracle_cards` | 2026-10-15T21:01:57.844+00:00 | https://data.scryfall.io/oracle-cards/oracle-cards-20261015210157.jsonl.gz |
| `oracle_tags` | 2026-10-15T21:00:33.145+00:00 | https://data.scryfall.io/oracle-tags/oracle-tags-20261015210033.jsonl.gz |
| `rulings` | 2026-10-15T21:00:37.150+00:00 | https://data.scryfall.io/rulings/rulings-20261015210037.jsonl.gz |

- **Oracle tags** are community annotations from [Scryfall Tagger](https://tagger.scryfall.com), such as `sacrifice-outlet`, `ramp` and `removal-creature`.
- **Rulings** are the official Wizards of the Coast rulings plus Scryfall's notes, as Scryfall publishes them.

### MTGJSON decklists

`data/precons.parquet` is built from MTGJSON's [`AllDeckFiles.tar.gz`](https://mtgjson.com/api/v5/AllDeckFiles.tar.gz) (build 5.3.0+20261015), downloaded with an identifying User-Agent and checked against its published `.sha256`; each refresh requests it again only when MTGJSON has a newer build. Only deck names, types, set codes, release dates, the publisher's decklist link, and per card the `oracle_id`, name, count and section are kept: MTGJSON's prices, purchase links, ranks and printing fields are dropped. Card bundles that are not decks (Secret Lair drops, land packs, MTGO redemption lists, Deck Builder's Toolkits and similar) are left out.

MTGJSON's data is MIT licensed:

> MIT License
>
> Copyright © 2018-Present Zachary Halpern
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

The card names in the decklists are © Wizards of the Coast, as below.

### Terms, copyright and affiliation

- **Card content.** Card names, rules text, mana costs and type lines are © **Wizards of the Coast LLC**. They are used under the [Wizards of the Coast Fan Content Policy](https://company.wizards.com/en/legal/fancontentpolicy). This is unofficial Fan Content, not approved or endorsed by Wizards. Portions of the materials used are property of Wizards of the Coast. © Wizards of the Coast LLC.
- **Scryfall.** Use of Scryfall data follows the [Scryfall API terms](https://scryfall.com/docs/api). This dataset is the search index of the mtg-search tool: the card tables ship together with the embeddings and tags they were built for. The `license: other` in the metadata reflects that it is derived data, not openly licensed.
- **No affiliation.** This project is **not affiliated with, endorsed by, or sponsored by Scryfall, MTGJSON or Wizards of the Coast**.
- **Images.** Card images are not stored here. `image_uri` and `scryfall_uri` are links to Scryfall.

### Model

| Model | Role | License | Revision |
|---|---|---|---|
| [lightonai/mLateOn](https://huggingface.co/lightonai/mLateOn) | Document embeddings and query encoder (fp32 `model.onnx`) | Apache-2.0 | `bbd883ae1fadb563056ff73b1061ae8172a3fcda` |

---

## Repository layout

| Path | Size | Description |
|---|---|---|
| `data/cards.parquet` | 8.5 MB | One row per card: text, costs, types, legalities, tags, deck-building fields. Config `cards`. |
| `data/tags.parquet` | 3.6 MB | One row per Tagger function tag, with its direct taggings. Config `tags`. |
| `data/rulings.parquet` | 2.3 MB | One row per ruling. Config `rulings`. |
| `data/precons.parquet` | 0.9 MB | One row per official preconstructed deck, from MTGJSON. Config `precons`. |
| `data/manifest.json` | 1 KB | Snapshot metadata: Scryfall `updated_at` and URIs per bulk file, row counts, build time, and the [content fingerprint](#content-fingerprint). |
| `embeddings/late/lightonai__mLateOn/tokens.npy` | 557.9 MB | float16 `(2,179,160, 128)`: per-token card embeddings, concatenated in card order. |
| `embeddings/late/lightonai__mLateOn/offsets.npy` | 0.3 MB | int64 `(34,647,)`: card *i* is `tokens[offsets[i]:offsets[i+1]]`. |
| `embeddings/late/lightonai__mLateOn/meta.json` | 1 KB | Model, revision, prompts, alignment hash, build info. |
| `eval/popularity.parquet` | 0.8 MB | `oracle_id`, `edhrec_rank`. **Evaluation only**: the server never loads, shows or ranks by it, and the Docker image does not contain it. Config `popularity`. |

---

## Schema

### `data/cards.parquet` (34,646 rows, 46 columns, sorted by `name`)

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

### `data/tags.parquet` (4,579 rows)

| Column | Type | Meaning |
|---|---|---|
| `slug` | string | Tag slug, as in Scryfall's `otag:<slug>`. |
| `label` | string | Display name. |
| `description` | string | Tagger description (often empty). |
| `aliases` | list[string] | Alternative slugs. |
| `parent_slugs` / `child_slugs` | list[string] | Tag hierarchy. Relevance in the eval uses the closure over `child_slugs`. |
| `card_count` | int64 | Number of cards in this corpus tagged *directly*. 4,410 tags have at least one. |
| `oracle_ids` | list[string] | Directly tagged oracle ids, restricted to this corpus. |

### `data/rulings.parquet` (78,973 rows)

| Column | Type | Meaning |
|---|---|---|
| `oracle_id` | string | Card (join to `cards`). |
| `source` | string | `wotc` or `scryfall`. |
| `published_at` | date | Ruling date. |
| `comment` | string | Ruling text. |

Only rulings for cards that appear in `cards.parquet` are kept. They are sorted by `(oracle_id, published_at)`.

### `data/precons.parquet` (1,871 rows)

Every preconstructed deck in MTGJSON's AllDeckFiles except card bundles: Jumpstart (570), Theme Deck (220), Commander Deck (197), Intro Pack (167), Arena Starter Deck (101) and 25 more types, released 1993-12-10 to 2026-10-02. Sorted by `release_date` (newest first), then `name`.

| Column | Type | Meaning |
|---|---|---|
| `file` | string | MTGJSON deck file name, e.g. `CallingAllAngels_FDC` (`mtgjson.com/api/v5/decks/<file>.json`). |
| `name` | string | Deck name. |
| `type` | string | MTGJSON deck type: `Commander Deck`, `Jumpstart`, `Duel Deck`, `Historic Brawl Precon Deck`, … |
| `set_code` | string | Set code of the product. |
| `release_date` | string | Release date (`YYYY-MM-DD`). |
| `source` | string (nullable) | The publisher's decklist page, when MTGJSON has one. |
| `commanders` | list[string] | Commander names (empty for non-Commander decks). |
| `cards` | list[struct] | Per card: `oracle_id` (join to `cards`), `name`, `qty`, `section` (`commander`, `main` or `sideboard`). Foil and non-foil copies of a card are merged. |

50,663 card rows; every one has an `oracle_id`. 164 of them (Planechase planes and Theros Challenge Deck cards in box sets) are cards that `cards.parquet` leaves out by layout.

### `eval/popularity.parquet` (34,646 rows, evaluation only)

| Column | Type | Meaning |
|---|---|---|
| `oracle_id` | string | Card (join to `cards`). |
| `edhrec_rank` | int64 (nullable) | EDHREC's overall popularity rank at snapshot time (1 = most played; null for 2,305 cards). Used only to measure popularity bias in evaluations. It is not part of the content fingerprint, so a rank change never triggers a refresh. |

### Content fingerprint

`data/manifest.json` carries deterministic hashes of the tables (`fingerprint_version` 1, computed by `mtg_search.ingest.fingerprint`). Each hash is SHA-256 over the selected columns of every row as canonical JSON (sorted keys), one line per row, in `cards.parquet` order. Every `cards.parquet` column belongs to exactly one group.

| Field | Covers | A change means |
|---|---|---|
| `text_hash` | `oracle_id`, `doc_text` | The embeddings must be re-encoded. |
| `oracle_ids_sha256` | `oracle_id` order (same digest as the embedding `meta.json` files) | Cards were added, removed or reordered: re-encode. |
| `gameplay_hash` | `oracle_id`, `name`, `face_names`, `faces`, `mana_cost`, `face_costs`, `pips_W` … `pips_C`, `generic`, `has_x`, `has_hybrid`, `has_phyrexian`, `cmc`, `colors`, `color_identity`, `type_line`, `supertypes`, `types`, `subtypes`, `keywords`, `legalities`, `game_changer`, `reserved`, `layout`, `produced_mana`, `is_funny`, `is_permanent`, `is_legendary`, `can_be_commander`, `partner_kind`, `power`, `toughness`, `loyalty`, `defense`, `tags` | Rules facts changed (e.g. a ban or new tags): publish new tables, reuse the embeddings. |
| `aux_hash` | all of `tags.parquet` and `rulings.parquet` (row order ignored) | Publish new tables, reuse the embeddings. |
| `precons_hash` | all of `precons.parquet`, as canonical JSON per row in `file` order (`mtg_search.precons.fingerprint`) | An official decklist changed or a deck was added: publish new tables, reuse the embeddings. |
| (not hashed) | `image_uri`, `scryfall_uri`, and the representative printing's `set`, `set_type`, `released_at`, `rarity` | Nothing: these are volatile and never trigger a refresh on their own. |

Current values: `text_hash` `18caede2…`, `gameplay_hash` `e4399316…`, `aux_hash` `df8d47a7…`, `precons_hash` `df3c39a0…`, `oracle_ids_sha256` `1bff7b49…`. The manifest's `precons` field records the MTGJSON build and the deck count.

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
- **Kept but flagged:** Un-cards. `is_funny = True` only when `set_type == "funny"` **and** the card is not legal or restricted in vintage. Eternal-legal cards from funny sets, such as Unfinity's non-acorn cards, therefore stay `False`. This gives 1,420 cards flagged, out of 1,467 with `set_type == "funny"`. The MCP server's tools always exclude them.
- **Join scope:** tags and rulings are restricted to the oracle ids that remain after filtering.

---

## Embedding artifacts

The embeddings were computed by [`jobs/encode.py`](https://github.com/kasperg3/mtg-search/blob/main/jobs/encode.py), a PEP-723 uv script, on Hugging Face Jobs (`t4-small`, Tesla T4), with torch 2.14.1+cu130, transformers 5.20.0 and sentence-transformers 6.1.0. The input is `data/cards.parquet` (`oracle_id`, `doc_text`).

| Artifact | Format | Dim | Query prompt | Doc prompt | Doc max len | Doc tokens p50 / max | Truncated | Encode time (T4) |
|---|---|---|---|---|---|---|---|---|
| late `lightonai/mLateOn` | `tokens.npy` fp16 + `offsets.npy` int64; 2,179,160 tokens | 128 | `[Q] ` | `[D] ` | 8192 | 59 / 398 | 0 | 213 s |

**Row alignment.** Row *i* of the embeddings belongs to row *i* of `data/cards.parquet` (sorted by `name`). `meta.json` records `oracle_ids_sha256`, the SHA-256 of all `oracle_id`s in that order, each followed by `\n`. Check it before using the embeddings with a `cards.parquet` from another revision:

```python
import hashlib, json, numpy as np, polars as pl
meta = json.load(open("embeddings/late/lightonai__mLateOn/meta.json"))
ids = pl.read_parquet("data/cards.parquet", columns=["oracle_id"])["oracle_id"]
h = hashlib.sha256(); [h.update(i.encode() + b"\n") for i in ids]
assert h.hexdigest() == meta["oracle_ids_sha256"]  # 1bff7b49... for this snapshot

tokens  = np.load("embeddings/late/lightonai__mLateOn/tokens.npy", mmap_mode="r")
offsets = np.load("embeddings/late/lightonai__mLateOn/offsets.npy")
card_i  = tokens[offsets[i]:offsets[i + 1]]          # (n_tokens_i, 128) float16
```

**Search.** A query is encoded with the `[Q] ` prompt by mLateOn's fp32 `model.onnx` (revision above), and each card is scored by MaxSim: for every query token, the best-matching card token, summed. All 34,646 cards are scored exactly, as one matrix product; at this size an approximate index buys nothing. Filters restrict the cards that compete before ranking. Measured parity with the benchmark reference on all 363 queries: 100% top-1 and top-10 agreement, identical nDCG@10.

---

## Evaluation summary

The search benchmark ([`kasper-cbx/mtg-search-bench`](https://huggingface.co/datasets/kasper-cbx/mtg-search-bench), config `queries`) has four query sets, scored against all 34,646 cards with funny cards excluded and per-query filters applied: **tags** (65, from Tagger labels), **handwritten** (46, rules-text wording, regex labels), **paraphrase** (222, the same needs in players' words), **deckneeds** (30, open-ended deck needs). nDCG@10 × 100:

| System | tags | handwritten | paraphrase | deckneeds | para+deck |
|---|---|---|---|---|---|
| **mLateOn** (this index) | **60.5** | **70.7** | 11.9 | 21.1 | 13.0 |
| dense (embeddinggemma-2) | 42.8 | 36.7 | 9.1 | 17.9 | 10.1 |
| sparse (opensearch doc-v3-gte) | 40.0 | 47.3 | 4.9 | 19.6 | 6.7 |
| BM25 | 36.4 | 61.1 | 2.5 | 8.5 | 3.2 |

The other systems' embeddings are in revisions up to `9776f07` (see [History](#history)). The answer lists come from tags and patterns, so absolute scores are lower bounds; compare systems with each other. With player-worded requests rewritten as rules text, mLateOn reaches 75.5. Agent-level results (search vs grep vs reading every card, deck building) are in the project's findings and experiment registry.

---

## Update process

[`scripts/refresh.py`](https://github.com/kasperg3/mtg-search/blob/main/scripts/refresh.py) keeps this repo in sync with Scryfall, run monthly by a GitHub Action (`.github/workflows/refresh.yml`, the 1st of each month at 06:00 UTC) and by hand when needed, e.g. after a set release or a ban announcement. Prices and popularity never trigger an update; legality and text changes always do.

| Tier | When | What happens |
|---|---|---|
| **No-op** | All content hashes equal: only volatile fields changed. | Nothing is uploaded or tagged. |
| **Metadata-only** | `gameplay_hash`, `aux_hash` or `precons_hash` changed, but `text_hash` and `oracle_ids_sha256` did not (bans, tags, rulings, precon lists, …). | Upload `data/*`, check that the embeddings still match `oracle_ids_sha256`, tag. No re-encoding. |
| **Re-encode** | `text_hash` or `oracle_ids_sha256` changed (errata, new cards). | Upload `data/*`, re-encode mLateOn on an HF Job, verify alignment, run the eval smoke test (fails if mLateOn nDCG@10 drops more than 3 points), tag. |

Every publishing refresh tags the new snapshot `scryfall-<YYYYMMDDHHMMSS>` (from Scryfall's `updated_at`), and tags the snapshot it replaces if it has no tag yet. To reproduce or cite a build, pin `revision="scryfall-…"`.

---

## Usage

The repo is public, so no login is needed.

```python
from datasets import load_dataset
cards = load_dataset("kasper-cbx/mtg-search-index", "cards", split="train", revision="scryfall-20261015210157")
```

```python
import polars as pl
from huggingface_hub import hf_hub_download

path = hf_hub_download("kasper-cbx/mtg-search-index", "data/cards.parquet", repo_type="dataset",
                       revision="scryfall-20261015210157")
cards = pl.read_parquet(path)
cards.filter(
    pl.col("legalities").struct.field("pauper").is_in(["legal", "restricted"]),
    pl.col("color_identity").list.eval(pl.element().is_in(["U", "B"])).list.all(),
    ~pl.col("is_funny"),
).select("name", "mana_cost", "type_line")
```

The MCP server (Docker; the build pulls this dataset with `HF_TOKEN` as a BuildKit secret):

```bash
git clone https://github.com/kasperg3/mtg-search && cd mtg-search
docker/build.sh --revision scryfall-20261015210157
claude mcp add mtg-search -- docker run -i --rm mtg-search
```

The server's read-only tools: `search_cards` (1–4 queries, filters for colour identity, format, types and mana value), `get_card`, `find_similar` and, from mtg-search-mcp 0.2.0, `get_deck` (a Moxfield or Archidekt link, an official precon from `data/precons.parquet`, or a pasted list, checked against its format's deck rules). Results are ordered by text relevance only. Tool output carries rules and print facts only: no popularity, price or other crowd-derived data. `game_changer` is shown because it is the official Commander bracket designation.

---

## Limitations

- **Tags are incomplete and noisy.** Tagger is community-maintained: 204 cards have no tags, newer or obscure cards are often under-tagged, and some tags are loose. `tags` holds direct taggings only; take the closure over `child_slugs` yourself.
- **One printing per card.** `rarity`, `set`, `released_at` and `image_uri` come from the representative printing Scryfall's `oracle_cards` file picks, not the first printing. No prices.
- **English Oracle text only.** No flavor text, art, artist or printing-specific text.
- **Un-cards are present,** flagged with `is_funny`.
- **Snapshot drift.** Card text and legality change with bans and errata: pin a `scryfall-…` tag, and check `oracle_ids_sha256` before mixing embeddings and tables from different revisions.
- **Derived data.** Card text is © Wizards of the Coast; see [Data sources and terms](#data-sources-and-terms).

## History

- **Up to revision `9776f07`:** this repo also held the retrievers compared in the search benchmark and then retired: dense `google/embeddinggemma-2`, learned sparse `opensearch-neural-sparse-encoding-doc-v3-gte` and `Linkup-Platform/linkup-sparseup-embed-v1`, late `answerdotai/answerai-colbert-small-v1`, and a re-quantized int8 mLateOn query encoder for the earlier NextPlaid image. They were removed from `main` when the server moved to exact mLateOn search. Pin `9776f07` to reproduce those experiments.
- **`scryfall-20261015210157`** (2026-10-16): the first snapshot with `data/precons.parquet` (MTGJSON's official decklists) and `precons_hash` in the manifest. The search and precon benchmarks stay on `scryfall-20261014090155`.
- **Schema 2** (manifest `schema_version` 2): prices were dropped. `edhrec_rank` moved from `data/cards.parquet` to `eval/popularity.parquet`, so the server's tables carry no popularity data.

---
pretty_name: MTG Precon Cut/Add Benchmark (EXP-007)
license: other
license_name: mixed-mtgjson-mit-edhrec
language:
  - en
tags:
  - magic-the-gathering
  - benchmark
  - deck-building
  - agents
size_categories:
  - 1K<n<10K
configs:
  - config_name: decks
    default: true
    data_files:
      - split: train
        path: data/decks.parquet
  - config_name: labels
    data_files:
      - split: train
        path: data/labels.parquet
  - config_name: precons
    data_files:
      - split: train
        path: data/precons.parquet
  - config_name: commanders
    data_files:
      - split: train
        path: data/commanders.parquet
  - config_name: splits
    data_files:
      - split: train
        path: data/splits.parquet
---

# MTG Precon Cut/Add Benchmark (EXP-007)

> Decklists come from MTGJSON; the `labels` and `commanders` configs come from [EDHREC](https://edhrec.com)
> precon pages. Card names are © Wizards of the Coast.

This is a benchmark of deck-building judgement on Magic: The Gathering Commander preconstructed
decks ("precons"). The agent sees a precon's 100 cards (as card text) and must pick the cards to
**cut** and the cards to **add**. Its picks are graded against what thousands of players actually
cut from and added to that precon, as aggregated by EDHREC.

Write-up: `eval/experiments/precon_benchmark.md` in the [mtg-search repo](https://github.com/kasperg3/mtg-search). Schema contract:
`bench/precon/schema.md`. Build code: `bench/precon/` (built from git `{{GIT_SHA}}`).

- **Snapshot:** EDHREC labels {{SNAPSHOT}} (Next.js buildId `{{EDHREC_BUILD}}`); MTGJSON
  `{{MTGJSON_VERSION}}`; Scryfall `cards.parquet` `{{SCRYFALL_DATE}}`.
- **{{N_PRECONS}} precons** (MTGJSON Commander decks; Collector's Edition duplicates dropped), of
  which **{{N_MATCHED}}** are matched to an EDHREC precon page.
- **{{N_LABELS}} labels** ({{LABELS_BY_KIND}}).
- **Splits** (cutoff `{{CUTOFF}}`): all precons {{SPLITS_ALL}}; scored {{SPLITS_SCORED}}.

Not affiliated with, endorsed by or sponsored by Wizards of the Coast, EDHREC or Scryfall.
Magic: The Gathering card names and text are © Wizards of the Coast (Fan Content Policy).

## Sources and terms

| Source | Used for | Terms |
|---|---|---|
| [MTGJSON](https://mtgjson.com) `DeckList.json`, `AllDeckFiles`, `SetList.json`, `AtomicCards.json` | official precon decklists, release dates, set families, first printings | MIT licence |
| [EDHREC](https://edhrec.com/precon) precon pages (`__NEXT_DATA__` JSON) | cut/add lists and rates, number of decks, commander counts | Collected once per snapshot, at one request every 5 s. |
| [Scryfall](https://scryfall.com/docs/api/bulk-data) bulk `oracle_cards` (our `cards.parquet`) | `oracle_id` mapping, colour identity, legality, land/basic flags | Scryfall bulk-data terms; WotC Fan Content Policy |

**EDHREC fetch policy (as run):** one pass over the `/precon` index plus each precon page (main
commander only, {{EDHREC_REQUESTS}} HTTP requests incl. `robots.txt`); at most 1 request per 5 s,
single-threaded; User-Agent `mtg-search research benchmark (contact: kaspergrontved@proton.me; ...)`;
robots.txt checked for every URL (`/precon/` is allowed); every response cached on disk and never
re-fetched; exponential back-off on errors / 429 (no 429s occurred). Fetch log and stats are in
`manifest.json`.

## Configs

All card references are Scryfall `oracle_id`s (oracle level). `precon_id` is an opaque id,
`pc-` + sha1(MTGJSON deck file name)[:10], identical across configs.

### `decks` — task input (safe to show the agent)
One row per precon: `precon_id`, `commander_oracle_ids`, `commander_names`, `color_identity`,
`cards` (list of `{oracle_id, name, qty, is_land, is_basic, is_commander}`, the full official
100-card list incl. commanders), `n_cards`. **No** product/deck name, set code, release date,
prices, EDHREC rank or tags. Card text is rendered by the harness from `cards.parquet`.

### `labels` — gold (never shown to the agent)
One row per (precon, card, kind): `kind` ∈ `cut`, `cut_land`, `add`, `add_land` (EDHREC
"Cards/Lands to Cut/Add"); `list_rank`; `rate` (cuts: EDHREC `unpopularity` *u*; adds:
*r* = `num_decks / potential_decks`); `num_decks`, `potential_decks` (sample size), `synergy`,
`lift`; `gain`, `gain_syn`; `is_land`; `headline`; `in_precon`; `is_new_in_product`;
`in_color_identity`; `commander_legal`.

### `precons` — metadata (reporting, slicing, memorisation probe; not a task input)
`deck_name`, `product`, `set_code`, `parent_set_code`, `release_date`, `mtgjson_file`,
`mtgjson_type`, `source` (WotC decklist URL), `edhrec_slug`, `edhrec_url`, `edhrec_group`,
`match_method`, `edhrec_deck_jaccard`, `potential_decks`, `n_colors`, `is_universes_beyond`,
`is_secret_lair`, `commander_names`.

### `commanders` — EDHREC `precon_commander_counts`
`precon_id`, `oracle_id`, `name`, `num_decks`, `is_precon_commander` (the commander whose page
supplies the labels). Labels are always for the precon's main commander page.

### `splits`
`precon_id`, `release_date`, `split` ∈ train/dev/gap/test, `scored`, `exclude_reason`, `cutoff`.

Extra files: `manifest.json` (snapshot dates, URLs, fetch stats, counts, parameters, git sha,
file hashes), `tag_denylist.json` (eval-mode oracle-tag deny-list), `tag_audit.parquet`
(tag-enrichment audit on train cuts).

## Gains (plan §6 defaults)

| | gain 3 | gain 2 | gain 1 | gain 0 |
|---|---|---|---|---|
| cut, by *u* | u ≥ 0.50 | 0.30 ≤ u < 0.50 | 0.15 ≤ u < 0.30 | below, or not listed |
| add, by *r* | r ≥ 0.30 | 0.15 ≤ r < 0.30 | 0.05 ≤ r < 0.15 | below, or not listed |

`gain_syn` = `gain · 1[synergy > 0]` for adds (generic-staple discount; equals `gain` for cuts).
The plan intends to re-fix thresholds on the train split; scorers may recompute grades from `rate`.
Headline metrics use `headline = True` rows (nonland `cut` / `add`); land kinds are kept for
diagnostics.

## Splits

Split key = MTGJSON release date relative to a model training cutoff *C* (default `{{CUTOFF}}`,
an assumption because GLM-5.3's cutoff is unknown): train < C−12 mo ≤ dev < C ≤ gap < C+2 mo ≤
test. A precon is `scored` only if it has an EDHREC page, `potential_decks ≥ 300`, was released
at least 8 weeks before the snapshot, is not an MTGO theme deck (`TD0`), and is not in the gap.
Recompute for another cutoff with
`python bench/precon/build_dataset.py --snapshot {{SNAPSHOT}} --splits-only --cutoff YYYY-MM-DD`.

Slices: `is_universes_beyond`, `is_secret_lair`, `n_colors` (precons), `headline` / `is_land`
and `is_new_in_product` (labels).

## Matching

- **MTGJSON → oracle_id:** each card's `identifiers.scryfallOracleId` (checked against
  `cards.parquet`), falling back to exact normalised name / face name.
- **EDHREC → oracle_id:** exact normalised name, then face name (DFC / split / adventure), then
  the front half of `A // B`, then MTGJSON `flavorName` (in-universe Universes Beyond names, e.g.
  "The Five Arrive"). Unmapped names are logged in the build's `unmapped.json` (0 in this build).
- **EDHREC page → MTGJSON deck:** same set of commander oracle_ids and same set code; fallback
  is the best card-overlap Jaccard ≥ 0.7 within the set code. `edhrec_deck_jaccard` records the
  overlap between EDHREC's embedded list and the MTGJSON list.

## Leakage controls (plan §7)

- Task inputs (`decks`) hold only card identifiers and names, never product or deck names,
  release dates, prices, popularity or tags; the build asserts the allowed column set.
- Labels live in separate files (`labels`, `commanders`) from inputs.
- `tag_denylist.json` lists evaluative / precon-identifying oracle tags (`meme`,
  `weaker-in-singleton-formats`, `stronger-in-singleton-formats`, `staple-with-set-s-mechanic`,
  `planeswalker-deck-staples`, patterns `cycle-*-face-commander`, `*precon*`, `*staple*`) that the
  harness must not render in eval mode; the train-split tag audit is in `tag_audit.parquet`.
- Test split is post-cutoff by release date; product names are available only in `precons` for
  the memorisation probe.

## Limitations

- Ground truth is crowd behaviour (a proxy for merit), aggregated by EDHREC; lists are truncated
  (about 50 adds / 30 cuts), so add/cut scores are lower bounds and unlisted cards score 0.
- `unpopularity` semantics are not documented by EDHREC; `potential_decks` counts decks EDHREC
  attributes to the precon's commander.
- Old precons often have few decks (< 300) and are not scored; the post-cutoff test pool is small
  (wide CIs).
- Universes Beyond top adds are often same-franchise cards (`is_new_in_product`); this is flagged,
  not removed. `is_new_in_product` uses the earliest MTGJSON printing in the precon's set family.
- EDHREC's embedded list can differ slightly from the official list (see `edhrec_deck_jaccard`);
  the input is always the official MTGJSON list.
- Labels are a single snapshot; rates drift as more decks are registered.

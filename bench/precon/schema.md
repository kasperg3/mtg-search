# EXP-007 precon benchmark: data schema (v1, final for snapshot 2026-10-14)

Contract between the data side (`bench/precon/collect_*.py`, `build_dataset.py`) and the
scorer / harness / baselines (`bench/precon/scorer.py`, `harness/`, `baselines/`).
Spec: `docs/benchmark_precon_plan.md`. Built locally under
`data/bench/precon/<snapshot_date>/` (gitignored) and uploaded to the **private** HF dataset
`kasper-cbx/mtg-precon-bench`. Load with
`load_dataset("kasper-cbx/mtg-precon-bench", "<config>")` or read the parquet files directly.

All card references use `oracle_id` from our `data/cards.parquet` (Scryfall oracle-level).
Card text, cost, type, legality, colour identity, `game_changer` are looked up there by the
harness/scorer; the dataset itself never carries card text, prices or popularity.

## Key: `precon_id`

Opaque, stable id: `"pc-" + sha1(mtgjson_deck_file_name)[:10]` (e.g. `pc-3f9a12bc0d`).
It deliberately carries **no product or deck name** (leakage control T2b). Same id in every file.

## Files / HF configs

### `decks` (task INPUT; safe to show the agent) — `decks.parquet`, one row per precon
| column | type | notes |
|---|---|---|
| precon_id | str | |
| commander_oracle_ids | list[str] | 1 (or 2 for partners / background). The commander of the EDHREC page the labels come from; equals MTGJSON's commander except 4 decks where EDHREC's face commander differs (then `precons.commander_source = "edhrec"`) |
| commander_names | list[str] | oracle names (cards.parquet `name`) |
| color_identity | list[str] | union of commanders' identity, WUBRG order |
| cards | list[struct{oracle_id: str, name: str, qty: int, is_land: bool, is_basic: bool, is_commander: bool}] | full 100-card list incl. commander(s); `qty>1` only for basics / any-number cards |
| n_cards | int | sum(qty), normally 100 |

No product name, deck name, set code, release date, prices, edhrec_rank, tags here.

### `labels` (GOLD; PRIVATE; never shown to the agent) — `labels.parquet`, one row per (precon, card, kind)
| column | type | notes |
|---|---|---|
| precon_id | str | |
| oracle_id | str | mapped from EDHREC card name |
| name | str | oracle name |
| kind | str | `cut` \| `cut_land` \| `add` \| `add_land` (EDHREC "Cards to Cut"/"Lands to Cut"/"Cards to Add"/"Lands to Add") |
| list_rank | int | 0-based position in EDHREC's list |
| rate | float | cuts: `u` = unpopularity; adds: `r` = num_decks / potential_decks |
| num_decks | int? | adds only |
| potential_decks | int | sample size behind the rate (decks with this commander) |
| synergy | float? | adds (null for cuts if absent) |
| lift | float? | adds |
| gain | int | 0-3, plan §6 thresholds (cut: u>=.50→3, >=.30→2, >=.15→1; add: r>=.30→3, >=.15→2, >=.05→1) |
| gain_syn | int | add-synergy variant: gain * 1[synergy > 0]; for cuts = gain |
| is_land | bool | from cards.parquet type line |
| headline | bool | True for `cut`/`add` (nonland); land kinds kept but excluded from headline (default, Q6) |
| in_precon | bool | sanity: cuts True, adds False |
| is_new_in_product | bool | earliest MTGJSON printing is in the precon's set family (set, parent set, and sets whose parent is either) — thematic-add slice |
| in_color_identity | bool | card identity within the deck's commander identity (all True in this build) |
| commander_legal | bool | Scryfall commander legality == legal |

Unlisted cards have gain 0 (scorer convention). Lists are truncated by EDHREC (~50 adds, ~30 cuts).

### `precons` (METADATA; not a task input; for slicing / reporting / memorisation probe) — `precons.parquet`
precon_id, deck_name, product (MTGJSON set name), set_code, parent_set_code, release_date (date),
mtgjson_file, mtgjson_type, edhrec_slug, edhrec_url, potential_decks (int, main commander),
edhrec_group, match_method (`commander+set` | `jaccard`), edhrec_deck_jaccard (EDHREC embedded list vs MTGJSON
list, 1.0 for all matched), n_colors, is_universes_beyond (commander printing has MTGJSON promoType
`universesbeyond` or a triangle security stamp), is_secret_lair, commander_names, mtgjson_commander_names,
commander_source (`mtgjson` | `edhrec`), n_out_of_identity. `source` = WotC decklist URL.

### `commanders` (GOLD side) — `commanders.parquet`
precon_id, oracle_id, name, num_decks, is_precon_commander — from EDHREC `precon_commander_counts`.

### `splits` — `splits.parquet`, one row per precon
precon_id, release_date, split (`train`|`dev`|`gap`|`test`), scored (bool: passes gates),
exclude_reason (str|null), cutoff (date used). Regenerate for another model cutoff with
`python bench/precon/build_dataset.py --splits-only --cutoff YYYY-MM-DD` (pure function of
release_date + gates, deterministic).

### `tag_denylist.json`
Evaluative / precon-identifying oracle tags that must not be rendered in eval mode (plan T3):
`exact` list + glob `patterns` (`cycle-*-face-commander`, `*precon*`, `*staple*`, and
`cycle-<set>-*` for every precon set code), `matched_tags_in_cards_parquet` (the expanded list,
186 tags) and the train-split audit flags (lift > 3, support >= 5). Full audit: `tag_audit.parquet`
(tag, support_cut, support_kept, lift, denied).

### `manifest.json`
snapshot_date, cards_parquet manifest date, source URLs, EDHREC buildId, fetch stats, counts,
cutoff parameter + split rule, gain thresholds, git sha, sha256 of each parquet.

## Defaults for open questions (plan §11; recorded, change via params)
- **Cutoff C (Q2):** GLM-5.3 cutoff unknown. Default `C = 2025-11-01` (so test = released
  >= 2026-01-01, matching the plan's candidate test pool). train: release < C-12mo; dev: C-12mo <= release < C;
  gap: C <= release < C+2mo (unused); test: release >= C+2mo.
- **Gates (§4):** scored only if potential_decks >= 300 and release <= snapshot - 8 weeks.
  Exclusions: MTGO theme decks (`td0`), non-Commander decks.
- **Bracket (Q4):** 3 (max 3 Game Changers) — scorer param, not data.
- **Lands (Q6):** land cuts/adds kept in `labels`, `headline=False`.
- **UB / Secret Lair (Q9):** included and flagged (`is_universes_beyond`, `is_secret_lair`).
- **Headline add metric (Q5):** raw gain; `gain_syn` provided as secondary.
- **Gain thresholds:** plan §6 values as stated. On this snapshot they give ~20 headline cuts and
  ~21 headline adds with gain >= 2 per scored precon (plan aimed for ~10), so the scorer may re-fix
  them on train from `rate`.
- **Collector's Edition decks** (16) are dropped as duplicates of the regular list. EDHREC
  `odds-and-ends` (SLD) has no MTGJSON list and is left out (inputs must not depend on EDHREC).

## On HF
`data/<config>.parquet` for the 5 configs (each loads as split `train` in `load_dataset`; the
benchmark split is the `split` column of `splits`), plus `manifest.json`, `tag_denylist.json`,
`tag_audit.parquet` at the repo root.

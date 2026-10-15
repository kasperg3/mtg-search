---
pretty_name: MTG Precon Cut/Add Benchmark (private)
license: other
license_name: mixed-mtgjson-mit-edhrec-permission-private
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
  - config_name: deck_fit
    data_files:
      - split: train
        path: data/deck_fit.parquet
---

# MTG Precon Cut/Add Benchmark (EXP-007)

> **Private dataset. Do not mirror, republish or make public.** The `labels` and `commanders`
> configs are derived from EDHREC data that was collected with permission for private
> evaluation only.

This is a benchmark of deck-building judgement on Magic: The Gathering Commander preconstructed
decks ("precons"). The agent sees a precon's 100 cards (as card text) and must pick the cards to
**cut** and the cards to **add**. Its picks are graded against what thousands of players actually
cut from and added to that precon, as aggregated by EDHREC.

Spec: `docs/benchmark_precon_plan.md` in the `mtg-search` repo (GitHub issue #3). Schema contract:
`bench/precon/schema.md`. Build code: `bench/precon/` (built from git `8c41e350c20f7eebeaa273319d30cbdcddf1421b`).

- **Snapshot:** EDHREC labels 2026-10-14 (Next.js buildId `PdCFALsQSMQepvm3iBv87`); MTGJSON
  `5.3.0+20261013`; Scryfall `cards.parquet` `2026-10-14T09:01:55.948+00:00`.
- **183 precons** (MTGJSON Commander decks; Collector's Edition duplicates dropped), of
  which **175** are matched to an EDHREC precon page.
- **16719 labels** (add 8370, add_land 2182, cut 3675, cut_land 2492).
- **Splits** (cutoff `2025-11-01`): all precons dev 15, test 20, train 148; scored dev 15, test 14, train 106.
- **Deck-fit grades** (config `deck_fit`, EXP-017): 2,385 grades by a blind Claude grader of whether proposed adds would upgrade the deck, on the 15 dev (tuning) and 14 test decks, with hidden controls. A second label source alongside the EDHREC rates, which favour popular cards.

## Citation

Cite the tagged release `v1.0`:

```bibtex
@misc{mtgsearch_preconbench_2026,
  author       = {Gr{\o}ntved, Kasper},
  title        = {{MTG Precon Cut/Add Benchmark}: Commander precon decklists with EDHREC cut/add rates and deck-fit grades},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/kasper-cbx/mtg-precon-bench}},
  note         = {Revision v1.0 (EDHREC snapshot 2026-10-14). Private dataset: EDHREC-derived labels collected with permission for private evaluation only}
}
```

**Revisions.** `v1.0` adds the `deck_fit` config and this card. The five data files `decks`, `labels`, `precons`, `commanders` and `splits` are byte-identical to revision `d6bb8ba`, on which every agent run (EXP-009 to EXP-017) was scored, and to `6f71dac`. Those two revisions differ only in `README.md`, `manifest.json` (build commit) and `tag_audit.parquet`.

Not affiliated with, endorsed by or sponsored by Wizards of the Coast, EDHREC or Scryfall.
Magic: The Gathering card names and text are © Wizards of the Coast (Fan Content Policy).

## Sources and terms

| Source | Used for | Terms |
|---|---|---|
| [MTGJSON](https://mtgjson.com) `DeckList.json`, `AllDeckFiles`, `SetList.json`, `AtomicCards.json` | official precon decklists, release dates, set families, first printings | MIT licence |
| [EDHREC](https://edhrec.com/precon) precon pages (`__NEXT_DATA__` JSON) | cut/add lists and rates, number of decks, commander counts | Collected **with permission from EDHREC** (owner's statement). Derived labels are kept **private** and are not redistributed; write-ups report aggregate scores only. |
| [Scryfall](https://scryfall.com/docs/api/bulk-data) bulk `oracle_cards` (our `cards.parquet`) | `oracle_id` mapping, colour identity, legality, land/basic flags | Scryfall bulk-data terms; WotC Fan Content Policy |

**EDHREC fetch policy (as run):** one pass over the `/precon` index plus each precon page (main
commander only, 178 HTTP requests incl. `robots.txt`); at most 1 request per 5 s,
single-threaded; User-Agent `mtg-search research benchmark (contact: kasper@colourbox.com; ...)`;
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

### `deck_fit` — deck-fit grades (EXP-017; never shown to the agent)

One row per graded (precon, card, pass):
- `precon_id`, `split` (`dev` = tuning decks, `test`), `oracle_id`, `name`;
- `kind`: `pool` (a card an agent proposed adding, from the runs in `proposed_by`), or a hidden control: `random` (a random legal in-identity card with no EDHREC add label), `top_label` (a top EDHREC add no agent picked) or `in_deck` (a card already in the deck);
- `grade` 0/1/2, `pass` (0; 1 = a seeded 10% of batches re-graded in a new order, for consistency only), `rubric` (`v3`), `grader_model`, `grading_run`.

**The grader.** Headless Claude Code (`claude-sonnet-5-5`), with no tools. It sees the format rules (colour identity, bracket 3: at most 3 Game Changers), the deck (the commander's rules text with its name replaced by "your commander", plus the other 99 card names), and 15 candidate cards per call with full rules text, in a seeded random order. It never sees which agent proposed a card, any EDHREC data, prices, or the product or deck name. Rubric `v3`:

> Grade each candidate for whether you would add it to this deck as an upgrade. An added card takes the place of one of the deck's weakest cards. Most legal cards would not improve a given Commander deck, so a card with no specific reason to be in this deck gets 0.
> 2 = a clear upgrade: it directly supports the commander or the deck's plan, or fills a role the list is short of (such as ramp, card draw, removal or interaction), and does so better than the deck's weakest cards.
> 1 = a reasonable but low-priority upgrade: it has a specific reason to be in this deck (it supports the plan or fills a gap) and is about as good as the deck's weakest cards.
> 0 = not an upgrade: no specific reason to be in this deck, filler, weaker than the cards it would replace, or works against the plan; or it breaks a deck-building rule.
> Judge only from the card text shown and the deck list. Do not reward a card for being widely played; judge what its text does for this deck.

**Checks (pass 0):**
- **Test decks:** 10/84 random controls graded ≥ 1, 0/42 in-deck cards, 87% of top EDHREC adds.
- **Tuning decks:** 14/90 random controls, 1/45 in-deck cards, 78% of top EDHREC adds.
- **Re-grading consistency on test:** κ(≥ 1) 0.80.

**Deck-fit score** (as used in EXP-017): the mean grade/2 over an episode's 20 ADD slots, where invalid picks (scorer rules, including the Game Changer limit in rank order) and missing slots count 0.

**Caveats:**
- **Tuning:** the rubric wording was settled on the dev decks over three versions before any test deck was graded.
- **Same model family:** the grader is the same model family as the agents it grades.
- **Recognition:** shown an anonymised deck, the grader named 12 of 15 dev decks.

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

Split key = MTGJSON release date relative to a model training cutoff *C* (default `2025-11-01`,
an assumption because GLM-5.3's cutoff is unknown): train < C−12 mo ≤ dev < C ≤ gap < C+2 mo ≤
test. A precon is `scored` only if it has an EDHREC page, `potential_decks ≥ 300`, was released
at least 8 weeks before the snapshot, is not an MTGO theme deck (`TD0`), and is not in the gap.
Recompute for another cutoff with
`python bench/precon/build_dataset.py --snapshot 2026-10-14 --splits-only --cutoff YYYY-MM-DD`.

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
- `deck_fit` grades are one model's judgment, with one pass, covering only the cards the graded runs proposed plus controls; cards outside that pool are unjudged, not 0.


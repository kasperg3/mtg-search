# `get_deck`: decklists for the agent (prototype)

**Status: prototype, not shipped, not yet tested with an agent.** Plan for that test:
[`eval/experiments/deck_tool_plan.md`](../eval/experiments/deck_tool_plan.md). Code: `src/mtg_search/decks.py`
(parsing, rules, rendering), `src/mtg_search/precons.py` (MTGJSON table), `CardService.get_deck` in `server.py`.

## The problem

When a prompt names a precon ("upgrade my Giada precon") or points at a decklist, the agent has no way to load
the list, so it works from memory. What exists today:

- **Shipped server:** per-card checks only. `search_cards`/`find_similar` filter by colour identity and format
  legality; `get_card` shows one card's legality in every format. Nothing reads a list or checks deck rules. A
  pasted list is read by the model itself.
- **`bench/precon/scorer.py`:** Commander only, and it checks picks, not a deck: the name resolves, the card is
  (not) in the deck, colour identity, Commander legality, singleton (basics and "any number" cards excepted) and
  the bracket-3 Game Changer limit.
- **`bench/precon/harness/`:** the deck is rendered into the prompt (`prompts.render_deck`); the Commander rules
  are prompt text. `check_cards` was tried (EXP-010) and deleted (EXP-014): agents called it instead of searching.

## The tool

```
get_deck(deck: str, format: str | None = None) -> str
```

- **`deck`:** "An official precon's name or its commander's name, or a whole decklist as text (one card per line)."
- **`format`:** "Format to check the deck against, e.g. 'commander', 'modern', 'brawl'. Leave empty to detect it."
- **Description:** `GET_DECK_DOC` in `server.py`, 1,046 characters, written like the other two trigger-first
  descriptions. It opens "Before working on a deck the user names or pastes, call this to load its exact list
  instead of recalling it from memory." and says "It does not suggest cards: call search_cards with the deck's
  colour identity and format for that."
- **Registration:** read-only, no network, **deferred** (no `alwaysLoad`), like `find_similar`. The three existing
  tool descriptions and the server instructions are unchanged. The surface key `omit` leaves the tool out, and
  `surfaces.SURFACES["three_tools"]` is the pre-`get_deck` surface for A/B runs.

**Why a fourth tool and not a parameter.** EXP-010 and EXP-014 found that extra tools and parameters cut
searching. Folding decks into `search_cards` or `get_card` would change the two always-loaded schemas that the
trigger sentences were tuned on. A separate deferred tool leaves those schemas alone: before it is loaded, Claude
Code shows only its name. The `check_cards` failure is the risk to measure. That tool returned a verdict that
could stand in for searching. `get_deck` returns the deck, which is the input to the search, and ends by naming
the `search_cards` call to make.

**Name resolution (pasted lists).**
1. Exact name or face name first (`CardService._name_index`).
2. Otherwise `CardService.resolve_name`: prefix, whole word, then fuzzy ≥ 70.
3. Every non-exact read is listed: `read (1): "Expresive Iteration" as Expressive Iteration`.
4. A line with a quantity that does not resolve is listed under `not found` with the closest names; it counts
   towards the deck size and is left out of every other check.
5. A line without a quantity must match a name exactly, or it goes under `ignored lines`.

## Sources (checked 2026-10-16)

| source | in the tool | why |
|---|---|---|
| Pasted text | yes | Arena, MTGO `.txt` and `.dek` XML, Moxfield and Archidekt text exports, Forge `.dck`, CSV with name and count columns, `1x Name (SET) 123`, `SB:` prefixes, `*F*`/`*CMDR*` markers, Archidekt `[Category]` tags (Commander, Sideboard, Maybeboard), section headers and type-group headers |
| MTGJSON precons | yes, bundled | MIT licence. `AllDeckFiles.tar.gz` becomes `precons.parquet`: 1,871 decks, 0.9 MB, built at index time. The server never calls MTGJSON. MTGJSON's `robots.txt` has `Disallow: /api/v5/*.json`. The archive and its `.sha256` are outside that rule, and the MTGJSON FAQ recommends the compressed files. A per-deck JSON fetch at tool time would fall under the rule. |
| MTGJSON deck URL | yes | `mtgjson.com/api/v5/decks/<file>.json` maps to the bundled table; nothing is fetched |
| Moxfield URL | no | `api.moxfield.com/robots.txt` and `api2.moxfield.com/robots.txt` say `Disallow: /`, and the deck pages load from that API. The terms page is JavaScript-only and could not be read without rendering it. |
| Archidekt URL | no | Terms (effective 2018-09-07) grant use "solely for your own personal, noncommercial use" and forbid using "automated agents or scripts … to generate automated searches, requests, or queries to the Site". Its `robots.txt` does not block deck pages; the terms do. |
| EDHREC | no | not a deck source for this tool |
| Local files | no | the Docker image cannot see them; the agent reads the file and passes the text |

For any URL other than MTGJSON's, the tool answers: "This server does not open web pages. Export the deck from
<host> as text … and pass that text as deck".

## Format rules

| formats | size | copies | sideboard | command zone |
|---|---|---|---|---|
| commander, duel, predh, paupercommander | exactly 100 | 1 | not part of the deck | 1 legendary creature (or "can be your commander"), or a valid pair |
| brawl / standardbrawl | exactly 100 / 60 | 1 | — | as Commander, or a legendary planeswalker |
| gladiator | at least 100 | 1 | — | — |
| standard, pioneer, modern, legacy, vintage, pauper, historic, timeless, alchemy, premodern, oldschool, penny, future | at least 60 | 4, restricted 1 | at most 15, companion included | — |
| any other Scryfall format (oathbreaker, competitivebrawl, …) | card legality only | | | |

**What Scryfall gives.** One status per card in 23 formats: `legal`, `banned`, `restricted` or `not_legal`. That
covers ban lists, the Vintage restricted list, rotation, Pauper rarity and the Brawl card pool. Pauper Commander's
status covers the 99 only, so the commander is not checked against it.

**What the tool adds.**
- Deck size and copy limits: basic lands and "any number" cards are exempt; "up to seven/nine" and "only one"
  limits come from the card's own text.
- Restricted cards limited to one copy, and the sideboard limited to 15 cards.
- Colour identity of the commander or pair.
- Commander eligibility, and partner pairs: Partner, Partner with (named), Friends forever, Partner—group,
  Background, and Doctor with Doctor's companion.
- The Game Changer count, with the bracket thresholds.

**Not checked:** companion deck conditions, bracket rules other than Game Changers, Duel Commander's
commander-only bans, the Pauper Commander commander's rarity, Oathbreaker construction, Limited.

**Detection** (when `format` is empty), first match wins:
1. The precon's MTGJSON type: Commander Deck → commander, Historic Brawl Precon Deck → brawl, Pioneer Challenger
   Deck → pioneer.
2. A commander section: Standard Brawl if the deck has 55–65 cards that are all legal in standardbrawl, otherwise
   Commander.
3. One or two commander-eligible cards in the sideboard, with 100 cards in all (the MTGO export): Commander.
4. About 100 singleton cards: Commander, with the problem "no commander marked".
5. A main deck of at least 55 cards: 60-card constructed. The tool checks copies and sizes and adds a line such
   as "every card is legal in vintage · not in standard (18 cards), …".
6. Otherwise no checks.

## Output

The first lines for an official precon (100-card output: 7.5k characters, about 2,000–2,500 tokens by character
count):

```
deck: Calling All Angels · Commander Deck · FDC · 2026-10-02 · list from MTGJSON
format: Commander (from the deck type) · 100 cards · colour identity W
problems: none found
   Game Changers (0) · brackets 1-2 allow none, bracket 3 up to three, brackets 4-5 any number
   not checked: companion deck conditions; bracket rules other than Game Changers
commander:
Giada, Font of Hope — {1}{W} (MV 2) · White · Legendary Creature — Angel · 2/2 · rare [Legendary] [Commander-eligible]
   …full text, roles, legal: commander ✓
main (99): 33 creatures · 5 instants · 7 sorceries · 11 artifacts · 5 enchantments · 38 lands
1 Segovian Angel {W} · Creature — Angel
1 Speaker of the Heavens {W} · Creature — Human Cleric · roles: lifegain-matters, repeatable-lifegain, repeatable-creature-tokens
…
32 Plains · Basic Land — Plains
To find cards for this deck, call search_cards with color_identity="W", format="commander"; every card above is already in it.
```

A pasted 60-card list with `format="modern"`:

```
deck: pasted list
format: Modern (given) · main 60 · sideboard 15
read (1): "Expresive Iteration" as Expressive Iteration
problems (2):
   - Lightning Bolt: 5 copies (main 4, sideboard 1); at most 4
   - banned in Modern (2): Ponder, Fury
main (60): 16 creatures · 16 instants · 6 sorceries · 2 artifacts · 20 lands
4 Dragon's Rage Channeler {R} · Creature — Human Shaman · roles: repeatable-surveil, gains-flying, card-types-in-graveyard-matter
…
sideboard (15):
…
To find cards for this deck, call search_cards with format="modern".
```

## Shipping it (not done)

1. `scripts/refresh.py`:
   - build `precons.parquet` with `python -m mtg_search.precons`;
   - add the file to `DATA_FILES`;
   - decide whether a precon-only change triggers a metadata-only publish (add a hash to the manifest
     fingerprint).
2. `docker/Dockerfile` and `.dockerignore`: copy `data/precons.parquet` (both copy explicit file lists).
3. `docs/datasets/mtg-search-index.md`: MTGJSON as a source, with its MIT notice ("Copyright © 2018 – Present,
   Zach Halpern").
4. README: a row in the tools table.
5. Benchmark runs: a precon lookup returns the deck's name, set and date, which the benchmark hides (leakage
   control T2b). `bench/precon/harness/tools.py` does not pass `precons_path`, so lookups stay off in benchmark
   runs unless an arm turns them on. The `shipped` surface now lists `get_deck`; `three_tools` reproduces the
   EXP-014 surface.

## Limitations

- Printing details (set code, collector number, foil) are dropped; everything is checked per card name.
- Precons are found by deck name or commander name, not by product type ("the Pioneer Challenger deck").
  MTGJSON lists each Duel Deck half as its own deck ("Jace", "Chandra").
- The commander section must be marked unless the list follows one of the layouts detection knows (commander
  first or last, after a blank line).
- The token figure is estimated from character counts, not measured with a tokenizer.

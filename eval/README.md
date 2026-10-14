# Retrieval evaluation sets

There are four query sets for comparing BM25, dense, sparse and late-interaction retrieval over `data/cards.parquet`. The first two reward finding precise wording. The last two reward generalisation: vague descriptions of what a card should do, or of the hole it should fill in a deck. Load them with:

```python
from eval.queries import load_queries, gains
qs = load_queries("tags")         # 65 functional queries (Tagger labels)
qs = load_queries("handwritten")  # 46 wording-sensitive queries (regex labels)
qs = load_queries("paraphrase")   # 222 = 2 low-overlap reformulations of each of the above
qs = load_queries("deckneeds")    # 30 open-ended deck-building needs, some with filters
gains(qs[0])                      # {oracle_id: 1|2} for nDCG
qs[0]["filters"]                  # {} or kwargs for mtg_search.filters.filter_expr
```

`uv run python eval/queries.py` checks every set. It confirms that each relevant oracle_id exists in `data/cards.parquet`, that no id appears twice, that no funny card is labelled, and that the labels satisfy the query's `filters`. It also prints how many relevant cards each query has.

### The `filters` field

`filters` is a dict of keyword arguments for `mtg_search.filters.filter_expr`. The keys are:
- `color_identity`: subset semantics, like commander deck building; `"C"` means colorless only.
- `colors`
- `format`: the card must be legal or restricted in that format.
- `types`: every listed type must match.
- `cmc_min`, `cmc_max`

The eval runner should restrict retrieval to the cards that pass the filter, the same way the MCP server does. The labels are already intersected with the filter. Only `deckneeds` uses it at present; queries in every other set load with `filters == {}`.

Both sets only label **non-funny** cards (`is_funny == False`). A funny card returned by a retriever counts as non-relevant. Filter them out, or accept a small penalty.

## `queries_tags.yaml`: from Scryfall oracle tags

- **Builder:** `uv run python eval/build_tag_queries.py`. It regenerates the YAML from the reviewed dict `TAG_QUERIES`.
- **Tag choice:** 65 *functional* Tagger tags covering gameplay roles: sacrifice outlets, edicts, soft and conditional counters, bounce, tuck, fight and bite, land destruction, one-sided wipes, graveyard hate, reanimation, wheels, loot and rummage, hand disruption, tutors, mana rocks and dorks, rituals, cost reducers, landfall, counters, clones, spell copy, flicker, lifegain payoffs, fog, protection, theft, goad, tax, extra turns and combats, and alt-wins. Art, meta, flavour and name-pun tags were not used.
- **Size filter:** each tag needs 20–500 relevant non-funny cards. Very broad tags such as `counterspell` (539), `sweeper` (933) and `removal-creature` fall outside this range, so a narrower child tag is used instead.
- **Query wording:** each query is phrased the way a deck builder would type it. Where possible it avoids the slug's own words. For example, `free-sacrifice-outlet` becomes "free way to sacrifice my own creatures whenever I want", and `rhystic` becomes "punisher effect that does something unless an opponent pays".
- **Relevance:** a card is relevant if it carries the tag **or any descendant tag**. The closure is taken over `child_slugs`, which matches Scryfall's `otag:` search. `data/tags.parquet` stores only direct taggings, so the builder computes the closure itself. The decision is intentional: the children are specialisations of the parent function. For example, `mana-rock` includes `moxen` and `utility-mana-rock`, and `flicker` includes `flicker-creature` and `flicker-slow`.
- **Grades and IDs:** relevance is binary (no `graded`). IDs are `tag-<slug>`, and `source: otag:<slug>`.

## `queries_handwritten.yaml`: wording-sensitive queries

- **Builder:** `uv run python eval/build_handwritten.py [--scryfall] [--show N]`.
- **Labelling rule:** each spec in `SPECS` has a case-insensitive regex that is run over `doc_text`, with reminder text `(...)` stripped. Stripping reminder text matches Scryfall `o:` semantics: without it, menace, extort and unearth reminder text would cause false hits. The exact rule is stored in each entry's `labeling_rule`.
- **Cross-check:** `--scryfall` runs the spec's Scryfall query (`scryfall_check`) through the API. It sends `User-Agent: mtg-search/0.1`, keeps to 2 requests per second or fewer, follows `next_page`, and caches results in `eval/.scryfall_cache/`. Scryfall cards are mapped to our ids through `oracle_id`. It then prints the cards found only by the regex and only by Scryfall. Each regex was revised until those differences were either true matches or deliberate exclusions.
  - The deliberate exclusions include: Thoughtseize-style queries require "discards", so they drop exile variants such as Tidehollow Sculler and Brain Maggot. Ward-like "becomes the target … counter that spell unless" text is dropped from the soft-counter query. The "exile instead" query does not include "your graveyard" self-replacement effects such as Yawgmoth's Will.
- **Per-card fixes:** the specs support `exclude` and `include` lists of card names for corrections made by hand. No current query needs them.
- **Grades:** `graded: {oracle_id: 2}` is set where a stricter regex picks out the exact wording. Examples: `counter target spell unless its controller pays` as opposed to "…target creature spell…", `double the number of +1/+1 counters` as opposed to "each kind of counter", and `each player sacrifices a creature` as opposed to "each opponent…". Every other relevant card has grade 1.
- **Contrast pairs:** some queries are minimal pairs, so a retriever has to respect small wording differences. Examples:
  - `each opponent loses` / `target player loses`
  - `flicker-own` (returns immediately) / `flicker-end-step` / `dies-return-end-step`
  - `cant-be-countered` (this spell) / `your-spells-uncounterable`
  - `one-blocker-max` ("can't be blocked by more than one creature") / `three-or-more-blockers`

## `queries_paraphrase.yaml`: reformulations of every original query

- **Builder:** `uv run python eval/build_paraphrase.py`, from the hand-written `REFORMULATIONS` dict.
- **Two variants per original query:**
  - `<id>-para` is a casual player paraphrase. For example, `hw-soft-counter` becomes "counter they can pay around", and `tag-removal-sacrifice` becomes "forces them to give up one of their own guys".
  - `<id>-need` is a deck-need or role phrasing. For example: "my control deck needs cheap ways to stop spells early, but I'm fine if they can pay through it late", or "my multiplayer commander deck wants life loss that scales with the number of players".
- **Labels:** `relevant` and `graded` are copied unchanged, and `origin: <id>` links back to the original query.
- **Wording rules:** the reformulations use no rules-text phrasing and no card names. Plain card-type words such as "spell", "land" and "graveyard" are allowed on their own, but not in distinctive pairs.
- **Retrieval-generalisation gap:** comparing nDCG on a paraphrase against nDCG on its `origin` measures how much a retriever loses when the query stops sharing wording with the cards.

### Lexical overlap (`eval/lexical_overlap.py`)

The script tokenises each query and each relevant card's rules text. Rules text here excludes the name and cost line, reminder text and the `Keywords:` line. It then removes English function words and applies light stemming. Rules words such as "target", "each" and "whenever" are kept as content words. It reports three measures:
- `card_hit`: the share of relevant cards whose rules text contains at least one of the query's content-word bigrams.
- `bigram`: the mean share of the query's bigrams found in each relevant card.
- `kw>0`: the share of queries that use a keyword or ability word, taken from the `keywords` column (for example ward, landfall, trample, mill), which occurs in at least 5% of the relevant cards.

A reformulation counts as flagged if `card_hit > 0.10` or `kw` is not empty. Every flagged reformulation was rewritten; six needed it in the first pass. The current figures (`uv run python eval/lexical_overlap.py --list`) are:

| set | n | card_hit | bigram | kw>0 | flagged |
|---|---|---|---|---|---|
| original (tags + handwritten) | 111 | 0.654 | 0.400 | 0.09 | 89 |
| para | 111 | 0.002 | 0.000 | 0.00 | 0 |
| need | 111 | 0.001 | 0.000 | 0.00 | 0 |
| deckneeds | 30 | 0.058 | 0.016 | 0.03 | 4 |

## `queries_deckneeds.yaml`: open-ended deck needs

- **Builder:** `uv run python eval/build_deckneeds.py [--show N]`.
- **Queries:** 30 new queries with no original behind them. Examples: "I keep losing to fliers in my mono-green deck" (filter `color_identity: G`), "cheap ways to refill my hand in an aggressive red deck" (`color_identity: R`, `cmc_max: 3`), "something to punish opponents who play lots of spells", "protect my commander from removal" (`format: commander`, `types: [Instant]`), and "how do I kill a big creature I can't target in modern".
- **Labels:** the union of oracle-tag closures and regexes, both as in the sets above. The union is intersected with `filters`, and funny cards are dropped. The rule used is stored in `labeling_rule`.
- **Hand review:** the 10 most popular cards for each query were checked by hand (`--show`), and false positives were fixed. For example, Colossus Hammer was dropped from the anti-flier query, and one-sided sweepers were dropped from the hexproof-answers query because the tag also contains artifact wipes.
- **Grades:** grade 2 marks the core answer where there is an obvious one. Examples: direct anti-flier removal, Blood Artist effects, Entomb-style tutors, and "each opponent loses" drains.
- **Size:** relevant sets hold 56–449 cards, because deck needs are broad. Use nDCG@10 and Recall@100.

## Caveats

- **Tag incompleteness.** Tagger is community-maintained, and newer or obscure cards are often untagged. Absolute recall on the tag set is therefore an *underestimate*: a retriever can be penalised for correct results. Comparisons *between* retrievers are still fair, because every system faces the same missing labels. Tags can also be loose (for example, `removal-nonland` includes cards like Red Elemental Blast), and some relevant sets are large (up to about 490 cards). Prefer nDCG@10 and Recall@50/100 over full recall.
- **Regex labels are lexical.** The hand-written labels come from string patterns, so they favour exact wording. A card with the same function in older or different templating can be missed. For example, cards that name themselves ("Exile Divergent Equation") get past `this creature` patterns. This biases the hand-written set slightly in favour of BM25. Read it as "can the model find the precise wording", not as "functional similarity".
- **Version drift.** Labels are tied to the current `data/cards.parquet` and `tags.parquet` snapshot. Re-run both builders after a data refresh. Query IDs are stable, because they are slugs or hand-chosen.
- **Deck-need labels are a union of proxies.** They are incomplete for vague needs: a card can fill the role without carrying any of the chosen tags or matching the regexes. Absolute scores on this set are lower bounds, so read them comparatively.
- **The paraphrase sets are not independent of the originals.** They use the same labels, so they measure robustness to rewording, not fresh coverage.
- **Graded relevance is partial.** Only some hand-written queries have grade-2 subsets. Tag queries are binary.

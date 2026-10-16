# mtg-search

**Semantic search over every Magic: The Gathering card, as an MCP server for deck-building agents.**

An LLM building a Commander deck from memory gets the rules wrong (off-colour cards, cards that are not
legal, cards already in the list) and keeps reaching for the same popular staples. mtg-search gives the
agent the official rules text of all 34,646 cards, a search engine over it, and the filters to check
its picks.

Write-up: [Does a search engine help a deck-building agent?](https://www.grontved.xyz/blog/agent-search-vs-grep/)

## Install

One line for Claude Code, with [uv](https://docs.astral.sh/uv/getting-started/installation/) installed. It
downloads the card index once (about 1.9 GB) and registers the server:

```bash
uvx mtg-search-mcp --download && claude mcp add --scope user mtg-search -- uvx mtg-search-mcp
```

Any other MCP client runs `uvx mtg-search-mcp` over stdio:

```json
{
  "mcpServers": {
    "mtg-search": { "command": "uvx", "args": ["mtg-search-mcp"] }
  }
}
```

The server uses about 2.7 GB of RAM and keeps the index in `~/.cache/mtg-search` (set `MTG_SEARCH_HOME` to
move it). To update, run `uvx mtg-search-mcp@latest --download`; a new card snapshot that has not been
downloaded yet is fetched in the background while the previous one keeps serving.

**With Docker instead** (no Python needed; the image already contains the index):

```bash
docker pull ghcr.io/kasperg3/mtg-search && claude mcp add --scope user mtg-search -- docker run -i --rm ghcr.io/kasperg3/mtg-search
```

Then ask for cards the way a player would: *"upgrade my Auntie Ool precon"*, *"cheap ways to protect my
commander in Selesnya"*, *"what else works like Smothering Tithe?"*, or point it at your own list:
*"swap ten cards in https://moxfield.com/decks/…"*

## Tools

| tool | what it does |
|---|---|
| `search_cards` | 1–4 rules-text queries, fused, with filters: `color_identity`, `format`, `types`, `mana_value_min` / `mana_value_max`. Only cards that pass the filters compete, so a filtered search never comes back empty because the best matches were cut. |
| `get_card` | One card by (fuzzy) name: full text, legality in every format, roles, official rulings and the Scryfall link. |
| `find_similar` | Cards whose text is closest to a given card's, with the same filters. |
| `get_deck` | Loads a deck to work on: a Moxfield or Archidekt link, an official precon by name or commander (lists from [MTGJSON](https://mtgjson.com)), or a pasted decklist. Returns every card with its roles and checks the deck against its format's rules (size, copies, commander and colour identity, bans, Game Changers). |

What the agent gets back, for `search_cards(queries=["Deals 3 damage to any target."], color_identity="R",
types=["instant"], mana_value_max=1, format="commander", limit=3)`:

```
search: "Deals 3 damage to any target." · filters: color_identity=R, format=commander, types=['instant'], mana_value_max=1 · 3 shown of 152 matching cards
1. Lightning Bolt — {R} (MV 1) · Red · Instant · uncommon
   Lightning Bolt deals 3 damage to any target.
   roles: burn-any
   legal: commander ✓
2. Collateral Damage — {R} (MV 1) · Red · Instant · common
   As an additional cost to cast this spell, sacrifice a creature.
   Collateral Damage deals 3 damage to any target.
   roles: bombard, sacrifice-outlet-creature, burn-any, lightning-bolt-redux
   legal: commander ✓
3. Shard Volley — {R} (MV 1) · Red · Instant · common
   ...
```

Results are ordered by text relevance only. Prices, EDHREC rank and other popularity signals are not in
the image at all, so the agent picks cards for what they do, not for how often they are played.

## How well it works

All runs use Claude Sonnet 5.5 in Claude Code, and cards were graded blind by a separate Claude
model. Details, intervals and every caveat: [write-up](https://www.grontved.xyz/blog/agent-search-vs-grep/)
and [`eval/experiments/`](https://github.com/kasperg3/mtg-search/blob/main/eval/experiments/REGISTRY.md).

**Search vs grep, same agent, same minimal instructions** (113 tasks; relevance over the 80 deck-need
and card-attribute tasks):

| | relevant cards in the top 10 | card-attribute tasks | tokens per task |
|---|---|---|---|
| **mtg-search** | **98.3%** | **100%** | **10.6k** |
| grep over the card text | 86.5% | 48.0% | 14.3k |
| grep, with a prompt tuned to the card file | 96.8% | 92.7% | 20.9k |
| reading every card (12-task sample) | | | 2.91M |

**Real decks.** Asked to add 20 cards to each of 14 Commander precons released after 1 November
2025, the agent's adds fit the deck better with mtg-search than from memory: deck-fit score
0.45 → 0.65 (0 to 1), better on 13 of 14 decks, p = 0.002.

**The engine.** On 46 rules-text queries, nDCG@10 is 70.7 for [`lightonai/mLateOn`](https://huggingface.co/lightonai/mLateOn)
(late interaction), 61.1 for BM25 and 36.7 for a dense embedding ([EXP-001](https://github.com/kasperg3/mtg-search/blob/main/eval/experiments/REGISTRY.md)).

## How it works

- **Data:** Scryfall's Oracle cards (one entry per card, current official wording), Tagger function
  tags, the official rulings and MTGJSON's official precon decklists, refreshed weekly by [`scripts/refresh.py`](https://github.com/kasperg3/mtg-search/blob/main/scripts/README.md).
- **Search:** exact late interaction with mLateOn. Queries go through the model's fp32 ONNX encoder;
  every card is scored by exact MaxSim against its precomputed token vectors, about 100–150 ms per query
  on a 6-core desktop CPU. Nothing is approximated, so the shipped server ranks exactly like the
  benchmarks' reference ([`docker/README.md`](https://github.com/kasperg3/mtg-search/blob/main/docker/README.md)).
- **Server:** Python, [MCP](https://modelcontextprotocol.io) over stdio, one Docker image.

## Datasets

All on Hugging Face; cite a tagged revision ([BibTeX](https://github.com/kasperg3/mtg-search/blob/main/docs/datasets/README.md)).

| dataset | contents |
|---|---|
| [`kasper-cbx/mtg-search-index`](https://huggingface.co/datasets/kasper-cbx/mtg-search-index) | the card tables, tags, rulings, precon decklists and mLateOn embeddings the server searches |
| [`kasper-cbx/mtg-search-bench`](https://huggingface.co/datasets/kasper-cbx/mtg-search-bench) | 363 search queries, 113 agent tasks, every relevance grade and the final labels |
| [`kasper-cbx/mtg-precon-bench`](https://huggingface.co/datasets/kasper-cbx/mtg-precon-bench) | Commander precon decklists, EDHREC add/cut rates and deck-fit grades |
| [`kasper-cbx/mtg-search-experiments`](https://huggingface.co/datasets/kasper-cbx/mtg-search-experiments) | raw run outputs: episodes and transcripts |

## Repository

```
src/mtg_search/   the MCP server, ingest, filters and the mLateOn search
docker/           the image (Dockerfile, build script, smoke test)
eval/             search benchmark, agent tasks and harness, experiment write-ups (eval/experiments/)
bench/precon/     the Commander precon benchmark: data build, agent harness, scorer, deck-fit judge
scripts/          weekly index refresh, dataset publishing, result sync
jobs/             the GPU encoding job for the card embeddings
docs/             dataset cards, findings page, plan and design notes
tests/            unit tests and the Docker integration test
```

## Development

```bash
uv sync
uv run pytest                    # unit tests
docker/build.sh                  # build the image from the HF dataset
uv run pytest -m docker          # run the built image over stdio
python docker/mcp_smoke.py       # sample calls with output and latency
```

Run the server from a checkout with `uv run mtg-search-mcp`; on first run it downloads the index into
`~/.cache/mtg-search`, like the published package.

## Releases

Every version bump on `main` is published automatically
([`publish.yml`](https://github.com/kasperg3/mtg-search/blob/main/.github/workflows/publish.yml)): tests,
then [PyPI](https://pypi.org/project/mtg-search-mcp/), a `vX.Y.Z` GitHub release and the matching Docker
image. Each release pins one card snapshot; when the index refresh tags a new one,
[`scripts/bump_index.py`](https://github.com/kasperg3/mtg-search/blob/main/scripts/bump_index.py) points the
package at it and bumps the patch version, which releases it.

## Data and terms

Card names and rules text are © Wizards of the Coast, used under the
[Wizards of the Coast Fan Content Policy](https://company.wizards.com/en/legal/fancontentpolicy). This
is unofficial Fan Content, not approved or endorsed by Wizards. Card data comes from
[Scryfall](https://scryfall.com), function tags from [Scryfall Tagger](https://tagger.scryfall.com),
precon decklists from [MTGJSON](https://mtgjson.com) and precon add/cut rates from
[EDHREC](https://edhrec.com). This project is not affiliated with any of them.

## License

The code is MIT licensed ([LICENSE](https://github.com/kasperg3/mtg-search/blob/main/LICENSE)). The card data and the benchmarks keep their sources'
terms, described above.

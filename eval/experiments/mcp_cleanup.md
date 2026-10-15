# EXP-014: A smaller MCP server (three tools, five filters) and what it does to tool use

**Status: done 2026-10-15 (dev split only).** This is a smoke check of a refactor, not a confirmation run. All runs used the 15 dev precons, ADD and SWAP, 1 repeat, Claude Code `claude-sonnet-5-5` with deferred tool loading, on the subscription.

## What changed (owner's request: "small and concise", no jargon)

- **Code.** mLateOn is the only retriever, with one implementation (`mtg_search.mlateon`) for the server, the evals and the benchmark. NextPlaid, BM25, dense, sparse, SparseUp, the PyTorch ranker, `check_cards`, `list_tags`, the unused filters and the spike code are deleted (commit `6601fd7`). The mLateOn eval reproduces 60.5 / 70.7 / 11.9 / 21.1 nDCG@10 exactly.
- **What the agent sees.**
  - **Tools:** three read-only tools (`search_cards`, `get_card`, `find_similar`) with tool annotations.
  - **Schemas:** `search_cards` has a 7-parameter schema (1.5k characters), down from 17 parameters (3.8k): `queries` (also accepts a string), `color_identity`, `format`, `types`, `mana_value_min`/`mana_value_max`, `limit`.
  - **Terminology:** "rules text" and "roles" instead of "Oracle".
  - **Why those parameters:** in 343 recorded calls, agents used only `query`/`queries`, `color_identity`, `format`, `types` and `limit`.

## Results

| surface | ADD searched | SWAP searched | ADD valid | SWAP valid | ADD nDCG@10 | SWAP add nDCG@10 | tokens/ep (ADD) |
|---|---|---|---|---|---|---|---|
| old (4 tools, 17 params), today | 6/15 (40%) | 13/15 (87%) | 0.977 | 0.993 | 0.255 | 0.155 | 52.5k |
| old, EXP-011 dev (yesterday) | 3/15 (20%) | 14/15 (93%) | 0.973 | 0.987 | 0.314 | 0.158 | 50.3k |
| new, first text | 2/15 (13%) | 8/15 (53%) | 0.950 | 0.987 | 0.248 | 0.194 | 46.0k |
| new, old instruction wording restored | 1/15 (7%) | 8/15 (53%) | 0.933 | 0.973 | 0.231 | 0.247 | 45.1k |
| **new + trigger sentence opening both descriptions (shipped)** | **11/15 (73%)** | **14/15 (93%)** | 0.977 | **1.000** | 0.189 | 0.176 | 53.1k |
| *ref: EXP-011 dev e6desc_slim (slim + same trigger)* | 9/15 (60%) | 14/15 (93%) | 0.990 | 0.993 | 0.279 | 0.177 | 54.0k |

1. **The slim schema lowered tool use; the wording was not the cause.** The first text searched in 10 of 30 episodes against 19 of 30 for the old surface (Fisher p = 0.038). Restoring the old instruction and description wording did not help (9 of 30). EXP-010 had already shown the same effect: slimming the schemas and dropping `list_tags` cut ADD searching from 57% to 29% (e4slim).
2. **A trigger sentence at the start of the `search_cards` and `get_card` descriptions restores it.** The sentences are "Before recommending cards for a deck, call this to find candidates instead of listing them from memory." and its `get_card` twin, as tested in EXP-010/011 (e1desc, e6desc_slim). With them the server searched in 25 of 30 episodes (vs 19 of 30 old, p = 0.14), with the same validity and similar tokens per episode.
3. **The trade-off is the one EXP-011 saw.** The shipped surface vs the old one, paired over the 15 precons:
   - ADD nDCG@10 against the community key: −0.066 [−0.108, −0.020], sign p = 0.27;
   - popularity drift: +0.50 [+0.07, +0.93], so less popular picks;
   - ADD validity: +0.000;
   - SWAP add nDCG@10: +0.021 (n.s.).

   Searching agents pick fewer staples, and the community key rewards staples. Under the EXP-011 dev selection rule (highest ADD nDCG@10), the old surface would win. The deck-aware judge (`precon_judging_plan.md`, EXP-017) would show whether the less popular picks are worse for the deck or only less popular.

## Caveats

- **This is the dev split, iterated three times.** That is selection, not confirmation.
- **Noise is large.** The same old surface searched 20% vs 40% of ADD episodes on two days, and n = 15 per task.
- **The benchmark proxy does not pass tool annotations or titles to Claude Code.** Only names, descriptions, schemas and `_meta` reach the agent, so the annotations play no part in these numbers.

Run records: `bench/precon/runs/20261015_exp014-dev_{before,after,after2,after3}.json`.

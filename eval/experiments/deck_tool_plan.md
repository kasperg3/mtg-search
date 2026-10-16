# EXP-020 plan: does `get_deck` let the agent work on a deck it is only told about?

**Status: pre-registered 2026-10-16, before any model run; run the same day, results in [`deck_tool.md`](deck_tool.md).** Design: [`docs/deck_tool.md`](../../docs/deck_tool.md).

`get_deck` gathers data: it loads the list of a deck the user names, links or pastes, so that the work the user
asked for (cutting, adding, swapping cards) is done on the real deck. It is not meant to replace searching. The
benchmark tests exactly that: the same three tasks, with the deck either pasted into the prompt (as in every run
so far) or only named by the user.

## Questions

1. **Help.** When the user names the precon and its commander but gives no list, are the agent's cuts, adds and
   swaps valid for the real deck once `get_deck` is on the surface? Without it the agent can only recall the list.
2. **Recovery.** With `get_deck`, is a named deck worked on about as well as a pasted one?
3. **Guard.** With the deck pasted, does the extra (deferred) tool change searching or the results? EXP-010's
   `check_cards` replaced searching and EXP-014 saw surface changes move the search rate, so this is checked even
   though `get_deck` returns data, not a verdict.

## Design

2 × 2, every cell on the same 21 precons × 3 tasks (CUT, ADD, SWAP) × 1 repeat = 63 episodes per cell, 252 in all.

| | surface `three_tools` (A): EXP-014's shipped server, no `get_deck` | surface `shipped` (B): `get_deck` added, deferred |
|---|---|---|
| **pasted**: deck card text in the prompt, as in EXP-007 to EXP-017 | PA (today's benchmark) | PB (guard) |
| **named**: "My deck is the official precon "<deck>" (commander: <name>), unchanged from the box." and no list | NA (memory only) | NB (help) |

- **Agent:** headless Claude Code, `claude-sonnet-5-5`, subscription, deferred tool loading (`--cc-tool-search`),
  3 workers. Prompts and budgets as in EXP-014 except the named-deck wording (`prompts.SYSTEM_NAMED`,
  `prompts.user_prompt_named`); the rules block (colour identity, bracket 3) is the same in both inputs.
- **Precons:** the 15 dev precons and the 6 unscored precons released on 2026-10-02 (FDC ×5, FRC ×1; excluded from
  scoring only for having < 8 weeks of EDHREC data). The 6 are after the model's training data, so on them NA
  cannot recall the list. The 14 scored test precons are not used.
- **`get_deck` in the benchmark:** never opens links (the harness gives it no fetcher). Official precon lookup
  (`data/precons.parquet`, built from MTGJSON's AllDeckFiles 5.3.0+20261013) is on in the named cells only: in the
  pasted cells a lookup by commander name would reveal the product, which those prompts hide (T2b). In named
  cells the deck's own name is allowed in the prompt; every other precon's identifiers stay forbidden.
- **Scoring:** unchanged (`scorer.score_episode`). Validity is checked against the bench deck (EDHREC's list,
  identical to MTGJSON's for all 21; World Shaper and Counter Intelligence are filed under a different face
  commander by EDHREC, and `get_deck` finds them by either).

## Metrics

| | per task (CUT, ADD, SWAP) |
|---|---|
| primary | validity: `cut.validity`, `add.validity`, `swap.cut.validity`, `swap.add.validity` |
| invalid reasons | cuts not in the deck (`not_in_deck`); adds already in it (`in_deck`), off identity, not legal |
| quality | nDCG@10 against the community key (`cut.ndcg@10`, `add.ndcg@10`, `swap.cut/add.ndcg@10`), dev precons only (the 6 new ones have thin labels) |
| tool use | share of episodes calling `get_deck`, `search_cards`, `get_card`; `get_deck` without `search_cards` |
| cost | tokens, turns, wall time per episode |

Comparisons are paired over precons (paired bootstrap 95% CI, sign test), per task.

## Decision rules (fixed before running)

1. **Help is established** if NB − NA validity is > 0 with a 95% CI above 0 on CUT and on ADD (pooled over the 21
   precons). Reported separately for the 6 new precons (smoke test, n = 6).
2. **Recovery** if NB validity is at least PA − 0.03 on every task, and NB calls `get_deck` in ≥ 80% of episodes.
   If NB calls it in < 50% of episodes, run cell NC: named, surface `deck_always` (`get_deck` loaded up front),
   before deciding.
3. **Guard passes** if, pooled over the 63 pasted episodes, PB searches (calls `search_cards`) in at least as many
   episodes as PA minus 6 (≈ 10 points), at most 3 PB episodes call `get_deck` without `search_cards`, and no task's
   PB − PA validity CI lies wholly below −0.03.
4. **Ship `get_deck` deferred** if 1 and 3 hold. If 3 fails, do not ship it in this form.

Community-key nDCG is reported, not decided on: EXP-017 showed it understates picks that fit the deck but are less
popular, and in the named cells the deck's name can cue recalled EDHREC lists.

## Caveats

- The 15 dev precons were released before the model's training cutoff, so NA can partly recall them; the 6 new
  ones are the clean slice, and n = 6 is small.
- Named prompts name the product by design (memory cue for both NA and NB).
- A deferred `get_deck` needs a `ToolSearch` turn before its first call; turns and tokens include it.
- One repeat per cell; EXP-014 saw 20-point search-rate swings at n = 15, so only large effects are read.

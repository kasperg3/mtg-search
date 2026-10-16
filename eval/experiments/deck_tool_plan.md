# Plan: does `get_deck` help or hurt tool use? (EXP-0xx, not run)

**Status: plan only, not run.** It needs the owner's go-ahead and the harness changes listed below. It takes the
next free ID in `REGISTRY.md` when it runs. Design: [`docs/deck_tool.md`](../../docs/deck_tool.md).

## Questions

1. **Harm.** With `get_deck` on the surface (deferred), does the agent still search when the deck is already in
   the prompt? In EXP-010, `check_cards` replaced searching in 28 of 28 episodes, and EXP-014 showed that surface
   changes move the search rate.
2. **Help.** When the user names a precon instead of pasting it, does the agent load it with `get_deck`? Are its
   adds then valid for the real deck, that is, not already in it and within its colour identity?

## Arms

| arm | surface |
|---|---|
| A | `three_tools`: the EXP-014 shipped surface, without `get_deck` |
| B | `shipped`: `get_deck` added, deferred |
| C (follow-up, only if B rarely calls `get_deck` in T2) | B with `alwaysLoad` on `get_deck` (needs a small hook in `build_mcp`) |

## Tasks

All runs: Claude Code, `claude-sonnet-5-5`, subscription, deferred loading (the default), 1 repeat.

| task | prompt | precons | episodes |
|---|---|---|---|
| T1: ADD as today | the deck is rendered in the prompt; precon lookup off (no `precons_path`), as in every benchmark run | 15 dev | 2 × 15 = 30 |
| T2: ADD by reference | the user names the deck and its commander ("Upgrade my precon 'Calling All Angels' (commander: Giada, Font of Hope). Propose 20 cards to add.") and gives no list; precon lookup on | 15 dev + the 6 unscored post-cutoff precons of snapshot 2026-10-14 (FDC ×5 and FRC ×1, released 2026-10-02, in no scored split) | 2 × 21 = 42 |

That is 72 episodes, about 15 minutes at 10–14 s each. The 14 scored test precons are not used.

**Harness changes needed (not made):**
- `prompts.py`: a by-reference mode for T2 that keeps the system-prompt rules and names the deck in the user
  turn instead of rendering it. The leakage pre-flight must allow the deck name and the commander name in T2 prompts.
- `tools.py`: pass `precons_path` to `CardService` for T2.
- Scoring: none. `validate_adds` already checks `in_deck` and `off_identity` against the real list, which the
  agent in T2 never saw unless it loaded it.

## Metrics

| | primary | secondary |
|---|---|---|
| T1 | share of episodes that call `search_cards`; share in which `get_deck` is called and `search_cards` is not | validity, ADD nDCG@10, popularity drift, tokens, turns, wall time |
| T2 | ADD validity, with the `in_deck` and `off_identity` counts; share of episodes that call `get_deck` | share that also call `search_cards`, tokens, turns, wall time |

## Decision rule (fixed before running)

- **No harm (T1):**
  - B searches in at least as many episodes as A minus 2 (EXP-010 and EXP-014 saw 20-point swings at n = 15);
  - at most 1 B episode calls `get_deck` without `search_cards`;
  - if B fails either, do not ship it in this form.
- **Help (T2):**
  - B's `in_deck` + `off_identity` invalid picks are at most half of A's, on the 6 new precons as well as on dev;
  - if B calls `get_deck` in fewer than half of the T2 episodes, run arm C before deciding.
- Read large effects only.

## Caveats

- **The dev precons were released before the model's training cutoff,** so the model may recall their lists; A
  is not blind on them. The 6 new precons are the clean slice, and n = 6 makes them a smoke test only.
- **T2 names the product by design,** so community-key nDCG means little there. Use validity.
- **Turn counts include a `ToolSearch` step:** a deferred `get_deck` has to be loaded before its first call.

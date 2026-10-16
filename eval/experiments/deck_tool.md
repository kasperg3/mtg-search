# EXP-020: `get_deck` lets the agent work on a deck it is only told about

**Status: done 2026-10-16.** Pre-registered in [`deck_tool_plan.md`](deck_tool_plan.md) (commit `342a15f`) before
any model run. All four cells ran at `342a15f`. Headless Claude Code `claude-sonnet-5-5`, subscription, deferred
tool loading, 3 workers. 21 precons (15 dev + 6 released 2026-10-02) × CUT/ADD/SWAP × 1 repeat = 63 episodes per
cell, 252 in all, 0 errors, every episode ended with `final_answer`. Report: `bench/precon/exp020_report.py`;
aggregates: [`runs/20261016_exp020_summary.json`](../../bench/precon/runs/20261016_exp020_summary.json).

## The cells

| | without `get_deck` (`three_tools`) | with `get_deck`, deferred (`shipped`) |
|---|---|---|
| **pasted**: the deck's card text is in the prompt | PA (today's benchmark) | PB |
| **named**: "My deck is the official precon "<deck>" (commander: <name>), unchanged from the box." | NA (memory only) | NB |

## Results

Validity (share of picks that are legal for the real deck), macro mean over the 21 precons:

| | CUT | ADD | SWAP cuts | SWAP adds |
|---|---|---|---|---|
| PA pasted, no `get_deck` | 1.000 | 0.976 | 0.971 | 0.990 |
| PB pasted, with `get_deck` | 1.000 | 0.990 | 0.995 | 0.995 |
| NA named, no `get_deck` | **0.300** | 0.855 | **0.243** | 0.819 |
| NB named, with `get_deck` | **1.000** | 0.988 | **1.000** | 1.000 |

What went wrong without the list (NA): 147 of 210 cuts and 158 of 210 swap cuts named cards that are not in the
deck; 61 of 420 adds and 38 of 210 swap adds were already in it. On the 6 new precons, which the model cannot
have seen, NA's cut validity was 0.233 (dev 0.327).

Tool use and cost per episode:

| | `get_deck` called | `search_cards` called (CUT / ADD / SWAP) | tokens | turns | wall |
|---|---|---|---|---|---|
| PA | – | 0 / 17 / 21 of 21 | 49.3k | 3.6 | 10.9 s |
| PB | 0 of 63 | 0 / 16 / 20 of 21 | 48.7k | 3.6 | 11.7 s |
| NA | – | 1 / 21 / 21 of 21 | 25.4k | 4.4 | 16.4 s |
| NB | 63 of 63 | 0 / 17 / 19 of 21 | 41.7k | 5.3 | 16.4 s |

NB called `get_deck` once in 62 episodes and twice in one: it first passed the deck name and the commander together
("Everyone's Invited! Morophon, the Boundless"), got "No precon named …" with the closest names, and retried with
the deck name. Every other call used the deck name alone. In the pasted cells the agent never called `get_deck`.

Paired differences over the 21 precons (95% bootstrap CI, two-sided sign test):

| comparison | CUT validity | ADD validity | SWAP cut validity | SWAP add validity |
|---|---|---|---|---|
| help: NB − NA | **+0.700** [+0.619, +0.786], p = 9.5e-7 | **+0.133** [+0.091, +0.181], p = 7.6e-6 | +0.757 [+0.690, +0.819], p = 9.5e-7 | +0.181 [+0.105, +0.262], p = 2.4e-4 |
| recovery: NB − PA | +0.000 | +0.012 [−0.002, +0.033] | +0.029 [+0.010, +0.048] | +0.009 [+0.000, +0.024] |
| guard: PB − PA | +0.000 | +0.014 [−0.007, +0.038] | +0.024 [+0.000, +0.048] | +0.005 [−0.010, +0.019] |

Community-key nDCG@10 (dev precons only, reported, not decided on):

| | CUT | ADD | SWAP cut | SWAP add |
|---|---|---|---|---|
| PA | 0.397 | 0.215 | 0.367 | 0.184 |
| PB | 0.406 | 0.253 | 0.435 | 0.174 |
| NA | 0.012 | 0.137 | 0.055 | 0.176 |
| NB | 0.406 | 0.279 | 0.437 | 0.228 |

NB − PA on these: +0.009, +0.064, +0.070, +0.044, none significant (sign p ≥ 0.30).

## Decision rules

1. **Help: established.** NB − NA validity is positive with the 95% CI above 0 on CUT (+0.700) and ADD (+0.133),
   and on both SWAP sides. On the 6 new precons alone: CUT +0.767 [+0.617, +0.900], ADD +0.208 [+0.100, +0.308].
2. **Recovery: holds.** NB validity is at least PA − 0.03 on every task (it is equal or higher on all four), and NB
   called `get_deck` in 63 of 63 episodes, so cell NC (`get_deck` always loaded) was not needed.
3. **Guard: passes.** PB searched in 36 of 63 pasted episodes vs PA's 38 (rule: at least 32); 0 PB episodes called
   `get_deck` without searching (rule: at most 3); no PB − PA validity CI lies below −0.03.
4. **Ship `get_deck` deferred.**

## What this shows

- Named but not pasted, the agent without the list works from a recalled list: 70% of its cuts are cards that
  are not in the deck. That is what the owner saw when they referenced a precon.
- With `get_deck` the agent loads the list in every episode and then does the task as if the deck had been pasted:
  same validity, the same searching on ADD and SWAP, about the same community-key score. It costs one or two more
  turns (the `ToolSearch` that loads the deferred tool, then the call) and about 5 seconds; tokens are lower than
  pasting on CUT and ADD because the list enters once, as a tool result.
- The extra tool changes nothing when the deck is already in the prompt: the agent does not call it, and it
  searches as often as before. Unlike EXP-010's `check_cards`, `get_deck` returns data, not a verdict.

## Caveats

- One repeat per cell; the effects that decide are large (+0.70 cut validity), the others are read as "no change".
- The 15 dev precons predate the model's training cutoff, so NA could partly recall them; the 6 new ones are the
  clean slice, and n = 6.
- Named prompts name the product, which also cues recalled EDHREC lists; that is why nDCG is not decided on.
- The benchmark's `get_deck` has no link fetcher (leakage rule T9), so Moxfield and Archidekt links were not part of
  this test; they were checked by hand on two public decks and by unit tests.

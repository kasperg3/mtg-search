# EXP-010: Smoke tests for improving the shipped MCP surface (after EXP-009)

**Question.** v3 + `alwaysLoad` is shipped (EXP-009). Does any cheap change to the shipped surface make the agent search more reliably, pick better or more validly, or cost fewer tokens, turns or seconds? The changes tried: wording in tool descriptions, instruction layout, a batch validity tool (`check_cards`, issue #4), and slimmer schemas.

**Answer (smoke tests, n = 14 per arm, read large effects only).**
- **Putting the trigger in the always-loaded tool descriptions works best.** Opening `search_cards`/`get_card` with "Before recommending … call this" made the agent search in **14/14** episodes (e1desc), against 8/14 for the shipped surface; Fisher p≈0.02. Quality and validity stayed at least as good.
- **That trigger plus slim schemas and no `list_tags` (e6desc_slim)** reached the highest smoke quality: nDCG@10 0.264, 98.9% valid, 79% searched. It also has 2.3k fewer surface characters.
- **`check_cards` replaces searching rather than adding to it.** With it, every episode called only `check_cards` and none searched (e3check, e5desc_check). Validity goes to 98.6–99.3%, but quality falls to the no-tools level (0.19–0.21) with staple-heavy picks. Don't ship it as a standalone tool in this form; see issue #4.
- **A shorter, trigger-first instruction block is worse** (e2lead): 14% searched and validity −3.2 points.
- **Turns and time:** a tool-using ADD episode takes about 3–4 Claude Code turns and about 10–14 s, against 1 turn and 5 s from memory. With the same "use your tools" hint, mLateOn and grep/BM25 need the same turns (3.0 vs 3.1) and time (10 vs 9 s). The agent searches about once with either, so this task does not separate them; see below.

## Setup

- **Task.** EXP-007 precon benchmark (`@d6bb8ba`), **ADD only** (the task where tools matter; CUT never triggered tool use), 14 test precons, 1 repeat.
- **Agent.** Headless Claude Code 2.1.299, `claude-sonnet-5-5`, subscription, **deferred loading on** (`ToolSearch`, the Claude Code default).
- **Retrieval.** mLateOn only.
- **Baseline.** Surface `shipped` = server.py after commit `5eb4fe7` (v3 text, `alwaysLoad` on `search_cards`/`get_card`).
- **Variants** (`bench/precon/harness/surfaces.py`):

| arm | change vs shipped |
|---|---|
| e1desc | `search_cards` and `get_card` descriptions start with a trigger sentence: "Before recommending cards for a deck, call this to find candidates instead of listing them from memory." / "Before recommending a card you know only from memory, call this to check its current text, colour identity and legality." |
| e2lead | Instructions rewritten shorter, with the procedural sentence first |
| e3check | Adds `check_cards` (always loaded); the instructions' last sentence says to search, then run `check_cards` on the final picks |
| e4slim | Drops `list_tags`; strips pydantic `title`s and `anyOf [T, null]` wrappers from the JSON schemas (tool surface 12.6k → 10.1k chars) |
| e5desc_check | e1desc + e3check |
| e6desc_slim | e1desc + e4slim |
| lexical-hint | **Retriever comparison**, not a surface variant. Harness `grep` + `bm25` tools, plus the EXP-007 system-prompt tool hint, upfront. Compared with the EXP-007 `mcp-hint` arm (mLateOn harness tools, same hint) |

## Results (ADD, 14 test precons)

| arm | searched | calls/ep | turns/ep | wall s/ep | prompt tok/ep | out tok/ep | ADD nDCG@10 | validity | pop. drift |
|---|---|---|---|---|---|---|---|---|---|
| shipped (v3 + alwaysLoad) | 57% | 0.57 | 3.6 | 12 | 53.7k | 927 | 0.243 | 0.979 | 3.30 |
| *same config, EXP-009 run* | *36%* | *0.43* | — | — | *52.6k* | — | *0.275* | *0.964* | *2.97* |
| **e1desc** | **100%** | 1.00 | 4.0 | 11 | 58.3k | 821 | 0.253 | 0.986 | **3.51** |
| e2lead | 14% | 0.14 | 3.1 | 13 | 48.1k | 1040 | 0.249 | 0.946 | 2.47 |
| e3check | 100% (all `check_cards`) | 1.14 | 4.1 | 13 | 53.3k | 1207 | 0.194 | 0.986 | 2.52 |
| e4slim | 29% | 0.29 | 3.3 | 12 | 48.5k | 889 | 0.215 | 0.964 | 2.72 |
| e5desc_check | 100% (all `check_cards`) | 1.07 | 4.1 | 14 | 52.2k | 1297 | 0.209 | **0.993** | 2.49 |
| **e6desc_slim** | 79% | 0.93 | 3.9 | 11 | 57.1k | 880 | **0.264** | 0.989 | 3.21 |
| *ref: v1 (pre-EXP-009 text)* | *0%* | 0 | 2.3 | 10 | 41.0k | 590 | *0.185* | *0.907* | *2.50* |
| *ref: no tools* | — | 0 | 1.0 | 5 | 11.3k | 297 | *0.189* | *0.904* | *2.42* |

- "searched" is the share of episodes with at least one tool call.
- Turns are Claude Code turns, which include the `ToolSearch` turn needed to load `final_answer` in deferred mode.
- Prompt tokens include cached tokens and tool results.

Paired vs shipped (ADD, 14 precons), nDCG@10 / validity / popularity drift:
- e1desc: +0.009 / +0.007 / +0.22
- e2lead: +0.006 / **−0.032 [−0.057, −0.011]** / **−0.83 [−1.29, −0.37]**
- e3check: −0.049 / +0.007 / **−0.77 [−1.19, −0.35]**
- e4slim: −0.028 / −0.014 / **−0.57 [−1.10, −0.04]**

The two shipped-config runs differ by about 20 points of search rate (36% vs 57%), which shows the run-to-run noise at n=14.

## Turns and time: mLateOn vs grep/BM25

With Claude Code, Sonnet 5.5, ADD and the same "use the tools" hint:

| tools | searched | calls/ep | turns/ep | wall s/ep | prompt tok/ep | tool-output chars/ep | ADD nDCG@10 | validity | pop. drift |
|---|---|---|---|---|---|---|---|---|---|
| mLateOn `search_cards` (EXP-007 mcp-hint) | 93% | 0.93 | 3.0 | 10 | 51.2k | 9.4k | 0.235 | 0.982 | 3.48 |
| `grep` + `bm25` | 100% | 1.07 (13 grep, 2 bm25) | 3.1 | 9 | 44.6k | 5.6k | 0.226 | 0.982 | 3.23 |

From EXP-006 (GLM-5.3, deck needs, forced 5-search budget; no wall time was recorded there):

| cond | LLM turns/ep | searches/ep | first labelled card found at search 1 | prompt tok/ep | output tok/ep | recall@5 |
|---|---|---|---|---|---|---|
| A mLateOn | 3.57 | 5.0 | 90% | 9.4k | 4.2k | 0.201 |
| F grep + BM25 | 3.64 | 5.0 | 94% | 12.1k | **7.1k** | 0.227 |

- On a one-shot ADD, Claude Code searches about once with either toolset, so turns and seconds are the same.
- When exploration is forced (EXP-006), the lexical agent needs the same number of turns but writes **1.7× the output tokens** (regex crafting) and reads 1.3× the input. It is slower and dearer per result, though not in turns.
- A task that rewards several searches would be needed to show a turns/time gap. Examples: known-item finding, or an unconstrained budget with a stop condition. EXP-006's known-item tasks never ran.
- From now on, every summary reports turns, wall time and tokens per episode (`smoke.py`-style table).

## Review caveats

From [`docs/review_2026-10-14.md`](../../docs/review_2026-10-14.md), M1–M3:
- **e6desc_slim was picked as the best of 7 arms on the same 14 test precons.** Its ADD nDCG@10 vs shipped is +0.021 [−0.091, +0.129], sign test p=0.79. The quality ranking between arms is noise-level.
- **The robust finding is the trigger rate.** e1desc 14/14 vs shipped 13/28 over both runs, Fisher p=0.0005. Separately, check_cards replaces searching in all 28 episodes.
- **Retrieval.** All arms used the exact in-process mLateOn ranker, not the shipped NextPlaid/int8 path.
- **Turns include a ToolSearch step.** Deferred arms spend one turn on `ToolSearch` to load `final_answer`, so turn and token counts include that constant.

## Recommendation

1. **Candidate, not yet established: e6desc_slim** (or e1desc). It adds the description trigger (e1desc), slims the schemas and drops `list_tags`. It improved the trigger rate and, in this smoke test, quality, while cutting surface size. The quality gain is not established (see caveats). Choose between the arms on the **dev** split, then confirm once on test. If `list_tags` should stay for human users, ship e1desc instead.
2. **Do not ship `check_cards` as a standalone tool.** Options for issue #4:
   - fold validity into what `search_cards` already guarantees (filters);
   - expose the check only after a search;
   - or only on request ("check my deck").
3. **Confirm on a full run before trusting small effects** (owner's decision): 14 test precons × CUT/ADD/SWAP, 2–3 repeats, shipped vs e6desc_slim, all through Claude Code on the subscription.

Run records: `bench/precon/runs/20261014_smoke-add_*.json`. Transcripts are on HF `kasper-cbx/mtg-search-experiments`.

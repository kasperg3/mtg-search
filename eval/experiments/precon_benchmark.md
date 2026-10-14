# EXP-007: Precon cut/add benchmark: first agent run (Claude Code, Sonnet 5.5)

**Question.** Given a Commander precon as card text, how well does an LLM agent with the popularity-free mtg-search MCP choose cards to cut, cards to add, and swaps? The grading is against EDHREC cut/add rates. Do the search tools help compared with no tools?

**Answer (small run, n = 14 precons, 1 repeat, wide CIs).**
- Sonnet 5.5 beats the text heuristic and matches or beats the popularity-oracle reference on ADD (nDCG@10 0.19–0.24 vs 0.16 heuristic, 0.19 oracle). On CUT it is only slightly above random (0.33–0.36 vs 0.28; oracle 0.66).
- **Given the search tools, it does not use them**: 0 tool calls in 42 episodes. It answers from its own card knowledge.
- **Told to use the tools (one added sentence), it makes 1–2 calls per episode.** ADD validity rises from 90.4% to 98.2% (+7.9 points [+2.1, +14.3]), and its adds become much less popularity-biased at equal or better quality. The quality differences are not significant at n=14.
- **Contamination looks low.** The forced-guess memorisation probe scores the same on the test precons as on older dev precons (overlapping CIs), at the level of a generic "popular staples" prior.

## Setup

- **Agent.** Headless Claude Code (`claude -p`, CLI 2.1.299) running `claude-sonnet-5-5` on the owner's Claude Code subscription. The API key is removed from the environment; the session init reports `apiKeySource: none`. No API spend.
  - Bridge: [`bench/precon/harness/claude_code.py`](../../bench/precon/harness/claude_code.py) + [`cc_proxy.py`](../../bench/precon/harness/cc_proxy.py), `--llm claude-code`.
  - Each episode runs in an empty temp dir with `--tools ""` (no file, shell or web tools) and `--strict-mcp-config`. One stdio MCP proxy exposes exactly the harness toolset plus `final_answer`, with the plan's per-task call budget (CUT 5, ADD 10, SWAP 12).
  - Every session's init event is asserted to contain only the proxy's tools and server.
  - System prompt, deck rendering, pre-flight leakage checks, scorer and run records are the same as for GLM. Temperature cannot be set (provider default).
- **Data.** `kasper-cbx/mtg-precon-bench@d6bb8ba`, the 14 scored **test** precons (released 2026-01 to 2026-08). Labels are EDHREC cut/add rates (snapshot 2026-10-14).
- **Conditions** (all 3 tasks × 14 precons × 1 repeat):
  - `none`: no tools, JSON answer in text;
  - `mcp`: search tools available;
  - `mcp-hint`: `mcp` plus one sentence: "Before you answer, use the tools: search for candidate cards and check exact card text, colour identity and legality of your picks." (`--tool-hint`). This was added after `mcp` showed 0 tool calls.
- **Probe** (plan T2c): no tools, no deck list, only product and deck name, asked for the common EDHREC cuts and adds.
  - The first version let the model decline. Sonnet 5.5 returned empty lists for all 14 test precons ("I don't have reliable memory of this deck"). That run is kept: [`20261014_cc-sonnet55_probe_test.json`](../../bench/precon/runs/20261014_cc-sonnet55_probe_test.json).
  - The probe now asks for best guesses. It was run on test and on the 15 dev precons (2025-06 to 2025-08, certainly in training) as the contamination reference.

## Leakage

Sonnet 5.5's training cutoff is June 2026. Under the plan's rule (test = released after cutoff + 2 months), only the 2026-08 Secret Lair would count as clean, so **all test precons here are formally pre-cutoff, and the results are labelled "possibly contaminated"**. The probe is the check:

| probe (forced guess) | CUT nDCG@10 | ADD nDCG@10 | ADD R@20 | ADD validity | popularity drift of adds |
|---|---|---|---|---|---|
| test (14) | 0.020 [0.00, 0.06] | 0.097 [0.05, 0.15] | 0.069 | 0.60 | 1.92 |
| dev (15, older) | 0.014 [0.00, 0.03] | 0.130 [0.09, 0.18] | 0.090 | 0.63 | 2.10 |

- The model cannot recall cuts, since it doesn't know the decklists.
- Its ADD guesses are generic staples. Popularity drift (mean log10 EDHREC rank of adds) is about 2, near the popularity oracle's 1.38.
- They score the same on test as on dev.
- There is no sign of deck-specific memorisation. The agent's ADD scores (0.19–0.24) come from reading the deck.

## Results (test, n = 14, macro mean [95% bootstrap CI])

| system | CUT nDCG@10 | CUT F1@10 | ADD nDCG@10 | ADD nDCG@20 | ADD R@20 | ADD validity | ADD popularity drift | SWAP cut nDCG@10 | SWAP add nDCG@10 | SWAP add validity |
|---|---|---|---|---|---|---|---|---|---|---|
| random-legal | 0.279 | 0.193 | 0.001 | 0.001 | 0.000 | 1.000 | 4.07 | 0.216 | 0.013 | 1.000 |
| text heuristic (mLateOn) | 0.295 | 0.176 | 0.161 | 0.124 | 0.069 | 1.000 | 3.94 | 0.295 | 0.161 | 1.000 |
| popularity oracle (reference) | 0.659 | 0.427 | 0.194 | 0.152 | 0.097 | 1.000 | 1.38 | 0.659 | 0.156 | 1.000 |
| Sonnet 5.5, `none` | 0.342 | 0.246 | 0.189 | 0.162 | 0.099 | 0.904 | 2.42 | 0.311 | 0.118 | 0.898 |
| Sonnet 5.5, `mcp` (0 tool calls) | 0.325 | 0.241 | 0.209 | 0.184 | 0.115 | 0.918 | 2.41 | 0.337 | 0.175 | 0.950 |
| Sonnet 5.5, `mcp-hint` | **0.362** [0.26, 0.47] | **0.267** | **0.235** [0.16, 0.31] | **0.192** | 0.094 | **0.982** | **3.48** | 0.327 | 0.176 | **0.971** |

The baselines are restricted to the same 14 test precons (`slices.split.test` of the all-precon runs). Lower popularity drift means more popular picks; random is 4.07.

Paired differences vs `none` (14 precons):

| metric | `mcp` − `none` | `mcp-hint` − `none` |
|---|---|---|
| CUT nDCG@10 | −1.7 [−5.2, +1.6] | +2.0 [−1.4, +5.4] |
| ADD nDCG@10 | +2.0 [−2.6, +6.9] | +4.6 [−9.5, +18.7] |
| ADD validity | +1.4 [−3.2, +6.4] | **+7.9 [+2.1, +14.3]** |
| SWAP add nDCG@10 | +5.7 [−1.1, +12.9] | +5.8 [−0.6, +12.9] |
| SWAP add validity | +5.2 [−0.7, +11.9] | +7.4 [−0.0, +14.8] |

Invalid ADD picks by reason (out of 280):

| reason | `none` | `mcp` | `mcp-hint` |
|---|---|---|---|
| off colour identity | 20 | 13 | 1 |
| Game Changer limit | 3 | 2 | 1 |
| unknown name | 2 | 4 | 1 |
| already in deck / duplicate | 1 | 2 | 2 |
| not Commander-legal | 1 | 2 | 0 |

Tool use in `mcp-hint`: 0.8 calls per CUT episode, 0.9 per ADD and 2.1 per SWAP (27 `search_cards`, 27 `get_card`). `find_similar` was never used.

## Reading

- **A strong model does not reach for the search tools by default.** In the shipped MCP, the tool descriptions and server instructions should say when to use them: to find candidates and to check colour identity and legality.
- **Using the tools mostly fixes validity.** Off-identity adds drop from 20 to 1. The remaining errors (Game Changer limit, already in deck) need deck context that no current tool has.
- **Searching shifts adds away from staples** (popularity drift 2.4 → 3.5) without losing nDCG. That matches the project's no-hype goal.
- **CUT is hard for every non-oracle system.** The community's cuts are largely the weakest *popular-knowledge* cards, which the popularity oracle captures and text-only reasoning only partly does.
- **Limits:**
  - n=14 and one repeat;
  - one model;
  - labels reflect EDHREC users, not optimal play;
  - formally pre-cutoff test set (see Leakage);
  - the `mcp-hint` prompt differs from the plan's prompt by one sentence.

Run records: `bench/precon/runs/20261014_cc-sonnet55_{none,mcp,mcp-hint,probe,probe-guess}_{test,dev}.json`. Transcripts are on the private HF dataset `kasper-cbx/mtg-search-experiments` under `bench/precon/runs/`.

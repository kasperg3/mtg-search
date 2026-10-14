# EXP-009: MCP server instructions: can the server's own text make an agent use the tools?

**Question.** In EXP-007, Sonnet 5.5 built decks from memory and made no tool calls unless the *system prompt* told it to search. Real clients carry no such hint. Can the shipped server's `instructions` and tool descriptions trigger tool use, valid picks and better picks on their own? At what token cost? Does it survive Claude Code's default deferred tool loading?

**Answer.**
- **General best-practice wording is not enough (v2).** A research-based rewrite triggered tool use in 1 of 42 episodes. It did make picks more careful even without searching: ADD validity +5.4 points.
- **One procedural sentence in the server instructions works when the tools load upfront (v3).** The sentence ties tool use to the moment of answering: "Before you answer a deck-building request, use the tools: search for candidate cards and check exact card text, colour identity and legality of your picks."
  - 29% of episodes searched.
  - ADD nDCG@10 +0.139 [+0.048, +0.236] vs v1, the best ADD score of any arm.
  - Validity +7.9 points; adds less popularity-biased.
  - The same sentence in the system prompt triggers more often (57%). The channel matters, but the server channel works.
- **Under deferred loading (Claude Code's default), instructions alone fail again (v3: 5%).** Marking `search_cards` and `get_card` with `_meta["anthropic/alwaysLoad"]` restores tool use (45%). It gives ADD nDCG@10 +0.090 [−0.002, +0.183], ADD validity +5.7 [+1.1, +11.8] and SWAP add validity +5.0 [+2.1, +8.6].
- **Recommendation:** ship the v3 text plus `alwaysLoad` on `search_cards` and `get_card`.
  - The text is 28% shorter than today's: 4.3k vs 6.0k chars of model-facing text.
  - It costs one extra instruction sentence and the two always-loaded schemas.

Research behind the rewrite: [`docs/mcp_instructions_research.md`](../../docs/mcp_instructions_research.md).

## Setup

- **Task.** EXP-007 precon benchmark, dataset `kasper-cbx/mtg-precon-bench@d6bb8ba`: 14 test precons × CUT/ADD/SWAP × 1 repeat = 42 episodes per arm. Paired bootstrap over precons (10k) and sign test.
- **Agent.** Headless Claude Code 2.1.299, `claude-sonnet-5-5`, on the owner's subscription (no API). Each episode is isolated: empty temp dir, no built-in tools, `--strict-mcp-config`, init event asserted.
- **Server.** `--tools shipped`: the real `build_mcp` surface over the in-process eval service, with **mLateOn-only retrieval**. The agent sees the server as `mtg-search`, with its `instructions`, tool names, descriptions and JSON schemas, plus the harness's `final_answer`. A proxy enforces the per-task call budget.
- **System prompt.** It contains task, rules and the final-answer instruction, and **no tool description and no call budget**, as in real use.
  - The first v1 arm still stated "You may make at most N tool calls"; it is kept as `shipped-v1-budgetline`.
  - Claude Code passes server instructions to the model even with `--system-prompt` (verified with a secret-word probe).
- **Deferred arms.** `--cc-tool-search` enables `ToolSearch`. Claude Code then shows only tool names and the instructions until the model loads a tool. Checks:
  - the agent had to `ToolSearch`-select `final_answer` before calling it;
  - with `alwaysLoad`, the marked tools are loaded at start (verified on a probe server).
- **Surfaces** ([`bench/precon/harness/surfaces.py`](../../bench/precon/harness/surfaces.py)):
  - `v1`: as shipped today. Short instructions; long descriptions that repeat the filter and output docs.
  - `v2`: the research rewrite (§3 of the research doc). The trigger and "use them even when you feel sure from memory" go in the instructions; descriptions are self-contained, put key terms first and drop the duplicated prose.
  - `v3`: `v2` plus the EXP-007 tool-hint sentence in the instructions.
  - `v3al`: `v3` plus `anthropic/alwaysLoad` on `search_cards` and `get_card`.

## Results

| arm | loading | episodes with ≥1 call | calls/ep (ADD / SWAP) | CUT nDCG@10 | ADD nDCG@10 | ADD validity | ADD popularity drift | SWAP add nDCG@10 | SWAP add validity | prompt tok/ep |
|---|---|---|---|---|---|---|---|---|---|---|
| v1 (shipped) | upfront | 0% | 0 / 0 | 0.345 | 0.185 | 0.907 | 2.50 | 0.174 | 0.936 | 37.8k |
| v1, budget line in prompt | upfront | 0% | 0 / 0 | 0.333 | 0.173 | 0.911 | 2.45 | 0.160 | 0.936 | 36.0k |
| v2 | upfront | 2% | 0.07 / 0 | 0.346 | 0.232 | 0.961 | 2.57 | 0.214 | 0.971 | 35.0k |
| **v3** | upfront | **29%** | 0.64 / 0.21 | 0.331 | **0.324** | 0.986 | 3.36 | 0.247 | 0.964 | 42.5k |
| v2 + hint in **system prompt** | upfront | 57% | 0.93 / 2.64 | 0.334 | 0.285 | **0.996** | **3.71** | 0.247 | 0.979 | 50.0k |
| v2 | deferred | 0% | 0 / 0 | 0.334 | 0.195 | 0.896 | 2.47 | 0.206 | 0.979 | 39.4k |
| v3 | deferred | 5% | 0 / 0.14 | 0.335 | 0.190 | 0.900 | 2.36 | 0.211 | 0.943 | 41.0k |
| **v3 + alwaysLoad** | deferred | **45%** | 0.43 / 1.0 | 0.338 | 0.275 | 0.964 | 2.97 | 0.143 | **0.986** | 51.8k |
| *ref: no tools (EXP-007)* | — | — | — | 0.342 | 0.189 | 0.904 | 2.42 | 0.118 | 0.898 | 11.4k |

- No arm called a tool on CUT. Cuts come from reading the deck, which needs no search.
- Popularity drift is the mean log10 EDHREC rank of the adds; higher means fewer staples.
- Prompt tokens include cached tokens and tool results.

Paired differences vs v1 (upfront), 14 precons:

| arm | ADD nDCG@10 | ADD validity | ADD popularity drift | SWAP add validity |
|---|---|---|---|---|
| v2 | +0.047 [−0.013, +0.116] | **+0.054 [+0.014, +0.093]** | +0.07 | +0.036 [−0.000, +0.079] |
| v3 | **+0.139 [+0.048, +0.236]** | **+0.079 [+0.025, +0.132]** | **+0.86 [+0.45, +1.28]** | +0.029 [−0.007, +0.064] |
| v2 + system-prompt hint | +0.100 [−0.026, +0.233] | **+0.089 [+0.036, +0.150]** | **+1.22 [+0.94, +1.49]** | **+0.043 [+0.021, +0.071]** |
| v2, deferred | +0.009 | −0.011 | −0.03 | **+0.043 [+0.007, +0.079]** |
| v3, deferred | +0.004 | −0.007 | −0.14 | +0.007 |
| v3 + alwaysLoad, deferred | +0.090 [−0.002, +0.183] | **+0.057 [+0.011, +0.118]** | **+0.47 [+0.08, +0.90]** | **+0.050 [+0.021, +0.086]** |

**Token efficiency** (model-facing text: instructions + 4 tool descriptions):

| | v1 | v2 | v3 |
|---|---|---|---|
| chars (≈ tokens) | 6,024 (≈1,506) | 4,171 (≈1,042) | 4,327 (≈1,081) |
| full tool surface incl. JSON schemas (chars) | 14,462 | 12,410 | 12,566 |

The input schemas are now the largest part, about 950 tokens each for the two filter tools. Dropping pydantic's per-property `title`s and the `anyOf [T, null]` wrappers would save roughly 200 tokens per filter tool. This is a code change, not done here.

## Reading

- **The agent knows about the tools and chooses not to use them.** In v1/v2 episodes it typically ends with "I picked them from memory and ran no searches; I didn't check colour identity or legality". Descriptions that say *what* a tool is good for don't change that.
- **What changes it is a procedural instruction placed at the decision point** ("Before you answer…, use the tools: search… and check…"). This matches Anthropic's note that Sonnet 5.5 tends to answer from training knowledge. Exhortations ("even when you feel sure") and capability claims did not work here.
- **Under deferred loading, `instructions` alone do not get the tools loaded.** The model would first have to decide to call `ToolSearch`, which it rarely does for a task it believes it can answer. `alwaysLoad` on the two core tools removes that step.
- **Using the tools pays off on ADD:** higher nDCG, near-perfect validity, fewer staples. On SWAP the add-side quality is noisy: v3 + alwaysLoad −0.031 n.s. vs v1, while validity improves.
- **Limits:**
  - one model (Sonnet 5.5) and one client (Claude Code);
  - n=14 precons × 1 repeat, so the tool-use rates have wide uncertainty (29% vs 45% are not distinguishable);
  - the benchmark always contains a full decklist and a forced `final_answer`, unlike open chat;
  - Claude apps (claude.ai/Desktop) may handle `instructions` and deferral differently (unknown, see research doc §1).

## Next

1. Ship v3 in `server.py` with `alwaysLoad` on `search_cards` and `get_card`. Optionally add the `readOnlyHint` / `title` annotations (research rule 10), plus a test that pins the instruction sentence.
2. Issue [#4](https://github.com/kasperg3/mtg-search/issues/4) (batch validity check). Then re-run v3 + alwaysLoad with the new tool to see whether the last invalid picks (Game Changer limit, already in deck) go away.
3. Optionally, an MCP *prompt* (e.g. `/mtg-search:upgrade-deck`). It would carry the system-prompt-strength hint for users who invoke it.

Run records: `bench/precon/runs/20261014_cc-sonnet55_shipped-{v1,v1-budgetline,v2,v2-deferred,v3,v3-deferred,v3al-deferred,v2-sphint}_test.json`. Transcripts are on HF `kasper-cbx/mtg-search-experiments`.

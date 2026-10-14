# MCP server instructions and tool descriptions: what makes an agent use the tools

Research notes, 2026-10-14. Motivation: in EXP-007 ([precon_benchmark.md](../eval/experiments/precon_benchmark.md)), Sonnet 5.5 made 0 tool calls in 42 episodes. One added system-prompt sentence led to 1–2 calls per episode, and invalid ADD picks fell from 9.6% to 1.8%. Real clients carry no such hint, so the server's own text has to do that job.

**Caveat for the benchmark.** EXP-007 never showed the model this server's text. [`cc_proxy.py`](../bench/precon/harness/cc_proxy.py) starts `Server("deck", …)` with no `instructions`, and [`tools.py`](../bench/precon/harness/tools.py) (`MCPToolset.schemas`) serves short hand-written descriptions instead of `server.py`'s. Also, `--tools ""` removes `ToolSearch`, so the tools were loaded upfront: the init event lists only the four proxy tools. That is not the default in real Claude Code (see below).

## 1. How clients consume `instructions` and descriptions

**Known**
- **Spec.** `InitializeResult.instructions` is optional: "a 'hint' to the model … this information MAY be added to the system prompt" ([schema 2025-11-25](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/schema/2025-11-25/schema.ts)).
  - In 2026-07-28 the field moved to `DiscoverResult`. Its comment now adds that it "should focus on information that helps the model use the server effectively and should not duplicate information already in tool descriptions" ([schema 2026-07-28](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/schema/2026-07-28/schema.ts)).
  - `Tool.description` is likewise a "hint".
- **Clients differ.** How a host uses instructions is "up to the implementer … not always guaranteed that they will be injected into the system prompt" ([MCP blog, Server Instructions, 2025-11](https://blog.modelcontextprotocol.io/posts/2025-11-03-using-server-instructions/)).
  - In that post's VS Code test (n=10 per cell), a workflow instruction raised GPT-5-mini from 2/10 to 8/10. Claude Sonnet 4 scored 10/10 without it and 9/10 with it.
- **Claude Code: tool search is on by default** ([docs: Scale with MCP tool search](https://code.claude.com/docs/en/mcp#scale-with-mcp-tool-search)).
  - "Only tool names and server instructions load at session start." A tool's description reaches the model only after a `ToolSearch` call matches it.
  - The docs tell server authors that instructions "help Claude understand when to search for your tools, similar to how skills work". They should say what category of tasks the tools handle, when to search for them, and the key capabilities.
- **Claude Code length caps.** Instructions and the descriptions of upfront tools are cut at **4,096 chars**; deferred descriptions at 16,384.
  - `ToolSearch` matches only the first 4,096 chars, so put search words first.
  - Before v2.1.296 the cap was 2,048 chars, which cut today's `search_cards` doc (2,918 chars) in the middle of its filter list.
  - A server can set `_meta["anthropic/alwaysLoad"]: true` on a tool to load it upfront (v2.1.285+).
- **Claude Code output limits.** A warning at 10k tokens, a 25k-token cap (`MAX_MCP_OUTPUT_TOKENS`), and text results over 50k chars are saved to a file ([docs](https://code.claude.com/docs/en/mcp#mcp-output-limits-and-warnings)).
- **Claude Code delivery of instructions (observed, not documented).**
  - In the local v2.1.299 binary they arrive in a meta (system-reminder) message headed `# MCP Server Instructions`, built from an `mcp_instructions_delta` attachment, not inside the system prompt.
  - They are therefore probably not removed by `--system-prompt`.
- **Claude apps (claude.ai, Desktop, mobile, Cowork).**
  - Tool access is "Auto" (the default), "Always available" or "On demand" ([help center](https://support.claude.com/en/articles/13730515-manage-claude-s-tool-access)).
  - Results are capped at ~150k chars ([connector docs](https://claude.com/docs/connectors/building)).
  - Directory review requires a `title` and `readOnlyHint`/`destructiveHint`, which "determine auto-permissions". Descriptions must "state precisely what the tool does and when to invoke it" and must not tell Claude how to behave beyond that ([review criteria](https://claude.com/docs/connectors/building/review-criteria)).

**Unknown**
- **Claude apps:** whether claude.ai and Desktop inject `instructions`, where they put them, and whether they truncate them. Also what the model sees for a deferred connector in Auto or On-demand mode. None of this is documented.
  - The MCP blog's check: connect the server and ask the model whether it has special instructions from that server.
- **Weight of a meta-message.** How much a meta-message instruction weighs against the client's own system prompt when the model decides whether to call a tool.
- **Keyword ranking.** How Claude Code's `ToolSearch` ranks keyword matches between tool name and description.
- **Other clients** (Cursor, ChatGPT, Codex). OpenAI's analogue: "the namespace helps the model choose what to load; the function description helps it use the loaded tool correctly" ([OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling)).

## 2. Rules

1. **Put the trigger ("use when …") in `instructions`.** Under Claude Code's default tool search, the instructions and the tool names are all the model sees until it searches. Name the task categories in the user's words, and name the tools so `ToolSearch` can select them. ([CC docs](https://code.claude.com/docs/en/mcp#for-mcp-server-authors), [tool search tips](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool#optimization-tips))
2. **Say why the model's memory is not enough, and add "even when you feel confident".** Anthropic says this about Sonnet 5.5, which "sometimes answers from its training knowledge" ([Prompting Sonnet 5.5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-sonnet-5-5#tool-use-in-chat-and-knowledge-work)). For this server, what memory gets wrong is errata, new cards, and colour identity or legality.
3. **Be a little pushy, but not shouty.** Claude tends to "undertrigger", so list the concrete triggers ([skill-creator](https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md)). Avoid "CRITICAL: You MUST": recent models overtrigger on it ([prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices#tool-usage)). Remove anything that reads as "minimize tool calls".
4. **Instructions carry cross-tool workflow, constraints and relationships.** They should not repeat the descriptions, make marketing or superiority claims, give unrelated behaviour rules, or turn into a manual ([MCP blog](https://blog.modelcontextprotocol.io/posts/2025-11-03-using-server-instructions/)). One added sentence claiming a tool is the most effective raised its use more than 7×, and combined edits more than 10× ([Faghih et al., EMNLP 2025](https://arxiv.org/abs/2505.18135)), but review rejects such claims.
5. **Each description stands alone:** what the tool does, when to use it (and when not), what it returns, its limits, and what its parameters mean. Start with key terms, as in the skills pattern "X. Use when …" ([define tools](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools#best-practices-for-tool-definitions), [skill descriptions](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices#writing-effective-descriptions)). Descriptions may load without instructions, and instructions may be dropped, so a one-line trigger in both is justified.
6. **Keep descriptions targeted, and don't repeat the parameter schema in prose.** 97% of 856 MCP tools had description smells. Fixing them added a median +5.85 points of task success but +67% steps, and trimmed variants matched the full ones ([arXiv 2602.14878](https://arxiv.org/abs/2602.14878)). Today `FILTER_DOC` (~265 tokens) appears twice and duplicates the schema's parameter descriptions.
7. **Use examples only where use isn't obvious.** For this server that is query phrasing. 1–5 realistic examples improved complex-parameter accuracy from 72% to 90% ([Advanced tool use](https://www.anthropic.com/engineering/advanced-tool-use)).
8. **Use a namespace and clear names.** Claude Code shows tools as `mcp__mtg-search__search_cards`, so the server name `mtg-search` is part of the signal. Names must be ≤64 chars ([Writing tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents), [review criteria](https://claude.com/docs/connectors/building/review-criteria)).
9. **Make responses token-efficient.**
   - Use concise defaults with a `detail` switch (in Anthropic's example, 72 vs 206 tokens), and keep `limit` defaults and pagination/truncation that steer toward small targeted searches.
   - Use readable names, not IDs, and give actionable errors ([Writing tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents)).
   - State the rough size of a result so the model can budget. In EXP-007, `search_cards` returned a median of ~2,000 tokens and `get_card` ~300.
10. **Annotate the tools:** `title`, `readOnlyHint: true`, `openWorldHint: false`, `idempotentHint: true`. Read-only tools skip per-call confirmation in Claude apps, so the user sees fewer prompts ([review criteria](https://claude.com/docs/connectors/building/review-criteria), [annotations post](https://blog.modelcontextprotocol.io/posts/2026-03-16-tool-annotations/)).
11. **Upfront vs deferred is a measurable trade-off.** Anthropic suggests keeping the 3–5 most-used tools non-deferred ([tool search](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool#deferred-tool-loading)). For a niche server, deferral plus good instructions is the cheap default. `alwaysLoad` on `search_cards`/`get_card` is the fallback if the deferred arm under-triggers.
12. **Evaluate on realistic held-out tasks, and in each client.** Track calls, tokens and errors, and read the transcripts ([Writing tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents)). Other things that change tool use:
    - Effort and thinking: thinking off or low effort reduces tool use ([Prompting Sonnet 5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-sonnet-5#tool-use-triggering)).
    - For GPT-4.1, three agentic reminders, one of them "use your tools rather than guess", raised SWE-bench Verified by close to 20% ([OpenAI GPT-4.1 guide](https://developers.openai.com/cookbook/examples/gpt4-1_prompting_guide)).

## 3. Proposed text

All proposed texts still pass the wording and soft-metric checks in `tests/test_server.py`. Optionally, have the server append `Card data as of YYYY-MM-DD.` to the instructions at startup, taking the date from the data snapshot.

| text | old chars (~tok) | new chars (~tok) |
|---|---|---|
| `instructions` | 293 (73) | 946 (237) |
| `search_cards` | 2,918 (729) | 1,526 (382) |
| `find_similar` | 2,065 (516) | 582 (146) |
| `get_card` | 285 (71) | 619 (155) |
| `list_tags` | 463 (115) | 498 (125) |
| **total** | **6,024 (1,506)** | **4,171 (1,043), −31%** |

**Session-start cost.**
- **Deferred mode (Claude Code default):** only the instructions load, +164 tokens over today.
- **Upfront mode:** the full `tools/list` payload drops from ~13,970 to ~11,460 chars (~3,490 → ~2,870 tokens).
  - Most of the rest is the input schema (~950 tokens each for `search_cards` and `find_similar`).
  - Dropping pydantic's per-property `title` and the `anyOf [T, null]` wrappers would save another ~200 tokens per filter tool (a code change).

`instructions`
```text
Magic: The Gathering (MTG) card search and lookup over the full Oracle card database (Scryfall data, which can include cards printed after your training data). Use these tools whenever a task involves choosing, suggesting or checking cards: building, upgrading or cutting a Commander/EDH or other deck, "cards that do X", replacements for a card, or a card's exact text, rulings, colour identity or format legality. Use them even when you feel sure from memory: errata, new cards, and colour-identity or legality slips are the usual errors.
Efficient use: one search_cards call per deck need, with color_identity (and format) set, returns candidates that are already in colour and legal; send 2-4 queries in that call for a broad need. get_card checks one card you named from memory or whose rulings matter. find_similar gives alternatives to a known card, e.g. for a swap. list_tags only helps find role vocabulary. Rankings use rules text only.
```

`search_cards` (drop `FILTER_DOC`/`OUTPUT_DOC`; the parameter descriptions already carry the filter details)
```text
Search every Magic: The Gathering (MTG) card by what it does: semantic search over Oracle rules text, with filters for Commander colour identity, format legality, card type and mana value. Use it to find candidates for a deck need (removal, ramp, card draw, sacrifice outlets, answers to fliers, ...) rather than listing cards from memory. With color_identity (and format="commander") set, every result is in colour and legal, so no separate check is needed.

Phrase each query the way the card itself is worded, not the way a player describes it, e.g. "Destroy target creature with flying.", "Whenever another creature you control dies, draw a card." Put colour, format, type and mana-value constraints in the filter arguments, not in the query text.
- Narrow effect: send 1 precise query.
- Open-ended need: send 2-4 `queries`, each a *different* mechanism, e.g. "I keep losing to fliers" -> ["Reach", "Destroy target creature with flying.", "Creatures you control gain reach until end of turn."]. Their rankings are fused.
Write one clean sentence per query, not a whole invented card, and don't add the user's own wording as a query.

Results are ranked by text relevance only, never by how widely played a card is. Filters apply before ranking, so even narrow filters return the best matches. Default: 15 cards, compact (oracle text cut at ~250 characters per face, plus roles, legality and Game Changer markers), about 2,000 tokens. Use limit 30-50 only to survey a role; use get_card or detail="full" for complete text.
```

`find_similar`
```text
Find Magic: The Gathering (MTG) cards that do what a given card does: replacements, upgrades and redundancy. Use it when the user wants "more cards like X", a stand-in for a card that is off-colour or not legal in their format, or, in a swap, an in-colour alternative to the card being cut. `card_name` is fuzzy-matched; that card's Oracle text is the query and the card itself is excluded. Takes the same filters as search_cards, e.g. find_similar("Rhystic Study", color_identity="BR", format="commander"). Ranked by rules-text similarity only; same compact output as search_cards.
```

`get_card`
```text
Look up one Magic: The Gathering (MTG) card by name (fuzzy: "fire ice", "jace mind sculptor" or one face of a double-faced card all work). Returns the current Oracle text of every face, mana cost and value, type line, P/T, colours and colour identity, legality in every format, markers (Game Changer, Reserved, Commander-eligible), functional tags, official rulings and the Scryfall link. Use it before recommending or judging a card you know only from memory, and when exact wording, a ruling or legality decides the answer. About 300-1,500 tokens per card; cards found with filtered search_cards need no second check.
```

`list_tags`
```text
List Scryfall oracle tags: community labels for what MTG cards do (e.g. "removal", "mana-rock", "free-sacrifice-outlet", "draw-engine"), with card counts and descriptions. Use it rarely: to find the usual name for a deck role before wording a search_cards query, or to see example cards for one tag. `prefix` filters by slug prefix (falling back to substring matches); an exact slug also lists up to 15 example cards (alphabetical). The same tags appear on the `roles:` line of every search result.
```

**Beyond text (code changes, not proposed here):**
- Add the annotations from rule 10.
- A batched `check_cards(names, color_identity, format)` would validate 20 remembered picks in one call instead of 20 `get_card` calls.
- The remaining EXP-007 errors (Game Changer limit, already in deck) need deck context that no current tool has.

## 4. Hypotheses for an A/B on the precon benchmark

**Setup.**
- Change the proxy to serve the real server's `instructions` and descriptions, under the server name `mtg-search`.
- In every arm, drop `TOOLS_TEXT` and the tool hint, and keep the budget enforced in the proxy but not stated in the prompt. A stated "at most N tool calls" may itself suppress use. The server's text is then the model's only tool guidance, as in real use.
- Arms: `srv-old` (today's `server.py`), `srv-new` (section 3), and `srv-new` with `ToolSearch` enabled (the deferred default; the isolation check must allow `ToolSearch`). Keep `none` and `mcp-hint` as references.
- Run 14 test precons × 3 repeats, with paired bootstrap CIs. Record Claude Code's effort level.

1. **H1 trigger.** `srv-new` gets ≥1 tool call in ≥50% of episodes and a mean of ≥1 call per ADD and SWAP episode, against ≈0 for `srv-old`. Secondary: `find_similar` is used in ≥25% of SWAP episodes (0 in EXP-007). Refuted if the share stays below 20%.
2. **H2 validity.** The ADD invalid-pick rate under `srv-new` is ≤4% (`none` 9.6%, `mcp` 8.2%, `mcp-hint` 1.8%), mostly through fewer off-colour picks (20/280 in `none`). The paired difference vs `none` has a CI excluding 0.
3. **H3 deferred loading.** With `ToolSearch` on, `srv-old` stays near 0 calls, while `srv-new` keeps at least ⅔ of its upfront-mode call share. If not, test `alwaysLoad` on `search_cards` and `get_card` as a further arm.
4. **H4 tokens.** In upfront mode, `srv-new` lowers first-request prompt tokens by ≥400 vs `srv-old` (about 460 expected), keeps tool-output tokens per episode ≤5k (median), and ADD nDCG@10 stays within 0.03 of `mcp-hint` (0.235).
5. **H5 popularity drift.** Episodes in `srv-new` that call tools pick fewer staples: ADD popularity drift ≥3.0 (`none` 2.4, `mcp-hint` 3.5), with ADD nDCG@10 ≥ `none` (0.189).

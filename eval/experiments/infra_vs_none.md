# EXP-013: Search infrastructure vs none vs everything in context

**Status: pre-registered 2026-10-15, before the main run.** Smoke runs only so far (`eval/agentic/runs/cc-smoke/`).

## Question (owner)

Ground the retrieval comparison in **"an agent without any search infrastructure vs one that has it"**. The no-infrastructure agent gets grep only: no BM25 tool, and no BM25 ranking inside grep. Add an **in-context retrieval** arm, where all card data is loaded into the agent up front. Track token usage in every arm.

**Owner hypothesis (H1):** the agent with retrieval tools uses the fewest tokens.

This replaces EXP-006 condition F (grep + BM25). F gave the lexical agent a second ranked retriever, and its significant edge came from regex-labelled tasks (see `agentic_search.md`, label-source check).

## Arms

All three arms use the same agent: headless Claude Code (`claude -p`), `claude-sonnet-5-5[1m]` (Sonnet 5.5 with its 1M-token context), on the owner's subscription with no API key. Each session is isolated as in the precon bridge:
- an empty temporary directory;
- no built-in tools: no web search or fetch, no file access, no shell;
- `--strict-mcp-config` with only our proxy server (A, G) or no server at all (I);
- the `init` event of every session is checked, and the episode aborts if any other tool is present.

| arm | infrastructure | tool / context |
|---|---|---|
| **A** search | yes | `search(query, filters?)`: mLateOn (exact) plus metadata filters, top 15 per call (EXP-006 A) |
| **G** grep | none | `grep(pattern, offset?)`: case-insensitive regex over the plain card text, matched line by line as in grep. Matches come in file order (alphabetical by name, with no correlation to popularity: Spearman ρ = 0.02 against EDHREC rank), 15 per call, with an offset for paging. No ranking and no filter parameters. |
| **I** in context | none, no tools | the plain card text of every card the task allows, in the system prompt |

**Common to all arms:**
- **"Card text"** is the same `doc_text` everywhere: name and mana cost, type line, rules text, P/T and keywords. It is what mLateOn indexes, what grep searches and what I reads. It contains no tags, popularity or price.
- **Task constraints** (colour identity, format, type, mana value: deck tasks only) are applied by the environment. A and G have them applied to every call; I's card file contains only the allowed cards. Only A can add further filters of its own.
- **Protocol.** At most 5 tool calls (A, G), then a ranked final list of at most 20 card names. The system prompts are EXP-006's, varying only in the tool paragraph. G's paragraph adds the per-line matching rule and one example file entry (Serra Angel, which is not relevant to any task). A coding agent would look at the file before grepping it; this was added after the smoke run.
- **I's corpus is too big for one context.** The whole file is about 7.5M characters, roughly 3.1M Sonnet 5.5 tokens at 2.42 characters per token. I therefore reads it in alphabetical parts of at most 1.8M characters (≤ ~750k tokens), and each part returns up to 20 candidates. With more than one part, one merge call ranks all the candidates, shown with their full text, into the final list. A task without constraints takes 5 parts plus the merge.

## Tasks and label sources

There are 113 tasks: `eval/agentic/tasks.yaml` (65 deck, 15 attr, 25 known) plus `tasks_known_easy.yaml` (8 known). Labels are incomplete proxies, and how they were built favours one arm or the other, so every result is reported **per stratum**.

| stratum | n | label | bias |
|---|---|---|---|
| deck / deckneeds | 30 | oracle tags OR regex over `doc_text` | partly regex, so partly favours G |
| deck / paraphrase-need | 35 | regex over `doc_text` only | **favours G**: grep can rebuild the labelling rule |
| attr | 15 | oracle-tag closure ∩ attribute filter | independent of both tools |
| known | 25 | one target card | the set was **selected where single-shot BM25 missed** (rank > 3), which disfavours lexical search |
| known-easy | 8 | one target card | the drafts dropped only because BM25 already ranked them top 3; added back here to offset that selection |

## Metrics

- **Tokens per episode (H1):** total = input + cache writes + cache reads + output (thinking included in output). Each component is also reported. They are summed over every session of an episode from Claude Code's per-model usage. Also reported: Claude Code turns, wall seconds and the notional list price. Runs are on the subscription, so nothing is actually spent.
- **Quality:**
  - deck and attr: nDCG@10, P@10 and final R@20 (`analyze.py`);
  - known: found-in-final and RR;
  - A and G only: recall@t and success@t.
- **Statistics:** paired over tasks, with 10k bootstrap CIs and two-sided sign tests.

## Hypotheses and decision rules (fixed before the main run)

- **H1, tokens.** Per-task total tokens: A < G, and A < I. It is supported if the paired sign test gives p < 0.05 for each pair. The A–I comparison uses I's sample only.
- **H2, quality with vs without infrastructure (A vs G).**
  - **Primary:** known-item found-in-final over known + known-easy, 33 tasks. These labels are independent of both tools, and including known-easy offsets the BM25 selection.
  - **Secondary:** attr nDCG@10 (15 tasks, tag labels).
  - Deck strata are reported, but no conclusion is drawn from paraphrase-need alone.
- **I, in context.** Run on a seeded stratified sample of 12 tasks (4 deck, 4 attr, 4 known; `random.Random(13)`). The sample is: `hw-donate-need, hw-flicker-end-step-need, need-red-enchantment-answers, need-survive-wipe, attr-1cmc-red-burn-instant, attr-2cmc-blue-counter, attr-black-cheap-tutor, attr-black-creature-discard, known-angrath-s-rampage, known-guardian-augmenter, known-nyxborn-hydra, known-smokebraider`. The full I run is **deferred to the owner**: it would read about 302M tokens (93 of 113 tasks have no constraints, about 3.1M tokens each), about $97 notional with the 1h prompt cache warm.
- **Repeats.** 1 repeat for each arm. Claude Code does not expose the sampling temperature.

## Smoke (`runs/cc-smoke/`, not part of the results)

- **Tasks:** 3 for A and G (known-abandoned-air-temple, need-fliers-mono-green, attr-1cmc-green-dork); I on the same 3.
- **A:** 3–5 turns, 6–11 s, 8.6k–17.9k tokens.
- **G:** 4–7 turns, 9–14 s, 13.2k–31.7k tokens.
- **G missed the known item.** It wrote patterns that span lines and assumed a cost of {4} where the card says {3}{W}, then returned an empty list. This led to the per-line note and the example entry. The smoke ran on the earlier G prompt.
- **I:**

  | task | parts | tokens | wall | result |
  |---|---|---|---|---|
  | mono-green task | 1 | 678k | 28 s | good anti-flier picks |
  | 1-mana green dork | 5 + merge | 3.11M | 62 s | Llanowar Elves, Elvish Mystic, Fyndhorn Elves … |
  | known item | 5 + merge | 3.11M | 48 s | target ranked first |

  The second unconstrained task read every part from the 1h cache: notional $0.34 instead of $12.46.

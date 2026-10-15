# EXP-013: Search infrastructure vs none vs everything in context

**Status: done 2026-10-15 (A and G on all 113 tasks; I on the pre-registered 12-task sample; the owner cancelled the full I run).** The design below was pre-registered in commit `1e04f45` before the main run.

**Answer.**
- **H1 (tokens): supported.** The agent with search infrastructure used the fewest tokens.
  - Against grep, G took 1.46× A's tokens (median ratio over tasks; 96 of 113 tasks higher, sign p = 1.5e-14).
  - Against reading everything in context, I took 283× A's tokens (12 of 12 tasks, p = 0.0005).
- **Search vs grep quality: about level.**
  - Known item (primary): found 97% vs 94% (p = 1).
  - Deck needs: tied in both label strata.
  - Attribute tasks: A ahead on nDCG@10 by −0.054 [−0.108, −0.008] for G, but the sign test is not significant (p = 0.45). On final R@20, A beat G 10 tasks to 2 (p = 0.04).
- **What grep costs.** Grep's real costs are effort and speed: 36% more tool calls, 0.8 more turns and 2.4 s more per task.
- **In context: best quality on the small sample, at a very high price.** Reading all ~3.1M tokens of card text gave the best quality on the 12-task sample (better than A on 5 tasks, worse on 1; not significant). It took about 300× the tokens and 6.5× the wall time.

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

## Results

Run data:
- `eval/agentic/runs/cc-main/` (A, G: 226 episodes, 0 failures, 0 retries);
- `eval/agentic/runs/cc-main-I/` (I: 12 episodes, 0 failures);
- metrics: `eval/agentic/results/cc_main.json` (`analyze_cc.py`).

The model was `claude-sonnet-5-5[1m]` on Claude Code 2.1.300, 1 repeat. Labels are the original incomplete ones, so compare arms, not absolute levels.

### Tokens, turns and time per episode (H1)

| arm | episodes | total tokens, mean / median | output tokens | Claude Code turns | tool calls | wall s | tool output chars |
|---|---|---|---|---|---|---|---|
| **A** search | 113 | **12.1k / 11.1k** | 622 | **4.14** | **2.14** | **8.6** | 5.8k |
| **G** grep | 113 | 20.9k / 16.6k | 836 | 4.90 | 2.90 | 10.9 | 10.1k |
| **I** in context | 12 | 2.91M / 3.11M | 3,456 | 5.58 (5 parts + merge) | 0 | 53.6 | — |

- **Total tokens** = input + cache writes + cache reads + output, summed over every session of an episode. Cache reads were 66% (A), 70% (G) and 98% (I) of the total.
- **Paired tests.**
  - **A vs G:**
    - tokens G − A = +8.8k [+7.0k, +10.6k]; G higher on 96 of 113 tasks (sign p = 1.5e-14); median ratio 1.46;
    - turns +0.76 (p = 1e-7);
    - wall +2.4 s (p = 2e-7).
  - **A vs I (12 tasks):** tokens higher on 12 of 12 (p = 0.0005), median ratio 283; wall 6.5×.
  - **G vs I:** median ratio 195.
- **Why grep costs more.** G makes 36% more calls (328 vs 242), because it has no ranking: it pages, or reformulates its regexes. Each call also returns about 28% more text (3.5k vs 2.7k characters), because grep echoes the matching line.
- **Why I costs what it costs.**
  - I's cost is fixed by the corpus: every task with no constraints reads the whole file, about 3.1M tokens.
  - The prompt cache made the parts cheap to re-read: notional list price $6.88 for the 12 episodes, against $2.38 for all 113 A episodes and $3.58 for all 113 G episodes. In tokens, though, I uses roughly 300× more.
  - All of this ran on the subscription, so the notional figures are list-price equivalents, not spend.

### Quality, A vs G (all 113 tasks)

| stratum (label source) | n | metric | A | G | G − A [95% CI] | tasks G better / A better | sign p |
|---|---|---|---|---|---|---|---|
| known + known-easy (**primary**) | 33 | found in final | 0.970 | 0.939 | −0.030 [−0.121, +0.061] | 1 / 2 | 1.0 |
| known (chosen against BM25) | 25 | found in final | 0.960 | 0.960 | 0 | 1 / 1 | 1.0 |
| known-easy | 8 | found in final | 1.000 | 0.875 | −0.125 | 0 / 1 | 1.0 |
| attr (tag closure; **secondary**) | 15 | nDCG@10 | 0.923 | 0.870 | −0.054 [−0.108, −0.008] | 2 / 5 | 0.45 |
| attr | 15 | final R@20 | 0.326 | 0.282 | −0.044 [−0.089, −0.007] | 2 / 10 | 0.04 |
| deck / deckneeds (tags or regex) | 30 | nDCG@10 | 0.805 | 0.824 | +0.019 [−0.057, +0.104] | 11 / 12 | 1.0 |
| deck / paraphrase (regex only) | 35 | nDCG@10 | 0.658 | 0.633 | −0.025 [−0.105, +0.051] | 10 / 12 | 0.83 |
| deck / paraphrase (regex only) | 35 | final R@20 | 0.121 | 0.146 | +0.025 [−0.001, +0.054] | 16 / 14 | 0.86 |

**Known-item misses:**
- **A missed Guardian Augmenter.** Its three searches described the card correctly, but it stopped with a list of 6 other hexproof effects.
- **G missed War Behemoth.** `^3/6$` matches 63 cards and the target is 56th in file order. G paged to offsets 15 and 30, then ran out of calls.
- **G missed Shadow of Mortality.** Its patterns guessed wording the card does not use, such as "life you've lost" and a {15} cost. The card says "costs {X} less to cast, where X is the difference" and costs {13}{B}{B}.
- **Agents rarely searched by card name:** A did in 2 of 33 known episodes, G in 1.

### Quality, in context (I) on the 12-task sample

The score below is nDCG@10, or found-in-final for known items. The sample happened to favour G over A (0.882 vs 0.794), whereas on all 113 tasks A and G are level, so treat these numbers as indicative.

| | A | G | I |
|---|---|---|---|
| mean score, 12 tasks | 0.794 | 0.882 | **0.919** |
| I better / worse than the arm | 5 / 1 | 5 / 1 | — |
| known items found (4) | 3 | 4 | 4 |
| attr nDCG@10 (4) | 0.804 | 0.785 | 0.888 |

In context, the model sees every candidate. It does well when the request is clear, as with 1-mana dorks, counterspells and named-effect known items. Its only weakness is cost.

### Reading

1. **H1 holds.** For the same task and protocol, the infrastructure agent is the cheapest: 1.46× fewer tokens than grep and about 280× fewer than reading everything in context. It is also the fastest (8.6 s vs 10.9 s vs 54 s).
2. **The value of the search infrastructure here is efficiency more than reach.** With a strong model, grep reaches about the same final quality on deck needs and known items, at about 1.5× the tokens and 25% more time. Its weak spot is tasks where the wording is generic and a ranking matters (attr R@20, the War Behemoth miss): without a ranking it must page through alphabetical matches.
3. **Reading everything is the quality ceiling, not a practical design.** About 3.1M tokens and about 54 s per request is two orders of magnitude more than either tool.
4. **Change from EXP-006.** There, a GLM agent with grep + BM25 appeared to beat mLateOn on final R@20. Here, with grep only and a stronger model, that advantage is gone, even on the regex-labelled tasks that favour grep (+0.025, p = 0.86).

### Caveats

- **Repeats.** 1 repeat, and Claude Code exposes no temperature, so run-to-run noise is unmeasured. EXP-010 saw swings of ~20 points in search rate between identical arms.
- **I sample.** I ran on 12 tasks. **The full I run is cancelled (owner, 2026-10-15)**: in-context retrieval will not be pursued further, so the I numbers stay a 12-task sample.
- **I's design.** I splits the file into parts and merges. A model with a context large enough to hold the 3.1M-token file in one pass might rank differently. The token count would be about the same.
- **Labels.** Still the original incomplete ones. Pooled judging (issue #2) would make the deck strata fairer to every arm.
- **Grep echo.** G's tool output includes the matched line, which a coding agent's grep would also show. Without it, G would use about 20% fewer tool-output tokens, still more than A per episode, because it makes more calls.


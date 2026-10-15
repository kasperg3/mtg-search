# EXP-018: Prompt ablation of search vs grep (EXP-013) with minimal, identical instructions

**Status: pre-registered 2026-10-15, before any episode of this run.** Results go below the pre-registration without changing it.

## Question (owner)

"Investigate whether the experiment where the agent only had access to grep also had instructions about how to structure the search … would it be worth it to rerun it using no prior knowledge about structuring the search, and only measuring the retrieval quality of the method and not the prompt that describes how to search efficiently?", then "run the ablation".

**Why.** In EXP-013 the grep agent (G) had more method-specific guidance than the search agent (A); see the correction in `infra_vs_none.md`.
- **G's extras:**
  - an example file entry showing the card layout and a per-line matching note, both added after the smoke run;
  - an example pattern in rules-text wording;
  - matching that let `\n` span the lines of a card, which grep cannot do.
- **Use of the layout:** 49% of G's patterns used the layout to stand in for filters, 88% on attribute tasks.
- **A's guidance:** A was told only that its search takes natural language.

**Question:** with identical minimal instructions for both tools, and tool descriptions limited to mechanics, does the EXP-013 result still hold? That result was the same relevance, with grep using more tokens.

## Design

- **Arms:**
  - **A0:** the `search` tool (mLateOn, exact, top 15, same filters);
  - **G0:** the `grep` tool (regex, 15 per call in file order, `offset` paging, no filters), now matched **truly line by line** (`env.grep(..., per_line=True)`).
- **Prompts** (`eval/agentic/cc_agent.py`, `SYSTEMS`, `TOOLSETS`):
  - **System prompt:** the same for both, with only the tool name changed. It covers the task protocol only: the 5-call budget, `final_answer` with up to 20 names, best first, only cards from the results, exact names, best guess first for a specific card. There is no search advice (EXP-013's "refine, rephrase, or explore a different angle" paragraph is gone).
  - **Tool descriptions:** mechanics only, with no examples and no file layout. Both name the same card fields (name, mana cost, type line, rules text, power/toughness, keywords).
    - search: "Ranked search … Returns the 15 best-matching cards";
    - grep: "Case-insensitive Python regular expression, matched separately against each line … Returns the number of matching cards and 15 of them in file order";
    - A0's filter parameters describe their syntax without examples.
- **Unchanged from EXP-013:** the user message (the request, plus the constraints the environment applies), the results format, the call budget enforced by the proxy, and the isolation checks.
  - **Agent:** headless Claude Code, `claude-sonnet-5-5[1m]`, on the subscription.
  - **Repeats:** 1.
- **Tasks:** 30, `random.Random(18)` over sorted ids, 10 per kind (`analyze_ablation.sample_ids`):
  - deck: `need-fliers-mono-green, hw-tax-spells-cost-more-need, need-ub-card-advantage, need-modern-blue-tempo, need-colorless-draw, tag-tutor-creature-need, hw-reanimate-own-need, need-protect-commander, need-team-haste, tag-hate-graveyard-need`;
  - attr: `attr-2cmc-blue-counter, attr-colorless-draw-artifact, attr-black-cheap-tutor, attr-white-instant-blink, attr-white-anthem-enchantment, attr-2cmc-rock, attr-green-land-ramp-sorcery, attr-1cmc-red-burn-instant, attr-white-wrath-sorcery, attr-green-cheap-fog`;
  - known: `known-easy-spoils-of-the-vault, known-forsaken-monument, known-dryad-militant, known-easy-biomancer-s-familiar, known-eager-glyphmage, known-pulse-of-the-forge, known-wall-of-faith, known-lagomos-hand-of-hatred, known-ghitu-amplifier, known-street-sweeper`.
- **Comparison base:** the EXP-013 A and G episodes on the same 30 tasks (`runs/cc-main`).
- **Command:**

  ```
  uv run python eval/agentic/cc_agent.py --run cc-ablation --conds A0,G0 --ids <the 30 ids> --workers 4 --experiment EXP-018
  ```

## Relevance grading

- **Rule:** the EXP-016 rule. A card in the deduplicated top 10 counts if it is on the task's list of correct cards, or the relevance grader graded it ≥ 1.
- **Reuse:** grades from EXP-016 (`runs/judge-full`) are reused.
- **New grades:** the cards in A0/G0 final top-20 lists that EXP-016 did not grade and that are not already on the list are graded with the same grader: `judge_cc.py`, EXP-016 rubric, `claude-sonnet-5-5`, 15 cards per call, one pass, `--only-new judge-full`, run `judge-ablation`.
- **Scope:** the 20 deck and attribute tasks only. Find-this-card tasks need no grading.

## Metrics and tests (fixed before running)

**Primary, under minimal prompts:**
1. **Tokens per task.** The per-task ratio G0 / A0 (median), and a two-sided sign test on G0 − A0 over the 30 tasks.
2. **Relevance.** P@10 G0 − A0 over the 20 deck and attribute tasks: mean, 10k paired bootstrap 95% CI (seed 0), and a two-sided sign test.

**Secondary:**
- **Find-this-card:** target found in the final list, G0 − A0 over 10 tasks.
- **Prompt effect within each method:** A0 vs A and G0 vs G on tokens (ratio), turns, tool calls, P@10 and found.
- **Gap change:** (G0 − A0) − (G − A) per task, for P@10, found and the log token ratio.
- **Grep diagnostics** (G vs G0): the shares of patterns that use the file layout, contain `\n`, match nothing, page with `offset`, error, or use alternation.

**Expectations (not decision rules):**
- G0 needs more calls and tokens than G, and loses relevance on attribute and find-this-card tasks, where G leaned on the layout.
- A0 ≈ A.
- So the token gap widens, and the relevance tie may turn into an A lead.

**Reading:**
- **Holds:** EXP-013's conclusion ("search finds the same cards as grep, with fewer tokens") holds if G0 uses more tokens than A0 (sign p < 0.05) and the P@10 difference does not favour G0 with a CI excluding 0.
- **Challenged:** the conclusion is challenged if G0 uses fewer tokens than A0, or beats A0 on P@10 with a CI excluding 0.

## Cost

About 60 episodes, about 1M tokens and about 10 minutes with 4 workers, plus at most about 30 grading calls. Everything runs on the subscription.

## Limits stated up front

- **Small:** 30 tasks, 1 repeat, and P@10 is near its ceiling in EXP-013 (about 0.97–0.99).
- **The model's own knowledge stays.** Sonnet knows the rules-text templates and card names. No prompt can remove that, so this ablation measures prompt guidance, not prior knowledge.
- **The minimal prompts are not the shipped server's.** The shipped server tells the agent to word queries like rules text.

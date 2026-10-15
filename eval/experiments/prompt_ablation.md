# EXP-018: Prompt ablation of search vs grep (EXP-013) with minimal, identical instructions

**Status: done 2026-10-15.** Pre-registered in commit `0f6568d` before any episode of this run; the pre-registration below is unchanged, and Results and Reading follow it.

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

## Results (2026-10-15, `runs/cc-ablation`, grades `runs/judge-ablation`)

**Status: done.**
- **Runs:** 60 tasks (30 per arm), no failures, no isolation alerts.
- **New grades:** 66 new cards graded in 14 grading calls; every card in a top 10 has a list entry or a grade.
- **Pre-registered reading:** neither "holds" nor "challenged". The relevance condition favours search, with a CI excluding 0. The token condition misses p < 0.05 (sign p = 0.099).

**Correction (2026-10-15, after the results).** The scorer resolved some answers to the wrong card: another card's face name beat a card's full name, so the answer "Lightning Bolt" was scored as "Emeritus of Conflict // Lightning Bolt" (fixed in `eval/agentic/env.py`; `eval/agentic/rescore.py` re-resolved every saved episode, and `runs/judge-facefix` graded the cards that newly counted). On these 30 tasks, 2 A0 answers and 2 of EXP-013's A and G answers changed. The numbers below are corrected. Before the fix they were: P@10 A 0.995, G 0.925, A0 0.970; G0 − A0 −0.26 [−0.43, −0.10]; G0 vs G −0.215 [−0.375, −0.07]; gap change −0.19 [−0.36, −0.03]; precision of returned cards A 0.995, G 0.980, A0 0.985; A0 on attribute tasks 0.990. G0 did not change. No sign-test count, p-value or conclusion changed.

| per task, 30 tasks | A (EXP-013) | G (EXP-013) | **A0** (minimal) | **G0** (minimal) |
|---|---|---|---|---|
| tokens (mean) | 11.8k | 20.4k | 9.7k | 15.7k |
| Claude Code turns | 4.03 | 4.80 | 3.77 | 4.73 |
| tool calls | 2.03 | 2.80 | 1.77 | 2.73 |
| seconds | 8.1 | 10.3 | 9.4 | 11.2 |
| P@10, 20 deck + attr tasks | 1.000 | 0.930 | **0.975** | **0.710** |
| specific card found, 10 tasks | 10/10 | 10/10 | 10/10 | 10/10 |

**Primary (minimal prompts):**
1. **Tokens:** G0 / A0 median ratio 1.69. G0 used more on 20 of 30 tasks, sign p = 0.099, so this is **not established at n = 30**. On the same 30 tasks EXP-013 had 1.46 and 25 of 30 (p = 0.0003).
2. **Relevance:** P@10 G0 − A0 = **−0.265 [−0.44, −0.10]**, sign p = 0.012 (A0 better on 10, G0 on 1, 9 tied). EXP-013 on the same tasks: −0.07 [−0.145, −0.015], p = 0.125.

**Secondary:**
- **Search barely needs the guidance.** A0 vs A: P@10 −0.025 [−0.065, 0], p = 0.5. A0 used slightly fewer tokens (median ratio 0.94, fewer on 28 of 30) and 0.27 fewer calls.
- **Grep loses relevance without it.** G0 vs G: P@10 −0.22 [−0.385, −0.075], p = 0.039. G0 used fewer tokens (median ratio 0.72), because it found less: 29% of its calls matched nothing (G: 7%), and its lists were shorter.
- **Gap change.** The P@10 gap (G0 − A0) − (G − A) moved by −0.195 [−0.37, −0.03], sign p = 0.065. The log token ratio did not change significantly (−0.10 [−0.26, +0.07]).
- **Find-this-card:** all four arms found all 10 targets.
- **Grep diagnostics** (G → G0):
  - layout features in patterns: 71% → 61%;
  - patterns with `\n`: 40% → 6% (these can no longer match);
  - patterns that matched nothing: 7% → 29%;
  - paging with `offset`: 8% → 4%.

**Exploratory (not pre-registered):**
- **Grep returns fewer cards, not wrong ones.** The precision of the cards it did return is about the same for every arm: A 1.000, G 0.985, A0 0.990, G0 0.978. G0's top 10 was shorter than 10 on 8 of 20 tasks, and empty on 2.
- **All of the loss is on attribute tasks.**
  - Deck needs: A0 0.950, G0 0.970, a tie.
  - Attribute tasks, such as "a 1-mana red burn instant": A0 1.000, G0 **0.450**.
- **Why attribute tasks fail.** Without the layout example, G0 assumed one-line records like the rendered results ("Name {cost} — Type — text"). It wrote patterns joining cost, type and rules text, such as `^[^{]*\{R\} — Instant.*deals \d+ damage to any target`, and those match nothing: a card's cost, type and rules text are on different lines. It then fell back to listing card names from memory (`^(Lightning Bolt|Shock|…)`). Search has filter parameters for exactly these attributes.

## Reading

- **What EXP-013's tie needed.** Grep matched search on relevance only when its prompt showed how the card file is laid out, and its matching could span lines.
- **With identical minimal instructions, the methods separate.** Search keeps its relevance (0.975). Grep falls to 0.71, because per-line grep cannot combine a card's mana cost, type and rules text unless they share a line.
- **On deck needs the two tie**, as before: the rules-text phrases a deck need calls for sit on one line.
- **Tokens.** Grep still tends to use more (median 1.7×), but that is not established at n = 30.
- **Revised conclusion:**
  - **Prompt-free method quality:** search is better; it finds the same deck-need cards and many more of the cards defined by attributes.
  - **EXP-013's "same cards, fewer tokens":** that described grep with a documented file layout.

**Limits:**
- 30 tasks, one repeat. The attribute stratum is 10 tasks.
- **The result depends on the file format.** A file with one card per line would let grep combine fields. The per-line rule is real grep's behaviour, but a grep user would normally look at the file first, which is what EXP-013's example entry stood in for.
- **The model's knowledge of card names** still rescued some grep tasks, through name lists.

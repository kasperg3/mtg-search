# EXP-019: Search vs grep on all 113 tasks, where the two grep arms differ only in the prompt

**Status: done 2026-10-15.** Pre-registered in commit `10d5374` before any episode of this run; the pre-registration below is unchanged, and Results and Reading follow it.

## Question (owner)

"Please redo the experiment so that we can update step 1 with the findings. Keep the old grep and state that the only difference is in the prompt between the two grep. Also state that the prompt for the embedding retrieval did not need any prompt tuning."

**Why.** EXP-018's minimal grep arm (G0) changed two things at once: the prompt, and the matching (truly per line). This run keeps EXP-013's grep tool exactly and changes only the prompt. It runs on all 113 tasks, so step 1 of the findings page can be updated like for like.

## Arms

| arm | tool | prompt | source |
|---|---|---|---|
| **A** | search (mLateOn, top 15, filters) | EXP-013 prompt (EXP-006's, never tuned) | EXP-013 `runs/cc-main` |
| **G** | grep, EXP-013's tool: regex over each card's whole text with `re.MULTILINE`, 15 per call in file order, `offset` paging, no filters | EXP-013 grep prompt: strategy paragraph, file-layout example, per-line note, example pattern | EXP-013 `runs/cc-main` |
| **A0** | the same search tool | minimal prompt (EXP-018 `SYSTEM_MIN`), mechanics-only tool description | this run |
| **G1** | **the same grep tool as G**: same matching, same tool description | minimal prompt, the same as A0's and G0's. The example pattern is dropped from the `pattern` description | this run |

- **G vs G1:** only the prompt differs. That covers the system prompt (strategy paragraph, layout example, per-line note) and the example pattern in the `pattern` parameter. The tool, its matching and its description are identical.
- **A0 vs G1:** the same minimal system prompt, and mechanics-only tool descriptions.
- **Unchanged from EXP-013:** the agent (headless Claude Code `claude-sonnet-5-5[1m]`, subscription, isolation checks), 1 repeat, the user message, the results format and the 5-call budget.

**Command:**
```
uv run python eval/agentic/cc_agent.py --run cc-promptonly --conds A0,G1 --workers 4 --experiment EXP-019
```

## Relevance grading

The EXP-016 rule: a card in the deduplicated top 10 counts if it is on the task's list or graded ≥ 1. EXP-016 and EXP-018 grades are reused. Cards in A0/G1 final top-20 lists that neither run graded, and that are not on the list, are graded with the same grader (`judge_cc.py --only-new judge-full,judge-ablation`, run `judge-promptonly`) on the 80 deck and attribute tasks.

## Metrics and tests (fixed before running)

Paired over tasks: mean difference, 10k bootstrap 95% CI (seed 0), two-sided sign test. Tokens are compared as per-task ratios (median), with a sign test.

1. **Prompt effect on grep, G1 vs G (primary).** P@10 over the 80 deck + attr tasks (also by kind); tokens, turns and tool calls over 113; specific card found over 33.
2. **Search needs no prompt tuning, A0 vs A.** The same metrics.
   - **Rule:** the claim "search needed no prompt tuning" is supported if the A0 − A 95% CI lower bound is above −0.05 for both P@10 (80 tasks) and found (33 tasks).
3. **Method under identical minimal prompts, G1 vs A0.** The same metrics.
4. **Reference:** EXP-013's G vs A on the same tasks.
5. **Grep diagnostics (G vs G1):** the shares of patterns that use the file layout, contain `\n`, match nothing, or page with `offset`.

**Expectations (not decision rules):**
- G1 loses relevance against G, mostly on attribute tasks, though less than G0, because cross-line matching is still available.
- A0 ≈ A.

## Cost

226 tasks (113 per arm), about 3M tokens and about 10–15 minutes with 4 workers, plus grading of the new cards. Everything runs on the subscription.

## Limits

- One repeat.
- P@10 is near its ceiling on deck tasks.
- The model's own knowledge of card wording and names is untouched.

## Results (2026-10-15, `runs/cc-promptonly`, grades `runs/judge-promptonly`)

**Status: done.**
- **Runs:** 226 tasks (113 per arm), no failures, no isolation alerts.
- **New grades:** 309 new cards graded in 58 calls; every top-10 card has a list entry or a grade.
- **Search rule ("search needed no prompt tuning"): met.**

**Correction (2026-10-15, after the results).** The scorer resolved some answers to the wrong card: another card's face name beat a card's full name, so the answer "Lightning Bolt" was scored as "Emeritus of Conflict // Lightning Bolt". The fix is in `eval/agentic/env.py`, and `eval/agentic/rescore.py` re-resolved every saved episode. In this run 5 answers changed (A0: 2, G1: 3), and 4 in EXP-013's (A, G). The 3 cards that newly counted were graded by the same grader (`runs/judge-facefix`, all grade 2). The numbers below are corrected. Before the fix they were: P@10 A 0.985, A0 0.981, G1 0.863, G 0.966; attribute tasks A 0.993, A0 0.993, G1 0.473, G 0.920; G1 vs G −0.104 [−0.161, −0.054]; G1 vs A0 −0.119 [−0.180, −0.061]. No sign-test count, p-value or conclusion changed.

| per task, 113 tasks | A: search, EXP-013 prompt | **A0: search, minimal prompt** | **G1: grep, minimal prompt** | G: grep, tuned prompt (EXP-013) |
|---|---|---|---|---|
| tokens (mean) | 12.1k | **10.6k** | **14.3k** | 20.9k |
| Claude Code turns | 4.14 | 3.98 | 4.55 | 4.90 |
| tool calls | 2.14 | 1.98 | 2.53 | 2.90 |
| seconds, mean (median) | 8.6 (8.3) | 10.3 (9.8) | 17.0 (11.3)* | 10.9 (10.1) |
| P@10, 80 deck + attr tasks | 0.986 | **0.983** | **0.865** | 0.968 |
| P@10, 65 deck tasks | 0.983 | 0.978 | 0.954 | 0.977 |
| P@10, 15 attr tasks | 1.000 | 1.000 | 0.480 | 0.927 |
| specific card found, 33 tasks | 0.970 | 1.000 | 0.848 | 0.939 |

\*Two G1 tasks took 159 s and 322 s; the median is 11.3 s.

**1. Prompt effect on grep, G1 vs G (primary).** Only the prompt differs.
- **P@10:** −0.103 [−0.160, −0.053], sign p = 0.0006 (G1 better on 5 tasks, G on 24, 51 tied). By kind: deck −0.023, attr −0.447.
- **Specific card found:** −0.091 [−0.242, +0.061], p = 0.45.
- **Effort:** G1 used fewer tokens (median ratio 0.72), turns (−0.35, p = 0.004) and calls (−0.37), because it found less. 29% of its calls matched nothing (G: 8%), and its lists were shorter (mean 12.7 cards on deck/attr tasks; 23 lists under 10, 1 empty).

**2. Search needs no prompt tuning, A0 vs A.**
- **P@10:** −0.004 [−0.015, +0.004], p = 1 (2 better, 2 worse, 76 tied).
- **Specific card found:** +0.030 [0, +0.091].
- **Effort:** A0 used slightly fewer tokens (median ratio 0.94) and 0.16 fewer turns.
- **Verdict:** both CI lower bounds are above −0.05, so the rule is met. Search's EXP-013 prompt was never tuned, and the minimal prompt does as well.

**3. Same minimal prompt, G1 vs A0.**
- **P@10:** −0.118 [−0.178, −0.061], sign p = 3.4×10⁻⁵ (G1 better on 4 tasks, A0 on 27). Deck −0.025, attr −0.520.
- **Specific card found:** −0.152 [−0.273, −0.030], p = 0.062.
- **Tokens:** G1 used more on 80 of 113 tasks (median ratio 1.24, p = 1.1×10⁻⁵).

**4. Reference, G vs A (EXP-013):** P@10 −0.019 [−0.051, +0.011], p = 0.42; tokens median ratio 1.46 (96 of 113).

**Grep diagnostics (G → G1):**
- patterns using the file layout: 50% → 53%;
- patterns with `\n`: 25% → 20%;
- patterns that matched nothing: 8% → 29%;
- paging with `offset`: 10% → 2%.

Without the layout example, G1 still writes layout-dependent patterns, but against a layout it has to guess, and a third of them miss.

## Reading

- **The tuned prompt is what made grep competitive.** With EXP-013's grep tool unchanged, removing the grep-specific prompt (layout example, matching note, strategy paragraph, example pattern) costs grep 10 points of P@10, and 45 points on attribute tasks.
- **Search needs no prompt tuning.** Its first, untuned prompt and the minimal prompt give the same relevance.
- **Like for like,** with the same minimal prompt for both tools, search finds more of the right cards than grep (0.98 vs 0.87 of the top 10, p = 3×10⁻⁵) with fewer tokens (10.6k vs 14.3k). It also finds every specific card, where grep finds 85%.
- **Grep can reach search's relevance on these tasks** only with a prompt that documents the card file, and then it uses about twice the tokens (20.9k vs 10.6k).
- **Deck needs stay close in every setup** (0.95–0.98): their rules-text phrases sit on one line, which grep handles well.

**Limits:**
- One repeat.
- The attribute stratum is 15 tasks.
- The model's own knowledge of card wording and names helps both tools and is untouched.
- The tuned grep prompt and EXP-013's results came from one round of tuning after a smoke run; more tuning might narrow the token gap.

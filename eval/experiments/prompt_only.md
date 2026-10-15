# EXP-019: Search vs grep on all 113 tasks, where the two grep arms differ only in the prompt

**Status: pre-registered 2026-10-15, before any episode of this run.** Results go below the pre-registration without changing it.

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

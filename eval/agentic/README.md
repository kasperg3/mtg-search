# Agentic multi-step search harness (EXP-006, EXP-013, EXP-015)

An agent acts as a deck-building search assistant. It gets up to 5 tool calls of 15 results each, then submits a ranked final list of up to 20 cards. Two conditions:

| cond | tool | |
|---|---|---|
| A | `search(query, filters?)` | search infrastructure: mLateOn |
| G | `grep(pattern, offset?)` | none: a regex over the plain card text, line by line, matches in file order (alphabetical), 15 per call; no ranking, no filter parameters |

`cc_agent.py` runs them (plus the in-context arm I) with headless Claude Code; `agent.py` runs A and G with GLM-5.3. EXP-006's SparseUp/BM25 conditions (B, C, D, F) were removed with those retrievers; their code is in git history up to commit 61071da and their data stays in `runs/main` and `runs/pilot`. Design and results: [`agentic_search.md`](../experiments/agentic_search.md), [`infra_vs_none.md`](../experiments/infra_vs_none.md), [`pooled_judging_smoke.md`](../experiments/pooled_judging_smoke.md).

Tool output never contains popularity, price or crowd signals. `Env.assert_no_leak` blocks `/edhrec|price|\$|rank|popular|salt/` in everything the harness writes. Card text is exempt, because card names can legitimately contain words such as "rank".

## Setup

- The key is `GLM_API_KEY` in the repo `.env`. It is never logged.
- The endpoint is `https://api.z.ai/api/paas/v4` (pay-as-you-go). The model is `glm-5.3` with `thinking: {type: enabled, effort: low}` (GLM-5.3 cannot turn thinking off). Temperature is 0.6.
- Search: `index/embeddings/late/lightonai__mLateOn/` and `index/models/lightonai__mLateOn-fp32/` (`python -m mtg_search.artifacts download --parts embeddings,model --dest index`).

## Run

```bash
# tasks (already committed): 65 deck needs + 15 attribute + 25 known-item
uv run python eval/agentic/build_tasks.py
# pilot / full run (resumable; already-finished episodes are skipped)
uv run python eval/agentic/agent.py --run <name> --conds A,G --limit 10        # GLM-5.3 (API key needed)
# metrics + paired bootstrap + recall@t plot
uv run --with matplotlib python eval/agentic/analyze.py --runs main \
    --out eval/agentic/results/main.json --plot eval/experiments/agentic_recall_at_t.png
```

Outputs go to `runs/<run>/`:
- `config.json`
- `episodes.jsonl`: one record per (task, condition, repeat): every tool call's query, filters and result card ids, the final list, and token usage, turns and time. (EXP-006 records also hold the SparseUp term feedback.)
- `transcripts/<task>__<cond>__r<rep>.jsonl`: every request (as message deltas) and every raw response, including `reasoning_content`.

## Claude Code runner (EXP-013)

`cc_agent.py` runs the same tasks and protocol with headless Claude Code (`claude-sonnet-5-5[1m]`, on the subscription, isolated as in the precon bridge). It has three arms:
- **A**: mLateOn `search`.
- **G**: grep only.
- **I**: no tools, all card text in context. The card file is read in parts of ≤1.8M characters, followed by a merge call.

Every episode records turns, wall time and token components. Design and results: [`eval/experiments/infra_vs_none.md`](../experiments/infra_vs_none.md).

```bash
uv run python eval/agentic/cc_agent.py --run cc-main --conds A,G --workers 4
uv run python eval/agentic/cc_agent.py --run cc-main-I --conds I --ids <task ids> --workers 1
uv run python eval/agentic/analyze_cc.py --runs cc-main --runs cc-main-I --out eval/agentic/results/cc_main.json
```

`tasks_known_easy.yaml` (built with `build_tasks.py --known-easy`) holds the 8 known-item drafts that were dropped only because BM25 found them easily. EXP-013 reports them as a separate stratum.

## Files

- `env.py`: mLateOn search, plain grep, compact card renderer, leak guard.
- `agent.py`: agent loop, conditions, runner.
- `glm.py`: HTTP client, retries, cost accounting (USD per 1M tokens: $1.40 input, $0.26 cached input, $4.40 output).
- `build_tasks.py`: task set. Known-item descriptions were drafted by GLM (`known_drafts.yaml`, `known_gen_transcript.jsonl`) and hand-reviewed (`known_reviewed.yaml`).
- `analyze.py`: metrics on the original labels, condensed-list nDCG, paired bootstrap, plot.
- `rescore.py`: re-resolves every saved episode's answer names with the current `Env.resolve_name` (dry run by default, `--write` to apply). Written for the 2026-10-15 fix, where another card's face name had beaten a card's full name; `runs/judge-facefix` grades the cards that newly counted.
- `judge.py`: the pre-registered judging rubric and prompt (GLM client, not run); `judge_cc.py` runs the same judge with Claude Code Sonnet (EXP-015), `judge_smoke_analyze.py` analyses it.

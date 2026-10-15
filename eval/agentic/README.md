# EXP-006: agentic multi-step search harness

GLM-5.3 acts as a deck-building search assistant. It gets up to 5 search calls of 15 results each, then submits a ranked final list of up to 20 cards. The conditions differ only in the tool output. Design and results: [`eval/experiments/agentic_search.md`](../experiments/agentic_search.md).

| cond | retriever | extra feedback in tool output |
|---|---|---|
| A | mLateOn | none |
| B | SparseUp | none |
| C | SparseUp | per-result matched SparseUp terms (`*` = expansion term, not a query word), plus the aggregate line "Frequent terms in these results not in your query" |
| D | mLateOn | the same aggregate line, computed with SparseUp over mLateOn's results |
| F | lexical only (legacy) | the tools are `grep(pattern)` and `bm25(query)` instead of `search` |
| G | none: grep only (EXP-013) | one tool, `grep(pattern, offset?)`. It matches line by line over the plain card text and returns matches in file order (alphabetical), 15 per call. No BM25, no ranking, no filter parameters |

Tool output never contains popularity, price or crowd signals. `Env.assert_no_leak` blocks `/edhrec|price|\$|rank|popular|salt/` in everything the harness writes. Card text is exempt, because card names can legitimately contain words such as "rank". F's grep orders its matches by the BM25 score of the pattern's literal words, with ties broken alphabetically. G's grep has no ranking at all.

## Setup

- The key is `GLM_API_KEY` in the repo `.env`. It is never logged.
- The endpoint is `https://api.z.ai/api/paas/v4` (pay-as-you-go). The model is `glm-5.3` with `thinking: {type: enabled, effort: low}` (GLM-5.3 cannot turn thinking off). Temperature is 0.6.
- Indexes: `index/embeddings/late/lightonai__mLateOn/` and `index/embeddings/sparse/Linkup-Platform__linkup-sparseup-embed-v1.*`.

## Run

```bash
# tasks (already committed): 65 deck needs + 15 attribute + 25 known-item
uv run python eval/agentic/build_tasks.py
# pilot / full run (resumable; already-finished episodes are skipped)
uv run --extra models python eval/agentic/agent.py --run pilot --conds A,C,F --limit 10
uv run --extra models python eval/agentic/agent.py --run main --conds A,B,C,D,F --repeats 2 --workers 6
# metrics + paired bootstrap + recall@t plot
uv run --with matplotlib python eval/agentic/analyze.py --runs main \
    --out eval/agentic/results/main.json --plot eval/experiments/agentic_recall_at_t.png
```

Outputs go to `runs/<run>/`:
- `config.json`
- `episodes.jsonl`: one record per (task, condition, repeat). It holds every search's query, filters, result oracle ids and the term feedback. Terms are computed for every condition, even where they are not shown, so they serve as a counterfactual baseline for term uptake. It also holds the final list, token usage and cost.
- `transcripts/<task>__<cond>__r<rep>.jsonl`: every request (as message deltas) and every raw response, including `reasoning_content`.

## Claude Code runner (EXP-013)

`cc_agent.py` runs the same tasks and protocol with headless Claude Code (`claude-sonnet-5-5[1m]`, on the subscription, isolated as in the precon bridge). It has three arms:
- **A**: mLateOn `search`.
- **G**: grep only.
- **I**: no tools, all card text in context. The card file is read in parts of ≤1.8M characters, followed by a merge call.

Every episode records turns, wall time and token components. Design and results: [`eval/experiments/infra_vs_none.md`](../experiments/infra_vs_none.md).

```bash
uv run --extra models python eval/agentic/cc_agent.py --run cc-main --conds A,G --workers 4
uv run --extra models python eval/agentic/cc_agent.py --run cc-main-I --conds I --ids <task ids> --workers 1
uv run python eval/agentic/analyze_cc.py --runs cc-main --runs cc-main-I --out eval/agentic/results/cc_main.json
```

`tasks_known_easy.yaml` (built with `build_tasks.py --known-easy`) holds the 8 known-item drafts that were dropped only because BM25 found them easily. EXP-013 reports them as a separate stratum.

## Files

- `env.py`: retrievers (mLateOn, SparseUp, BM25, grep), compact card renderer, SparseUp term feedback, leak guard.
- `agent.py`: agent loop, conditions, runner.
- `glm.py`: HTTP client, retries, cost accounting (USD per 1M tokens: $1.40 input, $0.26 cached input, $4.40 output).
- `build_tasks.py`: task set. Known-item descriptions were drafted by GLM (`known_drafts.yaml`, `known_gen_transcript.jsonl`) and hand-reviewed (`known_reviewed.yaml`).
- `analyze.py`: metrics on the original labels, condensed-list nDCG, paired bootstrap, plot.
- `judge.py`: blind TREC-style pooled GLM judging. **Deferred, not run** (tracked as a GitHub issue).

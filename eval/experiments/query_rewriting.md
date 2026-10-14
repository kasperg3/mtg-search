# EXP-003: Agent-side query rewriting

**Question.** On the paraphrase (222) and deckneeds (30) sets, the best system scores only about 13–14 nDCG@10, because vague player or deck-role wording doesn't match rules text. In production the caller is an LLM agent, which can rewrite the query before it calls the search tool. How much does that help, and how should the MCP tool be shaped to make use of it?

**Answer.** It helps a great deal. On the 252 queries, one rewrite phrased like rules text takes `late-mLateOn` from **13.0 to 75.5 nDCG@10** (+62.5 [+57.8, +67.1]). Recall@100 goes from 7.8 to 46.2. A blind control rewriter, which never saw the query ids, gets 73.1. Do not fuse the user's raw phrasing with the rewrites, because it drags results down. Use mLateOn alone: adding dense makes things worse. Query truncation is not a problem.

## Setup

- **Rewrites (`rewrites.yaml`).** I wrote these acting as the calling agent. I worked from `dump_queries.py` output only (`set`, `id`, `query`, `filters`). I never opened the YAML `relevant` fields or any card data. Each of the 252 queries has three fields:
  - `oracle`: one rewrite in rules-text style;
  - `hyde`: a hypothetical card, written as "type line. oracle text";
  - `multi`: three different rules-text rewrites, each covering a different way a card could fill the need.
- **Leak caveat and control (`rewrites_blind.yaml`).** The dump included query ids, and paraphrase ids encode the source tag slug (e.g. `tag-doom-blade-para`). As a control, a separate agent (same model family) rewrote all 252 queries from an **id-free** list: index, query text and filters only, with the instruction to open no other file. Its rewrites were mapped back to ids afterwards. 62% of its `oracle` strings are identical to mine, as are 31% of `hyde` and 45% of `multi`. Both rewriters converge on the canonical oracle wording. Results for both are reported below.
- **Retrievers.** Same index and code as `eval/run.py`:
  - `late-mLateOn` is `LateInMemory` with `lightonai/mLateOn@bbd883a`;
  - `dense` is `google/embeddinggemma-2@914f7f8`.

  Each text retrieves the top 100 within the query's `filters` mask; funny cards are excluded. Fusion is `mtg_search.fusion.rrf` (k=60, equal weights).
- **Variants.**

  | variant | what is retrieved |
  |---|---|
  | `orig` | the original query |
  | `oracle` | the one rules-text rewrite |
  | `hyde` | the hypothetical card |
  | `multi` | RRF over the 3 multi rewrites |
  | `orig+multi` | RRF over the original + 3 multi rewrites |
  | `orig+oracle`, `orig+hyde` | RRF over two lists |
  | `all` | RRF over all 6 lists |
  | `rrf(dense+mLateOn)` | RRF of both models' lists |
- **Statistics.** Cluster bootstrap 95% CI with 10,000 resamples, paired against `orig` on the same system. Paraphrase queries that share an origin form one cluster. Recall is binary; nDCG uses graded gains.
- **Environment.** CPU, 6 cores, torch 2.14.1, transformers 5.19, sentence-transformers 6.1. The machine was shared with other agents' jobs (load average about 16), so latencies are inflated. Git `6f1463d` (main run) and `30b0a3a` (blind run); `src/` was unchanged for retrieval between them.
- **Reproduce.**

  ```
  uv run --extra models python eval/experiments/query_rewriting.py
  uv run --extra models python eval/experiments/query_rewriting.py --rewrites eval/experiments/rewrites_blind.yaml --systems late-mLateOn dense
  ```
- **Run records.**
  - [runs/20261014T131954Z_query-rewriting.json](runs/20261014T131954Z_query-rewriting.json)
  - [runs/20261014T134119Z_query-rewriting_rewrites_blind.json](runs/20261014T134119Z_query-rewriting_rewrites_blind.json)

## Results

Cells show the score, then the difference from `orig` on the same system in parentheses, with its 95% CI. All differences vs `orig` have p < 0.001.

### Pooled paraphrase + deckneeds (252 queries), main rewrites

| system | variant | nDCG@10 | Recall@50 | Recall@100 |
|---|---|---|---|---|
| late-mLateOn | orig | 13.0 | 4.6 | 7.8 |
| late-mLateOn | **oracle** | **75.5** (+62.5 [+57.8, +67.1]) | 33.0 (+28.4 [+24.4, +32.2]) | 46.2 (+38.4 [+33.8, +42.9]) |
| late-mLateOn | hyde | 61.7 (+48.7 [+44.1, +53.3]) | 23.3 (+18.7 [+16.0, +21.5]) | 33.9 (+26.0 [+22.3, +29.9]) |
| late-mLateOn | multi | 68.8 (+55.8 [+50.7, +60.8]) | 31.3 (+26.7 [+23.3, +30.3]) | 45.7 (+37.9 [+33.9, +42.0]) |
| late-mLateOn | orig+multi | 67.0 (+54.0 [+49.2, +58.7]) | 30.3 (+25.6 [+22.3, +29.1]) | 42.6 (+34.7 [+30.7, +38.7]) |
| late-mLateOn | orig+oracle | 49.3 (+36.3 [+33.5, +39.1]) | 22.7 | 35.8 |
| late-mLateOn | all | 74.0 (+61.0 [+56.3, +65.6]) | **34.0** | **48.3** |
| dense | orig | 10.1 | 3.1 | 5.0 |
| dense | oracle | 50.4 (+40.2 [+36.1, +44.6]) | 16.3 | 24.9 |
| dense | hyde | 45.3 (+35.1 [+30.6, +39.6]) | 13.1 | 19.6 |
| dense | multi | 45.8 (+35.6 [+31.0, +40.2]) | 15.8 | 24.3 |
| dense | orig+multi | 43.4 (+33.2 [+28.9, +37.7]) | 15.2 | 22.5 |
| rrf(dense+mLateOn) | orig | 14.0 | 4.3 | 7.0 |
| rrf(dense+mLateOn) | multi | 64.2 (+50.1 [+45.4, +54.7]) | 26.0 | 39.1 |
| rrf(dense+mLateOn) | orig+multi | 62.6 (+48.5 [+44.0, +53.0]) | 25.3 | 37.6 |

### Per set (late-mLateOn; dense in brackets), main rewrites

| set | variant | nDCG@10 | Recall@50 | Recall@100 |
|---|---|---|---|---|
| paraphrase (222) | orig | 11.9 [9.1] | 4.4 [2.8] | 7.4 [4.6] |
| | oracle | **76.1** (+64.3 [+59.5, +69.0]) [49.9] | 35.2 [17.0] | 48.9 [25.7] |
| | hyde | 62.0 (+50.1 [+45.2, +54.8]) [44.7] | 24.7 [13.6] | 35.7 [20.1] |
| | multi | 69.8 (+57.9 [+52.6, +62.9]) [45.7] | 33.3 [16.4] | 48.2 [25.0] |
| | orig+multi | 68.1 (+56.2 [+51.2, +61.3]) [42.7] | 32.2 [15.8] | 44.8 [23.2] |
| deckneeds (30) | orig | 21.1 [17.9] | 6.1 [4.8] | 10.9 [7.9] |
| | oracle | **70.7** (+49.6 [+34.8, +63.8]) [53.8] | **17.1** [11.5] | 26.7 [19.0] |
| | hyde | 60.1 (+38.9 [+25.4, +52.0]) [49.6] | 13.1 [9.6] | 20.4 [16.0] |
| | multi | 61.7 (+40.5 [+26.3, +54.2]) [46.2] | 16.4 [11.7] | **27.3** [19.2] |
| | orig+multi | 59.1 (+37.9 [+25.6, +50.3]) [48.2] | 15.5 [11.2] | 26.0 [17.9] |

### Blind control (`rewrites_blind.yaml`, id-free), pooled

| system | orig | oracle | hyde | multi | orig+multi | all |
|---|---|---|---|---|---|---|
| late-mLateOn nDCG@10 | 13.0 | **73.1** (+60.1 [+55.1, +64.9]) | 62.1 | 64.1 | 61.0 | 71.5 |
| late-mLateOn Recall@100 | 7.8 | 44.7 | 33.9 | 41.9 | 38.5 | **46.1** |
| dense nDCG@10 | 10.1 | 47.5 | 44.4 | 40.6 | 39.2 | 47.8 |

The blind rewrites land 2–5 points below mine. Every conclusion below holds for both sets of rewrites. The ids therefore contributed little: the gain comes from translating the query into oracle wording, which the model does from the text alone.

### Breakdown (late-mLateOn, main rewrites, nDCG@10)

| subset | orig | oracle | multi |
|---|---|---|---|
| paraphrase, from tags (130) | 15.3 | 81.5 | 76.1 |
| paraphrase, from handwritten (92) | 7.1 | 68.6 | 60.7 |
| `-para` reformulations (111) | 18.4 | 77.0 | 74.2 |
| `-need` deck-role phrasings (111) | **5.4** | **75.3** | 65.3 |
| deckneeds (30) | 21.1 | 70.7 | 61.7 |

The `-need` phrasings ("my aristocrats list needs…") are the worst raw queries and gain the most. After rewriting they score about the same as the `-para` ones. For reference, the 111 *original* tags and handwritten queries score 64.8 with mLateOn, so a rules-text rewrite of a vague query retrieves better than the hand-written benchmark queries themselves.

Only 11 of 252 oracle rewrites score nDCG@10 below 0.2. The original query beats the oracle rewrite (by more than 0.1) on only 6 queries. These are mostly cases where the rewrite picked one narrow mechanism, such as "Mill four cards" for reanimator enablers, or "Draw two cards." for UB card advantage.

### Query length / truncation

These figures are mLateOn token counts including the `[Q] ` prefix.

| text | p50 | p95 | max | > 32 tokens |
|---|---|---|---|---|
| original | 13 | 20 | 26 | 0% |
| oracle | 15 | 26 | 37 | 1.6% |
| hyde | 21 | 33 | 39 | 7.1% (blind: 5.2%) |
| multi (each) | 16 | 27 | 40 | 1.5% (blind: 0.4%) |

There is no 32-token limit here. mLateOn's `config_sentence_transformers.json` sets `query_length: 8192` with `do_query_expansion: false`, and the NextPlaid spike found the same in `onnx_config.json`. So nothing was truncated, and raising the limit to 64 has nothing to test. As a sensitivity check, `late-mLateOn@32` caps queries at 32 tokens. It changes every variant by at most 0.6 nDCG@10 (pooled hyde 61.7 → 61.8, oracle unchanged). Even under a 32-token cap, truncation would not matter for these rewrites.

### Latency

Latency is the sum of sequential component searches, measured on a loaded CPU, so read the ratios rather than the absolute values. For late-mLateOn the p50 is 117 ms for one query, 126 ms for oracle, 401 ms for multi (3 searches) and 523 ms for orig+multi. On the NextPlaid server, a single query takes about 40 ms end to end (EXP-002), so 3–4 queries cost about 120–160 ms sequentially, or less if the queries are batch-encoded.

## Findings

1. **Rewriting is the single largest lever measured so far**: about +60 nDCG@10 on the vague sets. All earlier retriever and fusion choices (EXP-001, EXP-004) moved this metric by at most 1–4 points.
2. **Format matters.** Plain rules-text sentences (`oracle`) beat a hypothetical card (`hyde`) by about 14 points on mLateOn and 5 on dense. The type line and the extra invented clauses add noise to MaxSim.
3. **Do not fuse the raw user query.**
   - `orig+oracle` (49.3) is 26 points *below* `oracle` alone. With two lists, equal-weight RRF lets the bad list contribute half the score.
   - `orig+multi` is 1.8 points below `multi`: 62 queries got worse and 35 got better.
   - The raw phrasing almost never ranks something that the rewrites miss.
4. **One precise rewrite vs several diverse ones.** On these narrow, single-function labels, `oracle` wins on nDCG@10 (75.5 vs 68.8). `multi` is about equal on Recall@100 (45.7 vs 46.2) and slightly better on deckneeds (27.3 vs 26.7). `all` has the best recall (48.3).

   The labels reward exactly one mechanism, so the extra mechanisms in `multi` are partly "wrong" by construction. For example, for fliers, "Reach" and "destroy target creature with flying" are both valid, but a tag captures only one of them. For real, open-ended deck needs, the breadth is likely worth more than these numbers show.
5. **mLateOn alone beats mLateOn + dense once queries are rewritten.** `rrf(dense+mLateOn)` on multi scores 64.2, against 68.8 for mLateOn alone. Rules-text queries suit late interaction, and dense is the weaker list (50 vs 76 for oracle).

## Recommendation for the MCP tool

1. **Accept `queries: list[str]`** (1–4 items; a single string is still allowed). Run each query through mLateOn with the same filters, then fuse with RRF (k=60, equal weights) and return the fused list. Encode the queries in one batch, and keep dense out of the fusion.
2. **Don't add the user's raw request automatically.** Fuse only what the agent sends.
3. **Put guidance in the tool docstring.** The point is to make the agent write rules text, not chat. Proposed text:

   > Cards are matched against their Oracle rules text, so phrase each query the way the card itself would be worded, not the way a player would describe it. Write it as rules text, e.g. "Counter target spell unless its controller pays {3}.", "Destroy target creature with flying.", "Whenever another creature you control dies, draw a card.", "Creatures you control get +1/+1." Use "target", "you control", "an opponent controls", "each opponent", "until end of turn", and mana symbols like {2}{G} and {T}. Put colour, format, type and mana value constraints in the filter arguments, not in the query text. For a narrow effect, send one query. For an open-ended need (e.g. "I keep losing to fliers"), send 2–4 queries, each describing a *different* mechanism that would solve it (e.g. "Reach", "Destroy target creature with flying.", "Creatures you control gain reach until end of turn."). Results are fused. Don't include the user's original wording as a query, and don't write a whole invented card (type line plus several abilities); one clean sentence per query works best.
4. **Query length.** No change is needed: neither runtime truncates at 32 tokens. If the NextPlaid server is given an explicit `--query-length`, 64 is enough (the longest rewrite was 40 tokens).

## Caveats

- **The rewriter and the evaluator know MTG well.** Both rewrite sets were written by the same model family that has strong knowledge of card wording. A weaker calling model would gain less.
- **Some rewrites name real cards or near-verbatim templates.** For example, "Lightning Bolt deals 3 damage to any target" names a card. That is realistic agent behaviour, but it means part of the gain is the agent's own card recall.
- **Narrow labels.** Paraphrase labels come from single Tagger tags or regexes. Rewriting into canonical wording essentially recovers the original query's mechanism, so the benchmark rewards one-mechanism precision and under-credits diverse rewrites (see finding 4).
- **The rewrites were not tuned against results.** I did not iterate on them after seeing any results. The blind control is an independent second sample and lands within about 5 points.
- **Latencies are noisy** because the CPU was shared.

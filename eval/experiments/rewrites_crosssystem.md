# EXP-008: Does the sparse vs late vs fusion ranking hold under agent rewrites?

**Question.** EXP-003 showed that agent-side rewriting lifts mLateOn from 13.0 to 75.5 nDCG@10 on para+deck. Real agent usage lives in that rewritten regime, so the retriever choice should be made there. With the same rewrites, which of SparseUp, BM25 and mLateOn, alone or fused, retrieves best?

**Answer.**
- **mLateOn stays the best single retriever under rewriting.** SparseUp trails it on every rewrite variant: −11.5 nDCG@10 on `all` and −13.7 on `oracle` (para+deck, main rewrites).
- **SparseUp never helps as a fusion partner.** rrf(sparseup+mLateOn) is at or below mLateOn on every variant. Down-weighting SparseUp to 0.5 only brings the fusion back to parity.
- **BM25 + mLateOn is the one combination that wins.** Once queries are in rules-text wording, BM25 becomes a strong, complementary signal. rrf(bm25+mLateOn) beats mLateOn alone on the multi-query variants:
  - `multi`: +3.4 [+1.7, +5.1];
  - `all`: +3.5 [+1.8, +5.4] (main rewrites) and +4.1 [+2.5, +6.0] (blind rewrites);
  - it holds on the strict low-overlap paraphrase subset: `all` +3.8 [+1.0, +7.1] (main), +5.5 [+2.6, +8.7] (blind).
- **On un-rewritten vague queries the picture is reversed.** rrf(bm25+mLateOn) loses 3.3 to mLateOn on `orig`, and BM25 alone scores 3.2. The fusion gain needs the agent to rewrite, which the `search_cards` tool description already asks for (commit 6b09929).

## Setup

- **Script.** [`rewrites_crosssystem.py`](rewrites_crosssystem.py). It makes no new LLM calls and reuses the EXP-003 rewrites ([`rewrites.yaml`](rewrites.yaml) and the id-free blind control [`rewrites_blind.yaml`](rewrites_blind.yaml)).
- **Retrieval.** late-mLateOn comes from the EXP-003 retrieval cache. sparseup and bm25 are run here, with torch at 2 threads. Retrieval returns the top 100 under each query's filter mask.
- **Fusion.** Fusions use RRF with k=60 as one flat RRF over all component lists of both systems. Per-system weights are shown as `=0.5`.
- **Variants:**
  - `orig`: the vague query;
  - `oracle`: one rules-text rewrite;
  - `hyde`: a hypothetical card;
  - `multi`: RRF over 3 rewrites from different angles;
  - `all`: orig + oracle + hyde + 3 multi.
- **Statistics.** Paired cluster-bootstrap 95% CIs (10k resamples) are computed against late-mLateOn on the same variant and rewrites file.
- **Run record.** [`runs/20261014T142114Z_rewrites-crosssystem.json`](runs/20261014T142114Z_rewrites-crosssystem.json). The first run (13:50Z) lost its output to a machine OOM and was re-run unchanged from cache.

## Results: main rewrites, para+deck (n=252)

nDCG@10 / R@100 (×100). Δ is the nDCG@10 difference vs late-mLateOn, with its 95% CI.

| system | orig | oracle | hyde | multi | all |
|---|---|---|---|---|---|
| late-mLateOn | 13.0 / 7.8 | 75.5 / 46.2 | 61.7 / 33.9 | 68.8 / 45.7 | 74.0 / 48.3 |
| sparseup | 8.0 / 4.4 | 61.8 / 34.9 | 56.3 / 23.9 | 56.6 / 34.9 | 62.5 / 36.7 |
| bm25 | 3.2 / 2.5 | 70.6 / 44.1 | 66.6 / 38.8 | 67.2 / 42.8 | 72.5 / 46.0 |
| rrf(sparseup+mLateOn) | 10.9 / 6.8 | 70.6 / 43.1 | 60.4 / 31.1 | 66.1 / 43.1 | 72.3 / 45.4 |
| rrf(sparseup=0.5+mLateOn) | 11.5 / 7.5 | 72.4 / 46.1 | 61.0 / 33.7 | 67.8 / 44.7 | 73.8 / 47.3 |
| **rrf(bm25+mLateOn)** | 9.7 / 5.7 | **76.4 / 46.9** | **66.5 / 38.4** | **72.2 / 47.4** | **77.5 / 49.9** |

Δ nDCG@10 vs late-mLateOn:

| system | orig | oracle | hyde | multi | all |
|---|---|---|---|---|---|
| sparseup | −5.0 [−7.3, −2.8] | −13.7 [−17.7, −9.8] | −5.4 [−8.2, −2.7] | −12.2 [−15.3, −9.1] | −11.5 [−14.8, −8.3] |
| bm25 | −9.8 [−12.6, −7.2] | −4.9 [−7.9, −2.0] | +4.9 [+2.2, +7.7] | −1.6 [−4.3, +1.2] | −1.5 [−4.3, +1.2] |
| rrf(sparseup+mLateOn) | −2.1 [−3.6, −0.8] | −4.9 [−7.5, −2.5] | −1.4 [−3.0, +0.3] | −2.7 [−4.2, −1.1] | −1.7 [−3.6, +0.2] |
| rrf(sparseup=0.5+mLateOn) | −1.5 [−2.5, −0.5] | −3.1 [−5.2, −1.2] | −0.8 [−2.1, +0.5] | −1.0 [−2.1, +0.1] | −0.2 [−1.5, +1.1] |
| rrf(bm25+mLateOn) | −3.3 [−4.9, −1.9] | +0.9 [−0.4, +2.1] | +4.7 [+3.1, +6.5] | +3.4 [+1.7, +5.1] | +3.5 [+1.8, +5.4] |

## Blind rewrites (id-free control), para+deck

`all` variant: mLateOn 71.5, rrf(bm25+mLateOn) **75.6** (+4.1 [+2.5, +6.0]), bm25 70.7, sparseup 59.6, rrf(sparseup=0.5+mLateOn) 71.2. The ordering matches the main rewrites exactly, so the result does not depend on the rewriter seeing query ids.

## Leakage control: oracle rewrites that reconstruct the origin query

The paraphrase set was built by paraphrasing handwritten or tag-derived rules-text queries. A rewrite can therefore partly reconstruct that original wording.
- Of 222 paraphrase oracle rewrites, 47.7% have content-word Jaccard > 0.5 with the origin query (mean 0.54), and 16.2% reconstruct it exactly. The paraphrase queries themselves overlap the origin by only 0.07.
- The effect is concentrated in handwritten-origin queries: 80% have Jaccard > 0.5, against 25% of tag-origin queries.

On the **strict low-overlap subset** (n=106, main rewrites), absolute scores are *higher*, not lower: mLateOn `all` 80.7. The fusion ranking is unchanged:

| system (`all` variant) | low-overlap strict | high-overlap (n=106) | deckneeds (n=30, no origin query) |
|---|---|---|---|
| late-mLateOn | 80.7 | 67.9 | 70.3 |
| rrf(bm25+mLateOn) | 84.5 (+3.8 [+1.0, +7.1]) | 71.7 (+3.8 [+1.2, +6.6]) | 72.1 (+1.8 [−2.1, +6.0]) |
| rrf(sparseup=0.5+mLateOn) | 81.0 (+0.3) | 67.4 (−0.5) | 70.0 (−0.3) |
| sparseup | 70.4 (−10.1) | 53.7 (−14.3) | 63.0 (−7.3) |

So the rewriting gain is not an artifact of reconstruction. On deckneeds (n=30) all fusions are within noise of mLateOn. On that set the best multi-query result is rrf(bm25+mLateOn) `multi` +4.8 [+1.5, +8.0].

## Reference: original (already rules-text) queries

| system | handwritten nDCG@10 | tags nDCG@10 |
|---|---|---|
| late-mLateOn | 70.7 | 60.5 |
| rrf(bm25+mLateOn) | 70.0 (−0.7 [−3.5, +2.0]) | 54.0 (−6.6 [−12.2, −1.3]) |
| rrf(sparseup=0.5+mLateOn) | 63.6 (−7.1) | 57.8 (−2.8) |
| bm25 | 61.1 | 36.4 |
| sparseup | 47.3 | 47.2 |

On the tags set, BM25 fusion costs 6.6 points. Those queries are short role descriptions such as "card draw" or "ramp", where exact-word matching misfires.

## Conclusion

Under realistic agent use, where the agent rewrites into rules-text wording and sends several queries, **late interaction (mLateOn) is the backbone**. SparseUp adds nothing to ranking, alone or fused. The only combination worth considering is **mLateOn + BM25 via RRF, used for multi-query calls**: it gains about 3.5–5 nDCG@10 over mLateOn on rewritten queries. It loses about 3 points on raw vague queries and about 7 on short tag-style queries, so it should be gated on the agent having rewritten, or at least offered as an option, not made the default for every query. SparseUp's value stays in explanation (EXP-004) and in the agentic term-feedback question (EXP-006).

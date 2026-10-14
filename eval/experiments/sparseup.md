# EXP-004: SparseUp (learned sparse with a transformer query encoder)

**Question.** Does `Linkup-Platform/linkup-sparseup-embed-v1` do better than our sparse and dense baselines? It is a SPLADE-style model with a full query encoder. Three sub-questions:
- Does it fuse usefully with mLateOn?
- Are its per-term score explanations meaningful to a human?

**Answer.** Partly, and it is **not adopted** for the retrieval stack.
- It is the best *learned-sparse* model we have. Against gte-sparse it gains +1.3 nDCG@10 on para+deck (p=0.04) and +7.2 on tags.
- It loses to dense on para+deck (−2.1, p=0.04) and to mLateOn everywhere (−5.0 para+deck, −23.4 handwritten).
- Fusing it with mLateOn *lowers* mLateOn's score (−2.1 para+deck, −10.0 handwritten).
- Its explanations are its real asset. They are almost entirely whole words, decompose the score exactly, and make failure modes obvious. That makes it a candidate *explanation/diagnostic* layer (EXP-006), not a ranking component.

## Setup

- **Model.** `Linkup-Platform/linkup-sparseup-embed-v1` at revision `08314498d4f6a3a205b930ab9f27001404ea94b8`, Apache-2.0.
  - Architecture: ModernBERT-base (149M) MLM head, then `log1p(relu(logit − 15))`, then the top-12 vocab dims per position (`position_top_k`), then max-pool, then a case/`Ġ` vocab fold (50,370 → ~34k distinct dims).
  - Prefixes `[Q] ` / `[D] ` are attended but not pooled. Query max length is 128 tokens, document max length 512. Scoring is a dot product.
- **Loading.** We load it with `trust_remote_code=True` from a **local snapshot of the pinned revision**. The remote `SpladeSTModule.load()` ignores `revision`, so this is the only way to pin it.
  - Under transformers 5.19 / sentence-transformers 6.1, the model card's reference example reproduces exactly: `[[113.2993, 0.0], [7.0030, 99.1291]]`, and the query top-10 terms match.
  - **No transformers-5 repair** of the `_repair_gte` kind was needed.
  - Rows encoded by ST `encode_document` match the native `model.encode(kind="document")` to within 7e-6, with identical nnz.
- **Document encoding.** `jobs/encode.py --kind sparse` (new `MODEL_LOAD` entry: local snapshot, doc max 512, batch 16) on HF Jobs `t4-small`:
  - 327 s for 34,646 cards, job `6acf76525ccb47ba0f79a191`;
  - torch 2.14.1+cu130, transformers 5.19.0, ST 6.1.0;
  - doc tokens p50 64, max 426, 0 truncated;
  - uploaded to `kasper-cbx/mtg-search-index` `embeddings/sparse/` (commit `bbbd38b`);
  - `oracle_ids_sha256` = `1bff7b49…acf1`, which matches `data/cards.parquet`.
- **Retriever.** `src/mtg_search/retrievers/sparseup.py` (`SparseUpRetriever`). Each query goes through the transformer on CPU (fp32). Scoring is `docs_csc[:, q_ids] @ q_w` followed by top-k under the filter mask.
- **Eval.** `eval/run.py --only sparseup --label sparseup`, on all 4 sets.
  - Systems added: `sparseup`, `rrf(sparseup+mLateOn)`, `rrf(sparseup+dense)`.
  - Protocol: top-100 per component, RRF k=60, cluster bootstrap with 10k resamples.
  - Code: git `633cefa`, plus the run.py changes that are committed alongside this note.
  - Run record: [`runs/20261014T125242Z_sparseup.json`](runs/20261014T125242Z_sparseup.json).

## Results

nDCG@10 / Recall@50 / Recall@100 / MRR@10 (×100). `para+deck` = pooled nDCG@10 over the 252 paraphrase + deckneeds queries.

| system | para+deck | tags | handwritten | paraphrase | deckneeds |
|---|---|---|---|---|---|
| sparseup | **8.0** | 47.2 / 16.1 / 25.3 / 63.5 | 47.3 / 30.4 / 45.0 / 66.8 | 6.4 / 2.4 / 3.9 / 13.1 | 20.0 / 5.1 / 8.8 / 39.5 |
| sparse (gte doc-v3) | 6.7 | 40.0 / 14.8 / 23.2 / 59.0 | 47.3 / 33.9 / 48.2 / 75.4 | 4.9 / 1.6 / 2.7 / 11.0 | 19.6 / 4.9 / 8.5 / 33.0 |
| dense (embeddinggemma-2) | 10.1 | 42.8 / 12.4 / 18.7 / 63.5 | 36.7 / 19.8 / 29.9 / 58.4 | 9.1 / 2.8 / 4.6 / 18.1 | 17.9 / 4.8 / 7.9 / 37.7 |
| late-mLateOn | 13.0 | 60.5 / 21.9 / 32.1 / 73.8 | 70.7 / 52.7 / 71.3 / 89.5 | 11.9 / 4.4 / 7.4 / 22.2 | 21.1 / 6.1 / 10.9 / 37.3 |
| rrf(sparseup+mLateOn) | 10.9 | 56.2 / 20.0 / 30.9 / 73.3 | 60.7 / 43.2 / 64.8 / 80.5 | 9.6 / 3.6 / 6.3 / 18.9 | 20.1 / 6.1 / 10.4 / 39.3 |
| rrf(sparseup+dense) | 9.8 | 52.5 / 15.5 / 24.2 / 69.5 | 46.8 / 28.9 / 41.8 / 71.9 | 8.6 / 3.1 / 4.8 / 18.2 | 19.0 / 5.4 / 9.1 / 40.9 |
| rrf(dense+mLateOn) (ref) | 14.0 | 59.0 / 18.4 / 28.9 / 75.1 | 59.0 / 40.8 / 60.4 / 83.0 | 13.0 / 4.1 / 6.6 / 23.7 | 21.8 / 6.1 / 10.2 / 37.4 |
| bm25 (ref) | 3.2 | 36.4 / 16.0 / 26.0 / 50.4 | 61.1 / 46.3 / 63.5 / 76.0 | 2.5 / 1.2 / 2.1 / 5.7 | 8.5 / 3.0 / 5.7 / 17.5 |

### Paired comparisons (nDCG@10 points, A − B [95% CI], p)

| A | B | para+deck | handwritten |
|---|---|---|---|
| sparseup | sparse (gte) | +1.3 [+0.0, +2.6], p=0.043 | −0.0 [−5.7, +5.7], p=0.99 |
| sparseup | dense | −2.1 [−4.1, −0.0], p=0.043 | **+10.6** [+3.7, +17.9], p=0.007 |
| sparseup | late-mLateOn | **−5.0** [−7.3, −2.8], p=0.0001 | **−23.4** [−31.7, −15.2], p=0.0001 |
| rrf(sparseup+mLateOn) | late-mLateOn | −2.1 [−3.6, −0.8], p=0.004 | −10.0 [−15.2, −5.1], p=0.0006 |
| rrf(sparseup+mLateOn) | rrf(dense+mLateOn) | −3.2 [−4.8, −1.6], p=0.0003 | +1.6 [−4.3, +7.7], p=0.59 |
| rrf(sparseup+dense) | dense | −0.3 [−1.6, +1.0], p=0.68 | +10.1 [+5.2, +15.2], p=0.0002 |

### Latency (CPU, i5-9600K, `search()` = query encode + scoring + top-100, unfiltered)

The latency recorded during the main eval run is **not usable**. That run reported sparseup p50 2.1 s, because the machine was at load ≈ 20 on 6 cores (the query-rewriting run was going at the same time). Torch's 6 intra-op threads oversubscribed: the same forward pass took 1,078 ms with 6 threads and 129 ms with 2.

`eval/experiments/sparseup_latency.py` therefore re-measured all four systems back to back on the same 100 para+deck queries:

| system | quiet, 2 threads p50 / p95 | quiet, 6 threads p50 / p95 | loaded (≈12–18), 2 threads p50 / p95 |
|---|---|---|---|
| sparseup | **57 / 67 ms** | 77 / 152 ms | 107 / 203 ms |
| sparse (gte, inference-free) | 0.3 / 0.4 ms | 0.3 / 0.4 ms | 0.3 / 0.4 ms |
| dense (embeddinggemma-2) | 99 / 159 ms | 151 / 190 ms | 176 / 343 ms |
| late-mLateOn (in-process) | 164 / 201 ms | 141 / 217 ms | 642 / 1166 ms |

Run records: `runs/sparseup_latency_quiet.json` and `runs/sparseup_latency_loaded.json`.
- The query forward pass is about 99% of the cost. Sparse scoring takes 0.8 ms (query nnz p50 79).
- SparseUp is the cheapest neural query encoder we have (149M vs embeddinggemma's ~270M text tower). It is still ~200× slower than the inference-free gte-sparse.
- Under contention, use fewer torch threads (2) for single-query serving.

### Index

| | SparseUp | gte doc-v3 |
|---|---|---|
| vocab / active dims | 50,370 / 24,350 used | 30,522 |
| nnz per doc (mean / p50 / max) | 198 / 196 / 636 | ~223 (200-card sample) |
| `.npz` on disk | 35.5 MB | 36.7 MB |
| CSR in RAM (fp32) | 55 MB | — |
| query nnz (mean / p50 / max, 363 eval queries) | 79 / 79 / 146 | tokenizer-only (few terms) |

From EXP-005 ([doc_expansion.md](doc_expansion.md)): `k=12` is the model's internal per-position fan-out cap (`position_top_k`), and our index already applies it. Of the ~198 non-zero terms per card, ~168 (85%) are expansion terms that do not occur in the card text, and they carry ~76% of the weight mass.

## Explainability

`SparseUpRetriever.explain(query, row_idx, top_n=8)` is a **stable API** used by EXP-006. It returns `[(term, contribution, is_expansion)]`:
- `contribution = q[t] · d[t]`. Contributions over all shared terms sum exactly to the score.
- `is_expansion` is True when the term's folded vocab id is **not among the card text's own subtokens**.

`explain_detail()` adds the q/d weights, a query-side expansion flag, a subword flag and the coverage of the top-n terms. Run the demo with `eval/experiments/sparseup_explain.py`; its output is in `runs/sparseup_explain.json`.

How to read the examples below:
- Each term is shown as `term contribution`.
- `[+q]` means the term is not literally in the query; `[+d]` means it is not literally in the card.
- `·x` is a word-internal subword piece.
- ✓ means relevant, ✗ means not relevant.

**“counter they can pay around”** (paraphrase of `hw-soft-counter`). Query vector (42 terms): counter 3.32, counters[+q] 3.09, pay 2.95, around 2.87, payment[+q] 2.18, corner[+q], count[+q], paying[+q]…
1. Fraying Line ✗ — 51.6: counter 10.4, counters[+q] 9.3, pay 8.0, payment[+q+d] 4.2, paying[+q+d] 4.1, count[+q+d] 4.0…
2. Die Young ✗ — 51.4: counters[+q] 9.3, counter[+d] 8.9, pay 7.9, payment[+q+d] 4.4…
3. Aether Spike ✓ — 50.8: counters[+q] 9.2, counter 9.2, pay 7.7, counts[+q+d] 4.4…

The explanation shows the failure directly. The model expands "counter" to "counters"/"count", so a counterspell query matches cards about +1/+1 or energy **counters** that you **pay**. This is MTG-specific polysemy that a general-domain expansion gets wrong.

**“forces them to give up one of their own guys”** (paraphrase of the edict tag). Query: own, forces, surrender[+q], force[+q], guy[+q], sacrifice[+q]…
1. Force of Negation ✗ — force[+q] 7.0, forces[+d] 6.7, forcing[+q+d] 4.7…
2. Force Away ✗ — same pattern.
3. Force Bubble ✗ — same pattern.

The query expansion correctly *finds* "sacrifice" (weight 2.12). Literal "force" matches against **card names** (`doc_text` includes the name) outweigh it.

**“I need ways to get value from creatures dying in an aristocrats deck”** (deck need):
1. Death Frenzy ✓ — creatures 7.4, death[+q] 6.8, creature[+q] 5.5, dying[+d] 5.3, die[+q+d] 4.0, gain[+q] 3.4…
3. Death's Oasis ✓ — death[+q] 6.8, creatures 6.7, value 5.6, dying[+d] 5.6…

This is a sensible bridge: dying → death/die, plus value, and both hits are relevant.

**“I keep losing to fliers in my mono-green deck”** (filter G). "fliers" tokenises as `fl` + `·iers`, and nothing expands to *flying* or *reach*. The top hit, Summoner's Pact ✗, wins on green 9.1, keep 6.2, `mon` 6.2, deck 4.6 and losing 4.2. The explanation makes it obvious that the query's operative word was never understood.

**“counter target spell unless its controller pays”** (handwritten). Lofty Denial ✓✓, Dazzling Denial ✓✓ and Miscast ✓ all come from counter / controller / spell / pay. When the query uses rules wording, explanations are exactly what you'd expect.

**Is it human-meaningful? Mostly yes.**
- Over the 192 term contributions shown for the 8 demo queries × top-3 hits, **only 3% are subword fragments**. Examples are `·cery` (from "sorcery"), `·rifice`, `fl`/`·iers` and `mon`; they mostly come from rare or compound words.
- 58% of the shown terms are document-side expansions and 63% are query-side expansions.
- Many expansions are morphological variants (counters, sacrificed, spells). These count as "expansion" because the flag is strict about surface form.
- Others are genuine semantic bridges (dying→death/die, guys→sacrifice/surrender, creatures→animals/monsters/pets).
- The top-8 terms cover 40–94% of the score. Long matches on rules text spread over 40–57 shared terms.

**Caveat.** The literal check is per folded subtoken, so "counters" in a card that only says "counter" is flagged as an expansion. A word-level, lemmatised check would make the flag friendlier, but it would no longer be exact.

## Conclusion

- **Do not add SparseUp to the ranking stack.**
  - It is only marginally better than the gte-sparse model it would replace.
  - It is worse than dense on vague queries and much worse than mLateOn.
  - It *drags mLateOn down* in RRF (−2.1 para+deck, −10.0 handwritten). rrf(dense+mLateOn) stays the best fusion (14.0).
  - It needs a 149M query encoder (57–77 ms CPU), where gte-sparse needs none (0.3 ms).
- **Where it is strong.** It is the best single non-late system on deckneeds (20.0 nDCG@10, MRR 39.5) and beats dense by +10.6 on handwritten. It holds onto exact wording better than dense.
- **Where it could be useful.**
  - As an **explanation layer**: its term decompositions are short, readable and faithful (they sum exactly to the score). They expose MTG-specific failure modes: "counter" polysemy, card-name matches, untokenised jargon such as "fliers".
  - That makes it useful for diagnosing queries and for an agent that rewrites its own queries (EXP-006), or for showing users *why* a card matched, even when another model does the ranking.
- **Follow-ups (not done).**
  - Post-hoc pruning of document expansion terms (literal dims + top-N expansion) to measure what expansion buys; see EXP-005's note.
  - A domain fine-tune that teaches "counter" ≠ "counters", fliers → flying/reach, and similar.

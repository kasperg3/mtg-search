# EXP-006: Agentic multi-step search: does SparseUp term feedback steer the agent?

**Status: harness built, pilot done; the full run is on hold for a budget decision** (projected ≈ $33 > the $25 cap; see [Budget](#budget)).
Harness and how to run it: [`eval/agentic/README.md`](../agentic/README.md). Pilot data: `eval/agentic/runs/pilot/`, `eval/agentic/results/pilot.json`.

## Hypothesis (user)

In multi-step agentic retrieval, SparseUp's interpretable expansion terms let the agent diversify and steer its follow-up searches, because it is "informed about what was actually found". Dense and late-interaction results only give a binary found/not-found signal.

## Design

**Agent.** GLM-5.3 (Z.ai `api/paas/v4`, `glm-5.3`, thinking effort `low`; GLM-5.3 cannot disable thinking) runs at temperature 0.6. The system prompt casts it as a deck-building search assistant. It has the tool `search(query, filters?)`, which returns the top 15 compact cards: name, cost, type, P/T, and rules text truncated to 180 characters with reminder text removed. It gets at most X = 5 searches, then must call `final_answer` with a **ranked** list of at most 20 card names, best first. The system prompt is the same in every condition; only the tool output differs. The exception is F, whose tool paragraph describes its own tools. Reasoning is logged but not sent back. Every request delta and raw response is logged to `runs/<run>/transcripts/*.jsonl`.

**Conditions.**

| | retriever | tool output |
|---|---|---|
| A | mLateOn (LateInMemory) | cards only |
| B | SparseUp | cards only |
| C | SparseUp | cards + per-result "matched terms" (top-5 whole-word q·d contributors, `*` = not a query word) + aggregate line |
| D | mLateOn | cards + the **same** aggregate line, computed by SparseUp over mLateOn's results (isolates "terms as feedback" from the retriever) |
| F | lexical only | tools `grep(pattern)` (case-insensitive regex over card text; match count + 15 matches with the matched line highlighted) and `bm25(query)`; same budget and final-answer protocol |

E (SparseUp + role descriptions) was dropped when EXP-005 was cancelled.

**Aggregate line** ("Frequent terms in these results not in your query: …", top 15). Each vocab term t gets the score idf(t) · Σ_r d_r[t] over the SparseUp document vectors of the 15 results. A term counts only if it appears in at least 2 results, is a whole-word alphabetic token, is not a generic rules stopword, and is neither literally nor by crude stemming in the query. Terms are computed for **every** condition and logged, but shown only in C and D. A and B therefore give a chance baseline for "term uptake".

**No popularity signals.** No tool output contains EDHREC rank, prices, salt or inclusion counts, and no ordering uses them. The grep tool orders matches by the BM25 score of the pattern's literal words over the matching cards, with ties broken alphabetically by name. `Env.assert_no_leak` raises on `/edhrec|price|\$|rank|popular|salt/` in all harness-written tool text: headers, legends and term lists. Card renders are exempt, because card names can contain words such as "rank". SparseUp vocab terms that match the pattern are excluded from the term lists. `edhrec_rank` was used only offline, to pick known-item targets, and is never shown to the agent.
- *Pilot incident 1:* the first F pilot ordered grep matches by EDHREC rank, and its prompt said "most popular". Those 10 episodes are quarantined in `runs/pilot/tainted_F_edhrec_order/` and excluded; F was rerun with BM25 ordering.
- *Pilot incident 2:* in one C pilot episode (`need-punish-spells`), SparseUp produced the expansion terms "price" and "dollar" in the aggregate line. They are semantic expansions of "costs more", not a card-popularity signal. "price" is now filtered from the term vocab. That episode is in the pilot only.
- A and C transcripts match the pattern only inside the model's own reasoning (card names like *Price of Progress*), never in tool output.

**Tasks** (`eval/agentic/tasks.yaml`, 105 in total; labels are never shown to the agent).
- *deck* (65): the 30 `deckneeds` queries plus 35 seeded-sample `-need` paraphrases. Task filters such as colour identity, format or mana value are enforced by the environment and stated to the agent in plain words.
- *attr* (15): a function plus an attribute, e.g. "a one-mana green creature that taps for mana". The label is the oracle-tag closure ∩ the attribute filter (13–255 relevant cards). The agent is not given the filter, but may pass search filters itself.
- *known* (25): "Find the specific card that …". The target is one oracle_id: 13 cards with edhrec_rank < 5000 and 12 with rank > 15000 or unranked. GLM drafted the descriptions from the card text under these rules: no name or name words, no rules wording, describe what the card does rather than a deck role. I checked them by hand against the cards, then single-shot ranked each with BM25, mLateOn and SparseUp. I dropped the trivially lexical ones (BM25 rank ≤ 3: 8 cards) and 2 with over-obscure clues (`known_drafts.yaml` → `known_reviewed.yaml`).

**Metrics** (`analyze.py`). They use the **original task labels**, which are incomplete tag/regex proxies, so absolute numbers are lower bounds.
- Recall@t on the cumulative union of retrieved results, t = 1..5, carried forward after the agent stops.
- Final ranked list: graded nDCG@10 and nDCG@20 (grade 2 from `graded`, 1 otherwise), P@10, Recall@20.
- Known-item: success@t (target retrieved by iteration t), found-in-final, and reciprocal rank in the final list.
- Condensed-list nDCG@10 as a robustness check, using no LLM. Cards of unknown status are removed before scoring; "unknown" means unlabelled and carrying no Tagger tags at all.
- Diversity: mean pairwise Jaccard between iteration result sets, novel cards per iteration, and query word-set Jaccard.
- Iterations used, the share of final cards never retrieved, tokens and cost.
- Term uptake: the share of follow-up queries that introduce a new word (one absent from the task and earlier queries) matching a term from the previous search's aggregate line or starred per-result terms.
- Statistics: paired bootstrap over tasks (10k resamples; repeats averaged per task) for A vs C, B vs C, A vs D, A vs B, F vs C and F vs A.

**Pooled judging is future work.** TREC-style pooling of all conditions' top-20 with a blind GLM judge is implemented in `eval/agentic/judge.py` but **deferred and not run**; it is tracked as a GitHub issue. Until then, every reported metric uses the original, incomplete labels. The judge as written sees only the task text and card text, in a seeded random order, batched 15 cards per call, and never the condition. Its rubric:

> 2 = clearly satisfies it: the card itself performs the requested function directly and meets every stated constraint (colour, card type, mana value, format, cost).
> 1 = partially satisfies it: a weaker, narrower, conditional or indirect version of the requested function, or a card a reasonable player would still consider for this request; it must still meet the stated constraints.
> 0 = does not satisfy the request, or violates a stated constraint (wrong colour, card type or mana value, etc.).
> Judge only from the card text shown. Do not reward popularity or general power level.

## Pilot (10 tasks × A, C, F × 1 repeat: 4 deck, 3 attr, 3 known)

Original labels. With n = 10 these numbers are only a sanity check, not evidence.

| kind | cond | Recall@1 | Recall@5 | nDCG@10 | nDCG@20 | P@10 | novel/iter | result Jaccard | term uptake | iters | $/episode |
|---|---|---|---|---|---|---|---|---|---|---|---|
| deck (4) | A | 0.069 | 0.194 | 0.709 | 0.661 | 0.68 | 13.7 | 0.030 | 0.25 (hidden) | 5.0 | 0.023 |
| deck (4) | C | 0.098 | 0.242 | 0.590 | 0.630 | 0.55 | 14.2 | 0.016 | 0.06 (shown) | 5.0 | 0.030 |
| deck (4) | F | 0.068 | 0.226 | 0.586 | 0.633 | 0.68 | 12.9 | 0.010 | — | 5.0 | 0.055 |
| attr (3) | A | 0.351 | 0.542 | 0.905 | 0.756 | 0.93 | 10.1 | 0.279 | 0.38 (hidden) | 3.7 | 0.017 |
| attr (3) | C | 0.203 | 0.571 | 0.927 | 0.805 | 0.97 | 8.3 | 0.285 | 0.17 (shown) | 5.0 | 0.028 |
| attr (3) | F | 0.435 | 0.701 | 0.927 | 0.886 | 0.97 | 8.1 | 0.157 | — | 3.7 | 0.020 |

Known-item (3 popular targets): every condition retrieved the target at t = 1 or 2 and ranked it first (success@1 = RR = 1.0). Condensed nDCG@10 equals nDCG@10 in the pilot, because almost every card in the final lists carries Tagger tags. The tag-based condensed check is therefore close to vacuous, and pooled judging remains the real fix.

### What the pilot transcripts show

1. **The aggregate terms are mostly noise.** SparseUp's vocab-level expansions are dominated by BPE fragments and loose associations rather than MTG concepts. Two examples:
   - "green one-drop creature that taps to add green mana" → `colors, coloring, mai, ana, dru, colored, gent, manic, mang, men`
   - "instant spell that destroys target creature with flying" → `instantaneous, inert, inst, immediate, lying, flight, instinct, hovering, gravity`

   The per-result matched terms are more sensible (`tap*, color*, elf, dru`, `spider*, flight*, reach`), but they mostly restate what the card line already shows. An offline variant that keeps only words literally present in at least 3 result cards, with no IDF, is clean but generic (`color, cast, instant, gains`).
2. **The agent did not use the terms.** Term uptake when the terms were shown (C: 6% deck, 17% attr) was *lower* than the chance overlap with the same hidden terms in A (25% / 38%). GLM-5.3 steers from its own MTG knowledge instead. It names archetypal cards ("like Llanowar Elves or Elvish Mystic") and rewrites vague needs into rules wording ("instant that exiles target creature you control then returns it…").
3. **Parametric knowledge dominates known-item tasks.** For all 3 pilot targets (popular cards), the first query was already a rules-text reconstruction, e.g. "free colorless artifact that exiles a nonland nonartifact card from your hand" for Chrome Mox. Even F's grep and BM25 found every target at once. Known-item differences between retrievers are likely to be small for popular targets; the 12 obscure targets are the informative half.
4. The agent rarely lists cards it did not retrieve (≤ 2% of final cards), and it uses all 5 searches on deck needs.

## Budget

Pilot mean cost per episode (USD), by task kind, projected onto the full mix (65 deck, 15 attr, 25 known). B and D were not piloted; B ≈ A plus the SparseUp results, and D ≈ A plus the aggregate line. Both estimates come from the measured A and C costs.

| cond | deck | attr | known | per repeat (105 tasks) |
|---|---|---|---|---|
| A | 0.023 | 0.017 | 0.029 | $2.49 |
| B (est.) | ≈0.026 | ≈0.020 | ≈0.035 | ≈$2.8 |
| C | 0.030 | 0.028 | 0.058 | $3.83 |
| D (est.) | ≈0.026 | ≈0.020 | ≈0.040 | ≈$3.0 |
| F | 0.055 | 0.020 | 0.025 | $4.51 |
| **total** | | | | **≈ $16.6 per repeat → ≈ $33 for 2 repeats** |

Spent so far: $0.96 on the valid pilot episodes (A, C, and the F rerun), $0.37 on the quarantined F episodes and $0.14 on drafting the known-item descriptions. Total ≈ **$1.47**. The Z.ai API exposes no balance or quota endpoint for this key: `monitor/usage/quota/limit` answers "no coding plan".

Options within $25:
- (a) 2 repeats of A and C (the headline comparison), 1 repeat of B, D and F: ≈ $23.
- (b) 1 repeat of all five conditions: ≈ $17.
- (c) 2 repeats of A, B, C and D, without F: ≈ $24.

## Results

Pending the budget decision. Tables, recall@t curves (`agentic_recall_at_t.png`), bootstrap CIs, term-pivot excerpts and the conclusion will go here.

---
pretty_name: MTG Search Benchmark (queries, agent tasks, graded relevance)
license: other
license_name: derived-scryfall-wotc-fan-content
license_link: https://company.wizards.com/en/legal/fancontentpolicy
language:
  - en
tags:
  - magic-the-gathering
  - retrieval
  - benchmark
  - agents
  - llm-as-judge
task_categories:
  - text-retrieval
size_categories:
  - 1K<n<10K
configs:
  - config_name: queries
    default: true
    data_files:
      - split: train
        path: data/queries.parquet
  - config_name: agent_tasks
    data_files:
      - split: train
        path: data/agent_tasks.parquet
  - config_name: relevance_grades
    data_files:
      - split: train
        path: data/relevance_grades.parquet
  - config_name: qrels
    data_files:
      - split: train
        path: data/qrels.parquet
---

# MTG Search Benchmark

Benchmarks for finding Magic: The Gathering cards from a description of what they should do, built for the **mtg-search** project. It has two parts. Both use the card snapshot of [`kasper-cbx/mtg-search-index`](https://huggingface.co/datasets/kasper-cbx/mtg-search-index) at revision **`scryfall-20261014090155`** (34,646 cards). Cards are referenced by Scryfall `oracle_id`, with the card name alongside.

1. **Search benchmark** (`queries`): 363 queries for comparing search engines directly, with no agent and no prompt.
2. **Agent search tasks** (`agent_tasks`, `relevance_grades`, `qrels`): 113 requests for an AI agent with a search tool. The agent returns a ranked list of up to 20 cards.
   - **The grading problem:** the automatic lists of correct cards are incomplete, so every card the agents returned was graded blind by a separate Claude model.
   - **The result:** `qrels` holds the final relevance labels, the list plus the graded cards.

The precon deck benchmark (player add/cut rates from EDHREC, and deck-fit grades) is a separate dataset, [`kasper-cbx/mtg-precon-bench`](https://huggingface.co/datasets/kasper-cbx/mtg-precon-bench).

> Card names are © Wizards of the Coast, and the tag-based lists derive from Scryfall Tagger data.
> Not affiliated with Scryfall or Wizards of the Coast.

## Citation

Cite the tagged release `v1.1`. It adds 3 grades after a scoring fix (see `relevance_grades`); `v1.0` is kept unchanged.

```bibtex
@misc{mtgsearch_bench_2026,
  author       = {Gr{\o}ntved, Kasper},
  title        = {{MTG Search Benchmark}: card search queries, agent search tasks and graded relevance labels for Magic: The Gathering},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/kasper-cbx/mtg-search-bench}},
  note         = {Revision v1.1; cards from kasper-cbx/mtg-search-index@scryfall-20261014090155}
}
```

## Configs

### `queries`: the search benchmark (363 rows)

| column | meaning |
|---|---|
| `set` | `tags` (65), `handwritten` (46), `paraphrase` (222), `deckneeds` (30) |
| `id`, `query` | query id and text |
| `relevant` | `oracle_id`s of the correct cards |
| `graded` | `{oracle_id, grade}` for cards graded 2 (core answers); every other correct card has grade 1 |
| `filters` | JSON: hard constraints the search must apply (colour identity, format, types, mana value); `{}` when none |
| `origin` | for `paraphrase`: the id of the query it rewords |
| `source`, `labeling_rule` | how the correct cards were chosen |

- **The sets:**
  - **`tags`:** 65 Tagger function tags phrased as a player would ask. The correct cards carry the tag or one of its descendants.
  - **`handwritten`:** 46 queries in rules-text wording. The correct cards match a regex over the rules text, checked against Scryfall's own search.
  - **`paraphrase`:** each query from the first two sets reworded twice with no rules-text words (a casual phrasing and a deck-need phrasing). They keep the original's correct cards.
  - **`deckneeds`:** 30 open-ended deck needs. The correct cards are the union of tags and patterns, with the top answers reviewed by hand.
- **Metric:** nDCG@10 over all cards, with funny cards excluded and the query's filters applied.
- **The lists are incomplete,** so absolute scores are lower bounds: compare systems with each other.

### `agent_tasks`: 113 requests for an agent

| column | meaning |
|---|---|
| `id`, `prompt` | task id and the request shown to the agent |
| `kind` | `deck` (65 deck needs), `attr` (15: a card with given attributes, such as a 1-mana red burn instant), `known` (33: one specific card described in other words) |
| `source` | where the task comes from: `deckneeds`, `paraphrase-need`, `otag:<tag>` (attr), `edhrec_rank=<n>` / `known-easy …` (known) |
| `relevant`, `graded` | the automatic list of correct cards (for `known`: the one target) |
| `filters` | JSON: constraints the environment applies to every search |
| `popular` | for `known`: whether the target is a widely played card |

**Protocol:** the agent may make up to 5 tool calls, then submits a ranked list of up to 20 card names, using only cards from its results.
- **Deck and attribute tasks:** P@10, the share of the first 10 cards in the deduplicated list that are relevant under `qrels`. Empty slots count as misses.
- **Specific-card tasks:** whether the target is in the final list.

### `relevance_grades`: every grade from the relevance grader

| column | meaning |
|---|---|
| `experiment`, `grading_run` | `EXP-016` (`judge-full`), `EXP-018` (`judge-ablation`), `EXP-019` (`judge-promptonly`), `fix-2026-10-15` (`judge-facefix`) |
| `task_id`, `oracle_id`, `name` | the graded card |
| `grade` | 0, 1 or 2 |
| `on_list` | the card was already on the task's list |
| `kind` | `pool` (an agent returned it) or `decoy_random` (a hidden random card meeting the task's constraints, a check on the grader) |
| `pass`, `grader_model` | one pass; `claude-sonnet-5-5` |

**What was graded:**
- **EXP-016:** the union of the search agent's and the grep agent's final top 20 on the 80 deck and attribute tasks, including cards already on the list. Two random decoys per task were mixed in.
- **EXP-018 and EXP-019:** only cards that no earlier run had graded and that were not on the list.
- **fix-2026-10-15:** the scorer used to match some answers to the wrong card ("Lightning Bolt" to "Emeritus of Conflict // Lightning Bolt", a double-faced card with that back face). After the fix, 3 cards counted that no run had graded: Reanimate, Lightning Bolt and Sign in Blood, each on one task. They were graded in the same way.

**The grader:** headless Claude Code with no tools, 15 cards per call in a seeded random order. It sees the request (with its constraints) and each card's rules text. It never sees which agent returned a card, the list, or any popularity data. Its instructions:

> Grade each card for how well it satisfies the player's request:
> 2 = clearly satisfies it: the card itself performs the requested function directly and meets every stated constraint (colour, card type, mana value, format, cost).
> 1 = partially satisfies it: a weaker, narrower, conditional or indirect version of the requested function, or a card a reasonable player would still consider for this request; it must still meet the stated constraints.
> 0 = does not satisfy the request, or violates a stated constraint (wrong colour, card type or mana value, etc.).
> Judge only from the card text shown. Do not reward popularity or general power level.

**Checks (EXP-016):** 98.9% of cards already on a list were graded relevant, and 1 of 160 hidden random decoys (0.6%) was accepted.

### `qrels`: final relevance labels for the 113 tasks

| column | meaning |
|---|---|
| `task_id`, `oracle_id`, `name` | |
| `relevance` | 1 or 2 |
| `source` | `list` (on the task's list, with the list's grade) or `grader` (not on the list, graded ≥ 1 by the latest grading run; relevance = that grade) |

A card is relevant if it is on the list, or the grader graded it ≥ 1. A grade of 0 never removes a list card. Cards no agent returned are not graded, so `qrels` is complete only for the systems whose output was graded (the runs above). Grade a new system's unseen cards with the same grader and instructions before comparing.

## Usage

```python
import polars as pl
from huggingface_hub import hf_hub_download

def load(cfg):
    return pl.read_parquet(hf_hub_download("kasper-cbx/mtg-search-bench", f"data/{cfg}.parquet",
                                           repo_type="dataset", revision="v1.1"))

qrels = load("qrels")
rel = {t: set(g["oracle_id"]) for t, g in qrels.group_by("task_id")}  # task_id -> relevant oracle_ids
def p_at_10(task_id, ranked_oracle_ids):
    top = list(dict.fromkeys(ranked_oracle_ids))[:10]
    return sum(o in rel.get(task_id, ()) for o in top) / 10
```

## Results recorded with this benchmark

The agent is Claude Sonnet 5.5 in Claude Code, with one repeat per task.

**Search benchmark** (`queries`), nDCG@10 × 100, with funny cards excluded and filters applied (EXP-001):

| search engine | tags | handwritten | paraphrase | deckneeds |
|---|---|---|---|---|
| mLateOn (late interaction) | 60.5 | 70.7 | 11.9 | 21.1 |
| dense, embeddinggemma-2 | 42.8 | 36.7 | 9.1 | 17.9 |
| BM25 | 36.4 | 61.1 | 2.5 | 8.5 |

**Agent search tasks** (all 113 tasks), relevance by `qrels`:

| agent tools and prompt | P@10, 80 deck + attr | deck (65) | attr (15) | specific card found (33) | tokens / task | experiment |
|---|---|---|---|---|---|---|
| search (mLateOn), minimal prompt | 0.983 | 0.978 | 1.000 | 1.000 | 10.6k | EXP-019 |
| search (mLateOn), EXP-013 prompt | 0.986 | 0.983 | 1.000 | 0.970 | 12.1k | EXP-013 |
| grep, minimal prompt | 0.865 | 0.954 | 0.480 | 0.848 | 14.3k | EXP-019 |
| grep, tuned prompt (layout example, matching note, example pattern) | 0.968 | 0.977 | 0.927 | 0.939 | 20.9k | EXP-013 |

- **The two grep rows** use the same tool and differ only in the prompt.
- **"Minimal"** is the same prompt for search and grep: the task protocol, with no advice on searching.
- **Strictly line-by-line grep:** on a 30-task subset with the minimal prompt (EXP-018), it scored P@10 0.71.

## Limitations

- **The grader is a Claude model,** the same family as the agents it grades. The hidden decoys and the high agreement with the lists bound how lenient it is, but cannot prove it right.
- **Graded only where agents looked.** Cards outside the graded runs' output are unjudged, not irrelevant.
- **Regex lists favour grep,** which can rebuild the pattern. Tag lists miss cards that Tagger has not tagged.
- **One card snapshot.** `oracle_id`s refer to Scryfall's 2026-10-14 Oracle data; the lists are not updated when cards change.

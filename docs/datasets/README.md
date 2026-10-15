# Datasets

The project's data lives in three public Hugging Face datasets. Their cards are kept here (`docs/datasets/*.md`, uploaded as each repo's `README.md`). Cite a tagged revision, so the data is fixed.

| dataset | what | cite revision | commit |
|---|---|---|---|
| [`kasper-cbx/mtg-search-index`](https://huggingface.co/datasets/kasper-cbx/mtg-search-index) | The card corpus (Scryfall Oracle cards, 34,646), tags and rulings; mLateOn embeddings the server searches; eval-only EDHREC rank | `scryfall-20261014090155` | `c2b5455` |
| [`kasper-cbx/mtg-search-bench`](https://huggingface.co/datasets/kasper-cbx/mtg-search-bench) | The search benchmark (363 queries), the 113 agent search tasks, every relevance grade (EXP-016/018/019 and the 2026-10-15 rescore) and the final relevance labels | `v1.1` | `38028e1` |
| [`kasper-cbx/mtg-precon-bench`](https://huggingface.co/datasets/kasper-cbx/mtg-precon-bench) | Precon deck benchmark: official decklists, EDHREC add/cut rates, deck-fit grades (EXP-017) | `v1.0` | `c6fe41f` |

`mtg-search-bench` `v1.0` (`33e2c6b`) is kept; `v1.1` adds 3 grades and rescored qrels after a fix to the agent scorer's name matching (2026-10-15).

Raw run outputs (episodes, transcripts) are mirrored to `kasper-cbx/mtg-search-experiments` by `scripts/sync_results.py`. That is a backup, not a dataset to cite.

**Rebuilding:**
- `mtg-search-index` is rebuilt by `scripts/refresh.py`, weekly, which tags every new snapshot `scryfall-<stamp>`.
- The two benchmark datasets are built by `scripts/publish_datasets.py` (`bench`, `precon-fit`; the base precon files by `bench/precon/build_dataset.py` and `bench/precon/upload_hf.py`).

**Sources and terms:**
- **Card data** comes from [Scryfall](https://scryfall.com)'s bulk data. Card names and rules text are © Wizards of the Coast, used under the [Fan Content Policy](https://company.wizards.com/en/legal/fancontentpolicy). This project is not affiliated with Scryfall or Wizards of the Coast.
- **Precon decklists** come from [MTGJSON](https://mtgjson.com) (MIT); the precon add/cut rates and commander counts come from [EDHREC](https://edhrec.com).
- **The search benchmark** holds our own queries, tasks and grades, keyed by `oracle_id`; its tag-based lists derive from [Scryfall Tagger](https://tagger.scryfall.com).

## BibTeX

```bibtex
@misc{mtgsearch_index_2026,
  author       = {Gr{\o}ntved, Kasper},
  title        = {{MTG Search Index}: Scryfall Oracle cards with mLateOn late-interaction embeddings},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/kasper-cbx/mtg-search-index}},
  note         = {Revision scryfall-20261014090155 (Scryfall oracle\_cards 2026-10-14). Card data from Scryfall, \textcopyright{} Wizards of the Coast}
}

@misc{mtgsearch_bench_2026,
  author       = {Gr{\o}ntved, Kasper},
  title        = {{MTG Search Benchmark}: card search queries, agent search tasks and graded relevance labels for Magic: The Gathering},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/kasper-cbx/mtg-search-bench}},
  note         = {Revision v1.1; cards from kasper-cbx/mtg-search-index@scryfall-20261014090155}
}

@misc{mtgsearch_preconbench_2026,
  author       = {Gr{\o}ntved, Kasper},
  title        = {{MTG Precon Cut/Add Benchmark}: Commander precon decklists with EDHREC cut/add rates and deck-fit grades},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/kasper-cbx/mtg-precon-bench}},
  note         = {Revision v1.0 (EDHREC snapshot 2026-10-14). Add/cut rates from EDHREC; card names \textcopyright{} Wizards of the Coast}
}
```

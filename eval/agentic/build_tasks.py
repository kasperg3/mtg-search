"""Build eval/agentic/tasks.yaml for EXP-006.

    uv run python eval/agentic/build_tasks.py                    # deck + attribute + reviewed known-item tasks
    uv run python eval/agentic/build_tasks.py --gen-known        # (GLM) draft known-item descriptions -> known_drafts.yaml
    uv run python eval/agentic/build_tasks.py --known-easy       # tasks_known_easy.yaml (EXP-013 stratum)

Task kinds (labels are never shown to the agent):
  deck   30 deckneeds queries + 35 seeded-sample paraphrase "-need" queries. Task filters
         (deckneeds only) are enforced by the environment and stated to the agent in plain words.
  attr   15 function + attribute requests. Labels = oracle-tag closure ∩ attribute filter; the
         agent is NOT given the filter, it may pass search filters itself.
  known  25 "find the specific card that ..." tasks; target = 1 oracle_id. Descriptions were
         drafted by GLM-5.3 from the card text (no names, no rules wording, no deck-role framing),
         then hand-reviewed (known_reviewed.yaml).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import polars as pl
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from eval.queries import load_queries  # noqa: E402
from mtg_search.filters import filter_expr  # noqa: E402

N_NEED = 35
SEED = 6

# (id, prompt, tag slugs (union), label filter)
ATTR = [
    ("attr-1cmc-green-dork", "a one-mana green creature that taps for mana", ["mana-dork"], {"colors": "G", "colors_mode": "exact", "cmc_max": 1, "types": ["Creature"]}),
    ("attr-2cmc-rock", "a two-mana artifact that produces mana", ["mana-rock"], {"cmc_min": 2, "cmc_max": 2, "types": ["Artifact"]}),
    ("attr-1cmc-red-burn-instant", "a one-mana red instant that deals damage to any target", ["burn-any"], {"colors": "R", "colors_mode": "exact", "cmc_max": 1, "types": ["Instant"]}),
    ("attr-2cmc-blue-counter", "a two-mana blue instant that counters a spell", ["counterspell"], {"colors": "U", "colors_mode": "exact", "cmc_min": 2, "cmc_max": 2, "types": ["Instant"]}),
    ("attr-white-wrath-sorcery", "a white sorcery that clears the whole board of creatures", ["sweeper"], {"colors": "W", "colors_mode": "exact", "types": ["Sorcery"]}),
    ("attr-green-instant-artifact-hate", "a cheap green instant (two mana or less) that destroys an artifact", ["removal-artifact"], {"colors": "G", "colors_mode": "exact", "cmc_max": 2, "types": ["Instant"]}),
    ("attr-blue-creature-bounce", "a blue creature that returns a permanent to its owner's hand", ["bounce"], {"colors": "U", "colors_mode": "exact", "types": ["Creature"]}),
    ("attr-colorless-draw-artifact", "a colorless artifact costing three or less that draws cards", ["draw"], {"colors": "C", "cmc_max": 3, "types": ["Artifact"]}),
    ("attr-green-land-ramp-sorcery", "a green sorcery that puts lands from my library onto the battlefield", ["land-ramp"], {"colors": "G", "colors_mode": "exact", "types": ["Sorcery"]}),
    ("attr-red-cheap-loot", "a red card with mana value two or less that lets me discard and draw", ["loot", "rummage"], {"colors": "R", "colors_mode": "exact", "cmc_max": 2}),
    ("attr-black-creature-discard", "a black creature with mana value three or less that makes an opponent discard", ["discard"], {"colors": "B", "colors_mode": "exact", "cmc_max": 3, "types": ["Creature"]}),
    ("attr-green-cheap-fog", "a green spell costing two mana or less that prevents combat damage", ["fog"], {"colors": "G", "colors_mode": "exact", "cmc_max": 2}),
    ("attr-white-anthem-enchantment", "a white enchantment that makes all my creatures bigger", ["anthem"], {"colors": "W", "colors_mode": "exact", "types": ["Enchantment"]}),
    ("attr-white-instant-blink", "a white instant that exiles one of my creatures and returns it", ["flicker"], {"colors": "W", "colors_mode": "exact", "types": ["Instant"]}),
    ("attr-black-cheap-tutor", "a black instant or sorcery costing two or less that searches my library for a card", ["tutor-to-hand"], {"colors": "B", "colors_mode": "exact", "cmc_max": 2}),
]

KNOWN_SYSTEM = """You write search tasks for a Magic: The Gathering retrieval benchmark. Given one card, write ONE sentence starting "Find the specific card that" describing it the way a player who half-remembers it would ask a friend.
Rules:
- Never use the card's name or any word from its name, its set, or its creature types.
- Do not copy rules-text phrasing or keyword names; paraphrase in casual language (e.g. "can't be stopped by blockers" not "can't be blocked").
- Describe what the card does (its effect and anything distinctive about it: cost range, colour, card type, odd condition), not which deck it belongs in or what role it plays.
- Include enough detail that a careful player could identify this exact card, but stay a little vague (one or two concrete clues, not a full rules dump).
Reply with only the sentence."""


def tag_closure_map():
    from build_tag_queries import tag_closure

    return tag_closure(pl.read_parquet(ROOT / "data" / "tags.parquet"))


def deck_tasks() -> list[dict]:
    out = []
    for q in load_queries("deckneeds"):
        out.append({"id": q["id"], "kind": "deck", "source": "deckneeds", "prompt": q["query"], "filters": q["filters"],
                    "relevant": q["relevant"], "graded": q["graded"]})
    needs = [q for q in load_queries("paraphrase") if q["id"].endswith("-need")]
    for q in sorted(random.Random(SEED).sample(needs, N_NEED), key=lambda q: q["id"]):
        out.append({"id": q["id"], "kind": "deck", "source": "paraphrase-need", "prompt": q["query"], "filters": {},
                    "relevant": q["relevant"], "graded": q["graded"]})
    return out


def attr_tasks(cards: pl.DataFrame) -> list[dict]:
    cl = tag_closure_map()
    nonfunny = set(cards.filter(~pl.col("is_funny"))["oracle_id"])
    out = []
    for tid, prompt, slugs, filt in ATTR:
        lab = set().union(*(cl[s] for s in slugs))
        ok = set(cards.filter(filter_expr(cards, **filt))["oracle_id"])
        rel = sorted(lab & ok & nonfunny)
        assert len(rel) >= 3, (tid, len(rel))
        out.append({"id": tid, "kind": "attr", "source": "otag:" + "|".join(slugs), "prompt": prompt, "filters": {},
                    "label_filter": filt, "relevant": rel, "graded": {}})
    return out


def pick_known(cards: pl.DataFrame) -> pl.DataFrame:
    """13 popular (edhrec_rank < 5000) + 12 obscure (rank > 15000 or unranked) cards with
    substantive, single-faced rules text."""
    c = cards.filter(
        ~pl.col("is_funny") & (pl.col("layout") == "normal") & (pl.col("doc_text").str.len_chars().is_between(110, 420))
        & ~pl.col("type_line").str.contains("Basic|Token")
    )
    rng = random.Random(SEED)
    pop = c.filter(pl.col("edhrec_rank") < 5000)
    obs = c.filter(pl.col("edhrec_rank").is_null() | (pl.col("edhrec_rank") > 15000)).filter(
        pl.col("legalities").struct.field("legacy") == "legal")
    take = lambda df, n: df[sorted(rng.sample(range(df.height), n))]  # noqa: E731
    return pl.concat([take(pop, 18), take(obs, 17)])  # over-sample; review drops some


def gen_known(cards: pl.DataFrame) -> None:
    from glm import GLM, cost_usd

    from popularity import with_popularity  # eval/popularity.py: the rank is not in the shipped card table

    llm = GLM(temperature=0.3, effort="low")
    picks = pick_known(with_popularity(cards))
    drafts, cost = [], 0.0
    log = open(HERE / "known_gen_transcript.jsonl", "w")
    for r in picks.iter_rows(named=True):
        msgs = [{"role": "system", "content": KNOWN_SYSTEM}, {"role": "user", "content": f"Card:\n{r['doc_text']}"}]
        resp = llm.chat(msgs)
        log.write(json.dumps({"messages": msgs, "raw": resp}, ensure_ascii=False) + "\n")
        cost += cost_usd(resp.get("usage") or {})
        text = (resp["choices"][0]["message"].get("content") or "").strip().strip('"')
        drafts.append({"oracle_id": r["oracle_id"], "name": r["name"], "edhrec_rank": r["edhrec_rank"],
                       "card": r["doc_text"], "prompt": text})
        print(f"- {r['name']} ({r['edhrec_rank']}): {text}")
    (HERE / "known_drafts.yaml").write_text(yaml.safe_dump(drafts, sort_keys=False, allow_unicode=True, width=120))
    print(f"cost ${cost:.4f}")


def known_task(k: dict, prefix: str = "known") -> dict:
    return {"id": f"{prefix}-{re.sub(r'[^a-z0-9]+', '-', k['name'].lower()).strip('-')}", "kind": "known",
            "source": f"edhrec_rank={k['edhrec_rank']}", "prompt": k["prompt"], "filters": {},
            "relevant": [k["oracle_id"]], "graded": {}, "popular": (k["edhrec_rank"] or 10**9) < 5000}


def known_easy() -> None:
    """tasks_known_easy.yaml (EXP-013): the 8 drafts dropped from tasks.yaml only because single-shot BM25
    already ranked the target in its top 3. Dropping them biased the known-item set against lexical
    search, so EXP-013 reports them as a separate stratum. Checked by hand against the card text."""
    kept = {k["oracle_id"] for k in yaml.safe_load((HERE / "known_reviewed.yaml").read_text())}
    out = []
    for k in yaml.safe_load((HERE / "known_drafts.yaml").read_text()):
        r = (k.get("single_shot_rank") or {}).get("bm25")
        if k["oracle_id"] in kept or r is None or r > 3:
            continue
        k = dict(k, prompt=k["prompt"].replace("Find the specific card that cheap ", "Find the specific card that's a cheap "))
        out.append({**known_task(k, "known-easy"), "source": f"known-easy bm25@{r} edhrec_rank={k['edhrec_rank']}"})
    (HERE / "tasks_known_easy.yaml").write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=120))
    print(f"wrote {len(out)} known-easy tasks")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-known", action="store_true")
    ap.add_argument("--known-easy", action="store_true")
    args = ap.parse_args()
    cards = pl.read_parquet(ROOT / "data" / "cards.parquet")
    if args.gen_known:
        return gen_known(cards)
    if args.known_easy:
        return known_easy()
    tasks = deck_tasks() + attr_tasks(cards)
    known = yaml.safe_load((HERE / "known_reviewed.yaml").read_text())
    for k in known:
        tasks.append(known_task(k))
    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids))
    (HERE / "tasks.yaml").write_text(yaml.safe_dump(tasks, sort_keys=False, allow_unicode=True, width=120))
    from collections import Counter

    print(Counter(t["kind"] for t in tasks), "relevant sizes:",
          {t["id"]: len(t["relevant"]) for t in tasks if t["kind"] == "attr"})


if __name__ == "__main__":
    main()

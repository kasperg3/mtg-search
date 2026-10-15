"""EXP-017 deck-aware judging of precon add picks (pre-registration: eval/experiments/precon_judging_plan.md).

    # recognition probe: does the judge know which product an anonymised deck is?
    uv run python -m bench.precon.judge probe --run exp017-probe-dev --split dev --allow-llm
    # judge the add side of finished runs plus blind controls; then score the runs with the grades
    uv run python -m bench.precon.judge judge --run exp017-test --runs RUN_A,RUN_B --rejudge 0.1 --allow-llm
    uv run python -m bench.precon.judge analyze --run exp017-test --runs RUN_A,RUN_B

The judge is headless Claude Code (``claude-sonnet-5-5``, subscription, no tools, empty directory, init
checked; :class:`~bench.precon.harness.claude_code.ClaudeCode`). The system prompt holds the rubric and the
format rules; the user prompt holds the deck (the commander's full text with its name replaced, the other 99
cards as a name list), then BATCH candidate cards with full text, in a seeded random order. The judge never
sees which run proposed a card, any EDHREC data, prices, the product or the deck name; every prompt passes
the EXP-007 leakage pre-flight.

Pool per precon: every add-side pick of the given runs (ADD top 20, SWAP adds) that resolves and is
statically valid (legal, in colour identity, not in the deck). Invalid picks score 0 without a judge.
Blind controls per precon (N_CONTROLS): random legal in-identity cards with no add label (``random``),
the highest-rate EDHREC add-labelled cards no run picked (``top_label``), and non-basic cards already in
the deck (``in_deck``).

Outputs: ``bench/precon/runs/<run>/`` (gitignored: items, calls, transcripts) and the tracked aggregate
``bench/precon/runs/<run>.json`` written by ``analyze``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from bench.precon import scorer  # noqa: E402
from bench.precon.harness.leakage import identifying_strings, preflight  # noqa: E402
from bench.precon.harness.prompts import RULES, anonymise  # noqa: E402
from bench.precon.loader import CardDB, Deck, load_bench, norm_name  # noqa: E402
from mtg_search.render import render_card  # noqa: E402

RUNS = Path(__file__).resolve().parent / "runs"
DATA = "hf:kasper-cbx/mtg-precon-bench@d6bb8ba78f4d5d1d18e62d01fc7db795199586f6"  # dataset of the judged runs
MODEL = "claude-sonnet-5-5"
BATCH = 15
GC_LIMIT = 3
N_CONTROLS = {"random": 6, "top_label": 6, "in_deck": 3}
MAX_INVALID_ACCEPT = 0.15  # validity rule: random or in-deck controls graded >= 1 above this -> run invalid

RUBRICS = {
    # pre-registered; failed the dev validity rule (47% of random controls graded >= 1, mostly filler graded 1)
    "v1": """Grade each candidate for how well it would fit this deck if it were added:
2 = clearly fits: it works with the commander or the deck's evident plan, or fills a role the list is short of (such as ramp, card draw, removal or interaction).
1 = playable but a generic or weak fit: it does something useful for this deck, but neither supports its plan nor fills a clear gap.
0 = poor fit: it works against the plan or does nothing for it; or it breaks a deck-building rule: not legal in Commander, outside the colour identity, already in the deck, or a [Game Changer] when the deck already has {gc_limit} of them.
Judge only from the card text shown and the deck list. Do not reward a card for being widely played or for raw power on its own.""",
    # the one dev-only revision the pre-registration allows: credit means the add improves the deck
    "v2": """Grade each candidate for whether adding it would improve this deck. An added card takes the place of one of the deck's weakest cards, so it earns credit only if it is better for this deck than those.
2 = clearly improves the deck: it works with the commander or the deck's evident plan, or fills a role the list is short of (such as ramp, card draw, removal or interaction), and it is an upgrade over the deck's weakest cards.
1 = a marginal improvement: a reasonable option for this deck, but generic or only slightly better than the deck's weakest cards.
0 = would not improve the deck: filler, no better than what the deck already plays, works against the plan or does nothing for it; or it breaks a deck-building rule: not legal in Commander, outside the colour identity, already in the deck, or a [Game Changer] when the deck already has {gc_limit} of them.
Judge only from the card text shown and the deck list. Do not reward a card for being widely played; judge what its text does for this deck.""",
}
RUBRIC = "v2"

JUDGE_SYSTEM = """You are an expert Magic: The Gathering Commander deck builder grading candidate cards for one specific deck.

{rules}

{rubric}

Reply with JSON only: {{"grades": [{{"i": <card number>, "g": <0|1|2>}}, ...]}} with one entry per candidate."""

PROBE_SYSTEM = "You are an expert on Magic: The Gathering Commander preconstructed decks."
PROBE_USER = """{deck}

Which preconstructed Commander deck is this? Give the commander's card name, the deck's name and the product it was sold in. Give your best guess even if you are unsure. Reply with JSON only: {{"commander": "...", "deck_name": "...", "product": "...", "confidence": "high" | "medium" | "low"}}"""


# =========================================================================== prompts

def _render(row: dict) -> str:
    return render_card({**row, "tags": None}, detail="full", fmt="commander")


def deck_text(deck: Deck, rows: dict[str, dict], db: CardDB) -> str:
    """The commander(s) in full with the name replaced, then the other cards as names sorted by
    (land last, mana value, name), so the order carries no information about the source."""
    cmd = list(deck.commander_oids)

    def anon(t: str) -> str:
        for c in cmd:
            t = anonymise(t, rows[c]["name"])
        return t

    out = ["COMMANDER" + ("S" if len(cmd) > 1 else "")] + [anon(_render(rows[o])) for o in cmd]
    rest = sorted((o for o in deck.qty if o not in set(cmd)), key=lambda o: (db[o].is_land, db[o].cmc, db[o].name))
    out += ["", f"THE OTHER {sum(deck.qty[o] for o in rest)} CARDS"]
    out += [anon((f"{deck.qty[o]}x " if deck.qty[o] > 1 else "") + db[o].name) for o in rest]
    out += ["", f"Cards marked [Game Changer] in the deck: {scorer.deck_gc_count(deck, db)}."]
    return "\n".join(out)


def judge_system(deck: Deck, rubric: str = RUBRIC) -> str:
    ci = "".join(c for c in "WUBRG" if c in deck.color_identity) or "colourless"
    return JUDGE_SYSTEM.format(rules=RULES.format(ci=ci, bracket=3, gc_limit=GC_LIMIT),
                               rubric=RUBRICS[rubric].format(gc_limit=GC_LIMIT))


def judge_user(deck: Deck, oids: list[str], rows: dict[str, dict], db: CardDB) -> str:
    """The deck goes in the user prompt, as for the agents (the gold-name check covers the system prompt)."""
    cands = "\n\n".join(f"[{i}]\n{_render(rows[o])}" for i, o in enumerate(oids, 1))
    return f"THE DECK\n\n{deck_text(deck, rows, db)}\n\nCANDIDATE CARDS\n\n{cands}"


def parse_grades(text: str, n: int) -> dict[int, int]:
    """Grades from the first JSON object with a "grades" list; text around it is ignored."""
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text or ""):
        try:
            obj, _ = dec.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("grades"), list):
            try:
                out = {int(g["i"]): int(g["g"]) for g in obj["grades"]}
            except (ValueError, KeyError, TypeError):
                return {}
            return {i: g for i, g in out.items() if 1 <= i <= n and g in (0, 1, 2)}
    return {}


# =========================================================================== pool, controls, batches

def add_side(e: dict) -> list:
    """The ranked add picks of one episode, truncated as the scorer does."""
    out = e.get("output") or {}
    if e["task"] == "add":
        return list(out.get("adds") or [])[:20]
    if e["task"] == "swap":
        return [s.get("add") for s in list(out.get("swaps") or [])[:10] if isinstance(s, dict)]
    return []


def load_episodes(run_ids: list[str]) -> dict[str, list[dict]]:
    return {r: [json.loads(x) for x in (RUNS / r / "episodes.jsonl").read_text().splitlines()] for r in run_ids}


def build_pool(episodes: dict[str, list[dict]], decks: dict[str, Deck], db: CardDB) -> dict[str, set[str]]:
    pool: dict[str, set[str]] = {}
    for es in episodes.values():
        for e in es:
            picks = scorer.validate_adds(add_side(e), decks[e["precon_id"]], db, gc_limit=None)
            pool.setdefault(e["precon_id"], set()).update(p.oid for p in picks if p.valid)
    return pool


def controls(deck: Deck, pool: set[str], labels: list[dict], db: CardDB, seed: str) -> list[tuple[str, str]]:
    rng = random.Random(f"controls-{seed}-{deck.precon_id}")
    qty = deck.qty
    labelled = {lab["oracle_id"] for lab in labels if lab["kind"].startswith("add")}

    def addable(o: str) -> bool:
        c = db[o]
        return c.commander_legal and c.color_identity <= deck.color_identity and o not in qty and not c.is_basic and not c.is_funny

    rand = [o for o in sorted(db.cards) if o not in pool and o not in labelled and addable(o)]
    top, seen = [], set()
    for lab in sorted((lab for lab in labels if lab["kind"] == "add"), key=lambda lab: (-(lab["rate"] or 0), lab["oracle_id"])):
        o = lab["oracle_id"]
        if o in db and o not in pool and o not in seen and addable(o):
            top.append(o)
            seen.add(o)
    in_deck = [o for o in sorted(qty) if o not in set(deck.commander_oids) and not db[o].is_basic]
    return ([(o, "random") for o in rng.sample(rand, N_CONTROLS["random"])]
            + [(o, "top_label") for o in top[: N_CONTROLS["top_label"]]]
            + [(o, "in_deck") for o in rng.sample(in_deck, N_CONTROLS["in_deck"])])


def make_jobs(items: dict[str, list[tuple[str, str]]], rejudge: float, seed: str) -> list[tuple[str, int, int, list[str]]]:
    """Pass 0: each precon's cards shuffled and split into near-equal batches of at most BATCH.
    Pass 1: a seeded ``rejudge`` share of those batches again, in a new order."""
    jobs = []
    for pid in sorted(items):
        oids = [o for o, _ in items[pid]]
        random.Random(f"judge-{seed}-{pid}-p0").shuffle(oids)
        nb = -(-len(oids) // BATCH)
        sizes = [len(oids) // nb + (i < len(oids) % nb) for i in range(nb)]
        starts = [sum(sizes[:i]) for i in range(nb)]
        jobs += [(pid, 0, k, oids[s : s + n]) for k, (s, n) in enumerate(zip(starts, sizes, strict=True))]
    again = random.Random(f"rejudge-{seed}").sample(jobs, round(rejudge * len(jobs)))
    for pid, _, k, oids in sorted(again):
        b = list(oids)
        random.Random(f"judge-{seed}-{pid}-b{k}-p1").shuffle(b)
        jobs.append((pid, 1, k, b))
    return jobs


# =========================================================================== running the judge

class Setup:
    """Bench, card db, card rows and leakage inputs shared by probe / judge / analyze."""

    def __init__(self, data: str = DATA):
        import polars as pl

        self.bench, self.db = load_bench(data), CardDB.from_parquet()
        cards = pl.read_parquet(ROOT / "data" / "cards.parquet")
        self._cards = cards
        self.rows: dict[str, dict] = {}
        self.forbidden = self.bench.gold_names(["dev", "test"])
        self.identifying = identifying_strings(self.bench.precons)

    def need(self, oids) -> None:
        import polars as pl

        miss = [o for o in set(oids) if o not in self.rows]
        if miss:
            self.rows.update({r["oracle_id"]: r for r in self._cards.filter(pl.col("oracle_id").is_in(miss)).iter_rows(named=True)})

    def checked(self, deck: Deck, system: str, user: str, candidates: list[str] = ()) -> None:
        """EXP-007 pre-flight. Candidate cards are input like the deck's own cards, so an identifier that is
        part of a card's name (the card "The Masters of Evil") is exempt; one anywhere else is not."""
        names = [self.db[o].name for o in [*deck.qty, *candidates] if o in self.db]
        preflight(system, user, deck_names=names, forbidden_gold=self.forbidden, identifying=self.identifying)


def _usage(res: dict) -> dict:
    u = res.get("usage") or {}
    return {"input": u.get("input_tokens") or 0, "cache_write": u.get("cache_creation_input_tokens") or 0,
            "cache_read": u.get("cache_read_input_tokens") or 0, "output": u.get("output_tokens") or 0}


def _git_guard(allow_dirty: bool) -> dict:
    from bench.precon.run import git_info

    g = git_info()
    if g["dirty_bench"] and not allow_dirty:
        sys.exit("uncommitted changes under bench/precon or src: commit before a model run (or pass --allow-dirty)")
    return g


def cmd_judge(args) -> None:
    from bench.precon.harness.claude_code import ClaudeCode

    git = None if args.dry_run else _git_guard(args.allow_dirty)
    S = Setup(args.data)
    run_ids = args.runs.split(",")
    eps = load_episodes(run_ids)
    pool = build_pool(eps, S.bench.decks, S.db)
    out = RUNS / args.run
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    items = {pid: [(o, "pool") for o in sorted(pool[pid])]
             + controls(S.bench.decks[pid], pool[pid], S.bench.labels.get(pid, []), S.db, args.seed) for pid in sorted(pool)}
    (out / "items.json").write_text(json.dumps(items))
    jobs = make_jobs(items, args.rejudge, args.seed)
    S.need({o for v in items.values() for o, _ in v} | {o for pid in items for o in S.bench.decks[pid].qty})
    prompts = {}
    for pid, p, k, oids in jobs:  # pre-flight every prompt before the first call
        deck = S.bench.decks[pid]
        system, user = judge_system(deck, args.rubric), judge_user(deck, oids, S.rows, S.db)
        S.checked(deck, system, user, oids)
        prompts[(pid, p, k)] = (system, user)
    calls_path = out / "calls.jsonl"
    kept = [c for c in map(json.loads, calls_path.read_text().splitlines()) if len(c["grades"]) == len(c["oids"])] \
        if calls_path.exists() else []  # resume: keep fully graded calls, redo the rest
    calls_path.write_text("".join(json.dumps(c) + "\n" for c in kept))
    done = {(c["precon_id"], c["pass"], c["batch"]) for c in kept}
    todo = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    print(f"{sum(map(len, items.values()))} cards ({sum(len(v) for v in pool.values())} pool) over {len(items)} precons; "
          f"{len(jobs)} calls ({len(todo)} to run)", flush=True)
    if args.dry_run:
        return
    cc = ClaudeCode(model=args.model, timeout=600, retries=2, backoff=30)
    (out / "config.json").write_text(json.dumps({
        "experiment": "EXP-017", "runs": run_ids, "data": args.data, "model": args.model, "cli": cc.version, "git": git,
        "batch": BATCH, "rejudge": args.rejudge, "seed": args.seed, "n_controls": N_CONTROLS,
        "system_template": JUDGE_SYSTEM, "rubric": args.rubric, "rubric_text": RUBRICS[args.rubric]}, indent=1))
    lock = threading.Lock()

    def one(job):
        pid, p, k, oids = job
        system, user = prompts[(pid, p, k)]
        tpath = out / "transcripts" / f"{pid}__p{p}__b{k}.jsonl"
        tpath.write_text(json.dumps({"event": "config", "precon_id": pid, "pass": p, "batch": k, "oracle_ids": oids,
                                     "system": system, "user": user, "llm": cc.config()}, ensure_ascii=False) + "\n")
        t0, grades, res, n_asks = time.time(), {}, {}, 0
        for _ in range(2):  # re-ask once on malformed JSON
            n_asks += 1
            res = cc._retrying(system, user, tpath, None)
            grades = parse_grades(res.get("result") or "", len(oids))
            if len(grades) == len(oids):
                break
        row = {"precon_id": pid, "pass": p, "batch": k, "oids": oids, "grades": {str(i): g for i, g in grades.items()},
               "usage": _usage(res), "turns": res.get("num_turns"), "asks": n_asks, "wall_s": round(time.time() - t0, 1)}
        with lock, open(calls_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        for n, r in enumerate(ex.map(one, todo), 1):
            print(f"[{n}/{len(todo)}] {r['precon_id']} p{r['pass']} b{r['batch']}: {len(r['grades'])}/{len(r['oids'])} graded, "
                  f"{sum(r['usage'].values()):,} tokens, {r['wall_s']}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s; calls in {calls_path.relative_to(ROOT)}")


def _match(guess, truth) -> bool:
    g, t = norm_name(guess or ""), norm_name(truth or "")
    return bool(g and t) and (g == t or (len(g) >= 5 and g in t) or (len(t) >= 5 and t in g))


STOP = {"commander", "deck", "decks", "precon", "the", "of", "and", "edition", "collection"}


def _product_match(guess, truth) -> bool:
    toks = {w for w in norm_name(truth or "").split() if len(w) >= 4 and w not in STOP}
    return bool(toks) and bool(toks & set(norm_name(guess or "").split()))


def cmd_probe(args) -> None:
    from bench.precon.harness.claude_code import ClaudeCode

    git = _git_guard(args.allow_dirty)
    S = Setup(args.data)
    pids = S.bench.precon_ids(args.split)
    S.need({o for pid in pids for o in S.bench.decks[pid].qty})
    out = RUNS / args.run
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    users = {}
    for pid in pids:  # pre-flight every prompt before the first call
        users[pid] = PROBE_USER.format(deck=deck_text(S.bench.decks[pid], S.rows, S.db))
        S.checked(S.bench.decks[pid], PROBE_SYSTEM, users[pid])
    cc = ClaudeCode(model=args.model, timeout=600, retries=2, backoff=30)

    def one(pid):
        deck, user = S.bench.decks[pid], users[pid]
        tpath = out / "transcripts" / f"{pid}__probe.jsonl"
        tpath.write_text(json.dumps({"event": "config", "precon_id": pid, "system": PROBE_SYSTEM, "user": user,
                                     "llm": cc.config()}, ensure_ascii=False) + "\n")
        t0 = time.time()
        res = cc._retrying(PROBE_SYSTEM, user, tpath, None)
        m = re.search(r"\{.*\}", res.get("result") or "", re.DOTALL)
        try:
            ans = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            ans = {}
        meta = S.bench.precons.get(pid, {})
        cmd_names = list(deck.commander_names) + [n.split(",")[0] for n in deck.commander_names]
        return {"precon_id": pid, "answer": ans,
                "commander": any(_match(ans.get("commander"), n) for n in cmd_names),
                "deck_name": _match(ans.get("deck_name"), meta.get("deck_name")),
                "product": _product_match(ans.get("product"), meta.get("product")),
                "truth": {"deck_name": meta.get("deck_name"), "product": meta.get("product"), "commanders": deck.commander_names},
                "usage": _usage(res), "turns": res.get("num_turns"), "wall_s": round(time.time() - t0, 1)}

    with ThreadPoolExecutor(args.workers) as ex:
        rows = list(ex.map(one, pids))
    (out / "probe.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    n = len(rows)
    summary = {"experiment": "EXP-017", "run_id": args.run, "kind": "recognition probe", "split": args.split,
               "created": datetime.now(UTC).isoformat(timespec="seconds"), "git": git, "data": args.data,
               "model": args.model, "cli": cc.version, "n_decks": n,
               "recognised": {k: sum(r[k] for r in rows) / n for k in ("commander", "deck_name", "product")},
               "deck_or_product": sum(r["deck_name"] or r["product"] for r in rows) / n,
               "confidence": {c: sum((r["answer"].get("confidence") == c) for r in rows) for c in ("high", "medium", "low")},
               "cost": _cost(rows)}
    (RUNS / f"{args.run}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


# =========================================================================== analysis

def kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else None


def slot_grades(e: dict, deck: Deck, db: CardDB, grades: dict[str, int]) -> list[int | None]:
    """Judged grade per add slot of one episode; invalid picks (scorer rules, Game Changer limit in rank
    order) are 0 and valid picks the judge did not grade are None."""
    out = e.get("output") or {}
    if e["task"] == "add":
        picks = scorer.validate_adds(list(out.get("adds") or [])[:20], deck, db, gc_limit=GC_LIMIT)
    else:
        swaps = [s for s in list(out.get("swaps") or [])[:10] if isinstance(s, dict)]
        removed = [p.oid for p in scorer.validate_cuts([s.get("cut") for s in swaps], deck, db) if p.valid]
        picks = scorer.validate_adds([s.get("add") for s in swaps], deck, db, gc_limit=GC_LIMIT, removed=removed)
    return [grades.get(p.oid) if p.valid else 0 for p in picks]


def fit(gs: list[int | None], k: int, two: bool = False) -> float | None:
    """Mean of grade/2 (or share of grade 2) over k slots; missing slots count 0, ungraded are dropped."""
    slots = [g for g in (list(gs[:k]) + [0] * max(0, k - len(gs)))[:k] if g is not None]
    if not slots:
        return None
    return sum((g == 2) if two else g / 2 for g in slots) / len(slots)


def episode_metrics(e: dict, deck: Deck, db: CardDB, grades: dict[str, int]) -> dict:
    gs = slot_grades(e, deck, db, grades)
    if e["task"] == "add":
        return {"fit@20": fit(gs, 20), "fit@10": fit(gs, 10), "share2@20": fit(gs, 20, True),
                "ungraded": sum(g is None for g in gs)}
    return {"swap.fit@10": fit(gs, 10), "swap.share2@10": fit(gs, 10, True), "ungraded": sum(g is None for g in gs)}


def _cost(rows: list[dict]) -> dict:
    tot = {k: sum(r["usage"][k] for r in rows) for k in ("input", "cache_write", "cache_read", "output")}
    n = len(rows)
    return {"calls": n, "asks": sum(r.get("asks", 1) for r in rows), "tokens": tot, "tokens_total": sum(tot.values()),
            "tokens_per_call": sum(tot.values()) / n if n else None,
            "turns_per_call": sum(r.get("turns") or 0 for r in rows) / n if n else None,
            "wall_s_per_call": sum(r["wall_s"] for r in rows) / n if n else None,
            "wall_s_sum": sum(r["wall_s"] for r in rows)}


def cmd_analyze(args) -> dict:
    S = Setup(args.data)
    run_ids = args.runs.split(",")
    out = RUNS / args.run
    items = {pid: [tuple(x) for x in v] for pid, v in json.loads((out / "items.json").read_text()).items()}
    kind = {(pid, o): k for pid, v in items.items() for o, k in v}
    calls = [json.loads(x) for x in (out / "calls.jsonl").read_text().splitlines()]
    g: dict[int, dict[tuple[str, str], int]] = {0: {}, 1: {}}
    for c in calls:
        for i, o in enumerate(c["oids"], 1):
            if str(i) in c["grades"]:
                g[c["pass"]][(c["precon_id"], o)] = c["grades"][str(i)]
    by_pid = {pid: {o: gr for (p, o), gr in g[0].items() if p == pid} for pid in items}
    eps = load_episodes(run_ids)
    arms, per = {}, {}
    for r, es in eps.items():
        rows = [{"precon_id": e["precon_id"], "rep": e["rep"], "task": e["task"],
                 "metrics": {**episode_metrics(e, S.bench.decks[e["precon_id"]], S.db, by_pid.get(e["precon_id"], {})),
                             **{m: e["metrics"].get(m) for m in ("add.ndcg@10", "add.ndcg@20", "swap.add.ndcg@10",
                                                                 "add.validity", "swap.add.validity")}}}
                for e in es if e["task"] in ("add", "swap")]
        per[r] = rows
        add, swap = [x for x in rows if x["task"] == "add"], [x for x in rows if x["task"] == "swap"]
        arms[r] = {**scorer.aggregate(add, ["fit@20", "fit@10", "share2@20", "add.ndcg@10", "add.ndcg@20", "add.validity"]),
                   **scorer.aggregate(swap, ["swap.fit@10", "swap.share2@10", "swap.add.ndcg@10", "swap.add.validity"]),
                   "ungraded_slots": sum(x["metrics"]["ungraded"] for x in rows)}
    a, b = run_ids[0], run_ids[1]
    paired = {}
    for m, task in [("fit@20", "add"), ("fit@10", "add"), ("share2@20", "add"), ("add.ndcg@10", "add"),
                    ("swap.fit@10", "swap"), ("swap.share2@10", "swap"), ("swap.add.ndcg@10", "swap")]:
        pa = scorer.per_precon([x for x in per[a] if x["task"] == task], m)
        pb = scorer.per_precon([x for x in per[b] if x["task"] == task], m)
        paired[m] = scorer.paired_bootstrap(pa, pb, 10_000)
    # agreement with the EDHREC labels, on judged pool cards
    th = scorer.ScoreConfig().add_thresholds
    lab: dict[tuple[str, str], int] = {}
    for pid, xs in S.bench.labels.items():
        for x in xs:
            if x["kind"].startswith("add"):
                k = (pid, x["oracle_id"])
                lab[k] = max(lab.get(k, 0), scorer.grade(x["rate"], th))
    pool_j = [(k, gr) for k, gr in g[0].items() if kind.get(k) == "pool"]
    is_lab = [lab.get(k, 0) >= 1 for k, _ in pool_j]
    ok = [gr >= 1 for _, gr in pool_j]

    def share(xs):
        return sum(xs) / len(xs) if xs else None

    agreement = {"n_pool_judged": len(pool_j), "n_labelled": sum(is_lab),
                 "labelled_judged_ge1": share([o for o, li in zip(ok, is_lab, strict=True) if li]),
                 "unlabelled_judged_ge1": share([o for o, li in zip(ok, is_lab, strict=True) if not li]),
                 "labelled_judged_2": share([gr == 2 for (_, gr), li in zip(pool_j, is_lab, strict=True) if li]),
                 "unlabelled_judged_2": share([gr == 2 for (_, gr), li in zip(pool_j, is_lab, strict=True) if not li]),
                 "kappa_label_vs_judged_ge1": kappa(is_lab, ok)}
    ctrl = {}
    for kd in ("pool", *N_CONTROLS):
        gs = [gr for k, gr in g[0].items() if kind.get(k) == kd]
        ctrl[kd] = {"n": len(gs), "ge1": share([x >= 1 for x in gs]), "eq2": share([x == 2 for x in gs]),
                    "mean_grade": share(gs)}
    valid = ctrl["random"]["ge1"] <= MAX_INVALID_ACCEPT and ctrl["in_deck"]["ge1"] <= MAX_INVALID_ACCEPT
    both = [k for k in g[1] if k in g[0]]
    consistency = {"n": len(both), "same_grade": share([g[0][k] == g[1][k] for k in both]),
                   "same_ge1": share([(g[0][k] >= 1) == (g[1][k] >= 1) for k in both]),
                   "kappa_ge1": kappa([g[0][k] >= 1 for k in both], [g[1][k] >= 1 for k in both])}
    n_items = sum(map(len, items.values()))
    summary = {"experiment": "EXP-017", "run_id": args.run, "created": datetime.now(UTC).isoformat(timespec="seconds"),
               "config": json.loads((out / "config.json").read_text()) | {"system_template": None},
               "runs": run_ids, "n_precons": len(items), "n_cards": n_items, "n_graded_pass0": len(g[0]),
               "valid": valid, "validity_rule": f"random and in_deck controls graded >= 1 at most {MAX_INVALID_ACCEPT:.0%}",
               "arms": arms, "paired": {f"{a} - {b}": paired}, "agreement": agreement, "controls": ctrl,
               "consistency": consistency, "cost": _cost(calls)}
    (RUNS / f"{args.run}.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    _print(summary)
    return summary


def _print(s: dict) -> None:
    print(f"== {s['run_id']}: {s['n_precons']} precons, {s['n_cards']} cards, valid={s['valid']}")
    for r, m in s["arms"].items():
        print(f"  {r}: " + ", ".join(f"{k} {v['mean']:.3f}" for k, v in m.items() if isinstance(v, dict)))
    for pair, ms in s["paired"].items():
        print(f"  {pair}:")
        for k, v in ms.items():
            if v.get("n"):
                print(f"    {k:18s} {v['mean_diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] sign p={v['sign_test_p']:.3f} n={v['n']}")
    print("  controls:", {k: (v["n"], round(v["ge1"] or 0, 3), round(v["eq2"] or 0, 3)) for k, v in s["controls"].items()})
    print("  agreement:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s["agreement"].items()})
    print("  consistency:", s["consistency"])
    print("  cost:", s["cost"])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("probe", "judge", "analyze"):
        sp = sub.add_parser(name)
        sp.add_argument("--run", required=True)
        sp.add_argument("--data", default=DATA)
        if name != "probe":
            sp.add_argument("--runs", required=True, help="comma-separated precon run ids; analyze reports first - second")
        if name != "analyze":
            sp.add_argument("--model", default=MODEL)
            sp.add_argument("--workers", type=int, default=4)
            sp.add_argument("--allow-llm", action="store_true", help="required: judge calls run on the subscription")
            sp.add_argument("--allow-dirty", action="store_true")
        if name == "probe":
            sp.add_argument("--split", default="dev")
        if name == "judge":
            sp.add_argument("--rejudge", type=float, default=0.1)
            sp.add_argument("--rubric", default=RUBRIC, choices=sorted(RUBRICS))
            sp.add_argument("--seed", default="exp017")
            sp.add_argument("--dry-run", action="store_true", help="build pool, controls and prompts, run pre-flight, no calls")
    args = ap.parse_args(argv)
    if args.cmd != "analyze" and not getattr(args, "dry_run", False) and not args.allow_llm:
        sys.exit("refusing to call the judge without --allow-llm")
    {"probe": cmd_probe, "judge": cmd_judge, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()

"""EXP-007 data build: MTGJSON precon decklists + EDHREC precon cut/add labels -> parquet dataset.

    uv run python bench/precon/collect_mtgjson.py               # once: MTGJSON downloads (cached)
    uv run python bench/precon/collect_edhrec.py --snapshot D   # once per snapshot: polite EDHREC fetch
    uv run python bench/precon/build_dataset.py --snapshot D [--cutoff 2025-11-01]
    uv run python bench/precon/build_dataset.py --snapshot D --splits-only --cutoff YYYY-MM-DD

Output: data/bench/precon/<snapshot>/{decks,labels,precons,commanders,splits}.parquet,
tag_denylist.json, tag_audit.parquet, manifest.json, unmapped.json. Schema: bench/precon/schema.md.
Never fetches from EDHREC itself (reads the on-disk cache only).
"""
from __future__ import annotations

import argparse
import fnmatch
import gzip
import hashlib
import json
import re
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edhrec_fetch import BASE, MIN_INTERVAL, RAW, USER_AGENT, PoliteFetcher, next_data  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MTGJSON = ROOT / "data" / "bench_raw" / "mtgjson"
CARDS = ROOT / "data" / "cards.parquet"
OUT = ROOT / "data" / "bench" / "precon"

DEFAULT_CUTOFF = "2025-11-01"  # GLM-5.3 cutoff unknown; see schema.md
MIN_DECKS = 300
MIN_AGE_DAYS = 56  # 8 weeks
EXCLUDED_SETS = {"TD0"}  # MTGO theme decks (plan §8)
CUT_THRESH = (0.50, 0.30, 0.15)
ADD_THRESH = (0.30, 0.15, 0.05)
KIND_BY_HEADER = {
    "Cards to Cut": "cut",
    "Lands to Cut": "cut_land",
    "Cards to Add": "add",
    "Lands to Add": "add_land",
}
# Plan §7 T3: evaluative / popularity-like / precon-identifying oracle tags (eval-mode deny-list).
TAG_DENYLIST = [
    "meme",
    "weaker-in-singleton-formats",
    "stronger-in-singleton-formats",
    "staple-with-set-s-mechanic",
    "planeswalker-deck-staples",
]
TAG_DENY_PATTERNS = ["cycle-*-face-commander", "*precon*", "*staple*"]
WUBRG = "WUBRG"


# ----------------------------------------------------------------------------- helpers
def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = s.replace("’", "'").lower().strip()
    return re.sub(r"\s+", " ", s)


def precon_id(file_name: str) -> str:
    return "pc-" + hashlib.sha1(file_name.encode()).hexdigest()[:10]


def gain(rate: float | None, thresholds) -> int:
    if rate is None:
        return 0
    for g, t in zip((3, 2, 1), thresholds):
        if rate >= t:
            return g
    return 0


DATA_CODE = [f"bench/precon/{f}" for f in ("build_dataset.py", "collect_mtgjson.py", "collect_edhrec.py",
                                           "edhrec_fetch.py", "upload_hf.py", "dataset_card.md")]


def git_sha() -> str:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", *DATA_CODE], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    return sha + ("-dirty" if dirty else "")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def denied(tag: str) -> bool:
    return tag in TAG_DENYLIST or any(fnmatch.fnmatch(tag, p) for p in TAG_DENY_PATTERNS)


# ----------------------------------------------------------------------------- card index
class CardIndex:
    def __init__(self):
        df = pl.read_parquet(CARDS)
        self.df = df
        self.by_oid = {r["oracle_id"]: r for r in df.select(
            "oracle_id", "name", "face_names", "type_line", "supertypes", "types", "color_identity",
            "legalities", "is_funny", "layout", "faces", "tags",
        ).iter_rows(named=True)}
        full, face = {}, defaultdict(list)
        for oid, r in self.by_oid.items():
            k = norm(r["name"])
            if k not in full or (full[k][1] and not r["is_funny"]):
                full[k] = (oid, r["is_funny"])
            for f in r["face_names"] or []:
                face[norm(f)].append((oid, r["is_funny"]))
        self.full = {k: v[0] for k, v in full.items()}
        self.face = {}
        for k, lst in face.items():
            serious = [o for o, fun in lst if not fun] or [o for o, _ in lst]
            if len(set(serious)) == 1:
                self.face[k] = serious[0]
        self.flavor: dict[str, str] = {}  # MTGJSON flavorName (UB in-universe names) -> oracle_id

    def resolve(self, name: str) -> str | None:
        k = norm(name)
        if k in self.full:
            return self.full[k]
        if k in self.face:
            return self.face[k]
        if " // " in k:  # "A // B" where cards.parquet uses another joiner, or front-only
            front = k.split(" // ")[0]
            return self.full.get(front) or self.face.get(front)
        return self.flavor.get(k)

    def is_land(self, oid: str) -> bool:
        r = self.by_oid[oid]
        front = (r["faces"] or [{}])[0].get("type_line") or r["type_line"]
        return "Land" in front.split("—")[0]

    def is_basic(self, oid: str) -> bool:
        return "Basic" in (self.by_oid[oid]["supertypes"] or [])


# ----------------------------------------------------------------------------- MTGJSON
def load_mtgjson(ci: CardIndex):
    sets = {s["code"]: s for s in json.load(open(MTGJSON / "SetList.json"))["data"]}
    decklist = json.load(open(MTGJSON / "DeckList.json"))["data"]
    meta = json.load(open(MTGJSON / "Meta.json"))["data"]
    entries = [d for d in decklist if d["type"] in ("Commander Deck", "MTGO Commander Deck")]
    decks, unmapped, dup_editions = [], [], []
    for e in entries:
        if "Collector" in e["name"] and "Edition" in e["name"]:
            dup_editions.append(e["fileName"])  # same list as the regular deck, alt printings
            continue
        p = next((MTGJSON / "decks").rglob(e["fileName"] + ".json"), None)
        if p is None:
            raise FileNotFoundError(e["fileName"])
        d = json.load(open(p))["data"]
        cards: dict[str, dict] = {}
        commanders, ub = [], False
        for board in ("commander", "mainBoard"):
            seen_uuid = set()
            for c in d.get(board, []):
                if c["uuid"] in seen_uuid:
                    continue
                seen_uuid.add(c["uuid"])
                oid = c.get("identifiers", {}).get("scryfallOracleId")
                if c.get("flavorName") and oid in ci.by_oid:
                    ci.flavor.setdefault(norm(c["flavorName"]), oid)
                how = "oracle_id"
                if oid not in ci.by_oid:
                    oid, how = ci.resolve(c["name"]), "name"
                if oid is None:
                    unmapped.append({"source": "mtgjson", "deck": e["fileName"], "name": c["name"],
                                     "set": c.get("setCode"), "number": c.get("number")})
                    continue
                if board == "commander":
                    commanders.append(oid)
                    ub |= "universesbeyond" in (c.get("promoTypes") or []) or c.get("securityStamp") == "triangle"
                slot = cards.setdefault(oid, {"oracle_id": oid, "name": ci.by_oid[oid]["name"], "qty": 0,
                                              "is_land": ci.is_land(oid), "is_basic": ci.is_basic(oid),
                                              "is_commander": board == "commander", "_how": how})
                slot["qty"] += int(c.get("count", 1))
        code = d["code"]
        parent = sets.get(code, {}).get("parentCode")
        decks.append({
            "precon_id": precon_id(e["fileName"]),
            "mtgjson_file": e["fileName"],
            "mtgjson_type": e["type"],
            "deck_name": d["name"],
            "set_code": code,
            "parent_set_code": parent,
            "product": sets.get(code, {}).get("name"),
            "release_date": date.fromisoformat(d.get("releaseDate") or e["releaseDate"]),
            "source": d.get("source"),
            "commander_oracle_ids": commanders,
            "cards": list(cards.values()),
            "is_universes_beyond": ub,
            "is_secret_lair": code == "SLD",
        })
    return decks, unmapped, dup_editions, sets, meta


def first_printing_index(sets) -> dict[str, str]:
    """oracle_id -> set code of its earliest printing (MTGJSON AtomicCards)."""
    atomic = json.load(gzip.open(MTGJSON / "AtomicCards.json.gz"))["data"]
    out = {}
    for faces in atomic.values():
        for c in faces:
            oid = c.get("identifiers", {}).get("scryfallOracleId")
            if not oid or oid in out:
                continue
            prs = [s for s in c.get("printings", []) if s in sets]
            if prs:
                out[oid] = min(prs, key=lambda s: (sets[s]["releaseDate"], sets[s].get("isOnlineOnly", False)))
    return out


def product_family(code: str, sets) -> set[str]:
    parent = sets.get(code, {}).get("parentCode")
    roots = {code} | ({parent} if parent else set())
    return roots | {c for c, s in sets.items() if s.get("parentCode") in roots}


# ----------------------------------------------------------------------------- EDHREC
def load_edhrec(snapshot: str, ci: CardIndex):
    f = PoliteFetcher.__new__(PoliteFetcher)  # cache-only reader: never touches the network
    f.dir = RAW / snapshot
    cache = lambda url: f.dir / PoliteFetcher.cache_path(f, url).name  # noqa: E731
    idx_html = cache(f"{BASE}/precon")
    idx = next_data(idx_html.read_text())
    build_id = idx.get("buildId")
    entries = []
    for cl in idx["props"]["pageProps"]["data"]["container"]["json_dict"]["cardlists"]:
        for cv in cl["cardviews"]:
            entries.append({"group": cl["header"], **cv})
    pages, missing, unmapped = [], [], []
    for e in entries:
        p = cache(BASE + e["url"])
        if not p.exists():
            missing.append(e["url"])
            continue
        nd = next_data(p.read_text())
        dat = nd["props"]["pageProps"]["data"]
        jd = dat["container"]["json_dict"]
        slug = e["url"].removeprefix("/precon/")
        cmd_names = [c[0] for c in dat["deck"].get("commander_v2") or []] or dat["deck"].get("commander", [])
        cmd_oids = [ci.resolve(n) for n in cmd_names]
        deck_oids = {ci.resolve(row[0]) for rows in dat["deck"]["cards"].values() for row in rows} | set(cmd_oids)
        top = next((cl for cl in jd["cardlists"] if cl.get("tag") == "topcommanders"), None)
        potential = None
        if top and top["cardviews"]:
            potential = top["cardviews"][0].get("potential_decks") or top["cardviews"][0].get("num_decks")
        counts = dat.get("precon_commander_counts") or []
        main_count = next((c["count"] for c in counts if c.get("href") == e["url"]), None)
        potential = potential or main_count or (jd.get("card") or {}).get("num_decks")
        labels = []
        for cl in jd["cardlists"]:
            kind = KIND_BY_HEADER.get(cl["header"])
            if not kind:
                continue
            for rank, cv in enumerate(cl["cardviews"]):
                oid = ci.resolve(cv["name"])
                if oid is None:
                    unmapped.append({"source": "edhrec", "precon": slug, "kind": kind, "name": cv["name"],
                                     "set": cv.get("set"), "number": cv.get("collector_number")})
                    continue
                if kind.startswith("cut"):
                    rate, nd_, pd_ = cv.get("unpopularity"), None, potential
                else:
                    nd_, pd_ = cv.get("num_decks"), cv.get("potential_decks") or potential
                    rate = nd_ / pd_ if nd_ is not None and pd_ else None
                labels.append({"oracle_id": oid, "name": ci.by_oid[oid]["name"], "kind": kind,
                               "list_rank": rank, "rate": rate, "num_decks": nd_, "potential_decks": pd_,
                               "synergy": cv.get("synergy"), "lift": cv.get("lift")})
        commanders = []
        for c in counts:
            oid = ci.resolve(c["value"])
            if oid is None:
                unmapped.append({"source": "edhrec", "precon": slug, "kind": "commander", "name": c["value"]})
                continue
            commanders.append({"oracle_id": oid, "name": ci.by_oid[oid]["name"], "num_decks": c["count"],
                               "is_precon_commander": c.get("href") == e["url"]})
        pages.append({"slug": slug, "url": BASE + e["url"], "group": e["group"], "set": e.get("set") or
                      (e.get("cards") or [{}])[0].get("set"), "commander_oids": [o for o in cmd_oids if o],
                      "deck_oids": deck_oids - {None}, "potential_decks": potential, "labels": labels,
                      "commanders": commanders})
    return pages, missing, unmapped, build_id


# ----------------------------------------------------------------------------- join + splits
def join(decks, pages):
    """Match each EDHREC page to one MTGJSON deck: same commander oracle_id set (and same set code,
    case-insensitive); fallback = best card-overlap Jaccard >= 0.7 within the same set code."""
    by_cmd = defaultdict(list)
    for d in decks:
        by_cmd[frozenset(d["commander_oracle_ids"])].append(d)
    matches, unmatched = {}, []
    for p in pages:
        cands = by_cmd.get(frozenset(p["commander_oids"]), [])
        same_set = [d for d in cands if p["set"] and d["set_code"].lower() == p["set"].lower()]
        cands = same_set or cands
        how = "commander+set" if same_set else "commander"
        if not cands:
            cands = [d for d in decks if p["set"] and d["set_code"].lower() == p["set"].lower()]
            how = "jaccard"
        best, bj = None, -1.0
        for d in cands:
            ids = {c["oracle_id"] for c in d["cards"]}
            j = len(ids & p["deck_oids"]) / max(1, len(ids | p["deck_oids"]))
            if j > bj:
                best, bj = d, j
        if best is None or (how == "jaccard" and bj < 0.7):
            unmatched.append({"slug": p["slug"], "set": p["set"], "best_jaccard": round(bj, 3)})
            continue
        if best["precon_id"] in matches:
            unmatched.append({"slug": p["slug"], "set": p["set"], "dup_of": matches[best["precon_id"]][0]["slug"]})
            continue
        matches[best["precon_id"]] = (p, how, bj)
    return matches, unmatched


def add_months(d: date, months: int) -> date:
    y, m = divmod(d.month - 1 + months, 12)
    return d.replace(year=d.year + y, month=m + 1, day=min(d.day, 28))


def make_splits(precons: pl.DataFrame, cutoff: str, snapshot: str) -> pl.DataFrame:
    C = date.fromisoformat(cutoff)
    lo, hi = add_months(C, -12), add_months(C, 2)
    snap = date.fromisoformat(snapshot)
    rows = []
    for r in precons.sort("release_date", "precon_id").iter_rows(named=True):
        rd = r["release_date"]
        split = "train" if rd < lo else "dev" if rd < C else "gap" if rd < hi else "test"
        reason = None
        if r["set_code"] in EXCLUDED_SETS or r["mtgjson_type"] != "Commander Deck":
            reason = "excluded_product(td0/mtgo)"
        elif r["edhrec_slug"] is None:
            reason = "no_edhrec_page"
        elif (r["potential_decks"] or 0) < MIN_DECKS:
            reason = f"potential_decks<{MIN_DECKS}"
        elif (snap - rd).days < MIN_AGE_DAYS:
            reason = f"released<{MIN_AGE_DAYS}d_before_snapshot"
        elif split == "gap":
            reason = "cutoff_gap(C..C+2mo)"
        rows.append({"precon_id": r["precon_id"], "release_date": rd, "split": split,
                     "scored": reason is None, "exclude_reason": reason, "cutoff": C})
    return pl.DataFrame(rows)


# ----------------------------------------------------------------------------- tag audit (T3)
def tag_audit(ci: CardIndex, decks_df: pl.DataFrame, labels: pl.DataFrame, train_ids: set[str]):
    """lift = P(tag | gold cut) / P(tag | kept card), over train precons (headline cuts, gain>=1)."""
    tags = {oid: set(r["tags"] or []) for oid, r in ci.by_oid.items()}
    cut_set = {(r["precon_id"], r["oracle_id"]) for r in labels.filter(
        pl.col("kind") == "cut", pl.col("gain") >= 1).iter_rows(named=True)}
    n_cut = n_keep = 0
    c_cut, c_keep = Counter(), Counter()
    for r in decks_df.filter(pl.col("precon_id").is_in(list(train_ids))).iter_rows(named=True):
        for c in r["cards"]:
            if c["is_commander"] or c["is_land"]:
                continue
            t = tags.get(c["oracle_id"], set())
            if (r["precon_id"], c["oracle_id"]) in cut_set:
                n_cut += 1
                c_cut.update(t)
            else:
                n_keep += 1
                c_keep.update(t)
    rows = []
    for t, k in c_cut.items():
        p1, p0 = k / max(1, n_cut), (c_keep[t] + 1) / (n_keep + 1)
        rows.append({"tag": t, "support_cut": k, "support_kept": c_keep[t], "lift": p1 / p0,
                     "denied": denied(t)})
    return pl.DataFrame(rows).sort("lift", descending=True), n_cut, n_keep


# ----------------------------------------------------------------------------- main
def build(snapshot: str, cutoff: str):
    out = OUT / snapshot
    out.mkdir(parents=True, exist_ok=True)
    ci = CardIndex()
    decks, unm_mj, dup_editions, sets, mj_meta = load_mtgjson(ci)
    pages, missing, unm_ed, build_id = load_edhrec(snapshot, ci)
    matches, unmatched = join(decks, pages)
    fp = first_printing_index(sets)
    # Precon-identifying cycle tags (e.g. cycle-c20-alt-commander, cycle-c17-curse): deny every
    # `cycle-<commander set>-*` tag for the precon set codes in this build (plan T3).
    for code in sorted({d["set_code"].lower() for d in decks}):
        pat = f"cycle-{code}-*"
        if pat not in TAG_DENY_PATTERNS:
            TAG_DENY_PATTERNS.append(pat)

    deck_rows, meta_rows, label_rows, cmd_rows = [], [], [], []
    for d in decks:
        pid = d["precon_id"]
        d["mtgjson_commander_oracle_ids"] = list(d["commander_oracle_ids"])
        d["commander_source"] = "mtgjson"
        m = matches.get(pid)
        if m and set(m[0]["commander_oids"]) != set(d["commander_oracle_ids"]):
            # Labels come from the EDHREC page of the precon's *face* commander; when MTGJSON names a
            # different commander, the task uses EDHREC's (it is in the official 100) so inputs and
            # labels refer to the same commander.
            e_oids = m[0]["commander_oids"]
            in_deck = {c["oracle_id"] for c in d["cards"]}
            if e_oids and set(e_oids) <= in_deck:
                d["commander_oracle_ids"] = list(e_oids)
                d["commander_source"] = "edhrec"
                for c in d["cards"]:
                    c["is_commander"] = c["oracle_id"] in e_oids
        ci_set = set()
        for o in d["commander_oracle_ids"]:
            ci_set |= set(ci.by_oid[o]["color_identity"] or [])
        cid = [c for c in WUBRG if c in ci_set]
        deck_rows.append({
            "precon_id": pid,
            "commander_oracle_ids": d["commander_oracle_ids"],
            "commander_names": [ci.by_oid[o]["name"] for o in d["commander_oracle_ids"]],
            "color_identity": cid,
            "cards": [{k: v for k, v in c.items() if not k.startswith("_")} for c in d["cards"]],
            "n_cards": sum(c["qty"] for c in d["cards"]),
        })
        m = matches.get(pid)
        p, how, jac = m if m else (None, None, None)
        meta_rows.append({
            "precon_id": pid, "deck_name": d["deck_name"], "product": d["product"], "set_code": d["set_code"],
            "parent_set_code": d["parent_set_code"], "release_date": d["release_date"],
            "mtgjson_file": d["mtgjson_file"], "mtgjson_type": d["mtgjson_type"], "source": d["source"],
            "edhrec_slug": p["slug"] if p else None, "edhrec_url": p["url"] if p else None,
            "edhrec_group": p["group"] if p else None, "match_method": how,
            "edhrec_deck_jaccard": round(jac, 4) if jac is not None else None,
            "potential_decks": p["potential_decks"] if p else None,
            "n_colors": len(cid), "is_universes_beyond": d["is_universes_beyond"],
            "is_secret_lair": d["is_secret_lair"],
            "commander_names": [ci.by_oid[o]["name"] for o in d["commander_oracle_ids"]],
            "mtgjson_commander_names": [ci.by_oid[o]["name"] for o in d["mtgjson_commander_oracle_ids"]],
            "commander_source": d["commander_source"],
            "n_out_of_identity": sum(1 for c in d["cards"]
                                     if not set(ci.by_oid[c["oracle_id"]]["color_identity"] or []) <= set(cid)),
        })
        if not p:
            continue
        in_deck = {c["oracle_id"] for c in d["cards"]}
        fam = product_family(d["set_code"], sets)
        for lab in p["labels"]:
            oid = lab["oracle_id"]
            card = ci.by_oid[oid]
            is_cut = lab["kind"].startswith("cut")
            g = gain(lab["rate"], CUT_THRESH if is_cut else ADD_THRESH)
            syn = lab["synergy"]
            label_rows.append({
                "precon_id": pid, **lab,
                "gain": g,
                "gain_syn": g if is_cut else (g if (syn is not None and syn > 0) else 0),
                "is_land": ci.is_land(oid),
                "headline": lab["kind"] in ("cut", "add"),
                "in_precon": oid in in_deck,
                "is_new_in_product": fp.get(oid) in fam,
                "in_color_identity": set(card["color_identity"] or []) <= set(cid),
                "commander_legal": (card["legalities"] or {}).get("commander") == "legal",
            })
        for c in p["commanders"]:
            cmd_rows.append({"precon_id": pid, **c})

    decks_df = pl.DataFrame(deck_rows)
    precons_df = pl.DataFrame(meta_rows)
    labels_df = pl.DataFrame(label_rows, schema_overrides={"num_decks": pl.Int64, "synergy": pl.Float64,
                                                            "lift": pl.Float64, "rate": pl.Float64})
    cmd_df = pl.DataFrame(cmd_rows)
    splits_df = make_splits(precons_df, cutoff, snapshot)

    # ---- leakage hygiene asserts (inputs carry no product names / popularity / price) ----
    allowed = {"precon_id", "commander_oracle_ids", "commander_names", "color_identity", "cards", "n_cards"}
    assert set(decks_df.columns) == allowed, decks_df.columns
    card_fields = {f.name for f in decks_df.schema["cards"].inner.fields}
    assert card_fields == {"oracle_id", "name", "qty", "is_land", "is_basic", "is_commander"}, card_fields
    bad = [r for r in decks_df.iter_rows(named=True) if not re.fullmatch(r"pc-[0-9a-f]{10}", r["precon_id"])]
    assert not bad

    # ---- T3 tag audit on train ----
    train_ids = set(splits_df.filter(pl.col("split") == "train", pl.col("scored"))["precon_id"])
    audit, n_cut, n_keep = tag_audit(ci, decks_df, labels_df, train_ids)
    flagged = audit.filter(pl.col("lift") > 3, pl.col("support_cut") >= 5)

    for name, df in [("decks", decks_df), ("labels", labels_df), ("precons", precons_df),
                     ("commanders", cmd_df), ("splits", splits_df), ("tag_audit", audit)]:
        df.write_parquet(out / f"{name}.parquet")
    (out / "tag_denylist.json").write_text(json.dumps(
        {"exact": TAG_DENYLIST, "patterns": TAG_DENY_PATTERNS,
         "matched_tags_in_cards_parquet": sorted({t for r in ci.by_oid.values() for t in (r["tags"] or []) if denied(t)}),
         "audit_flagged_lift_gt3_support_ge5": flagged.select("tag", "lift", "support_cut", "denied").to_dicts(),
         }, indent=2))
    unmapped = unm_mj + unm_ed
    (out / "unmapped.json").write_text(json.dumps(unmapped, indent=2))

    # ---- manifest ----
    fetch_log = RAW / snapshot / "fetch_log.jsonl"
    log = [json.loads(line) for line in fetch_log.read_text().splitlines()] if fetch_log.exists() else []
    statuses = Counter(str(r["status"]) for r in log)
    ts = sorted(r["ts"] for r in log)
    cards_manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
    scored = splits_df.filter(pl.col("scored"))
    head = labels_df.filter(pl.col("headline"))
    manifest = {
        "dataset": "kasper-cbx/mtg-precon-bench",
        "schema_version": 1,
        "snapshot_date": snapshot,
        "built_from_git_sha": git_sha(),
        "sources": {
            "mtgjson": {"url": "https://mtgjson.com/api/v5/", "files": ["DeckList.json", "SetList.json",
                        "AllDeckFiles.tar.gz", "AtomicCards.json.gz"], "version": mj_meta.get("version"),
                        "date": mj_meta.get("date"), "license": "MIT"},
            "edhrec": {"index_url": f"{BASE}/precon", "page_url_pattern": f"{BASE}/precon/<slug>",
                       "build_id": build_id, "user_agent": USER_AGENT, "min_interval_s": MIN_INTERVAL,
                       "requests": len(log), "status_counts": dict(statuses),
                       "first_fetch": ts[0] if ts else None, "last_fetch": ts[-1] if ts else None,
                       "pages_missing_from_cache": missing,
                       "permission_note": "Owner states EDHREC granted permission to collect precon cut/add data; "
                                          "derived labels kept private, no redistribution."},
            "scryfall_cards_parquet": {"scryfall_updated_at": cards_manifest.get("scryfall_updated_at"),
                                       "built_at": cards_manifest.get("built_at"),
                                       "card_count": cards_manifest.get("card_count")},
        },
        "split_rule": {"cutoff": cutoff, "cutoff_note": "GLM-5.3 training cutoff unknown; default assumption",
                       "train": "release < cutoff-12mo", "dev": "cutoff-12mo <= release < cutoff",
                       "gap": "cutoff <= release < cutoff+2mo (unused)", "test": "release >= cutoff+2mo",
                       "gates": {"min_potential_decks": MIN_DECKS, "min_age_days": MIN_AGE_DAYS,
                                 "excluded_sets": sorted(EXCLUDED_SETS)}},
        "gain_thresholds": {"cut_unpopularity": dict(zip(("3", "2", "1"), CUT_THRESH)),
                            "add_rate": dict(zip(("3", "2", "1"), ADD_THRESH))},
        "defaults": {"bracket": 3, "max_game_changers": 3, "lands_in_headline": False,
                     "universes_beyond": "included, flagged", "secret_lair": "included, flagged",
                     "headline_add_gain": "gain (raw); gain_syn secondary"},
        "counts": {
            "mtgjson_commander_decks": len(decks) + len(dup_editions),
            "collector_edition_duplicates_dropped": len(dup_editions),
            "precons": precons_df.height,
            "edhrec_index_pages": len(pages) + len(missing),
            "edhrec_matched": len(matches),
            "edhrec_unmatched": unmatched,
            "deck_cards_total": int(sum(len(r["cards"]) for r in deck_rows)),
            "unmapped_mtgjson_cards": len(unm_mj),
            "unmapped_edhrec_names": len(unm_ed),
            "labels": labels_df.height,
            "labels_by_kind": dict(Counter(labels_df["kind"].to_list())),
            "headline_labels_gain_ge2_per_scored_precon": {
                k: round(head.filter(pl.col("kind") == k, pl.col("gain") >= 2,
                                     pl.col("precon_id").is_in(scored["precon_id"].to_list())).height
                         / max(1, scored.height), 2) for k in ("cut", "add")},
            "splits_all": dict(Counter(splits_df["split"].to_list())),
            "splits_scored": dict(Counter(scored["split"].to_list())),
            "exclude_reasons": dict(Counter(splits_df["exclude_reason"].drop_nulls().to_list())),
            "tag_audit": {"n_cut": n_cut, "n_kept": n_keep, "flagged": flagged.height},
        },
        "files": {f.name: sha256(f) for f in sorted(out.glob("*.parquet"))},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps(manifest["counts"], indent=2, default=str))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=date.today().isoformat())
    ap.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    ap.add_argument("--splits-only", action="store_true",
                    help="recompute splits.parquet for another cutoff into splits_<cutoff>.parquet")
    a = ap.parse_args()
    if a.splits_only:
        out = OUT / a.snapshot
        s = make_splits(pl.read_parquet(out / "precons.parquet"), a.cutoff, a.snapshot)
        s.write_parquet(out / f"splits_{a.cutoff}.parquet")
        print(s.group_by("split", "scored").len().sort("split"))
        return
    build(a.snapshot, a.cutoff)


if __name__ == "__main__":
    main()

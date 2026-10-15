"""EXP-007 deterministic scorer for the precon CUT / ADD / SWAP tasks (plan §5-§6).

Entry points
------------
``score_episode(task, output, deck, labels, db, cfg)`` -> flat ``{metric: value}`` dict for one
episode (``value`` is ``None`` where a metric is undefined, e.g. nDCG with no labels).
``aggregate(episodes)`` -> macro means over precons with bootstrap CIs (+ repeat sd).
``slice_aggregate``, ``paired_bootstrap``, ``sign_test``, ``kendall_tau`` for comparisons.

Validity (every pick is checked; an invalid pick keeps its slot with gain 0)
  cuts: name resolves exactly; card is in the deck; not a commander; not a duplicate pick.
  adds: name resolves; not already in the deck (basics / "any number" cards excepted);
        within the commanders' colour identity; commander-legal; not a duplicate pick;
        Game Changer count of the final deck <= ``cfg.gc_limit`` (bracket 3 -> 3), checked in
        rank order after removing the valid cuts (SWAP), so the first GC adds over the limit
        are the invalid ones.

Gains (plan §6): thresholded grades (``cfg.cut_thresholds`` on u, ``cfg.add_thresholds`` on r),
or continuous g = rate (``.cont``), or synergy-filtered g·1[synergy>0] (``.syn``, adds), or
thematic-free g·1[not is_new_in_product] (``.nothematic``, adds). Unlisted cards gain 0.

Headline vs land: picks are partitioned, order preserved, into a headline (nonland) list and a
land list; each list is scored against labels of the matching kinds (``cut``/``add`` vs
``cut_land``/``add_land``). A pick's side is the side of its label if it is labelled for this
precon, else whether its front face is a Land. Unresolved names count on the headline side.
Land picks are allowed by the prompt, so they do not cost headline slots (review m3, rescored
2026-10-15): with S requested slots and L land picks, headline metrics @k use k' = min(k, S - L)
(slots the agent left empty still count as misses). Land-side metrics keep k.

Metrics on a ranked list g_1..g_n (n may be < k; missing slots gain 0):
  DCG@k   = sum_{i<=k} g_i / log2(i+1)  (linear gain)
  nDCG@k  = DCG@k / IDCG@k, IDCG from the k largest label gains of the precon (None if 0)
  P@k     = #{i<=k : grade_i >= 1} / k
  R@k     = |top-k ∩ G2| / |G2|, G2 = labels with grade >= 2 (None if G2 empty)
  F1@k    = 2PR/(P+R) with P = |top-k ∩ G2| / k (set-based, against G2)
  false-cut rate = valid cut picks with no cut/cut_land label / valid cut picks
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field

import numpy as np

from .loader import ROLE_MINIMUMS, CardDB, Deck

CUT_KINDS = {"headline": "cut", "land": "cut_land"}
ADD_KINDS = {"headline": "add", "land": "add_land"}
COLORS = "WUBRG"


@dataclass(frozen=True)
class ScoreConfig:
    cut_thresholds: tuple[float, float, float] = (0.50, 0.30, 0.15)  # grade 3 / 2 / 1 lower bounds on u
    add_thresholds: tuple[float, float, float] = (0.30, 0.15, 0.05)  # ... on r
    gc_limit: int = 3  # bracket 3 (plan default, Q4)
    n_cuts: int = 10
    n_adds: int = 20
    n_swaps: int = 10

    def to_dict(self) -> dict:
        return asdict(self)


def grade(rate: float | None, thresholds: Sequence[float]) -> int:
    if rate is None or (isinstance(rate, float) and math.isnan(rate)):
        return 0
    for g, t in zip((3, 2, 1), thresholds, strict=True):
        if rate >= t:
            return g
    return 0


# =========================================================================== ranking metrics

def dcg(gains: Sequence[float], k: int) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(list(gains)[:k]))


def ndcg(gains: Sequence[float], ideal: Iterable[float], k: int) -> float | None:
    best = dcg(sorted((g for g in ideal if g > 0), reverse=True), k)
    return dcg(gains, k) / best if best > 0 else None


def precision_at(grades: Sequence[int], k: int, min_grade: int = 1) -> float:
    return sum(1 for g in list(grades)[:k] if g >= min_grade) / k


def recall_at(picked: Sequence[str | None], relevant: set[str], k: int) -> float | None:
    if not relevant:
        return None
    return len({p for p in list(picked)[:k] if p} & relevant) / len(relevant)


def f1_at(picked: Sequence[str | None], relevant: set[str], k: int) -> float | None:
    if not relevant:
        return None
    hit = len({p for p in list(picked)[:k] if p} & relevant)
    p, r = hit / k, hit / len(relevant)
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


# =========================================================================== validity

@dataclass
class Pick:
    raw: str
    oid: str | None
    valid: bool
    reason: str | None = None  # why invalid
    side: str = "headline"  # headline | land


def _side(oid: str | None, labels_by_oid: dict[str, dict], db: CardDB) -> str:
    if oid is None or oid not in db:
        return "headline"
    lab = labels_by_oid.get(oid)
    if lab is not None and "headline" in lab and lab["headline"] is not None:
        return "headline" if lab["headline"] else "land"
    if lab is not None:
        return "land" if lab["kind"].endswith("_land") else "headline"
    return "land" if db[oid].is_land else "headline"


def validate_cuts(names: Sequence, deck: Deck, db: CardDB, labels_by_oid: dict | None = None) -> list[Pick]:
    labels_by_oid = labels_by_oid or {}
    qty, cmd = deck.qty, set(deck.commander_oids)
    seen: set[str] = set()
    out = []
    for raw in names:
        oid = db.resolve(raw)
        reason = None
        if oid is None:
            reason = "unresolved"
        elif oid not in qty:
            reason = "not_in_deck"
        elif oid in cmd:
            reason = "is_commander"
        elif oid in seen:
            reason = "duplicate"
        if oid:
            seen.add(oid)
        out.append(Pick(str(raw), oid, reason is None, reason, _side(oid, labels_by_oid, db)))
    return out


def deck_gc_count(deck: Deck, db: CardDB, removed: Iterable[str] = ()) -> int:
    rem = set(removed)
    return sum(1 for oid in deck.qty if oid not in rem and oid in db and db[oid].game_changer)


def validate_adds(names: Sequence, deck: Deck, db: CardDB, labels_by_oid: dict | None = None,
                  gc_limit: int | None = 3, removed: Iterable[str] = ()) -> list[Pick]:
    labels_by_oid = labels_by_oid or {}
    qty = deck.qty
    ci = deck.color_identity
    gc = deck_gc_count(deck, db, removed)
    seen: set[str] = set()
    out = []
    for raw in names:
        oid = db.resolve(raw)
        reason = None
        if oid is None:
            reason = "unresolved"
        else:
            c = db[oid]
            multi_ok = c.is_basic or c.any_number
            if oid in seen and not multi_ok:
                reason = "duplicate"
            elif oid in qty and not multi_ok:  # also a card cut in the same swap set
                reason = "in_deck"
            elif not c.color_identity <= ci:
                reason = "off_identity"
            elif not c.commander_legal:
                reason = "not_legal"
            elif c.game_changer and gc_limit is not None and gc >= gc_limit:
                reason = "game_changer_limit"
            if reason is None and c.game_changer:
                gc += 1
            seen.add(oid)
        out.append(Pick(str(raw), oid, reason is None, reason, _side(oid, labels_by_oid, db)))
    return out


def validity_stats(picks: Sequence[Pick], expected: int, prefix: str) -> dict:
    n = len(picks)
    reasons: dict[str, int] = {}
    for p in picks:
        if not p.valid:
            reasons[p.reason] = reasons.get(p.reason, 0) + 1
    out = {
        f"{prefix}.n_submitted": n,
        f"{prefix}.n_valid": sum(p.valid for p in picks),
        f"{prefix}.n_missing": max(0, expected - n),
        f"{prefix}.validity": (sum(p.valid for p in picks) / n) if n else None,
    }
    for r in ("unresolved", "not_in_deck", "is_commander", "duplicate", "in_deck", "off_identity",
              "not_legal", "game_changer_limit"):
        if r in reasons:
            out[f"{prefix}.invalid.{r}"] = reasons[r]
    return out


# =========================================================================== scoring one side

def _gain_fns(kind_family: str, cfg: ScoreConfig) -> dict[str, Callable[[dict], float]]:
    th = cfg.cut_thresholds if kind_family == "cut" else cfg.add_thresholds

    def raw(lab):
        return float(grade(lab.get("rate"), th))

    fns = {"": raw, ".cont": lambda lab: float(lab.get("rate") or 0.0)}
    if kind_family == "add":
        fns[".syn"] = lambda lab: raw(lab) * (1.0 if (lab.get("synergy") or 0) > 0 else 0.0)
        fns[".nothematic"] = lambda lab: raw(lab) * (0.0 if lab.get("is_new_in_product") else 1.0)
    return fns


def score_side(picks: Sequence[Pick], labels: Sequence[dict], family: str, ks: Sequence[int],
               cfg: ScoreConfig, prefix: str, recall_k: int, f1_k: int | None = None,
               n_slots: int | None = None) -> dict:
    """Score one ranked list (already validated) for one task family (``cut``|``add``),
    separately on the headline and the land side. ``n_slots`` = picks requested; land picks then
    shrink the headline cut-off instead of costing headline slots (review m3)."""
    kinds = CUT_KINDS if family == "cut" else ADD_KINDS
    th = cfg.cut_thresholds if family == "cut" else cfg.add_thresholds
    n_land = sum(1 for p in picks if p.side == "land")
    out: dict = {}
    for side, kind in kinds.items():
        p_side = [p for p in picks if p.side == side]

        def kk(k: int, side=side) -> int:
            return min(k, n_slots - n_land) if (side == "headline" and n_slots is not None) else k

        labs = {lab["oracle_id"]: lab for lab in labels if lab["kind"] == kind}
        pre = prefix if side == "headline" else f"{prefix}_land"
        out[f"{pre}.n_picks"] = len(p_side)
        out[f"{pre}.n_labels"] = len(labs)
        grades = [grade(labs[p.oid]["rate"], th) if p.valid and p.oid in labs else 0 for p in p_side]
        picked = [p.oid if p.valid else None for p in p_side]
        g2 = {o for o, lab in labs.items() if grade(lab.get("rate"), th) >= 2}
        for suffix, fn in _gain_fns(family, cfg).items():
            gains = [fn(labs[p.oid]) if p.valid and p.oid in labs else 0.0 for p in p_side]
            ideal = [fn(lab) for lab in labs.values()]
            for k in ks:
                out[f"{pre}.ndcg@{k}{suffix}"] = ndcg(gains, ideal, kk(k)) if kk(k) > 0 else None
        for k in ks:
            out[f"{pre}.p@{k}"] = precision_at(grades, kk(k)) if kk(k) > 0 else None
        out[f"{pre}.r@{recall_k}"] = recall_at(picked, g2, kk(recall_k)) if kk(recall_k) > 0 else None
        if f1_k:
            out[f"{pre}.f1@{f1_k}"] = f1_at(picked, g2, kk(f1_k)) if kk(f1_k) > 0 else None
        out[f"{pre}.n_g2"] = len(g2)
    return out


def false_cut_rate(picks: Sequence[Pick], labels: Sequence[dict]) -> float | None:
    listed = {lab["oracle_id"] for lab in labels if lab["kind"] in ("cut", "cut_land")}
    valid = [p for p in picks if p.valid]
    return sum(1 for p in valid if p.oid not in listed) / len(valid) if valid else None


def popularity_drift(picks: Sequence[Pick], db: CardDB) -> float | None:
    """Mean log10 edhrec_rank of valid adds (offline only; lower = more popular)."""
    vals = [math.log10(db[p.oid].edhrec_rank) for p in picks if p.valid and db[p.oid].edhrec_rank]
    return sum(vals) / len(vals) if vals else None


# =========================================================================== structural diagnostics

def deck_profile(qty: dict[str, int], db: CardDB, colors: Iterable[str]) -> dict:
    lands = sum(q for o, q in qty.items() if db[o].is_land)
    nonland = [(o, q) for o, q in qty.items() if not db[o].is_land]
    n_nl = sum(q for _, q in nonland)
    avg_mv = sum(db[o].cmc * q for o, q in nonland) / n_nl if n_nl else 0.0
    sources = {c: sum(q for o, q in qty.items() if c in db[o].produced_mana) for c in colors}
    pips = {c: sum(db[o].pips.get(c, 0) * q for o, q in nonland) for c in colors}
    roles = {r: sum(q for o, q in qty.items() if r in db.diag_roles(o)) for r in ROLE_MINIMUMS}
    return {"lands": lands, "avg_mv": avg_mv, "sources": sources, "pips": pips, "roles": roles}


def structural_diagnostics(deck: Deck, cuts: Sequence[Pick], adds: Sequence[Pick], db: CardDB) -> dict:
    """Diagnostics only (never in the headline): land count, coloured sources vs pips, avg MV
    and role coverage of the final deck (precon - valid cuts + valid adds) vs the precon."""
    base = {o: q for o, q in deck.qty.items() if o in db}
    final = dict(base)
    for p in cuts:
        if p.valid and p.oid in final:
            final[p.oid] -= 1
            if final[p.oid] <= 0:
                del final[p.oid]
    for p in adds:
        if p.valid:
            final[p.oid] = final.get(p.oid, 0) + 1
    cols = [c for c in COLORS if c in deck.color_identity]
    b, f = deck_profile(base, db, cols), deck_profile(final, db, cols)
    out = {
        "diag.deck_size": sum(final.values()),
        "diag.lands": f["lands"],
        "diag.lands_delta": f["lands"] - b["lands"],
        "diag.lands_flag": abs(f["lands"] - b["lands"]) > 2,
        "diag.avg_mv_delta": round(f["avg_mv"] - b["avg_mv"], 4),
    }
    for c in cols:
        out[f"diag.sources.{c}"] = f["sources"][c]
        out[f"diag.sources_delta.{c}"] = f["sources"][c] - b["sources"][c]
        tot = sum(f["pips"].values())
        out[f"diag.pip_share.{c}"] = round(f["pips"][c] / tot, 4) if tot else None
    for r, mn in ROLE_MINIMUMS.items():
        out[f"diag.role.{r}"] = f["roles"][r]
        out[f"diag.role_delta.{r}"] = f["roles"][r] - b["roles"][r]
        out[f"diag.role_below_min.{r}"] = f["roles"][r] < mn
    return out


def pair_coherence(pairs: Sequence[tuple[Pick, Pick]], db: CardDB) -> float | None:
    """Share of valid (cut, add) pairs sharing at least one core functional role (diagnostic)."""
    ok = [(c, a) for c, a in pairs if c.valid and a.valid]
    if not ok:
        return None
    return sum(1 for c, a in ok if db.core_roles(c.oid) & db.core_roles(a.oid)) / len(ok)


# =========================================================================== episode

@dataclass
class EpisodeOutput:
    """Parsed final answer. ``cuts``/``adds`` are ranked name lists; ``swaps`` a list of
    {cut, add, reason} dicts."""
    cuts: list = field(default_factory=list)
    adds: list = field(default_factory=list)
    swaps: list = field(default_factory=list)


def score_episode(task: str, output: EpisodeOutput | dict, deck: Deck, labels: Sequence[dict], db: CardDB,
                  cfg: ScoreConfig = ScoreConfig()) -> dict:
    if isinstance(output, dict):
        output = EpisodeOutput(**{k: output.get(k) or [] for k in ("cuts", "adds", "swaps")})
    by_oid: dict[str, dict] = {}
    for lab in labels:
        by_oid.setdefault(lab["oracle_id"], lab)
    cut_l = {lab["oracle_id"]: lab for lab in labels if lab["kind"].startswith("cut")}
    add_l = {lab["oracle_id"]: lab for lab in labels if lab["kind"].startswith("add")}
    m: dict = {}
    if task == "cut":
        cuts = validate_cuts(list(output.cuts)[: cfg.n_cuts], deck, db, cut_l)
        m.update(validity_stats(cuts, cfg.n_cuts, "cut"))
        m.update(score_side(cuts, labels, "cut", (10,), cfg, "cut", recall_k=10, f1_k=10, n_slots=cfg.n_cuts))
        m["cut.false_cut_rate"] = false_cut_rate(cuts, labels)
    elif task == "add":
        adds = validate_adds(list(output.adds)[: cfg.n_adds], deck, db, add_l, cfg.gc_limit)
        m.update(validity_stats(adds, cfg.n_adds, "add"))
        m.update(score_side(adds, labels, "add", (10, 20), cfg, "add", recall_k=20, n_slots=cfg.n_adds))
        m["add.pop_drift"] = popularity_drift(adds, db)
        m["add.gc_final"] = deck_gc_count(deck, db) + sum(1 for p in adds if p.valid and db[p.oid].game_changer)
    elif task == "swap":
        swaps = [s for s in list(output.swaps)[: cfg.n_swaps] if isinstance(s, dict)]
        cuts = validate_cuts([s.get("cut") for s in swaps], deck, db, cut_l)
        removed = [p.oid for p in cuts if p.valid]
        adds = validate_adds([s.get("add") for s in swaps], deck, db, add_l, cfg.gc_limit, removed)
        m.update(validity_stats(cuts, cfg.n_swaps, "swap.cut"))
        m.update(validity_stats(adds, cfg.n_swaps, "swap.add"))
        m.update(score_side(cuts, labels, "cut", (10,), cfg, "swap.cut", recall_k=10, f1_k=10, n_slots=cfg.n_swaps))
        m.update(score_side(adds, labels, "add", (10,), cfg, "swap.add", recall_k=10, n_slots=cfg.n_swaps))
        m["swap.cut.false_cut_rate"] = false_cut_rate(cuts, labels)
        m["swap.add.pop_drift"] = popularity_drift(adds, db)
        m["swap.pair_coherence"] = pair_coherence(list(zip(cuts, adds, strict=True)), db)
        m.update(structural_diagnostics(deck, cuts, adds, db))
        m["swap.gc_final"] = deck_gc_count(deck, db, removed) + sum(1 for p in adds if p.valid and db[p.oid].game_changer)
    else:
        raise ValueError(f"unknown task {task!r}")
    return m


def picks_detail(task: str, output: EpisodeOutput | dict, deck: Deck, labels: Sequence[dict], db: CardDB,
                 cfg: ScoreConfig = ScoreConfig()) -> list[dict]:
    """Per-pick validity + gain (for transcripts / qualitative review; contains gold -> private)."""
    if isinstance(output, dict):
        output = EpisodeOutput(**{k: output.get(k) or [] for k in ("cuts", "adds", "swaps")})
    cut_l = {lab["oracle_id"]: lab for lab in labels if lab["kind"].startswith("cut")}
    add_l = {lab["oracle_id"]: lab for lab in labels if lab["kind"].startswith("add")}
    rows = []
    if task in ("cut", "swap"):
        names = output.cuts if task == "cut" else [s.get("cut") for s in output.swaps if isinstance(s, dict)]
        for p in validate_cuts(names, deck, db, cut_l):
            lab = cut_l.get(p.oid) if p.valid else None
            rows.append({"role": "cut", **asdict(p), "rate": lab and lab["rate"], "grade": grade(lab and lab["rate"], cfg.cut_thresholds)})
    if task in ("add", "swap"):
        names = output.adds if task == "add" else [s.get("add") for s in output.swaps if isinstance(s, dict)]
        for p in validate_adds(names, deck, db, add_l, cfg.gc_limit):
            lab = add_l.get(p.oid) if p.valid else None
            rows.append({"role": "add", **asdict(p), "rate": lab and lab["rate"], "grade": grade(lab and lab["rate"], cfg.add_thresholds)})
    return rows


# =========================================================================== aggregation

def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def bootstrap_ci(values: Sequence[float], n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean (seeded, numpy)."""
    if not values:
        return (float("nan"), float("nan"))
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def per_precon(episodes: Sequence[dict], metric: str) -> dict[str, float]:
    """Mean over repeats per precon (``episodes``: dicts with precon_id, rep, metrics)."""
    acc: dict[str, list[float]] = {}
    for e in episodes:
        v = e["metrics"].get(metric)
        if v is None or isinstance(v, bool):
            continue
        acc.setdefault(e["precon_id"], []).append(float(v))
    return {k: _mean(v) for k, v in acc.items()}


def numeric_metrics(episodes: Sequence[dict]) -> list[str]:
    keys: set[str] = set()
    for e in episodes:
        keys |= {k for k, v in e["metrics"].items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return sorted(keys)


def aggregate(episodes: Sequence[dict], metrics: Sequence[str] | None = None, n_boot: int = 10_000,
              seed: int = 0) -> dict[str, dict]:
    """Macro mean over precons (repeats averaged first), 95% bootstrap CI over precons, and the
    sd across repeats of the per-repeat macro mean."""
    metrics = metrics or numeric_metrics(episodes)
    out = {}
    reps = sorted({e.get("rep", 0) for e in episodes})
    for m in metrics:
        pp = per_precon(episodes, m)
        if not pp:
            continue
        vals = [pp[k] for k in sorted(pp)]
        lo, hi = bootstrap_ci(vals, n_boot, seed)
        rep_means = []
        for r in reps:
            rv = per_precon([e for e in episodes if e.get("rep", 0) == r], m)
            if rv:
                rep_means.append(_mean(list(rv.values())))
        sd = (math.sqrt(sum((x - _mean(rep_means)) ** 2 for x in rep_means) / (len(rep_means) - 1))
              if len(rep_means) > 1 else None)
        out[m] = {"mean": _mean(vals), "ci95": [lo, hi], "n_precons": len(vals), "repeat_sd": sd}
    return out


def slice_aggregate(episodes: Sequence[dict], slice_of: Callable[[str], str], metrics: Sequence[str],
                    n_boot: int = 2_000, seed: int = 0) -> dict[str, dict]:
    """``{slice_value: aggregate(...)}``; ``slice_of(precon_id)`` gives the slice value."""
    groups: dict[str, list[dict]] = {}
    for e in episodes:
        groups.setdefault(str(slice_of(e["precon_id"])), []).append(e)
    return {g: aggregate(es, metrics, n_boot, seed) for g, es in sorted(groups.items())}


def paired_bootstrap(a: dict[str, float], b: dict[str, float], n_boot: int = 10_000, seed: int = 0) -> dict:
    """Paired comparison of two systems' per-precon values (a - b) on their common precons."""
    common = sorted(set(a) & set(b))
    d = [a[k] - b[k] for k in common]
    if not d:
        return {"n": 0}
    lo, hi = bootstrap_ci(d, n_boot, seed)
    return {"n": len(d), "mean_diff": _mean(d), "ci95": [lo, hi], "sign_test_p": sign_test(d)}


def sign_test(diffs: Sequence[float]) -> float:
    """Two-sided exact sign test p-value (ties dropped)."""
    pos = sum(1 for x in diffs if x > 0)
    neg = sum(1 for x in diffs if x < 0)
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Kendall tau-b between two score vectors over the same systems."""
    n = len(x)
    if n < 2:
        return None
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx * dy > 0:
                conc += 1
            else:
                disc += 1
    den = math.sqrt((conc + disc + tx) * (conc + disc + ty))
    return (conc - disc) / den if den else None


HEADLINE = {
    "cut": ["cut.ndcg@10", "cut.ndcg@10.cont", "cut.p@10", "cut.r@10", "cut.f1@10", "cut.false_cut_rate",
            "cut.validity", "cut_land.ndcg@10"],
    "add": ["add.ndcg@10", "add.ndcg@20", "add.ndcg@10.cont", "add.ndcg@20.cont", "add.p@10", "add.p@20", "add.r@20",
            "add.ndcg@20.syn", "add.ndcg@20.nothematic", "add.validity", "add.pop_drift", "add_land.ndcg@10"],
    "swap": ["swap.cut.ndcg@10", "swap.add.ndcg@10", "swap.cut.ndcg@10.cont", "swap.add.ndcg@10.cont",
             "swap.add.ndcg@10.syn", "swap.cut.validity", "swap.add.validity", "swap.pair_coherence",
             "diag.lands_delta", "diag.avg_mv_delta"],
}

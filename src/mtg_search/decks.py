"""Decklists for ``get_deck``: parse pasted text, check a deck against a format's construction rules, render it.

Pure functions over card rows (``cards.parquet`` dicts). :meth:`mtg_search.server.CardService.get_deck` resolves
the names and loads official precons (:mod:`mtg_search.precons`).

Scryfall's legality data already says whether each card is legal, banned, restricted or not legal in every
format; that covers ban lists, the restricted list, rotation, Pauper's rarity and the Brawl card pool. What this
module adds is what a single card's legality cannot say: deck size, copy limits (basics and "any number" cards
excepted), sideboard size, restricted counts, commanders (eligibility and partner pairs), colour identity and the
Game Changer count.
"""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .render import COLOR_ORDER, render_card, role_tags
from .tag_filter import TagFilter

MAX_LINES = 600
MAX_CHARS = 40_000

# --------------------------------------------------------------------------- formats


@dataclass(frozen=True)
class Rules:
    label: str
    size: int  # cards in the deck, commanders included
    exact: bool  # exactly ``size`` (Commander) or at least ``size`` (constructed)
    copies: int  # copies per card name, basic lands and "any number" cards excepted
    sideboard: int | None = None  # largest sideboard; None: the format has no sideboard rule here
    commander: bool = False  # one or two commanders, colour identity
    walker_commander: bool = False  # a legendary planeswalker may be the commander (Brawl)
    game_changers: bool = False  # report the Game Changer count (Commander brackets)
    commander_legality: bool = True  # False: the format's legality covers the 99 only (Pauper Commander)
    unchecked: str = ""  # rules of this format the checks leave out


_SIXTY = {"size": 60, "exact": False, "copies": 4, "sideboard": 15}
RULES: dict[str, Rules] = {
    "commander": Rules("Commander", 100, True, 1, commander=True, game_changers=True,
                       unchecked="companion deck conditions; bracket rules other than Game Changers"),
    "duel": Rules("Duel Commander", 100, True, 1, commander=True, unchecked="cards banned only as a commander"),
    "predh": Rules("PreDH", 100, True, 1, commander=True),
    "paupercommander": Rules("Pauper Commander", 100, True, 1, commander=True, commander_legality=False,
                             unchecked="that the commander has an uncommon printing"),
    "brawl": Rules("Brawl", 100, True, 1, commander=True, walker_commander=True),
    "standardbrawl": Rules("Standard Brawl", 60, True, 1, commander=True, walker_commander=True),
    "gladiator": Rules("Gladiator", 100, False, 1),
    **{f: Rules(label, **_SIXTY) for f, label in (
        ("standard", "Standard"), ("pioneer", "Pioneer"), ("modern", "Modern"), ("legacy", "Legacy"),
        ("vintage", "Vintage"), ("pauper", "Pauper"), ("historic", "Historic"), ("timeless", "Timeless"),
        ("alchemy", "Alchemy"), ("premodern", "Premodern"), ("oldschool", "Old School"), ("penny", "Penny Dreadful"),
        ("future", "Future Standard"))},
}
# a 60-card list whose format nobody named: size, copies and sideboard only
CONSTRUCTED = Rules("60-card constructed", **_SIXTY)
LEGAL_IN = ("standard", "pioneer", "modern", "legacy", "vintage", "pauper")  # the "every card is legal in" line

FORMAT_ALIASES = {
    "edh": "commander", "cedh": "commander", "historic brawl": "brawl", "arena brawl": "brawl",
    "standard brawl": "standardbrawl", "duel commander": "duel", "pauper commander": "paupercommander",
    "pdh": "paupercommander", "pre dh": "predh", "old school": "oldschool", "penny dreadful": "penny",
}


def normalize_format(fmt: str | None, known: Sequence[str]) -> str | None:
    """'Modern' / 'EDH' / 'standard brawl' -> the Scryfall format key; ValueError for an unknown format."""
    if fmt is None or not str(fmt).strip():
        return None
    f = re.sub(r"[\s_-]+", " ", str(fmt).strip().lower())
    f = FORMAT_ALIASES.get(f, f).replace(" ", "")
    if f not in known:
        raise ValueError(f"unknown format {fmt!r}; known formats: {', '.join(sorted(known))}")
    return f


# --------------------------------------------------------------------------- parsing

SECTIONS = ("commander", "companion", "main", "sideboard", "maybe")
_HEADERS = {
    "commander": "commander", "commanders": "commander", "command zone": "commander",
    "companion": "companion", "companions": "companion",
    "deck": "main", "main": "main", "mainboard": "main", "main deck": "main", "maindeck": "main",
    "sideboard": "sideboard", "side": "sideboard", "sb": "sideboard", "side board": "sideboard",
    "maybeboard": "maybe", "maybe": "maybe", "considering": "maybe", "wishlist": "maybe",
    "tokens": "other", "token": "other", "attractions": "other", "stickers": "other", "planes": "other",
    "schemes": "other", "about": "about", "metadata": "about",
}
_TYPE_GROUPS = {
    "creature", "creatures", "instant", "instants", "sorcery", "sorceries", "artifact", "artifacts",
    "enchantment", "enchantments", "planeswalker", "planeswalkers", "land", "lands", "battle", "battles",
    "spell", "spells", "other spells", "nonland", "kindred", "tribal", "other",
}
_HEADER = re.compile(r"(?://+|#+)?\s*\[?([a-z][a-z' ]*?)\]?\s*(?:\(\s*\d+\s*\)|:\s*\d*|\s\d+)?\s*:?")
_DEK = re.compile(r'<Cards\b[^>]*?Quantity="(\d+)"[^>]*?Sideboard="(true|false)"[^>]*?Name="([^"]+)"', re.I)
_DEK_ALT = re.compile(r'<Cards\b[^>]*?Quantity="(\d+)"[^>]*?Name="([^"]+)"[^>]*?Sideboard="(true|false)"', re.I)
_LEAD_QTY = re.compile(r"(\d{1,3})\s*[xX]?\s+(.+)")
_TRAIL_QTY = re.compile(r"(.+?)\s+[xX]\s?(\d{1,3})")
_CATEGORY = re.compile(r"\s+\[([^\]]*)\](?:\s*\^[^^]*\^)?\s*$")  # Archidekt: 1x Sol Ring (cmr) 472 [Ramp]
_MARKER = re.compile(r"\s+\*([A-Za-z]+)\*\s*$")  # *F* foil, *E* etched, *CMDR* (TappedOut)
_SET_NUM = re.compile(r"\s+\(([A-Za-z0-9]{2,6})\)(?:\s+[\w★†-]+)?\s*$")  # (SET) 123 / (PLST) ABC-123
_SET_LEAD = re.compile(r"\[[A-Za-z0-9]{2,6}\]\s+")  # MWS: 4 [M10] Lightning Bolt
_FORGE_SET = re.compile(r"\|[A-Za-z0-9]{2,6}(?:\|\d+)?\s*$")  # Forge .dck: 1 Sol Ring|C21
_TAGS = re.compile(r"\s+#\S.*$")  # Moxfield: 1 Sol Ring #ramp #!staple


@dataclass
class Line:
    qty: int
    name: str
    section: str
    lineno: int
    counted: bool = True  # a quantity was written: an unresolved name is reported, not ignored


@dataclass
class Decklist:
    lines: list[Line] = field(default_factory=list)
    name: str | None = None
    ignored: list[str] = field(default_factory=list)
    sections_marked: bool = False


def _header(line: str) -> str | None:
    m = _HEADER.fullmatch(line.strip().lower())
    if not m:
        return None
    word = m.group(1).strip()
    if word in _HEADERS:
        return _HEADERS[word]
    return "group" if word in _TYPE_GROUPS else None


def _clean(rest: str) -> tuple[str, str | None]:
    """Card name without set/number, markers, tags and categories; plus a section the line itself names."""
    section = None
    rest = _TAGS.sub("", rest.strip())
    while True:
        m = _CATEGORY.search(rest) or _MARKER.search(rest)
        if not m:
            break
        tag = m.group(1).lower()
        if "commander" in tag or tag == "cmdr":
            section = "commander"
        elif "sideboard" in tag:
            section = "sideboard"
        elif "maybeboard" in tag or "nodeck" in tag:
            section = "maybe"
        rest = rest[: m.start()]
    rest = _FORGE_SET.sub("", _SET_NUM.sub("", rest))
    rest = _SET_LEAD.sub("", rest.strip(), count=1)
    return rest.strip().strip('"').strip(), section


def _parse_csv(text: str) -> Decklist | None:
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 2:
        return None
    head = [h.strip().lower() for h in rows[0]]
    name_col = next((i for i, h in enumerate(head) if h in ("name", "card", "card name")), None)
    qty_col = next((i for i, h in enumerate(head) if h in ("count", "quantity", "qty", "amount")), None)
    if name_col is None or qty_col is None:
        return None
    sec_col = next((i for i, h in enumerate(head) if h in ("board", "section", "zone", "category", "categories")), None)
    out = Decklist(sections_marked=sec_col is not None)
    for n, row in enumerate(rows[1:], 2):
        if len(row) <= max(name_col, qty_col) or not row[name_col].strip():
            continue
        try:
            qty = int(float(row[qty_col]))
        except ValueError:
            out.ignored.append(",".join(row))
            continue
        sec = "main"
        if sec_col is not None and sec_col < len(row):
            s = row[sec_col].strip().lower()
            sec = next((v for k, v in _HEADERS.items() if k in s and v in SECTIONS), "main")
        out.lines.append(Line(qty, row[name_col].strip(), sec, n))
    return out


def parse_decklist(text: str) -> Decklist:
    """Parse a pasted decklist: Arena/MTGO/Moxfield/Archidekt/Forge text exports, MTGO .dek XML, or CSV with
    a name and a count column. Sections come from headers (Commander, Companion, Deck, Sideboard, Maybeboard,
    "SB:" prefixes, Archidekt categories, *CMDR*); without headers, a short block after the last blank line
    is the sideboard (MTGO export). Lines without a quantity count as one copy."""
    text = (text or "").replace("\r\n", "\n").replace("\t", " ")
    if len(text) > MAX_CHARS or text.count("\n") > MAX_LINES:
        raise ValueError(f"decklist too long (at most {MAX_LINES} lines); pass one deck at a time")
    dek = [(int(q), n, sb == "true") for q, sb, n in _DEK.findall(text)] or \
          [(int(q), n, sb == "true") for q, n, sb in _DEK_ALT.findall(text)]
    if dek:
        return Decklist([Line(q, n, "sideboard" if sb else "main", i) for i, (q, n, sb) in enumerate(dek, 1)],
                        sections_marked=True)
    first = next((ln for ln in text.split("\n") if ln.strip()), "")
    if first.count(",") >= 1 and re.search(r"\b(count|quantity|qty)\b", first, re.I):
        parsed = _parse_csv(text)
        if parsed is not None:
            return parsed

    out = Decklist()
    section, blocks, block = "main", [], []
    for lineno, raw in enumerate(text.split("\n"), 1):
        line = raw.strip()
        if not line:
            if block:
                blocks.append(block)
                block = []
            continue
        sb = re.match(r"SB:\s*", line, re.I)
        if sb:
            line = line[sb.end():]
        h = None if sb else _header(line)
        if h is not None and not _LEAD_QTY.fullmatch(line):
            if h == "group":
                section = "main" if section in ("commander", "companion", "about") else section
            else:
                section = h
                out.sections_marked = out.sections_marked or h in SECTIONS
            continue
        forced = "sideboard" if sb else None
        inline = None if sb else re.match(r"(commanders?|companion)\s*:\s*(\S.*)", line, re.I)
        if inline:  # Commander: Atraxa, Praetors' Voice
            forced = "commander" if inline.group(1).lower().startswith("commander") else "companion"
            line = inline.group(2)
        if section == "about" or line.startswith("[") and line.endswith("]"):
            m = re.match(r"name\s*[=:]?\s+(.+)|name=(.+)", line, re.I)
            if m:
                out.name = (m.group(1) or m.group(2)).strip()
            continue
        if line.startswith(("//", "#")) or section == "other":
            continue
        m = _LEAD_QTY.fullmatch(line)
        qty, counted = (int(m.group(1)), True) if m else (1, bool(inline))
        rest = m.group(2) if m else line
        if not m:
            t = _TRAIL_QTY.fullmatch(line)
            if t:
                rest, qty, counted = t.group(1), int(t.group(2)), True
        name, own = _clean(rest)
        if not name or not re.search(r"[A-Za-z]", name):
            out.ignored.append(raw.strip())
            continue
        own = own or forced
        out.sections_marked = out.sections_marked or own is not None
        ln = Line(qty, name, own or section, lineno, counted)
        out.lines.append(ln)
        block.append(ln)
    if block:
        blocks.append(block)
    if not out.sections_marked and len(blocks) >= 2:
        first, last = blocks[0], blocks[-1]
        n_first, n_last = sum(ln.qty for ln in first), sum(ln.qty for ln in last)
        n_all = sum(ln.qty for b in blocks for ln in b)
        if n_first <= 2 and n_all - n_first >= 90:  # commander, blank line, the other 99
            for ln in first:
                ln.section = "commander"
        elif n_last <= 15 and n_all - n_last >= 40:  # MTGO / plain export: main, blank line, sideboard
            for ln in last:
                ln.section = "sideboard"
    return out


# --------------------------------------------------------------------------- checks


@dataclass
class Entry:
    """One resolved (``row`` set) or unresolved (``row`` None) deck line."""

    qty: int
    section: str
    name: str  # card name, or the text as written when unresolved
    row: Mapping[str, Any] | None = None


_COPY_RULE = re.compile(r"a deck can have (any number|up to (\w+)|only one) (?:of )?cards? named", re.I)
_NUMBERS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten".split())}


def copy_limit(row: Mapping[str, Any], default: int) -> int | None:
    """Copies allowed per name: None = any number (basic lands, Relentless Rats ...), or the card's own limit."""
    if "Basic" in (row.get("supertypes") or []) and "Land" in (row.get("types") or []):
        return None
    m = _COPY_RULE.search(row.get("doc_text") or "")
    if not m:
        return default
    if m.group(1).lower() == "any number":
        return None
    if m.group(1).lower() == "only one":
        return 1
    w = m.group(2).lower()
    return int(w) if w.isdigit() else _NUMBERS.get(w, default)


def front_type(row: Mapping[str, Any]) -> str:
    return (row.get("type_line") or "").split(" // ")[0]


def category(row: Mapping[str, Any]) -> str:
    front = front_type(row)
    types = front.split("—")[0]
    if "Land" in types:
        return "land"
    for t in ("Creature", "Planeswalker", "Battle", "Instant", "Sorcery", "Artifact", "Enchantment"):
        if t in types:
            return t.lower()
    return "other"


CATEGORY_ORDER = ("creature", "planeswalker", "battle", "instant", "sorcery", "artifact", "enchantment", "other", "land")
PLURAL = {"sorcery": "sorceries", "other": "other"}


def _letters(ci) -> str:
    return "".join(c for c in COLOR_ORDER if c in (ci or [])) or "C"


def _eligible(row: Mapping[str, Any], rules: Rules, partner: Mapping[str, Any] | None) -> bool:
    if row.get("can_be_commander"):
        return True
    if rules.walker_commander and "Legendary" in front_type(row) and "Planeswalker" in front_type(row):
        return True
    return row.get("partner_kind") == "background" and bool(partner) and partner.get("partner_kind") == "choose a background"


def pair_problem(a: Mapping[str, Any], b: Mapping[str, Any]) -> str | None:
    """Why two cards cannot share the command zone, or None if they can."""
    ka, kb = a.get("partner_kind"), b.get("partner_kind")
    if ka and ka == kb and (ka in ("partner", "friends forever") or ka.startswith("partner—")):
        return None
    if {ka, kb} in ({"choose a background", "background"}, {"doctor", "doctor's companion"}):
        return None
    if ka == kb == "partner with":
        ta, tb = (a.get("doc_text") or "").lower(), (b.get("doc_text") or "").lower()
        if f"partner with {b['name'].lower()}" in ta and f"partner with {a['name'].lower()}" in tb:
            return None
        return f"{a['name']} and {b['name']} have Partner with, but not with each other"
    kinds = f"{ka or 'no partner ability'} / {kb or 'no partner ability'}"
    return f"{a['name']} and {b['name']} cannot both be commanders ({kinds})"


@dataclass
class Report:
    fmt: str | None
    rules: Rules | None
    how: str
    entries: list[Entry]
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    identity: str | None = None
    size: int = 0


def _qty(entries: Sequence[Entry], *sections: str) -> int:
    return sum(e.qty for e in entries if e.section in sections)


def _singleton(entries: Sequence[Entry]) -> bool:
    seen: dict[str, int] = {}
    for e in entries:
        if e.row is not None and e.section in ("main", "commander") and copy_limit(e.row, 1) is not None:
            seen[e.name] = seen.get(e.name, 0) + e.qty
    return all(v <= 1 for v in seen.values())


def detect(entries: list[Entry], source_format: str | None = None,
           source_label: str = "the deck type") -> tuple[str | None, Rules | None, str]:
    """(format, rules, how it was found) when the caller gave none. ``source_format``: the format the deck's
    source names (a precon's type, a deck site's format field), named in the output as ``source_label``."""
    if source_format in RULES:
        return source_format, RULES[source_format], f"from {source_label}"
    n_cmd, n_main, n_side = _qty(entries, "commander"), _qty(entries, "main"), _qty(entries, "sideboard")
    if n_cmd:
        total = n_cmd + n_main
        rows = [e.row for e in entries if e.row is not None and e.section in ("commander", "main")]
        brawl = 55 <= total <= 65 and all((r.get("legalities") or {}).get("standardbrawl") == "legal" for r in rows)
        fmt = "standardbrawl" if brawl else "commander"
        return fmt, RULES[fmt], f"detected: a commander and {total} cards"
    if _sideboard_commanders(entries, RULES["commander"]):
        return "commander", RULES["commander"], "detected: 100 cards with the commander in the sideboard"
    if n_main + n_side >= 95 and _singleton(entries):
        return "commander", RULES["commander"], "detected: 100 singleton cards"
    if n_main >= 55:
        return None, CONSTRUCTED, "detected from the deck size"
    return None, None, "not detected"


def _sideboard_commanders(entries: list[Entry], rules: Rules) -> list[Entry]:
    """MTGO / Moxfield-MTGO exports put the commander(s) in the sideboard: 98-99 main + 1-2 eligible cards."""
    side = [e for e in entries if e.section == "sideboard"]
    n_side, n_main = sum(e.qty for e in side), _qty(entries, "main")
    if not (1 <= n_side <= 2 and n_main + n_side == rules.size) or any(e.row is None for e in side):
        return []
    other = side[1].row if len(side) == 2 else None
    if all(_eligible(e.row, rules, other if e is side[0] else side[0].row) for e in side):
        return side
    return []


def check(entries: list[Entry], fmt: str | None, rules: Rules | None, how: str) -> Report:
    """Every construction rule of ``fmt`` the deck breaks. ``entries`` may be re-sectioned (sideboard commanders,
    companion counted in a 60-card sideboard)."""
    rep = Report(fmt, rules, how, entries)
    unresolved = [e for e in entries if e.row is None]
    if unresolved:
        rep.notes.append("cards that were not found count towards the deck size only")
    if rules is None:
        rep.size = _qty(entries, "commander", "main")
        if fmt:
            rep.notes.append(f"deck-size and copy rules for {fmt} are not checked; card legality is")
            _legality(rep, [e for e in entries if e.section != "maybe"])
        return rep
    if rules.commander:
        _check_commander(rep)
    else:
        _check_constructed(rep)
    if rules.unchecked:
        rep.notes.append("not checked: " + rules.unchecked)
    return rep


def _legality(rep: Report, entries: Sequence[Entry]) -> None:
    fmt = rep.fmt
    if not fmt:
        return
    label = rep.rules.label if rep.rules else fmt
    bad: dict[str, list[str]] = {}
    for e in entries:
        if e.row is None:
            continue
        st = (e.row.get("legalities") or {}).get(fmt) or "not_legal"
        if st in ("banned", "not_legal"):
            bad.setdefault(st, [])
            if e.name not in bad[st]:
                bad[st].append(e.name)
    for st, names in bad.items():
        word = "banned" if st == "banned" else "not legal"
        rep.problems.append(f"{word} in {label} ({len(names)}): " + ", ".join(names))


def _copies(rep: Report, entries: Sequence[Entry], default: int, label: str) -> None:
    total: dict[str, list[Entry]] = {}
    for e in entries:
        if e.row is not None:
            total.setdefault(e.name, []).append(e)
    for name, es in total.items():
        n = sum(e.qty for e in es)
        row = es[0].row
        lim = copy_limit(row, default)
        restricted = rep.fmt and (row.get("legalities") or {}).get(rep.fmt) == "restricted"
        if restricted:
            lim = 1 if lim is None else min(lim, 1)
        if lim is not None and n > lim:
            where = ""
            if len({e.section for e in es}) > 1:
                where = " (" + ", ".join(f"{e.section} {e.qty}" for e in es) + ")"
            why = f"restricted in {label}, at most 1" if restricted else f"at most {lim}"
            rep.problems.append(f"{name}: {n} copies{where}; {why}")


def _check_commander(rep: Report) -> None:
    rules, entries = rep.rules, rep.entries
    if not any(e.section == "commander" for e in entries):
        moved = _sideboard_commanders(entries, rules)
        for e in moved:
            e.section = "commander"
        if moved:
            rep.notes.append("commander read from the sideboard (MTGO export)")
    cmds = [e for e in entries if e.section == "commander"]
    deck = [e for e in entries if e.section in ("commander", "main")]
    rep.size = sum(e.qty for e in deck)
    wrong = rep.size != rules.size if rules.exact else rep.size < rules.size
    if wrong:
        need = f"exactly {rules.size}" if rules.exact else f"at least {rules.size}"
        rep.problems.append(f"{rep.size} cards (commander included); {rules.label} needs {need}")
    known = [e for e in cmds if e.row is not None]
    if not cmds:
        rep.problems.append("no commander marked: put it under a line that says Commander")
    elif sum(e.qty for e in cmds) > 2:
        rep.problems.append(f"{sum(e.qty for e in cmds)} commanders; at most 2 (a partner pair)")
    else:
        for i, e in enumerate(known):
            other = known[1 - i].row if len(known) == 2 else None
            if not _eligible(e.row, rules, other):
                what = "a legendary creature or planeswalker" if rules.walker_commander else "a legendary creature"
                rep.problems.append(f"{e.name} cannot be a commander (not {what} and not marked as able to be one)")
        if len(known) == 2:
            p = pair_problem(known[0].row, known[1].row)
            if p:
                rep.problems.append(p)
    if known:
        ci = set().union(*(set(e.row.get("color_identity") or []) for e in known))
        rep.identity = _letters(ci)
        off = [e for e in entries if e.section in ("main", "companion") and e.row is not None
               and not set(e.row.get("color_identity") or []) <= ci]
        if off:
            rep.problems.append(f"outside the colour identity {rep.identity} ({len(off)}): "
                                + ", ".join(f"{e.name} ({_letters(e.row.get('color_identity'))})" for e in off))
    _copies(rep, deck, rules.copies, rules.label)
    checked = ("commander", "main", "companion") if rules.commander_legality else ("main", "companion")
    _legality(rep, [e for e in entries if e.section in checked])
    n_side = _qty(entries, "sideboard")
    if n_side:
        rep.notes.append(f"sideboard ({n_side} cards) is not part of a {rules.label} deck and was not checked")
    if rules.game_changers:
        gc = [e.name for e in deck if e.row is not None and e.row.get("game_changer")]
        line = f"Game Changers ({len(gc)})" + (": " + ", ".join(gc) if gc else "")
        rep.notes.append(line + " · brackets 1-2 allow none, bracket 3 up to three, brackets 4-5 any number")


def _check_constructed(rep: Report) -> None:
    rules, entries = rep.rules, rep.entries
    for e in entries:
        if e.section == "commander":
            e.section = "main"
    side_names = {e.name for e in entries if e.section == "sideboard"}
    for e in entries:  # a companion starts the game in the sideboard and counts towards its 15 cards
        if e.section == "companion":
            e.section = "sideboard"
            if e.name in side_names:
                e.qty = 0
    entries[:] = [e for e in entries if e.qty > 0]
    n_main, n_side = _qty(entries, "main"), _qty(entries, "sideboard")
    rep.size = n_main
    if n_main < rules.size:
        rep.problems.append(f"main deck has {n_main} cards; at least {rules.size}")
    if rules.sideboard is not None and n_side > rules.sideboard:
        rep.problems.append(f"sideboard has {n_side} cards; at most {rules.sideboard}")
    deck = [e for e in entries if e.section in ("main", "sideboard")]
    _copies(rep, deck, rules.copies, rules.label)
    _legality(rep, deck)
    if rep.fmt is None:
        rows = [e.row for e in deck if e.row is not None]
        ok, no = [], []
        for f in LEGAL_IN:
            n = sum(1 for r in rows if (r.get("legalities") or {}).get(f) not in ("legal", "restricted"))
            (no.append(f"{f} ({n} card{'s' if n > 1 else ''})") if n else ok.append(f))
        line = "every card is legal in " + ", ".join(ok) if ok else "no format of " + "/".join(LEGAL_IN) + " allows every card"
        if ok and no:
            line += " · not in " + ", ".join(no)
        rep.notes.append(line + " (pass format to check one)")


# --------------------------------------------------------------------------- rendering


def card_line(e: Entry, tag_counts: Mapping[str, int] | TagFilter) -> str:
    r = e.row
    s = f"{e.qty} {r['name']}"
    if r.get("mana_cost"):
        s += f" {r['mana_cost']}"
    s += f" · {r.get('type_line') or ''}"
    if r.get("game_changer"):
        s += " [Game Changer]"
    roles = role_tags(r.get("tags"), tag_counts, n=3)
    if roles:
        s += " · roles: " + ", ".join(roles)
    return s


def _sort_key(e: Entry):  # by card type, basic lands last, then mana value and name
    basic = "Basic" in (e.row.get("supertypes") or [])
    return (CATEGORY_ORDER.index(category(e.row)), basic, float(e.row.get("cmc") or 0), e.name)


def render(rep: Report, *, title: str, extra: Sequence[str] = (), unresolved: Sequence[tuple[str, int, list[str]]] = (),
           read_as: Sequence[tuple[str, str]] = (), ignored: Sequence[str] = (),
           tag_counts: Mapping[str, int] | TagFilter | None = None) -> str:
    """``extra``: lines after the title; ``unresolved``: (text, line number or 0, closest names); ``read_as``:
    (text as written, card name it was read as)."""
    tc = tag_counts or {}
    entries = rep.entries
    out = [f"deck: {title}", *extra]
    fmt_label = rep.rules.label if rep.rules else (rep.fmt or "unknown")
    head = f"format: {fmt_label} ({rep.how}) · {rep.size} cards"
    if rep.rules and not rep.rules.commander:
        head = f"format: {fmt_label} ({rep.how}) · main {_qty(entries, 'main')} · sideboard {_qty(entries, 'sideboard')}"
    if rep.identity:
        head += f" · colour identity {rep.identity}"
    out.append(head)
    if unresolved:
        items = [f'"{t}"' + (f" (line {n})" if n else "") + (f" (closest: {', '.join(a)})" if a else "")
                 for t, n, a in unresolved]
        out.append(f"not found ({len(unresolved)}): " + "; ".join(items) + ". Fix these names and call again.")
    if read_as:
        shown = [f'"{t}" as {n}' for t, n in read_as[:10]]
        more = f" (+{len(read_as) - 10} more)" if len(read_as) > 10 else ""
        out.append(f"read ({len(read_as)}): " + "; ".join(shown) + more)
    if ignored:
        out.append(f"ignored lines ({len(ignored)}): " + "; ".join(f'"{t[:60]}"' for t in ignored[:5]))
    if rep.rules is None and rep.fmt is None:
        out.append("checks: none (format not detected; pass format to check the deck)")
    elif rep.problems:
        out.append(f"problems ({len(rep.problems)}):")
        out += [f"   - {p}" for p in rep.problems]
    else:
        out.append("problems: none found")
    out += [f"   {n}" for n in rep.notes]

    in_full = bool(rep.rules and rep.rules.commander)
    for sec in ("commander", "companion", "main", "sideboard"):
        es = [e for e in entries if e.section == sec and e.row is not None]
        if not es:
            continue
        if sec == "commander" and in_full:  # the commanders' full text: the deck is built around it
            out.append("commander" + ("s" if len(es) > 1 else "") + ":")
            out += [render_card(e.row, detail="full", fmt=rep.fmt, tag_counts=tc) for e in es]
            continue
        es.sort(key=_sort_key)
        n = sum(e.qty for e in es)
        if sec == "main":
            counts: dict[str, int] = {}
            for e in es:
                counts[category(e.row)] = counts.get(category(e.row), 0) + e.qty
            parts = [f"{counts[c]} {PLURAL.get(c, c + 's')}" for c in CATEGORY_ORDER if c in counts]
            out.append(f"main ({n}): " + " · ".join(parts))
        else:
            out.append(f"{sec} ({n}):")
        out += [card_line(e, tc) for e in es]
    n_maybe = _qty(entries, "maybe")
    if n_maybe:
        out.append(f"maybeboard: {n_maybe} cards (not part of the deck, not checked)")
    if rep.fmt and rep.identity:
        out.append(f'To find cards for this deck, call search_cards with color_identity="{rep.identity}", '
                   f'format="{rep.fmt}"; every card above is already in it.')
    elif rep.fmt:
        out.append(f'To find cards for this deck, call search_cards with format="{rep.fmt}".')
    return "\n".join(out)

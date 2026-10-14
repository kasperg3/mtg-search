"""Compact, LLM-friendly text rendering of cards for the MCP tools.

One block per card (~60-120 tokens in compact mode)::

    3. Counterspell — {U}{U} (MV 2) · Blue · Instant · uncommon
       Counter target spell.
       roles: counterspell, interrupt
       legal: pauper ✓ · EDHREC #14 · ~$1.20

Columns that are missing from an older ``cards.parquet`` are simply not rendered.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .tag_filter import TagFilter

COLOR_WORDS = {"W": "White", "U": "Blue", "B": "Black", "R": "Red", "G": "Green"}
COLOR_ORDER = "WUBRG"
MAJOR_FORMATS = ("standard", "pioneer", "modern", "legacy", "vintage", "pauper", "commander")
LAYOUT_MARKERS = {
    "modal_dfc": "MDFC",
    "transform": "transform",
    "split": "split",
    "adventure": "adventure",
    "flip": "flip",
    "meld": "meld",
    "prototype": "prototype",
}
COMPACT_ORACLE_CHARS = 250


def role_tags(tags: list[str] | None, tag_counts: Mapping[str, int] | TagFilter, n: int = 5) -> list[str]:
    """Up to ``n`` functional oracle tags (see :mod:`mtg_search.tag_filter`): trivia/flavour/
    template tags and too rare / too generic tags dropped; core deck roles and mid-specificity
    tags first. Pass a :class:`TagFilter` to use the tag hierarchy; a plain ``{slug: count}``
    mapping falls back to the pattern/list rules only."""
    tf = tag_counts if isinstance(tag_counts, TagFilter) else TagFilter(tag_counts)
    return tf.roles(tags, n)


def color_words(colors: list[str] | None) -> str:
    cs = [c for c in COLOR_ORDER if c in (colors or [])]
    if not cs:
        return "Colorless"
    return "/".join(COLOR_WORDS[c] for c in cs)


def _letters(colors: list[str] | None) -> str:
    return "".join(c for c in COLOR_ORDER if c in (colors or [])) or "C"


def _mv(cmc: Any) -> str:
    if cmc is None:
        return "?"
    f = float(cmc)
    return str(int(f)) if f.is_integer() else f"{f:g}"


def _stats(face: Mapping[str, Any]) -> str | None:
    if face.get("power") is not None or face.get("toughness") is not None:
        return f"{face.get('power') or ''}/{face.get('toughness') or ''}"
    if face.get("loyalty") is not None:
        return f"loyalty {face['loyalty']}"
    if face.get("defense") is not None:
        return f"defense {face['defense']}"
    return None


def faces_of(card: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Per-face dicts {name, mana_cost, type_line, oracle_text, power, ...}.

    Uses the ``faces`` column when present; otherwise parses ``doc_text`` (blocks joined by
    ``\\n//\\n``: ``name cost`` / type line / oracle lines / stats)."""
    faces = card.get("faces")
    if faces:
        return [dict(f) for f in faces]
    out = []
    doc = card.get("doc_text") or ""
    doc = re.sub(r"\nKeywords: .*$", "", doc)
    names = card.get("face_names") or [card.get("name")]
    blocks = doc.split("\n//\n")
    for i, block in enumerate(blocks):
        lines = block.split("\n")
        name = names[i] if i < len(names) else lines[0]
        cost = lines[0][len(name) :].strip() if lines[0].startswith(name) else ""
        f = {"name": name, "mana_cost": cost, "type_line": lines[1] if len(lines) > 1 else "",
             "oracle_text": "\n".join(lines[2:])}
        if len(blocks) == 1:
            for k in ("power", "toughness", "loyalty", "defense"):
                f[k] = card.get(k)
        out.append(f)
    return out


def _oracle(text: str | None, compact: bool, indent: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if compact and len(text) > COMPACT_ORACLE_CHARS:
        cut = text[:COMPACT_ORACLE_CHARS].rsplit(" ", 1)[0]
        text = cut.rstrip(" ,.;:") + "…"
    return [indent + line for line in text.split("\n")]


def legality_text(legalities: Mapping[str, Any] | None, fmt: str | None) -> str:
    leg = dict(legalities or {})
    if fmt:
        st = leg.get(fmt.lower())
        if st == "legal":
            return f"legal: {fmt.lower()} ✓"
        if st == "restricted":
            return f"legal: {fmt.lower()} (restricted)"
        return f"legal: {fmt.lower()} ✗ ({(st or 'unknown').replace('_', ' ')})"
    ok = [f for f in MAJOR_FORMATS if leg.get(f) in ("legal", "restricted")]
    return "legal: " + (", ".join(ok) if ok else "none of " + "/".join(MAJOR_FORMATS))


def price_text(prices: Mapping[str, Any] | None) -> str | None:
    p = dict(prices or {})
    if p.get("usd") is not None:
        return f"~${p['usd']:.2f}"
    if p.get("eur") is not None:
        return f"~€{p['eur']:.2f}"
    if p.get("tix") is not None:
        return f"~{p['tix']:.2f} tix"
    return None


def markers(card: Mapping[str, Any], faces: list[dict[str, Any]]) -> list[str]:
    m = []
    if card.get("is_legendary"):
        m.append("Legendary")
    pk = card.get("partner_kind")
    if card.get("can_be_commander"):
        m.append(f"Commander-eligible: {pk}" if pk else "Commander-eligible")
    elif pk:
        m.append(pk.capitalize() if pk == "background" else pk)
    if card.get("game_changer"):
        m.append("Game Changer")
    if card.get("reserved"):
        m.append("Reserved")
    lay = LAYOUT_MARKERS.get(card.get("layout") or "")
    if lay:
        back_land = any("Land" in (f.get("type_line") or "") for f in faces[1:])
        front_land = "Land" in (faces[0].get("type_line") or "") if faces else False
        if lay == "MDFC" and back_land and not front_land:
            lay = "MDFC: land back face"
        m.append(lay)
    return m


def render_card(
    card: Mapping[str, Any],
    *,
    rank: int | None = None,
    fmt: str | None = None,
    detail: str = "compact",
    tag_counts: Mapping[str, int] | TagFilter | None = None,
) -> str:
    compact = detail != "full"
    faces = faces_of(card)
    multi = len(faces) > 1
    head = f"{rank}. " if rank is not None else ""
    head += str(card.get("name"))
    cost = card.get("mana_cost") or ""
    head += f" — {cost} (MV {_mv(card.get('cmc'))})" if cost else f" — (MV {_mv(card.get('cmc'))})"
    head += f" · {color_words(card.get('colors'))} · {card.get('type_line') or ''}"
    if not multi:
        st = _stats(faces[0]) if faces else None
        if st:
            head += f" · {st}"
    if card.get("rarity"):
        head += f" · {card['rarity']}"
    mk = markers(card, faces)
    if mk:
        head += " " + " ".join(f"[{x}]" for x in mk)
    lines = [head]
    ind = "   "
    if multi:
        for f in faces:
            fl = f"{ind}// {f.get('name')}"
            if f.get("mana_cost"):
                fl += f" — {f['mana_cost']}"
            fl += f" · {f.get('type_line') or ''}"
            st = _stats(f)
            if st:
                fl += f" · {st}"
            lines.append(fl)
            lines += _oracle(f.get("oracle_text"), compact, ind + "  ")
    elif faces:
        lines += _oracle(faces[0].get("oracle_text"), compact, ind)

    info = []
    roles = role_tags(card.get("tags"), tag_counts or {}, n=5 if compact else 12)
    if roles:
        info.append("roles: " + ", ".join(roles))
    if card.get("keywords"):
        info.append("keywords: " + ", ".join(card["keywords"]))
    if card.get("produced_mana"):
        info.append("produces: " + "".join("{" + c + "}" for c in card["produced_mana"]))
    if info:
        lines.append(ind + " · ".join(info))

    foot = [legality_text(card.get("legalities"), fmt)]
    if card.get("edhrec_rank") is not None:
        foot.append(f"EDHREC #{card['edhrec_rank']}")
    pr = price_text(card.get("prices"))
    if pr:
        foot.append(pr)
    ci, co = card.get("color_identity"), card.get("colors")
    if ci is not None and set(ci or []) != set(co or []):
        foot.append(f"identity: {_letters(ci)}")
    lines.append(ind + " · ".join(foot))
    return "\n".join(lines)


def all_legalities(legalities: Mapping[str, Any] | None) -> str:
    groups: dict[str, list[str]] = {}
    for f, st in sorted(dict(legalities or {}).items()):
        groups.setdefault(st or "unknown", []).append(f)
    order = ["legal", "restricted", "banned", "not_legal"]
    parts = [f"{k.replace('_', ' ')}: {', '.join(groups[k])}" for k in order if k in groups]
    parts += [f"{k}: {', '.join(v)}" for k, v in groups.items() if k not in order]
    return "\n".join("   " + p for p in parts)

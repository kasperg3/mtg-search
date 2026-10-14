"""Turn a Scryfall card object into the text that gets embedded / indexed.

Format per face::

    {name} {mana_cost}
    {type_line}
    {oracle_text}
    {P/T | Loyalty: n | Defense: n}

Multi-face cards repeat the block per face, joined with ``"\\n//\\n"``.
Card-level keywords are appended once as ``Keywords: a, b``. Flavor text is
never included. Empty lines are omitted.
"""

from __future__ import annotations

from typing import Any

FACE_SEPARATOR = "\n//\n"


def _stats_line(obj: dict[str, Any]) -> str | None:
    if obj.get("power") is not None or obj.get("toughness") is not None:
        return f"{obj.get('power', '')}/{obj.get('toughness', '')}"
    if obj.get("loyalty") is not None:
        return f"Loyalty: {obj['loyalty']}"
    if obj.get("defense") is not None:
        return f"Defense: {obj['defense']}"
    return None


def face_block(face: dict[str, Any]) -> str:
    """Render one face (or a single-faced card) as a text block."""
    head = face.get("name", "")
    cost = face.get("mana_cost") or ""
    if cost:
        head = f"{head} {cost}"
    lines = [head]
    for key in ("type_line", "oracle_text"):
        val = face.get(key)
        if val:
            lines.append(val)
    stats = _stats_line(face)
    if stats:
        lines.append(stats)
    return "\n".join(line for line in lines if line)


def faces(card: dict[str, Any]) -> list[dict[str, Any]]:
    """Faces of a card; a single-faced card is its own only face."""
    return card.get("card_faces") or [card]


def doc_text(card: dict[str, Any]) -> str:
    text = FACE_SEPARATOR.join(face_block(f) for f in faces(card))
    keywords = card.get("keywords") or []
    if keywords:
        text += "\nKeywords: " + ", ".join(keywords)
    return text

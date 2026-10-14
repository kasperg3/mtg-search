"""Prompts, deck rendering and ``final_answer`` schemas for the EXP-007 tasks (plan §5).

The deck is rendered from card text only: ``render_card`` (compact for the 99, full for the
commander) on a copy of each card with oracle tags removed; ``render_card`` itself never shows
prices or popularity (allow-list). No product name, deck name, set code or release date is
available to this module at all: it only receives a :class:`~bench.precon.loader.Deck`.
"""

from __future__ import annotations

import hashlib
import re

from mtg_search.render import render_card

TASKS = ("cut", "add", "swap")

# Tool budgets per task (plan §5): (max tool calls, max turns)
BUDGETS = {"cut": (5, 10), "add": (10, 15), "swap": (12, 20)}

RULES = """Deck-building rules (Commander format):
- Singleton: at most one copy of each card, except basic lands and cards that say a deck can have any number of them.
- Every card must be within the commander's colour identity ({ci}) and legal in Commander.
- Power bracket {bracket}: the final deck may contain at most {gc_limit} cards marked [Game Changer].
- Judge cards by what their rules text does for this commander and this deck."""

INSTRUCTIONS = {
    "cut": ("Identify the {n} cards whose removal most improves the deck; you will replace them with stronger "
            "cards for this commander. Each must be a card in the deck below and must not be the commander. "
            "Basic lands are allowed but discouraged."),
    "add": ("Propose {n} cards to add to the deck (cards not already in it), best first. Lands are allowed."),
    "swap": ("Propose {n} one-for-one swaps that keep the deck at 100 cards: each swap cuts one card that is in the "
             "deck (not the commander) and adds one card that is not, with a one-sentence reason. Best swap first."),
}
N_PICKS = {"cut": 10, "add": 20, "swap": 10}

TOOLS_TEXT = {
    "mcp": ("You can use a card search engine over all Magic cards: `search_cards` (natural-language search with "
            "optional filters such as color_identity, format, types and mana value), `find_similar` (cards that play "
            "like a given card) and `get_card` (full text, rulings and legality of one card). Results show only "
            "rules text and printed facts."),
    "lexical": ("You can use two lexical tools over the full text of all Magic cards: `grep` (case-insensitive "
                "regular expression; returns the match count and 15 matching cards ordered by keyword relevance) and "
                "`bm25` (keyword search, top 15). Both accept optional filters such as color_identity and format."),
    "none": "You have no tools: answer from your own knowledge of Magic cards.",
}

SYSTEM = """You are an expert Magic: The Gathering Commander deck builder. You are given a complete 100-card Commander deck as card text and asked to improve it.

{tools}
You may make at most {max_calls} tool calls. {final}

{rules}"""

TOOL_HINT = ("Before you answer, use the tools: search for candidate cards and check exact card text, colour identity "
             "and legality of your picks. ")
FINAL_TOOL = "When you are done, call `final_answer` exactly once with your answer. Copy card names exactly."
FINAL_TEXT = "Reply with only a JSON object in a ```json code block, in this shape: {shape}. Copy card names exactly."
SHAPES = {
    "cut": '{"cuts": ["card name", ...]}',
    "add": '{"adds": ["card name", ...]}',
    "swap": '{"swaps": [{"cut": "card name", "add": "card name", "reason": "..."}, ...]}',
}


def final_answer_tool(task: str) -> dict:
    n = N_PICKS[task]
    if task == "swap":
        props = {"swaps": {"type": "array", "description": f"Exactly {n} swaps, best first.", "items": {
            "type": "object", "properties": {"cut": {"type": "string"}, "add": {"type": "string"},
                                              "reason": {"type": "string"}}, "required": ["cut", "add"]}}}
    else:
        key = "cuts" if task == "cut" else "adds"
        props = {key: {"type": "array", "items": {"type": "string"}, "description": f"Exactly {n} card names, best first."}}
    return {"type": "function", "function": {"name": "final_answer", "description": "Submit your final answer.",
                                             "parameters": {"type": "object", "properties": props, "required": list(props)}}}


def _strip_tags(row: dict) -> dict:
    return {**row, "tags": None}


def anonymise(text: str, name: str) -> str:
    """Replace the commander's name (and its short name before a comma) with neutral words."""
    out = text.replace(name, "your commander")
    short = name.split(",")[0].strip()
    if short and short != name and len(short) > 3:
        out = re.sub(rf"\b{re.escape(short)}\b", "your commander", out)
    return out


def render_deck(deck, card_rows: dict[str, dict], anonymise_commander: bool = False) -> str:
    """``card_rows``: oracle_id -> cards.parquet row (dict). Commander(s) in full, then the rest
    sorted by (land last, mana value, name) so the order carries no information about the source."""
    cmd_ids = list(deck.commander_oids)
    lines = ["COMMANDER" + ("S" if len(cmd_ids) > 1 else "")]
    for oid in cmd_ids:
        txt = render_card(_strip_tags(card_rows[oid]), detail="full", fmt="commander")
        if anonymise_commander:
            txt = anonymise(txt, card_rows[oid]["name"])
        lines.append(txt)
    qty = deck.qty
    rest = [o for o in qty if o not in set(cmd_ids)]
    rest.sort(key=lambda o: ("Land" in (card_rows[o].get("type_line") or "").split("//")[0],
                             float(card_rows[o].get("cmc") or 0), card_rows[o]["name"]))
    n = sum(qty[o] for o in rest)
    lines.append("")
    lines.append(f"DECK ({n} cards)")
    for o in rest:
        txt = render_card(_strip_tags(card_rows[o]), detail="compact", fmt="commander")
        if anonymise_commander:
            for c in cmd_ids:
                txt = anonymise(txt, card_rows[c]["name"])
        lines.append((f"{qty[o]}x " if qty[o] > 1 else "") + txt)
    return "\n".join(lines)


def system_prompt(task: str, tools: str, color_identity: str, bracket: int = 3, gc_limit: int = 3,
                  tool_hint: bool = False) -> str:
    """``tool_hint`` adds one sentence telling the agent to use its tools (added after Sonnet 5.5
    answered all 42 MCP episodes without a single tool call)."""
    max_calls = BUDGETS[task][0] if tools != "none" else 0
    return SYSTEM.format(
        tools=TOOLS_TEXT[tools], max_calls=max_calls,
        final=(TOOL_HINT if tool_hint else "") + FINAL_TOOL if tools != "none" else FINAL_TEXT.format(shape=SHAPES[task]),
        rules=RULES.format(ci=color_identity or "colourless", bracket=bracket, gc_limit=gc_limit),
    ).replace("You may make at most 0 tool calls. ", "")


def user_prompt(task: str, deck_text: str, few_shots: list[str] | None = None) -> str:
    parts = []
    for fs in few_shots or []:
        parts.append("EXAMPLE\n" + fs)
    parts.append("TASK: " + INSTRUCTIONS[task].format(n=N_PICKS[task]))
    parts.append(deck_text)
    return "\n\n".join(parts)


PROBE_SYSTEM = """You are an expert on Magic: The Gathering Commander preconstructed decks and the deck-building community."""
PROBE_USER = """Preconstructed Commander deck: "{deck_name}" from the product "{product}".

From memory only, list the cards that players most commonly cut from this precon when upgrading it, and the cards they most commonly add (as aggregated by EDHREC). This is a memory test for research: give your best guesses even if you are unsure, wrong guesses are expected and fine, and do not leave the lists empty. Reply with only a JSON object in a ```json code block: {{"cuts": [10 card names, most commonly cut first], "adds": [20 card names, most commonly added first]}}."""


def probe_prompt(deck_name: str, product: str) -> list[dict]:
    """Memorisation probe (plan T2c): product + deck name only, no deck list, no tools."""
    return [{"role": "system", "content": PROBE_SYSTEM},
            {"role": "user", "content": PROBE_USER.format(deck_name=deck_name, product=product)}]


def prompt_hash(*parts: str) -> str:
    return hashlib.sha256("\x1e".join(parts).encode()).hexdigest()[:16]

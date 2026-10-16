"""Decklists behind Moxfield and Archidekt links, for ``get_deck``.

A link the user gives is read the way their browser reads the deck page: one GET to the JSON endpoint that page
loads (``api2.moxfield.com/v3/decks/all/<id>``, ``archidekt.com/api/decks/<id>/``), with an identifying
User-Agent. Nothing is cached, stored or followed further. Only names, counts and sections are kept; the
sites' prices, ranks and comments are never read into the result. A private or missing deck, a rate limit or a
network error comes back as :class:`LinkError`, whose message asks for the exported text instead.

MTGGoldfish and TappedOut answer scripted requests with a Cloudflare challenge (HTTP 403, checked 2026-10-16),
so the server does not try them; :func:`fetch` raises :class:`LinkError` for any host it does not know.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .decks import Decklist, Line
from .precons import USER_AGENT

TIMEOUT = 15.0
MAX_BYTES = 8_000_000  # a 100-card Moxfield deck is ~0.5 MB of JSON

MOXFIELD = re.compile(r"https?://(?:www\.)?moxfield\.com/decks/([A-Za-z0-9_-]{6,})")
ARCHIDEKT = re.compile(r"https?://(?:www\.)?archidekt\.com/(?:api/)?decks/(\d+)")
SITES = "Moxfield and Archidekt"

# Moxfield board -> get_deck section; other boards (tokens, attractions, stickers, planes ...) are not the deck
MOXFIELD_BOARDS = {"commanders": "commander", "companions": "companion", "mainboard": "main",
                   "sideboard": "sideboard", "maybeboard": "maybe"}
MOXFIELD_FORMATS = {"pauperedh": "paupercommander", "duelcommander": "duel", "historicbrawl": "brawl",
                    "edh": "commander", "oldschool": "oldschool", "pennydreadful": "penny"}
# Archidekt's deckFormat ids, checked against its public decks on 2026-10-16 (7 is "Custom", 8 "Frontier")
ARCHIDEKT_FORMATS = {1: "standard", 2: "modern", 3: "commander", 4: "legacy", 5: "vintage", 6: "pauper",
                     9: "future", 10: "penny"}


class LinkError(ValueError):
    """The deck behind a link could not be read; the message says what to do instead."""


@dataclass
class LinkedDeck:
    site: str  # "Moxfield" | "Archidekt"
    name: str
    url: str
    format: str | None  # the site's format as a Scryfall key, None when it has none we know
    decklist: Decklist


def site_of(url: str) -> str | None:
    if MOXFIELD.match(url):
        return "Moxfield"
    if ARCHIDEKT.match(url):
        return "Archidekt"
    return None


def fetch(url: str, client: Any = None) -> LinkedDeck:
    """Read one public deck from a Moxfield or Archidekt link (``client``: an ``httpx.Client``, for tests)."""
    m_mox, m_arc = MOXFIELD.match(url), ARCHIDEKT.match(url)
    if not (m_mox or m_arc):
        host = re.sub(r"^www\.", "", (re.match(r"https?://([^/?#]+)", url) or re.match(r"(.*)", url)).group(1))
        raise LinkError(f"This server reads {SITES} deck links, not {host}. Export the deck from {host} as text "
                        '(one card per line, e.g. "1 Sol Ring") and pass that text as deck.')
    site = "Moxfield" if m_mox else "Archidekt"
    api = (f"https://api2.moxfield.com/v3/decks/all/{m_mox.group(1)}" if m_mox
           else f"https://archidekt.com/api/decks/{m_arc.group(1)}/")
    data = _get_json(api, site, client)
    return _moxfield(data, url) if m_mox else _archidekt(data, url)


def _get_json(api: str, site: str, client: Any) -> dict:
    import httpx

    own = client is None
    client = client or httpx.Client(timeout=TIMEOUT, follow_redirects=True)
    instead = f"Ask the user to export the deck from {site} as text and pass that text as deck."
    try:
        r = client.get(api, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    except httpx.HTTPError as e:
        raise LinkError(f"Could not reach {site} ({type(e).__name__}). {instead}") from e
    finally:
        if own:
            client.close()
    if r.status_code in (401, 403, 404):
        raise LinkError(f"{site} did not return this deck (HTTP {r.status_code}): it may be private, deleted or "
                        f"the link may be wrong. {instead}")
    if r.status_code == 429:
        raise LinkError(f"{site} is limiting requests right now (HTTP 429). Try again in a minute, or: {instead}")
    if r.status_code != 200 or len(r.content) > MAX_BYTES:
        raise LinkError(f"{site} answered HTTP {r.status_code}. {instead}")
    try:
        return r.json()
    except ValueError as e:
        raise LinkError(f"{site} did not return a deck. {instead}") from e


def _format(key: str | None, known: dict[str, str]) -> str | None:
    if not key:
        return None
    k = re.sub(r"[\s_-]+", "", str(key).lower())
    return known.get(k, k)


def _moxfield(data: dict, url: str) -> LinkedDeck:
    out, n = Decklist(sections_marked=True), 0
    boards = data.get("boards") or {}
    for board, section in MOXFIELD_BOARDS.items():
        for c in ((boards.get(board) or {}).get("cards") or {}).values():
            name = ((c.get("card") or {}).get("name") or "").strip()
            if name:
                n += 1
                out.lines.append(Line(int(c.get("quantity") or 1), name, section, n))
    fmt = _format(data.get("format"), MOXFIELD_FORMATS)
    if fmt == "brawl":  # Moxfield's "brawl" is Standard Brawl when the deck has 60 cards
        size = sum(ln.qty for ln in out.lines if ln.section in ("commander", "main"))
        fmt = "standardbrawl" if size <= 65 else "brawl"
    return LinkedDeck("Moxfield", data.get("name") or "Moxfield deck", url, fmt, out)


def _archidekt(data: dict, url: str) -> LinkedDeck:
    """A card's first category is its primary one; categories marked not in the deck (Maybeboard, Sideboard,
    any custom one) leave the card out of the main deck."""
    excluded = {(c.get("name") or "").lower() for c in data.get("categories") or [] if not c.get("includedInDeck", True)}
    out, n = Decklist(sections_marked=True), 0
    for c in data.get("cards") or []:
        if c.get("deletedAt"):
            continue
        name = (((c.get("card") or {}).get("oracleCard") or {}).get("name") or "").strip()
        if not name:
            continue
        cats = [x.lower() for x in c.get("categories") or []]
        if "commander" in cats:
            section = "commander"
        elif c.get("companion"):
            section = "companion"
        elif cats and cats[0] in excluded:
            section = "sideboard" if cats[0].startswith("sideboard") else "maybe"
        else:
            section = "main"
        n += 1
        out.lines.append(Line(int(c.get("quantity") or 1), name, section, n))
    fmt = ARCHIDEKT_FORMATS.get(data.get("deckFormat"))
    return LinkedDeck("Archidekt", data.get("name") or "Archidekt deck", url, fmt, out)

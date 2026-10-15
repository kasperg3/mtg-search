"""Pre-flight leakage checks for EXP-007 prompts (plan §7: T1, T2b, T4, T6, T9).

``preflight(...)`` raises :class:`PromptLeak` before any LLM call if
  * a name from any dev/test gold label set appears in the system prompt or a few-shot (T4;
    the episode's own deck is input and exempt),
  * a few-shot comes from a precon outside the train split (T4),
  * any precon's product name, deck name, set code or EDHREC slug appears anywhere in the prompt
    (T2b; strings that are also names of cards in this deck are exempt),
  * a popularity / price string appears anywhere in the prompt (T1/T6),
  * a tool outside the allow-list is offered (T9: no web/fetch tool),
  * the model-facing tool text (server instructions, tool names, descriptions and every string in
    the parameter schemas) names a dev/test gold card, a precon identifier or a popularity / price
    string (review m1: the shipped ``find_similar`` example named a dev gold card).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from .tools import LEAK

ALLOWED_TOOLS = {"search_cards", "find_similar", "get_card", "list_tags", "check_cards", "grep", "bm25", "final_answer"}


class PromptLeak(AssertionError):
    pass


def tool_text(schemas: list[dict], instructions: str | None = None) -> str:
    """Everything a toolset shows the model: server instructions plus every string (keys and values)
    in the tool schemas: names, descriptions, parameter descriptions, enums, examples."""
    out = [instructions or ""]

    def walk(o) -> None:
        if isinstance(o, str):
            out.append(o)
        elif isinstance(o, dict):
            for k, v in o.items():
                out.append(str(k))
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(schemas)
    return "\n".join(out)


def _contains(text_lc: str, needle: str) -> bool:
    n = needle.lower().strip()
    return bool(n) and re.search(rf"(?<![\w']){re.escape(n)}(?![\w'])", text_lc) is not None


def identifying_strings(precons: dict[str, dict]) -> set[str]:
    out = set()
    for meta in precons.values():
        for k in ("deck_name", "product", "edhrec_slug"):
            v = meta.get(k)
            if isinstance(v, str) and len(v) >= 6:
                out.add(v)
        sc = meta.get("set_code")
        if isinstance(sc, str) and len(sc) >= 3:
            out.add(f"({sc})")  # bare set codes are too short to search for as words
    return out


def preflight(system: str, user: str, *, deck_names: Iterable[str], forbidden_gold: Iterable[str],
              identifying: Iterable[str], few_shots: Iterable[str] = (), few_shot_splits: Iterable[str] = (),
              tool_names: Iterable[str] = (), tool_text: str = "") -> None:  # noqa: F811 - kwarg named like the helper
    deck_lc = {n.lower() for n in deck_names}
    sys_fs = "\n".join([system, *few_shots]).lower()
    for name in forbidden_gold:
        if name.lower() not in deck_lc and _contains(sys_fs, name):
            raise PromptLeak(f"dev/test gold card in system prompt or few-shot: {name!r}")
    bad = [s for s in few_shot_splits if s != "train"]
    if bad:
        raise PromptLeak(f"few-shot from non-train split: {bad}")
    full = (system + "\n" + user).lower()
    for s in identifying:
        if any(s.lower() in d for d in deck_lc):
            continue
        if _contains(full, s):
            raise PromptLeak(f"product/deck identifier in prompt: {s!r}")
    m = LEAK.search(system + "\n" + user)
    if m:
        raise PromptLeak(f"popularity/price string in prompt: {m.group(0)!r}")
    extra = set(tool_names) - ALLOWED_TOOLS
    if extra:
        raise PromptLeak(f"tool outside allow-list: {sorted(extra)}")
    check_tool_text(tool_text, forbidden_gold=forbidden_gold, identifying=identifying)


def check_tool_text(text: str, *, forbidden_gold: Iterable[str], identifying: Iterable[str]) -> None:
    """Static tool text is the same for every deck, so there is no deck exemption here."""
    if not text:
        return
    lc = text.lower()
    hits = sorted(n for n in forbidden_gold if _contains(lc, n))
    if hits:
        raise PromptLeak(f"dev/test gold card in tool text: {hits}")
    ids = sorted(s for s in identifying if _contains(lc, s))
    if ids:
        raise PromptLeak(f"product/deck identifier in tool text: {ids}")
    m = LEAK.search(text)
    if m:
        raise PromptLeak(f"popularity/price string in tool text: {m.group(0)!r}")

"""EXP-007 harness plumbing: parsing, leakage pre-flight, deck rendering, agent loop with a
fake LLM, tool-call and spending caps. No network, no data files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.precon.harness.agent import parse_final, run_episode  # noqa: E402
from bench.precon.harness.leakage import PromptLeak, identifying_strings, preflight  # noqa: E402
from bench.precon.harness.llm import Budget, BudgetExceeded, FakeLLM, cost_usd  # noqa: E402
from bench.precon.harness.prompts import render_deck, system_prompt, user_prompt  # noqa: E402
from bench.precon.harness.tools import LeakError, Toolset, check_tool_output, denied_tags  # noqa: E402
from bench.precon.loader import Deck  # noqa: E402


# ------------------------------------------------------------------ parsing

@pytest.mark.parametrize("payload,expect,how", [
    ({"cuts": ["A", "B"]}, {"cuts": ["A", "B"]}, "tool_args"),
    ({"cards": ["A"]}, {"cuts": ["A"]}, "tool_args"),  # single wrong key tolerated
    ({"cuts": [{"name": "A"}, "B"]}, {"cuts": ["A", "B"]}, "tool_args"),
    ('Sure!\n```json\n{"cuts": ["A", "B"]}\n```', {"cuts": ["A", "B"]}, "json_in_text"),
    ('my answer {"cuts": ["A"]} done', {"cuts": ["A"]}, "json_in_text"),
    ("1. A\n2. **B**\n- C", {"cuts": ["A", "B", "C"]}, "list_in_text"),
    ("I cannot help", {"cuts": []}, "unparsed"),
])
def test_parse_cut(payload, expect, how):
    assert parse_final("cut", payload) == (expect, how)


def test_parse_swap_shapes():
    out, how = parse_final("swap", {"swaps": [{"cut": "A", "add": "B", "reason": "r"}, "C -> D", ["E", "F"], {"cut": "X"}]})
    assert how == "tool_args"
    assert [(s["cut"], s["add"]) for s in out["swaps"]] == [("A", "B"), ("C", "D"), ("E", "F")]
    out, how = parse_final("swap", "1. Cut A → B\n2. G -> H")
    assert how == "list_in_text" and out["swaps"][1] == {"cut": "G", "add": "H", "reason": ""}
    out, _ = parse_final("add", '{"adds": "Sol Ring"}')
    assert out == {"adds": ["Sol Ring"]}


# ------------------------------------------------------------------ leakage

def test_preflight_rules():
    ok = dict(deck_names=["Sol Ring", "Commander X"], forbidden_gold={"Rhystic Study", "Sol Ring"},
              identifying={"Avengers Assemble", "Marvel Super Heroes Commander"}, tool_names=["grep", "final_answer"])
    preflight("system", "deck with Sol Ring", **ok)
    preflight("example uses Sol Ring", "deck", **ok)  # gold name that is in this deck is input, exempt
    with pytest.raises(PromptLeak, match="gold"):
        preflight("e.g. cut for Rhystic Study", "deck", **ok)
    with pytest.raises(PromptLeak, match="gold"):
        preflight("s", "u", **{**ok, "few_shots": ["add Rhystic Study"]})
    with pytest.raises(PromptLeak, match="few-shot"):
        preflight("s", "u", **{**ok, "few_shot_splits": ["train", "test"]})
    with pytest.raises(PromptLeak, match="identifier"):
        preflight("s", "This is the avengers assemble precon", **ok)
    with pytest.raises(PromptLeak, match="popularity"):
        preflight("s", "Sol Ring (EDHREC rank #1)", **ok)
    with pytest.raises(PromptLeak, match="allow-list"):
        preflight("s", "u", **{**ok, "tool_names": ["web_fetch"]})


def test_preflight_tool_text():
    """Review m1: the model-facing tool text is checked too, with no deck exemption (it is static)."""
    from bench.precon.harness.leakage import tool_text

    ok = dict(deck_names=["Sol Ring"], forbidden_gold={"Rhystic Study", "Sol Ring"},
              identifying={"Avengers Assemble"}, tool_names=["find_similar", "final_answer"])
    schemas = [{"type": "function", "function": {"name": "find_similar", "description": "Cards like a given card.",
                                                  "parameters": {"properties": {"card_name": {"description": "e.g. {}"}}}}}]

    def text(example: str, instructions: str = "") -> str:
        sc = [{**schemas[0], "function": {**schemas[0]["function"], "parameters": {
            "properties": {"card_name": {"description": f"e.g. {example}"}}}}}]
        return tool_text(sc, instructions)

    preflight("s", "u", **ok, tool_text=text("Divination"))
    with pytest.raises(PromptLeak, match="tool text"):  # nested parameter description
        preflight("s", "u", **ok, tool_text=text("Rhystic Study"))
    with pytest.raises(PromptLeak, match="tool text"):  # a deck card is not exempt in static tool text
        preflight("s", "deck with Sol Ring", **ok, tool_text=text("Sol Ring"))
    with pytest.raises(PromptLeak, match="tool text"):
        preflight("s", "u", **ok, tool_text=text("Divination", "Sorted by EDHREC rank"))


def test_shipped_surfaces_pass_tool_text_preflight():
    """Every shipped/frozen surface's text is free of card names from a gold-like set (here: the old example)."""
    from bench.precon.harness.leakage import check_tool_text
    from bench.precon.harness.surfaces import SURFACES
    from mtg_search.server import SIMILAR_DOC

    texts = [SIMILAR_DOC] + [v for s in SURFACES.values() for v in s.values() if isinstance(v, str)]
    for t in texts:
        check_tool_text(t, forbidden_gold={"Rhystic Study"}, identifying=set())


def test_identifying_strings_skip_short():
    ids = identifying_strings({"p": {"deck_name": "Aura of Courage", "product": "X", "set_code": "AFC",
                                     "edhrec_slug": "aura-of-courage"}})
    assert ids == {"Aura of Courage", "aura-of-courage", "(AFC)"}


def test_tool_output_guard():
    check_tool_output("1. Price of Progress — {1}{R} · Instant\n   Price of Progress deals damage")
    for bad in ("usd: 3.20", "$3.50", "EDHREC rank 12", "most played in Golgari", "a staple"):
        with pytest.raises(LeakError):
            check_tool_output(bad)


def test_denied_tags_globs():
    vocab = ["meme", "cycle-c16-face-commander", "aer-precon-tutor", "ramp", "draw"]
    assert denied_tags(vocab, {"exact": ["meme"], "patterns": ["cycle-*-face-commander", "*precon*"]}) == {
        "meme", "cycle-c16-face-commander", "aer-precon-tutor"}
    assert denied_tags(vocab, None) == set()


# ------------------------------------------------------------------ rendering

def _row(oid, name, cost, type_line, text, tags=("ramp",)):
    return {"oracle_id": oid, "name": name, "mana_cost": cost, "cmc": float(len(cost) // 3), "colors": ["G"],
            "color_identity": ["G"], "type_line": type_line, "rarity": "rare", "legalities": {"commander": "legal"},
            "tags": list(tags), "faces": [{"name": name, "mana_cost": cost, "type_line": type_line, "oracle_text": text}]}


ROWS = {
    "c": _row("c", "Tana, the Bloodsower", "{2}{R}{G}", "Legendary Creature — Elf Druid",
              "Whenever Tana, the Bloodsower deals combat damage to a player, create that many 1/1 Saprolings."),
    "a": _row("a", "Llanowar Elves", "{G}", "Creature — Elf Druid", "{T}: Add {G}."),
    "f": _row("f", "Forest", "", "Basic Land — Forest", "({T}: Add {G}.)"),
}
DECK = Deck("pc-x", ["c"], ["Tana, the Bloodsower"], frozenset("RG"),
            [{"oracle_id": "c", "qty": 1}, {"oracle_id": "f", "qty": 98}, {"oracle_id": "a", "qty": 1}])


def test_render_deck_card_text_only():
    txt = render_deck(DECK, ROWS)
    assert txt.startswith("COMMANDER\nTana, the Bloodsower — {2}{R}{G}")
    assert "DECK (99 cards)" in txt
    assert "98x Forest" in txt
    assert txt.index("Llanowar Elves") < txt.index("Forest —")  # nonland before land
    assert "roles:" not in txt and "ramp" not in txt  # oracle tags never rendered in the deck
    anon = render_deck(DECK, ROWS, anonymise_commander=True)
    assert "Tana" not in anon and "your commander" in anon


def test_system_prompt_budget_and_rules():
    s = system_prompt("add", "mcp", "RG")
    assert "at most 10 tool calls" in s and "(RG)" in s and "at most 3 cards marked [Game Changer]" in s
    s = system_prompt("cut", "none", "RG")
    assert "tool calls" not in s and '"cuts"' in s
    u = user_prompt("cut", "DECK ...")
    assert u.startswith("TASK: Identify the 10 cards")


# ------------------------------------------------------------------ agent loop

class StubTools(Toolset):
    name = "stub"
    tool_names = ("bm25",)

    def __init__(self):
        self.calls = []

    def schemas(self):
        return [{"type": "function", "function": {"name": "bm25", "description": "", "parameters": {
            "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}]

    def call(self, name, args):
        self.calls.append((name, args))
        return "bm25: x · 2 results\n1. Elvish Mystic — {G} (MV 1) · Green · Creature\n2. Fyndhorn Elves — {G} (MV 1) · Green · Creature"


def test_episode_with_fake_llm(tmp_path, monkeypatch):
    monkeypatch.setenv("GLM_API_KEY", "sk-SECRET-should-never-appear")
    tools = StubTools()
    sys_p, user = system_prompt("add", "lexical", "RG"), user_prompt("add", render_deck(DECK, ROWS))
    tp = tmp_path / "t.jsonl"
    res = run_episode(FakeLLM(n_tool_calls=2), tools, "add", sys_p, user, transcript_path=tp)
    assert res.stop == "final_answer" and res.parse == "tool_args"
    assert len(tools.calls) == 2 and res.llm_calls == 3
    assert res.output["adds"][:2] == ["Elvish Mystic", "Fyndhorn Elves"]
    assert res.usage["completion"] == 3 * 1650 and res.cost_usd > 0
    log = tp.read_text()
    assert "SECRET" not in log
    events = [json.loads(line)["event"] for line in log.splitlines()]
    assert events[0] == "config" and events[-1] == "final"


def test_tool_call_cap_enforced():
    tools = StubTools()
    res = run_episode(FakeLLM(n_tool_calls=9), tools, "cut", "s", "u", max_calls=3, max_turns=10)
    assert len(tools.calls) == 3  # after the cap only final_answer is offered
    assert res.stop == "final_answer"


def test_no_tools_text_answer():
    deck_user = user_prompt("cut", render_deck(DECK, ROWS))
    res = run_episode(FakeLLM(), Toolset(), "cut", "s", deck_user)
    assert res.stop == "text_answer" and res.parse == "json_in_text"
    assert res.output["cuts"] == ["Forest"]  # fake cuts = deck lines after the first


def test_budget_cap():
    b = Budget(max_usd=0.001)
    with pytest.raises(BudgetExceeded):
        run_episode(FakeLLM(n_tool_calls=5), StubTools(), "add", "s", "u", budget=b)
    assert b.usd > 0
    with pytest.raises(BudgetExceeded):
        b.check()


def test_cost_formula():
    u = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000, "prompt_tokens_details": {"cached_tokens": 500_000}}
    assert cost_usd(u, "glm-5.3") == pytest.approx(0.5 * 1.40 + 0.5 * 0.26 + 4.40)


def test_with_popularity_joins_eval_only_rank(tmp_path):
    """Evaluation code gets EDHREC rank only from the eval-only file, never from the card table."""
    import polars as pl

    from eval.popularity import with_popularity

    pl.DataFrame({"oracle_id": ["b", "a"], "edhrec_rank": [7, None]}).write_parquet(tmp_path / "pop.parquet")
    cards = pl.DataFrame({"oracle_id": ["a", "b", "c"], "name": ["A", "B", "C"]})
    out = with_popularity(cards, tmp_path / "pop.parquet")
    assert out["oracle_id"].to_list() == ["a", "b", "c"] and out["edhrec_rank"].to_list() == [None, 7, None]
    with pytest.raises(ValueError, match="already has edhrec_rank"):
        with_popularity(out, tmp_path / "pop.parquet")

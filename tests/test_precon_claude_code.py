"""EXP-007 Claude Code bridge: budget proxy logic and session isolation check (no CLI calls)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from bench.precon.harness.cc_proxy import CallBudget  # noqa: E402
from bench.precon.harness.claude_code import check_isolation  # noqa: E402
from bench.precon.harness.tools import LeakError  # noqa: E402


def test_call_budget_counts_and_caps():
    b = CallBudget(2)
    calls = []

    def call(name, args):
        calls.append(name)
        return "result"

    c1, r1 = b.handle("search_cards", {"query": "x"}, call)
    assert c1 == "result" and r1["n"] == 1
    c2, r2 = b.handle("get_card", {"name": "y"}, call)
    assert "call final_answer now" in c2 and r2["n"] == 2
    c3, r3 = b.handle("search_cards", {"query": "z"}, call)
    assert c3 == "Tool budget exhausted. Call final_answer now." and r3["event"] == "over_budget"
    assert calls == ["search_cards", "get_card"]
    c4, r4 = b.handle("final_answer", {"cuts": ["A"]}, call)
    assert r4 == {"event": "final_answer", "args": {"cuts": ["A"]}} and b.final == {"cuts": ["A"]}
    c5, r5 = b.handle("final_answer", {"cuts": ["B"]}, call)
    assert c5 == "Already received." and r5 is None and b.final == {"cuts": ["A"]}


def test_call_budget_tool_error_is_reported_not_raised():
    b = CallBudget(3)

    def call(name, args):
        raise ValueError("unknown card")

    content, rec = b.handle("get_card", {"name": "nope"}, call)
    assert content == "Tool error: unknown card" and rec["error"] == "unknown card" and b.n == 1


def test_isolation_accepts_proxy_only():
    check_isolation({"tools": ["mcp__deck__search_cards", "mcp__deck__final_answer"],
                     "mcp_servers": [{"name": "deck", "status": "connected"}]}, with_proxy=True)
    check_isolation({"tools": [], "mcp_servers": []}, with_proxy=False)


@pytest.mark.parametrize("init,with_proxy", [
    ({"tools": ["mcp__deck__search_cards", "WebSearch"], "mcp_servers": [{"name": "deck", "status": "connected"}]}, True),
    ({"tools": ["mcp__deck__search_cards", "mcp__github__x"],
      "mcp_servers": [{"name": "deck", "status": "connected"}, {"name": "github", "status": "connected"}]}, True),
    ({"tools": ["Read"], "mcp_servers": []}, False),
    ({"tools": [], "mcp_servers": [{"name": "claude.ai Drive", "status": "connected"}]}, False),
])
def test_isolation_rejects_extra_tools_or_servers(init, with_proxy):
    with pytest.raises(LeakError):
        check_isolation(init, with_proxy)

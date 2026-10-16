"""Model-facing text variants of the MCP server for A/B tests on the precon benchmark.

A variant overrides ``build_mcp(service, surface)``: keys ``instructions``, ``search_cards``, ``find_similar``,
``get_card``, ``get_deck`` (tool descriptions) and ``omit`` (tools to leave out); missing keys keep the shipped
text. "shipped" is ``server.py`` as is.

The variants tested in EXP-009/010/011 (v1, v2, v3, v3al, e1desc ... e6desc_slim) ran on the earlier
four-tool server; their exact text is in git history up to commit 61071da and in the EXP-009/010 write-ups.
"""

SURFACES: dict[str, dict[str, str]] = {
    "shipped": {},
    # the three-tool surface of EXP-014, before get_deck (control arm of eval/experiments/deck_tool_plan.md)
    "three_tools": {"omit": "get_deck"},
    # get_deck loaded up front instead of deferred (arm C of deck_tool_plan.md, only if B rarely calls it)
    "deck_always": {"always_load": "get_deck"},
}

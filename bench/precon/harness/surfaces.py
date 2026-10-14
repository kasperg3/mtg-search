"""Model-facing text variants of the shipped MCP server for A/B tests (EXP-009).

Each variant overrides ``build_mcp(service, surface)``: keys ``instructions``, ``search_cards``,
``find_similar``, ``get_card``, ``list_tags``. "v1" = the text as shipped in server.py.
"""

SURFACES: dict[str, dict[str, str]] = {
    "v1": {},
}

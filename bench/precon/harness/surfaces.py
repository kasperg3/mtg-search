"""Model-facing text variants of the MCP server for A/B tests on the precon benchmark.

A variant overrides ``build_mcp(service, surface)``: keys ``instructions``, ``search_cards``, ``find_similar``,
``get_card`` (tool descriptions); missing keys keep the shipped text. "shipped" is ``server.py`` as is.

The variants tested in EXP-009/010/011 (v1, v2, v3, v3al, e1desc ... e6desc_slim) ran on the earlier
four-tool server; their exact text is in git history up to commit 61071da and in the EXP-009/010 write-ups.
"""

SURFACES: dict[str, dict[str, str]] = {
    "shipped": {},
}

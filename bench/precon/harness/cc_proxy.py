"""Per-episode stdio MCP server that a headless Claude Code session talks to (EXP-007, ``--llm claude-code``).

It is lightweight (no models): the tool list and every tool call are forwarded over HTTP to
the :class:`~bench.precon.harness.claude_code.ToolService` that the runner hosts in-process, so
mLateOn is loaded once for the whole run. The proxy enforces the same per-task tool-call budget
and budget messages as :func:`bench.precon.harness.agent.run_episode`, records ``final_answer``
and appends every call to the episode log (JSONL).

Environment: ``CC_TOOL_URL``, ``CC_TASK``, ``CC_MAX_CALLS``, ``CC_LOG``.
"""

from __future__ import annotations

import json
import os
import urllib.request

import anyio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server


class CallBudget:
    """Tool-call accounting shared with the GLM loop's wording (agent.py)."""

    def __init__(self, cap: int):
        self.cap, self.n, self.final = cap, 0, None

    def handle(self, name: str, args: dict, call) -> tuple[str, dict | None]:
        """-> (content for the model, log record or None)."""
        if name == "final_answer":
            if self.final is not None:
                return "Already received.", None
            self.final = args
            return "Received. Your answer is recorded; reply with a one-line confirmation and stop.", {
                "event": "final_answer", "args": args}
        if self.n >= self.cap:
            return "Tool budget exhausted. Call final_answer now.", {"event": "over_budget", "tool": name}
        self.n += 1
        rec = {"event": "tool_call", "n": self.n, "tool": name, "args": args}
        try:
            content = call(name, args)
        except ValueError as e:
            content = f"Tool error: {e}"
            rec["error"] = str(e)
        rec["chars"] = len(content)
        if self.n >= self.cap:
            content += f"\n(That was tool call {self.n}/{self.cap}; call final_answer now.)"
        return content, rec


def _http(url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def main() -> None:
    base, task = os.environ["CC_TOOL_URL"].rstrip("/"), os.environ["CC_TASK"]
    budget = CallBudget(int(os.environ["CC_MAX_CALLS"]))
    log_path = os.environ["CC_LOG"]
    meta = _http(f"{base}/schemas?task={task}")
    schemas = meta["tools"]

    def log(rec: dict) -> None:
        with open(log_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def call(name: str, args: dict) -> str:
        out = _http(f"{base}/call", {"name": name, "args": args})
        if out.get("leak"):
            log({"event": "leak", "tool": name, "error": out["error"]})
            raise ValueError("tool unavailable")
        if "error" in out:
            raise ValueError(out["error"])
        return out["text"]

    async def on_list_tools(ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=[
            types.Tool(name=s["function"]["name"], description=s["function"]["description"],
                       input_schema=s["function"]["parameters"], meta=s.get("meta")) for s in schemas])

    async def on_call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        args = dict(params.arguments or {})
        content, rec = await anyio.to_thread.run_sync(budget.handle, params.name, args, call)
        if rec:
            log(rec)
        return types.CallToolResult(content=[types.TextContent(type="text", text=content)],
                                    is_error=bool(rec and rec.get("error")))

    server = Server(os.environ.get("CC_SERVER_NAME", "deck"), instructions=meta.get("instructions"),
                    on_list_tools=on_list_tools, on_call_tool=on_call_tool)

    async def run() -> None:
        async with stdio_server() as (r, w):
            await server.run(r, w, server.create_initialization_options())

    anyio.run(run)


if __name__ == "__main__":
    main()

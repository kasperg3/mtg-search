"""Toolsets for the precon benchmark agent (OpenAI function-calling schemas + executors).

``shipped``  the real mtg-search MCP surface (``src/mtg_search/server.py``: instructions, tool names,
             descriptions, schemas), served in-process over ``mlateon.MLateOn`` (``backend="local"``) or by
             the built Docker image over MCP stdio (``backend="docker"``). ``surface`` swaps in variant
             model-facing text from ``surfaces.SURFACES`` for A/B tests.
``none``     no tools.

Every tool output passes :func:`check_tool_output` (no popularity / price / crowd strings), and evaluative
or precon-identifying tags from the dataset's ``tag_denylist.json`` are removed before rendering
(in-process) or from ``roles:`` lines (Docker).
"""

from __future__ import annotations

import fnmatch
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parents[3]

# T1/T6 guard: strings that must never appear in a tool result.
# Patterns avoid bare words that occur in card names/oracle text ("Price of Progress").
LEAK = re.compile(r"edhrec|\$\s?\d|\bprices?\s*[:=]|\busd\b|\bpopularity\b|most played|rank #|salt score|"
                  r"inclusion rate|num_decks|potential_decks|synergy score|\bstaple\b", re.IGNORECASE)


class LeakError(AssertionError):
    pass


def check_tool_output(text: str) -> None:
    m = LEAK.search(text)
    if m:
        raise LeakError(f"popularity/price-like string in tool output: {m.group(0)!r}")


def denied_tags(all_slugs: list[str], denylist: dict | list | None) -> set[str]:
    """Resolve ``tag_denylist.json`` (exact slugs + glob patterns; accepts a list, or a dict
    with any of ``exact``/``tags``/``slugs`` and ``patterns``/``globs``) against the vocabulary."""
    if not denylist:
        return set()
    if isinstance(denylist, list):
        items = denylist
    else:
        items = []
        for k in ("exact", "tags", "slugs", "deny", "patterns", "globs", "glob"):
            v = denylist.get(k)
            if isinstance(v, list):
                items += [x if isinstance(x, str) else x.get("slug") or x.get("pattern") for x in v]
    out = set()
    for it in filter(None, items):
        if any(ch in it for ch in "*?["):
            out |= {s for s in all_slugs if fnmatch.fnmatch(s, it)}
        else:
            out.add(it)
    return out


def cards_for_eval(cards: pl.DataFrame, deny: set[str]) -> pl.DataFrame:
    if not deny or "tags" not in cards.columns:
        return cards
    return cards.with_columns(pl.col("tags").list.eval(pl.element().filter(~pl.element().is_in(list(deny)))))


def _fn(name: str, desc: str, params: dict) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": params}}


class Toolset:
    name = "none"
    tool_names: tuple[str, ...] = ()
    instructions: str | None = None

    def schemas(self) -> list[dict]:
        return []

    def call(self, name: str, args: dict) -> str:
        raise ValueError(f"unknown tool {name}")

    def config(self) -> dict:
        return {"tools": self.name}

    def close(self) -> None:
        pass


class NoTools(Toolset):
    pass


class _DockerMCP:
    """Minimal synchronous MCP stdio client for ``docker run -i --rm <image>``."""

    def __init__(self, image: str):
        self.p = subprocess.Popen(["docker", "run", "-i", "--rm", image], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._id = 0
        init = self._req("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                        "clientInfo": {"name": "precon-bench", "version": "0"}})
        self.instructions = init.get("instructions")
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _send(self, msg: dict) -> None:
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()

    def _req(self, method: str, params: dict) -> dict:
        self._id += 1
        rid = self._id
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("mtg-search MCP container exited")
            msg = json.loads(line)
            if msg.get("id") == rid:
                if "error" in msg:
                    raise ValueError(msg["error"].get("message"))
                return msg["result"]

    def list_tools(self) -> list[dict]:
        return self._req("tools/list", {})["tools"]

    def call(self, name: str, args: dict) -> str:
        res = self._req("tools/call", {"name": name, "arguments": args})
        text = "\n".join(c.get("text", "") for c in res.get("content", []))
        if res.get("isError"):
            raise ValueError(text)
        return text

    def close(self) -> None:
        if self.p.poll() is None:
            self.p.stdin.close()
            try:
                self.p.wait(30)
            except subprocess.TimeoutExpired:
                self.p.kill()


class ShippedToolset(Toolset):
    """The shipped MCP server's own surface: instructions, tool names, descriptions, schemas and argument
    handling (``build_mcp``). ``retriever`` replaces mLateOn in tests (any object with ``search``)."""

    name = "shipped"

    def __init__(self, surface: str = "shipped", backend: str = "local", denylist: dict | None = None,
                 data_dir: Path = ROOT / "data", retriever: Any = None, image: str = "mtg-search"):
        from .surfaces import SURFACES

        self.surface, self.backend_name = surface, backend
        tags_df = pl.read_parquet(data_dir / "tags.parquet") if (data_dir / "tags.parquet").exists() else None
        self.deny = denied_tags(tags_df["slug"].to_list(), denylist) if tags_df is not None else set()
        if backend == "docker":
            if SURFACES[surface]:
                raise ValueError("surface variants need backend='local'; the image serves its own text")
            self._docker = _DockerMCP(image)
            listed = self._docker.list_tools()
            self.instructions = self._docker.instructions
            self._schemas = [{**_fn(t["name"], t.get("description") or "", t["inputSchema"]),
                              **({"meta": t["_meta"]} if t.get("_meta") else {})} for t in listed]
        elif backend == "local":
            import anyio

            from mtg_search.server import CardService, build_mcp, server_columns

            cards = pl.read_parquet(data_dir / "cards.parquet")
            cards = cards_for_eval(cards.select(server_columns(cards.columns)), self.deny)
            if retriever is None:
                from mtg_search.mlateon import MLateOn

                retriever = MLateOn.local(cards["oracle_id"].to_list())
            tags = tags_df.filter(~pl.col("slug").is_in(list(self.deny))) if tags_df is not None else None
            self.svc = CardService(cards, retriever, tags=tags, rulings_path=data_dir / "rulings.parquet")
            self.mcp = build_mcp(self.svc, SURFACES[surface])
            self.instructions = self.mcp.instructions
            listed = anyio.run(self.mcp.list_tools)
            self._schemas = [{**_fn(t.name, t.description or "", dict(t.input_schema)),
                              **({"meta": dict(t.meta)} if t.meta else {})} for t in listed]
        else:
            raise ValueError(f"backend must be 'local' or 'docker', got {backend!r}")
        self.tool_names = tuple(sc["function"]["name"] for sc in self._schemas)

    def config(self) -> dict:
        return {"tools": "shipped", "surface": self.surface, "backend": self.backend_name, "n_denied_tags": len(self.deny),
                "surface_chars": len(self.instructions or "") + sum(len(json.dumps(s)) for s in self._schemas)}

    def schemas(self) -> list[dict]:
        return self._schemas

    def call(self, name: str, args: dict) -> str:
        if name not in self.tool_names:
            raise ValueError(f"unknown tool {name}")
        if self.backend_name == "docker":
            text = self._docker.call(name, args)
        else:
            import anyio

            try:
                res = anyio.run(self.mcp.call_tool, name, args)
            except Exception as e:  # ToolError / validation error -> shown to the model, like the real server
                raise ValueError(str(e)) from e
            text = "\n".join(getattr(c, "text", "") for c in getattr(res, "content", []) or [])
            if getattr(res, "is_error", False):
                raise ValueError(text)
        text = self._strip_denied_roles(text)
        check_tool_output(text)
        return text

    def _strip_denied_roles(self, text: str) -> str:
        if not self.deny:
            return text

        def fix(m):
            kept = [t for t in m.group(2).split(", ") if t.strip() not in self.deny]
            return m.group(1) + ", ".join(kept)

        return re.sub(r"(roles: )([^\n·]*)", fix, text)

    def close(self) -> None:
        if self.backend_name == "docker":
            self._docker.close()


def make_toolset(kind: str, denylist: dict | None = None, backend: str = "local", **kw: Any) -> Toolset:
    if kind == "none":
        return NoTools()
    if kind == "shipped":
        return ShippedToolset(backend=backend, denylist=denylist, **kw)
    raise ValueError(f"unknown toolset {kind!r} (shipped | none)")

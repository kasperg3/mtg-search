"""Pluggable toolsets for the EXP-007 agent (OpenAI function-calling schemas + executors).

``mcp``      the shipped mtg-search tools (``search_cards`` / ``find_similar`` / ``get_card``),
             either in-process (``CardService`` from ``src/mtg_search/server.py`` over an
             in-memory mLateOn backend: ``backend="late"``) or the built Docker image over MCP
             stdio (``backend="docker"``).
``lexical``  EXP-006 condition F style: ``grep`` (regex over card text; matches ordered by BM25
             of the pattern's literal words, ties alphabetical) and ``bm25``; cards rendered with
             the same ``render_card`` as the MCP tools.
``none``     no tools.

Every tool output passes :func:`check_tool_output` (no popularity / price / crowd strings), and
evaluative or precon-identifying oracle tags from the dataset's ``tag_denylist.json`` are removed
from the card table before rendering (in-process), or from ``roles:`` lines (Docker).
"""

from __future__ import annotations

import fnmatch
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
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


def _fn(name: str, desc: str, props: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": props, "required": required}}}


def _filter_props() -> dict:
    from mtg_search.filters import FILTER_PARAMS

    return {k: {kk: vv for kk, vv in v.items() if kk in ("type", "items", "enum", "description", "additionalProperties")}
            for k, v in FILTER_PARAMS.items()}


class Toolset:
    name = "none"
    tool_names: tuple[str, ...] = ()

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


class _MaskBackend:
    """Adapts a retriever that takes a boolean mask to CardService's row-index protocol."""

    def __init__(self, retriever, n: int):
        self.r, self.n = retriever, n

    def search(self, query: str, k: int, allowed_rows: np.ndarray | None = None):
        mask = None
        if allowed_rows is not None:
            mask = np.zeros(self.n, dtype=bool)
            mask[np.asarray(allowed_rows, dtype=np.int64)] = True
        return self.r.search(query, k, mask)


class MCPToolset(Toolset):
    name = "mcp"
    tool_names = ("search_cards", "find_similar", "get_card")

    def __init__(self, backend: str = "late", denylist: dict | None = None, data_dir: Path = ROOT / "data",
                 retriever=None, image: str = "mtg-search"):
        self.backend_name = backend
        self.deny: set[str] = set()
        tags_df = pl.read_parquet(data_dir / "tags.parquet") if (data_dir / "tags.parquet").exists() else None
        if tags_df is not None:
            self.deny = denied_tags(tags_df["slug"].to_list(), denylist)
        if backend == "docker":
            self._docker = _DockerMCP(image)
            self.svc = None
            return
        from mtg_search.server import CardService, server_columns

        cards = pl.read_parquet(data_dir / "cards.parquet")
        cards = cards_for_eval(cards.select(server_columns(cards.columns)), self.deny)
        if retriever is None:
            if backend == "late":
                from mtg_search.retrievers.late import LateInMemory

                retriever = LateInMemory.from_dir(ROOT / "index" / "embeddings", "lightonai/mLateOn")
            elif backend == "bm25":  # cheap stand-in for tests / dry runs; NOT the shipped ranking
                from mtg_search.retrievers.bm25 import BM25Retriever

                retriever = BM25Retriever(cards["doc_text"].to_list())
            else:
                raise ValueError(f"unknown backend {backend!r}")
        self.svc = CardService(cards, _MaskBackend(retriever, cards.height),
                               tags=tags_df.filter(~pl.col("slug").is_in(list(self.deny)))
                               .select("slug", "label", "description", "card_count", "parent_slugs")
                               if tags_df is not None else None,
                               rulings_path=data_dir / "rulings.parquet")

    def config(self) -> dict:
        return {"tools": "mcp", "backend": self.backend_name, "n_denied_tags": len(self.deny)}

    def schemas(self) -> list[dict]:
        fp = _filter_props()
        lim = {"limit": {"type": "integer", "description": "Cards to return (1-50, default 15)."},
               "detail": {"type": "string", "enum": ["compact", "full"]}}
        return [
            _fn("search_cards", "Semantic search over all Magic cards. Write the query like the rules text you want "
                "or describe the effect; put colours/format/types/mana value in filters. Results are ordered by text "
                "relevance only.", {"query": {"type": "string"}, "queries": {"type": "array", "items": {"type": "string"},
                                    "description": "Several phrasings; rankings are fused."}, **lim, **fp}, []),
            _fn("find_similar", "Cards that play like a given card (functional substitutes, upgrades).",
                {"card_name": {"type": "string"}, **lim, **fp}, ["card_name"]),
            _fn("get_card", "Full oracle text, legalities, functional tags and rulings of one card.",
                {"name": {"type": "string"}}, ["name"]),
        ]

    def call(self, name: str, args: dict) -> str:
        if name not in self.tool_names:
            raise ValueError(f"unknown tool {name}")
        if self.svc is None:
            out = self._docker.call(name, args)
            out = self._strip_denied_roles(out)
        else:
            fn = {"search_cards": self.svc.search_cards, "find_similar": self.svc.find_similar,
                  "get_card": self.svc.get_card}[name]
            out = fn(**args)
        check_tool_output(out)
        return out

    def _strip_denied_roles(self, text: str) -> str:
        if not self.deny:
            return text

        def fix(m):
            kept = [t for t in m.group(2).split(", ") if t.strip() not in self.deny]
            return m.group(1) + ", ".join(kept)

        return re.sub(r"((?:roles|oracle tags): )([^\n·]*)", fix, text)

    def close(self) -> None:
        if self.svc is None:
            self._docker.close()


class _DockerMCP:
    """Minimal synchronous MCP stdio client for ``docker run -i --rm <image>``."""

    def __init__(self, image: str):
        self.p = subprocess.Popen(["docker", "run", "-i", "--rm", image], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._id = 0
        self._req("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                 "clientInfo": {"name": "precon-bench", "version": "0"}})
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


class LexicalToolset(Toolset):
    """``grep`` + ``bm25`` over ``doc_text`` (EXP-006 condition F semantics, popularity-free)."""

    name = "lexical"
    tool_names = ("grep", "bm25")
    K = 15

    def __init__(self, denylist: dict | None = None, data_dir: Path = ROOT / "data"):
        from mtg_search.retrievers.bm25 import BM25Retriever
        from mtg_search.server import server_columns
        from mtg_search.tag_filter import TagFilter

        cards = pl.read_parquet(data_dir / "cards.parquet")
        tags_df = pl.read_parquet(data_dir / "tags.parquet") if (data_dir / "tags.parquet").exists() else None
        self.deny = denied_tags(tags_df["slug"].to_list(), denylist) if tags_df is not None else set()
        self.cards = cards_for_eval(cards.select(server_columns(cards.columns)), self.deny)
        self.docs = self.cards["doc_text"].to_list()
        self.names = self.cards["name"].to_list()
        self.funny = np.asarray(self.cards["is_funny"].to_list(), dtype=bool)
        self.bm25 = BM25Retriever(self.docs)
        self.tf = TagFilter.from_frame(tags_df) if tags_df is not None else None

    def config(self) -> dict:
        return {"tools": "lexical", "n_denied_tags": len(self.deny)}

    def schemas(self) -> list[dict]:
        fp = {k: v for k, v in _filter_props().items() if k in ("color_identity", "colors", "colors_mode", "format",
                                                                   "types", "cmc_min", "cmc_max")}
        return [
            _fn("grep", "Case-insensitive regex over card text. Returns the match count and 15 matching cards.",
                {"pattern": {"type": "string"}, **fp}, ["pattern"]),
            _fn("bm25", "BM25 keyword search over card text. Returns the top 15 cards.",
                {"query": {"type": "string"}, **fp}, ["query"]),
        ]

    def _mask(self, flt: dict) -> np.ndarray:
        from mtg_search.filters import FILTER_PARAMS, allowed_indices

        flt = {k: v for k, v in flt.items() if k in FILTER_PARAMS and v not in (None, "", [], {})}
        m = np.zeros(len(self.docs), dtype=bool)
        m[allowed_indices(self.cards, **flt)] = True
        return m & ~self.funny

    def _render(self, rows: list[int]) -> list[str]:
        from mtg_search.render import render_card

        return [render_card(self.cards.row(r, named=True), rank=i, detail="compact", tag_counts=self.tf)
                for i, r in enumerate(rows, 1)]

    def call(self, name: str, args: dict) -> str:
        args = dict(args)
        if name == "bm25":
            q = str(args.pop("query", "")).strip()
            if not q:
                raise ValueError("empty query")
            hits = self.bm25.search(q, self.K, self._mask(args))
            out = "\n".join([f'bm25: "{q}" · {len(hits)} results', *self._render([r for r, _ in hits])])
        elif name == "grep":
            pat = str(args.pop("pattern", ""))
            try:
                rx = re.compile(pat, re.IGNORECASE)
            except re.error as e:
                raise ValueError(f"invalid regex: {e}") from e
            mask = self._mask(args)
            match = [i for i in np.flatnonzero(mask) if rx.search(self.docs[i])]
            words = " ".join(re.findall(r"[A-Za-z][A-Za-z'+/-]*", re.sub(r"\\[a-zA-Z]", " ", pat)))
            score = np.zeros(len(self.docs), dtype=np.float32)
            if words and match:
                mm = np.zeros(len(self.docs), dtype=bool)
                mm[match] = True
                for r, s in self.bm25.search(words, len(match), mm):
                    score[r] = s
            order = sorted(match, key=lambda i: (-score[i], self.names[i]))[: self.K]
            out = "\n".join([f"grep: /{pat}/ · {len(match)} cards match; showing {len(order)}", *self._render(order)])
        else:
            raise ValueError(f"unknown tool {name}")
        check_tool_output(out)
        return out


class ShippedToolset(MCPToolset):
    """The shipped MCP server's own surface (instructions, tool names, descriptions, schemas and
    argument handling via ``build_mcp``) over the in-process eval service. ``surface`` selects the
    model-facing text: "v1" = as shipped in server.py, otherwise a variant from ``surfaces.SURFACES``."""

    name = "shipped"
    tool_names = ("search_cards", "find_similar", "get_card", "list_tags")

    def __init__(self, surface: str = "v1", **kw: Any):
        super().__init__(**kw)
        if self.svc is None:
            raise ValueError("shipped toolset needs an in-process backend (late or bm25)")
        import anyio

        from mtg_search.server import build_mcp

        from .surfaces import SURFACES

        self.surface = surface
        self.mcp = build_mcp(self.svc, SURFACES[surface])
        self.instructions = self.mcp.instructions
        self._schemas = [_fn(t.name, t.description or "", t.input_schema.get("properties", {}),
                             t.input_schema.get("required", [])) for t in anyio.run(self.mcp.list_tools)]
        for sc, t in zip(self._schemas, anyio.run(self.mcp.list_tools)):
            sc["function"]["parameters"] = dict(t.input_schema)  # keep $defs / constraints verbatim
        always = [x for x in SURFACES[surface].get("_always_load", "").split(",") if x]
        for sc in self._schemas:  # MCP tool _meta, passed through by the Claude Code proxy only
            if sc["function"]["name"] in always:
                sc["meta"] = {"anthropic/alwaysLoad": True}

    def config(self) -> dict:
        return {"tools": "shipped", "surface": self.surface, "backend": self.backend_name, "n_denied_tags": len(self.deny),
                "surface_chars": len(self.instructions or "") + sum(len(json.dumps(s)) for s in self._schemas)}

    def schemas(self) -> list[dict]:
        return self._schemas

    def call(self, name: str, args: dict) -> str:
        import anyio

        if name not in self.tool_names:
            raise ValueError(f"unknown tool {name}")
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


def make_toolset(kind: str, denylist: dict | None = None, backend: str = "late", **kw: Any) -> Toolset:
    if kind == "none":
        return NoTools()
    if kind == "mcp":
        return MCPToolset(backend=backend, denylist=denylist, **kw)
    if kind == "lexical":
        return LexicalToolset(denylist=denylist, **kw)
    if kind == "shipped":
        return ShippedToolset(backend=backend, denylist=denylist, **kw)
    raise ValueError(f"unknown toolset {kind!r}")

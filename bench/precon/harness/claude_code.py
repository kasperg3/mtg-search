"""Headless Claude Code as the EXP-007 agent (``--llm claude-code``): subscription, not the API.

One episode = one ``claude -p`` session in an empty temporary directory with:

* ``--system-prompt``: the harness system prompt (replaces Claude Code's own), user prompt on stdin;
* ``--tools ""``: no built-in tools (no file access, no shell, no web);
* ``--strict-mcp-config`` + one stdio MCP server, :mod:`bench.precon.harness.cc_proxy`, that exposes
  exactly the harness toolset plus ``final_answer`` and enforces the per-task call budget
  (the ``none`` condition gets no MCP server at all);
* ``--no-session-persistence``; ``ANTHROPIC_API_KEY`` is removed from the environment, so the
  session runs on the logged-in Claude Code subscription.

The ``init`` event of every session is checked: any tool or MCP server beyond the proxy aborts the
episode. Tool calls are served by :class:`ToolService`, which wraps the runner's in-process
toolset (mLateOn loaded once) behind a localhost HTTP endpoint; ``check_tool_output`` still runs
on every result. Sampling temperature cannot be set through Claude Code (provider default).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .agent import KEYS, parse_final
from .prompts import BUDGETS, N_PICKS, final_answer_tool, probe_prompt
from .tools import LeakError, Toolset

ROOT = Path(__file__).resolve().parents[3]
PROXY_SERVER = "deck"
DEFAULT_MODEL = "claude-sonnet-5-5"


class ToolService:
    """Localhost HTTP front for a Toolset: ``GET /schemas?task=`` and ``POST /call``."""

    def __init__(self, toolset: Toolset):
        self.toolset, self.lock = toolset, threading.Lock()
        svc = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def _send(self, obj: dict) -> None:
                body = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                task = parse_qs(urlparse(self.path).query).get("task", ["cut"])[0]
                self._send({"tools": svc.toolset.schemas() + [final_answer_tool(task)]})

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                try:
                    with svc.lock:
                        self._send({"text": svc.toolset.call(req["name"], req.get("args") or {})})
                except LeakError as e:
                    self._send({"leak": True, "error": str(e)})
                except (ValueError, TypeError) as e:
                    self._send({"error": str(e)})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()


def _env() -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    env.pop("CLAUDECODE", None)  # allow launching from inside a Claude Code session
    return env


def _usage(result: dict) -> dict:
    u = result.get("usage") or {}
    cached = u.get("cache_read_input_tokens") or 0
    prompt = (u.get("input_tokens") or 0) + cached + (u.get("cache_creation_input_tokens") or 0)
    return {"prompt": prompt, "cached": cached, "completion": u.get("output_tokens") or 0, "reasoning": 0}


def check_isolation(init: dict, with_proxy: bool) -> None:
    """The session must see only the proxy's tools (or none) and no other MCP server."""
    tools = init.get("tools") or []
    servers = [s.get("name") for s in init.get("mcp_servers") or []]
    bad = [t for t in tools if not (with_proxy and t.startswith(f"mcp__{PROXY_SERVER}__"))]
    if bad or servers != ([PROXY_SERVER] if with_proxy else []):
        raise LeakError(f"Claude Code session not isolated: tools={tools} mcp_servers={servers}")
    if with_proxy and any(s.get("status") != "connected" for s in init.get("mcp_servers") or []):
        raise RuntimeError(f"proxy MCP server not connected: {init.get('mcp_servers')}")


class ClaudeCode:
    """Runs episodes and probes as headless ``claude -p`` sessions."""

    def __init__(self, model: str = DEFAULT_MODEL, timeout: float = 1200, retries: int = 2, backoff: float = 60):
        if not shutil.which("claude"):
            sys.exit("claude CLI not found")
        self.model, self.timeout, self.retries, self.backoff = model, timeout, retries, backoff
        self.version = subprocess.run(["claude", "--version"], capture_output=True, text=True).stdout.strip()

    def config(self) -> dict:
        return {"llm": "claude-code", "model": self.model, "cli": self.version, "billing": "subscription",
                "temperature": "provider default"}

    def _session(self, system: str, user: str, transcript_path: Path | None, mcp: dict | None) -> tuple[dict, list]:
        cmd = ["claude", "-p", "--model", self.model, "--system-prompt", system, "--tools", "",
               "--strict-mcp-config", "--no-session-persistence", "--output-format", "stream-json", "--verbose"]
        if mcp:
            cmd += ["--mcp-config", json.dumps({"mcpServers": {PROXY_SERVER: mcp}}), "--allowedTools", f"mcp__{PROXY_SERVER}"]
        cwd = tempfile.mkdtemp(prefix="cc-precon-")
        try:
            p = subprocess.run(cmd, input=user, capture_output=True, text=True, cwd=cwd, env=_env(), timeout=self.timeout)
        finally:
            shutil.rmtree(cwd, ignore_errors=True)
        events = []
        for line in p.stdout.splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        if transcript_path:
            with open(transcript_path, "a") as f:
                for ev in events:
                    f.write(json.dumps({"event": "cc", "data": ev}, ensure_ascii=False) + "\n")
                if p.stderr.strip():
                    f.write(json.dumps({"event": "stderr", "text": p.stderr[-4000:]}) + "\n")
        init = next((e for e in events if e.get("type") == "system" and e.get("subtype") == "init"), None)
        result = next((e for e in reversed(events) if e.get("type") == "result"), None)
        if init is None or result is None:
            raise RuntimeError(f"claude -p failed (exit {p.returncode}): {p.stderr[-500:] or p.stdout[-500:]}")
        check_isolation(init, with_proxy=mcp is not None)
        if result.get("is_error"):
            raise RuntimeError(f"claude -p error: {result.get('subtype')} {str(result.get('result'))[:300]}")
        return result, events

    def _retrying(self, system: str, user: str, transcript_path: Path, mcp: dict | None) -> dict:
        """Retry transient CLI failures (rate limits, crashes); each attempt is marked in the log so
        only the last attempt's proxy events count. Isolation failures are never retried."""
        for attempt in range(self.retries + 1):
            with open(transcript_path, "a") as f:
                f.write(json.dumps({"event": "attempt", "n": attempt}) + "\n")
            try:
                return self._session(system, user, transcript_path, mcp)[0]
            except (RuntimeError, subprocess.TimeoutExpired):
                if attempt == self.retries:
                    raise
                time.sleep(self.backoff * (attempt + 1))
        raise AssertionError("unreachable")

    def episode(self, task: str, system: str, user: str, *, tool_url: str | None, transcript_path: Path) -> dict:
        t0 = time.time()
        with open(transcript_path, "w") as f:
            f.write(json.dumps({"event": "config", "task": task, "llm": self.config(), "tool_url": bool(tool_url),
                                "max_calls": BUDGETS[task][0], "system": system, "user": user}, ensure_ascii=False) + "\n")
        mcp = None
        if tool_url:
            mcp = {"type": "stdio", "command": sys.executable, "args": ["-m", "bench.precon.harness.cc_proxy"],
                   "env": {"PYTHONPATH": str(ROOT), "CC_TOOL_URL": tool_url, "CC_TASK": task,
                           "CC_MAX_CALLS": str(BUDGETS[task][0]), "CC_LOG": str(transcript_path)}}
        rec = {"task": task, "output": {KEYS[task]: []}, "stop": "error", "tool_calls": [], "llm_calls": 0,
               "usage": {"prompt": 0, "cached": 0, "completion": 0, "reasoning": 0}, "cost_usd": 0.0,
               "notional_usd": 0.0, "wall_s": 0.0, "parse": "", "error": None}
        try:
            result = self._retrying(system, user, transcript_path, mcp)
            log = [json.loads(x) for x in transcript_path.read_text().splitlines()]
            last = max(i for i, x in enumerate(log) if x.get("event") == "attempt")
            proxy = [x for x in log[last:] if x.get("event") in ("tool_call", "final_answer", "over_budget", "leak")]
            if any(x["event"] == "leak" for x in proxy):
                raise LeakError("tool output failed check_tool_output")
            rec["tool_calls"] = [{k: x.get(k) for k in ("n", "tool", "args", "chars", "error") if k in x}
                                 for x in proxy if x["event"] == "tool_call"]
            final = next((x["args"] for x in proxy if x["event"] == "final_answer"), None)
            if final is not None:
                rec["output"], rec["parse"] = parse_final(task, final)
                rec["stop"] = "final_answer"
            else:
                rec["output"], rec["parse"] = parse_final(task, result.get("result") or "")
                rec["stop"] = "text_answer" if rec["parse"] != "unparsed" else "no_answer"
            rec.update(llm_calls=result.get("num_turns") or 0, usage=_usage(result),
                       notional_usd=round(result.get("total_cost_usd") or 0.0, 6))
        except (LeakError, RuntimeError, subprocess.TimeoutExpired) as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        rec["wall_s"] = time.time() - t0
        if isinstance(rec["output"].get(KEYS[task]), list):
            rec["output"][KEYS[task]] = rec["output"][KEYS[task]][: N_PICKS[task]]
        with open(transcript_path, "a") as f:
            f.write(json.dumps({"event": "final", **{k: rec[k] for k in ("output", "stop", "usage", "error")}},
                               ensure_ascii=False) + "\n")
        if rec["error"] and rec["error"].startswith("LeakError"):
            raise LeakError(rec["error"])
        return rec

    def probe(self, deck_name: str, product: str, *, transcript_path: Path) -> dict:
        t0 = time.time()
        msgs = probe_prompt(deck_name, product)
        with open(transcript_path, "w") as f:
            f.write(json.dumps({"event": "request", "messages": msgs, "llm": self.config()}, ensure_ascii=False) + "\n")
        result = self._retrying(msgs[0]["content"], msgs[1]["content"], transcript_path, None)
        text = result.get("result") or ""
        cuts, how_c = parse_final("cut", text)
        adds, how_a = parse_final("add", text)
        return {"output": {"cuts": cuts["cuts"][:10], "adds": adds["adds"][:20]}, "parse": f"{how_c}/{how_a}",
                "usage": _usage(result), "cost_usd": 0.0, "notional_usd": round(result.get("total_cost_usd") or 0.0, 6),
                "llm_calls": result.get("num_turns") or 1, "wall_s": time.time() - t0, "stop": "text_answer"}

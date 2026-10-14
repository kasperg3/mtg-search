"""LLM clients for the EXP-007 harness: GLM-5.3 (Z.ai, OpenAI-compatible) and a scripted fake.

Adapted from ``eval/agentic/glm.py`` (same endpoint, pricing table and retry policy) so the
benchmark does not import an experiment directory. The key is read from ``GLM_API_KEY`` in the
environment or the repo ``.env``; it is never logged, printed or written to a transcript.

:class:`Budget` is the hard spending cap shared by every episode of a run: each response's
cost is charged *before* the next request, and :meth:`Budget.check` raises
:class:`BudgetExceeded` once the cap is reached, so a run can overshoot by at most one call.
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE_URL = os.environ.get("GLM_BASE_URL", "https://api.z.ai/api/paas/v4")
MODEL = os.environ.get("GLM_MODEL", "glm-5.3")
# USD per 1M tokens: (input, cached input, output); docs.z.ai pricing, as in eval/agentic/glm.py (2026-10-14)
PRICE = {"glm-5.3": (1.40, 0.26, 4.40), "glm-5.3-flash": (0.15, 0.03, 0.50), "fake": (1.40, 0.26, 4.40)}


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    """Thread-safe hard cap on USD (and optionally tokens) for a whole run."""

    def __init__(self, max_usd: float, max_tokens: int | None = None):
        self.max_usd, self.max_tokens = max_usd, max_tokens
        self.usd, self.tokens = 0.0, 0
        self._lock = threading.Lock()

    def check(self) -> None:
        with self._lock:
            if self.usd >= self.max_usd:
                raise BudgetExceeded(f"run budget ${self.max_usd:.2f} reached (${self.usd:.4f})")
            if self.max_tokens is not None and self.tokens >= self.max_tokens:
                raise BudgetExceeded(f"run token budget {self.max_tokens} reached")

    def charge(self, usd: float, tokens: int) -> None:
        with self._lock:
            self.usd += usd
            self.tokens += tokens


def _api_key() -> str:
    k = os.environ.get("GLM_API_KEY")
    if k:
        return k
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("GLM_API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
    raise RuntimeError("GLM_API_KEY not set")


def usage_tokens(u: dict) -> dict:
    return {
        "prompt": u.get("prompt_tokens", 0) or 0,
        "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0,
        "completion": u.get("completion_tokens", 0) or 0,
        "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0,
    }


def cost_usd(u: dict, model: str = MODEL) -> float:
    p_in, p_cached, p_out = PRICE.get(model, PRICE["glm-5.3"])
    t = usage_tokens(u)
    return ((t["prompt"] - t["cached"]) * p_in + t["cached"] * p_cached + t["completion"] * p_out) / 1e6


class GLM:
    """POST /chat/completions with tools; returns the raw OpenAI-style JSON response."""

    name = "glm"

    def __init__(self, model: str = MODEL, temperature: float = 0.0, effort: str = "low", timeout: float = 300):
        import httpx

        self._httpx = httpx
        self.model, self.temperature, self.effort = model, temperature, effort
        self.client = httpx.Client(base_url=BASE_URL, headers={"Authorization": f"Bearer {_api_key()}"}, timeout=timeout)

    def config(self) -> dict:
        return {"client": "glm", "model": self.model, "temperature": self.temperature, "effort": self.effort,
                "base_url": BASE_URL}

    def chat(self, messages: list[dict], tools: list[dict] | None = None, retries: int = 6) -> dict:
        body = {"model": self.model, "temperature": self.temperature, "messages": messages,
                "thinking": {"type": "enabled", "effort": self.effort}}
        if tools:
            body["tools"] = tools
        err = ""
        for attempt in range(retries):
            try:
                r = self.client.post("/chat/completions", json=body)
                if r.status_code == 200:
                    return r.json()
                err = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code not in (429, 500, 502, 503, 504):
                    raise RuntimeError(err)
            except (self._httpx.TransportError, self._httpx.TimeoutException) as e:
                err = repr(e)
            time.sleep(min(60, 2**attempt * 2) + random.random() * 2)
        raise RuntimeError(f"GLM request failed after {retries} attempts: {err}")


# =========================================================================== fake LLM (dry runs)

def approx_tokens(text: str) -> int:
    """~4 characters per token (English + card text); used only for fake-LLM usage."""
    return max(1, len(text) // 4)


_DECK_LINE = re.compile(r"^(?:\d+x )?(?P<name>[^—\n ][^—\n]*?) — ", re.M)
_RESULT_LINE = re.compile(r"^\s*\d+\. (?P<name>[^—\n]+?) — ", re.M)


class FakeLLM:
    """Deterministic scripted policy that exercises the full loop without any network call:
    turn 1..n call each available search tool once (with a query derived from the task), then
    ``final_answer`` built from names seen in the deck (cuts) and in tool results (adds).
    Usage is synthesised (~4 chars/token for the prompt, plus ``completion_per_call`` output
    tokens) so the dry run yields token estimates for the cost model."""

    name = "fake"

    def __init__(self, completion_per_call: int = 1650, n_tool_calls: int = 2):
        self.model, self.completion_per_call, self.n_tool_calls = "fake", completion_per_call, n_tool_calls
        self.temperature, self.effort = 0.0, "n/a"
        self._n = 0

    def config(self) -> dict:
        return {"client": "fake", "completion_per_call": self.completion_per_call, "n_tool_calls": self.n_tool_calls}

    def _call(self, name: str, args: dict) -> dict:
        self._n += 1
        return {"id": f"call_{self._n}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        tools = tools or []
        names = [t["function"]["name"] for t in tools]
        user = next(m["content"] for m in messages if m["role"] == "user")
        tool_out = "\n".join(m["content"] for m in messages if m["role"] == "tool")
        n_done = sum(1 for m in messages if m["role"] == "tool")
        search_tools = [n for n in names if n != "final_answer"]
        if search_tools and n_done < self.n_tool_calls:
            tool = search_tools[n_done % len(search_tools)]
            arg = {"search_cards": {"query": "draw a card whenever a creature enters", "limit": 15},
                   "bm25": {"query": "draw a card creature enters"},
                   "grep": {"pattern": "draw a card"},
                   "find_similar": {"card_name": "Sol Ring"},
                   "get_card": {"name": "Sol Ring"}}.get(tool, {})
            calls = [self._call(tool, arg)]
            content = ""
        else:
            deck_names = [m.group("name").strip() for m in _DECK_LINE.finditer(user.split("DECK", 1)[-1])]
            seen = list(dict.fromkeys(m.group("name").strip() for m in _RESULT_LINE.finditer(tool_out)))
            seen = [n for n in seen if n not in set(deck_names)] or ["Sol Ring", "Arcane Signet", "Command Tower"]
            cuts = deck_names[1:11]
            adds = (seen * 20)[:20]
            if "swaps" in json.dumps([t for t in tools if t["function"]["name"] == "final_answer"]):
                args = {"swaps": [{"cut": c, "add": a, "reason": "fake"} for c, a in zip(cuts, adds, strict=False)]}
            elif "adds" in json.dumps(tools):
                args = {"adds": adds}
            elif "cuts" in json.dumps(tools):
                args = {"cuts": cuts}
            else:
                args = {"cuts": cuts, "adds": adds}
            if "final_answer" in names:
                calls, content = [self._call("final_answer", args)], ""
            else:  # no tools at all: answer in content as JSON
                calls, content = [], "```json\n" + json.dumps(args) + "\n```"
        prompt_text = json.dumps(messages, ensure_ascii=False) + json.dumps(tools)
        usage = {"prompt_tokens": approx_tokens(prompt_text), "completion_tokens": self.completion_per_call,
                 "completion_tokens_details": {"reasoning_tokens": int(self.completion_per_call * 0.9)},
                 "prompt_tokens_details": {"cached_tokens": 0}}
        msg = {"role": "assistant", "content": content}
        if calls:
            msg["tool_calls"] = calls
        return {"choices": [{"message": msg, "finish_reason": "tool_calls" if calls else "stop"}], "usage": usage}

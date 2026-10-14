"""Memorisation probe (plan T2c): no tools, no deck list, only the product and deck name.

The model is asked to recall the community's common cuts and adds for the named precon; its
answer is scored with the same CUT (cuts) and ADD (adds) metrics. High probe scores on a split
mean the model has memorised that split's upgrade guides / EDHREC pages (contamination).
The prompt deliberately contains the product name, so the ``preflight`` identifier check is
NOT applied to it; it is never used for the scored deck-building conditions.
"""

from __future__ import annotations

import json
import time

from .agent import parse_final
from .llm import Budget, cost_usd, usage_tokens
from .prompts import probe_prompt


def run_probe(llm, deck_name: str, product: str, *, budget: Budget | None = None, transcript_path=None) -> dict:
    messages = probe_prompt(deck_name, product)
    if budget:
        budget.check()
    t0 = time.time()
    resp = llm.chat(messages, None)
    u = resp.get("usage") or {}
    c = cost_usd(u, getattr(llm, "model", "glm-5.3"))
    if budget:
        budget.charge(c, (u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or 0))
    text = resp["choices"][0]["message"].get("content") or ""
    cuts, how_c = parse_final("cut", text)
    adds, how_a = parse_final("add", text)
    if transcript_path:
        with open(transcript_path, "w") as f:
            f.write(json.dumps({"event": "request", "messages": messages}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"event": "response", "raw": resp}, ensure_ascii=False) + "\n")
    return {"output": {"cuts": cuts["cuts"][:10], "adds": adds["adds"][:20]}, "parse": f"{how_c}/{how_a}",
            "usage": usage_tokens(u), "cost_usd": round(c, 6), "llm_calls": 1, "wall_s": time.time() - t0, "stop": "text_answer"}

"""Minimal GLM (Z.ai, OpenAI-compatible) chat client with tool calling and cost accounting.

The key is read from ``GLM_API_KEY`` in the environment or the repo's ``.env``; it is never
logged. Endpoint: https://api.z.ai/api/paas/v4 (pay-as-you-go). GLM-5.3 always thinks; the
``thinking`` param only selects effort (low/high/max).
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
BASE_URL = os.environ.get("GLM_BASE_URL", "https://api.z.ai/api/paas/v4")
MODEL = os.environ.get("GLM_MODEL", "glm-5.3")
# USD per 1M tokens (docs.z.ai/guides/overview/pricing, checked 2026-10-14)
PRICE = {"glm-5.3": (1.40, 0.26, 4.40), "glm-5.3-flash": (0.15, 0.03, 0.50)}


def _api_key() -> str:
    k = os.environ.get("GLM_API_KEY")
    if k:
        return k
    for line in (ROOT / ".env").read_text().splitlines():
        if line.strip().startswith("GLM_API_KEY="):
            return line.split("=", 1)[1].strip().strip("\"'")
    raise RuntimeError("GLM_API_KEY not set")


def cost_usd(usage: dict, model: str = MODEL) -> float:
    p_in, p_cached, p_out = PRICE[model]
    prompt = usage.get("prompt_tokens", 0)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    out = usage.get("completion_tokens", 0)
    return ((prompt - cached) * p_in + cached * p_cached + out * p_out) / 1e6


class GLM:
    def __init__(self, model: str = MODEL, temperature: float = 0.6, effort: str = "low", timeout: float = 300):
        self.model, self.temperature, self.effort = model, temperature, effort
        self.client = httpx.Client(
            base_url=BASE_URL, headers={"Authorization": f"Bearer {_api_key()}"}, timeout=timeout
        )

    def body(self, messages: list[dict], tools: list[dict] | None = None, **extra) -> dict:
        b = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": messages,
            "thinking": {"type": "enabled", "effort": self.effort},
        }
        if tools:
            b["tools"] = tools
        b.update(extra)
        return b

    def chat(self, messages: list[dict], tools: list[dict] | None = None, retries: int = 6, **extra) -> dict:
        """POST /chat/completions; returns the raw JSON response. Retries 429/5xx/network errors."""
        body = self.body(messages, tools, **extra)
        for attempt in range(retries):
            try:
                r = self.client.post("/chat/completions", json=body)
                if r.status_code == 200:
                    return r.json()
                err = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code not in (429, 500, 502, 503, 504):
                    raise RuntimeError(err)
            except (httpx.TransportError, httpx.TimeoutException) as e:
                err = repr(e)
            time.sleep(min(60, 2**attempt * 2) + random.random() * 2)
        raise RuntimeError(f"GLM request failed after {retries} attempts: {err}")

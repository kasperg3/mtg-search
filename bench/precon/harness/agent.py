"""EXP-007 agent loop for the precon CUT / ADD / SWAP tasks.

One episode = system prompt (task rules, tool description, budget) + user prompt (task
instruction + the deck as card text, no product name) -> up to ``max_turns`` LLM turns with at
most ``max_calls`` tool calls (after that only ``final_answer`` is offered) -> parsed final answer.

* Final output: the ``final_answer`` tool call's JSON (tool conditions) or a JSON block in the
  reply (no-tools condition); :func:`parse_final` also accepts loosely shaped JSON, ``"A -> B"``
  swap strings and plain numbered lists, so a formatting slip costs validity, not the episode.
* Transcripts: every request delta, raw response (incl. reasoning) and tool result is appended
  to ``transcript_path`` (JSONL). The API key never enters a message, so it cannot be logged.
* Cost: tokens and USD per episode (GLM price table); the run-level :class:`Budget` is checked
  before every LLM call and raises ``BudgetExceeded`` at the cap.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .llm import Budget, cost_usd, usage_tokens
from .prompts import BUDGETS, N_PICKS, final_answer_tool
from .tools import LeakError, Toolset

KEYS = {"cut": "cuts", "add": "adds", "swap": "swaps"}


@dataclass
class EpisodeResult:
    task: str
    output: dict
    stop: str
    tool_calls: list = field(default_factory=list)
    llm_calls: int = 0
    usage: dict = field(default_factory=lambda: {"prompt": 0, "cached": 0, "completion": 0, "reasoning": 0})
    cost_usd: float = 0.0
    wall_s: float = 0.0
    parse: str = ""
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# =========================================================================== parsing

def _json_objects(text: str) -> list:
    """Every top-level JSON value found in ``text`` (code blocks first, then raw scanning)."""
    out = []
    for block in re.findall(r"```(?:json)?\s*(.*?)```", text, re.S):
        try:
            out.append(json.loads(block))
        except json.JSONDecodeError:
            pass
    dec = json.JSONDecoder()
    i = 0
    while i < len(text):
        j = min([p for p in (text.find("{", i), text.find("[", i)) if p >= 0], default=-1)
        if j < 0:
            break
        try:
            obj, end = dec.raw_decode(text, j)
            out.append(obj)
            i = end
        except json.JSONDecodeError:
            i = j + 1
    return out


def _names(v) -> list[str]:
    if isinstance(v, str):
        return [v]
    out = []
    for x in v or []:
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, dict):
            n = x.get("name") or x.get("card") or x.get("card_name")
            if n:
                out.append(str(n))
    return out


def _swap(x) -> dict | None:
    if isinstance(x, dict):
        c = x.get("cut") or x.get("out") or x.get("remove")
        a = x.get("add") or x.get("in")
        return {"cut": str(c), "add": str(a), "reason": str(x.get("reason") or "")} if c and a else None
    if isinstance(x, str):
        m = re.split(r"\s*(?:->|→|=>| for )\s*", x, maxsplit=1)
        if len(m) == 2:
            return {"cut": m[0].strip(), "add": m[1].strip(), "reason": ""}
    if isinstance(x, (list, tuple)) and len(x) >= 2:
        return {"cut": str(x[0]), "add": str(x[1]), "reason": ""}
    return None


def coerce(task: str, obj) -> dict | None:
    key = KEYS[task]
    if isinstance(obj, dict):
        v = obj.get(key)
        if v is None and len(obj) == 1:
            v = next(iter(obj.values()))
        if v is None:
            return None
    elif isinstance(obj, list):
        v = obj
    else:
        return None
    if task == "swap":
        sw = [s for s in (_swap(x) for x in v) if s]
        return {"swaps": sw} if sw else None
    names = _names(v)
    return {key: names} if names else None


def parse_final(task: str, payload) -> tuple[dict, str]:
    """-> (output dict, how it was parsed). ``payload`` is tool-call args (dict/str) or reply text."""
    if isinstance(payload, dict):
        got = coerce(task, payload)
        if got:
            return got, "tool_args"
        payload = json.dumps(payload)
    text = str(payload or "")
    for obj in _json_objects(text):
        got = coerce(task, obj)
        if got:
            return got, "json_in_text"
    bullet = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+")
    lines = [bullet.sub("", ln).strip().strip("*`").strip() for ln in text.splitlines() if bullet.match(ln)]
    lines = [ln for ln in lines if ln]
    if lines:
        if task == "swap":
            sw = [s for s in (_swap(ln) for ln in lines) if s]
            if sw:
                return {"swaps": sw}, "list_in_text"
        else:
            return {KEYS[task]: lines}, "list_in_text"
    return {KEYS[task]: []}, "unparsed"


# =========================================================================== episode

def run_episode(llm, toolset: Toolset, task: str, system: str, user: str, *, budget: Budget | None = None,
                transcript_path: Path | None = None, max_calls: int | None = None, max_turns: int | None = None) -> EpisodeResult:
    calls_cap, turns_cap = BUDGETS[task]
    calls_cap = calls_cap if max_calls is None else max_calls
    turns_cap = turns_cap if max_turns is None else max_turns
    tools_all = toolset.schemas() + ([final_answer_tool(task)] if toolset.schemas() else [])
    tf = open(transcript_path, "w") if transcript_path else None

    def log(rec: dict) -> None:
        if tf:
            tf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            tf.flush()

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    res = EpisodeResult(task=task, output={KEYS[task]: []}, stop="max_turns")
    t0, sent, nudged, n_tools = time.time(), 0, False, 0
    log({"event": "config", "task": task, "llm": llm.config(), "toolset": toolset.config(),
         "max_calls": calls_cap, "max_turns": turns_cap})
    try:
        for turn in range(turns_cap):
            if budget:
                budget.check()
            tools = tools_all if n_tools < calls_cap else [t for t in tools_all if t["function"]["name"] == "final_answer"]
            log({"event": "request", "turn": turn, "new_messages": messages[sent:], "tools": [t["function"]["name"] for t in tools]})
            sent = len(messages)
            resp = llm.chat(messages, tools or None)
            res.llm_calls += 1
            log({"event": "response", "turn": turn, "raw": resp})
            u = resp.get("usage") or {}
            for k, v in usage_tokens(u).items():
                res.usage[k] += v
            c = cost_usd(u, getattr(llm, "model", "glm-5.3"))
            res.cost_usd += c
            if budget:
                budget.charge(c, (u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or 0))
            msg = resp["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            amsg = {"role": "assistant", "content": msg.get("content") or ""}
            if calls:
                amsg["tool_calls"] = [{"id": tc["id"], "type": "function", "function": tc["function"]} for tc in calls]
            messages.append(amsg)
            if not calls:
                out, how = parse_final(task, msg.get("content") or "")
                if how != "unparsed" or not tools_all or nudged:
                    res.output, res.parse, res.stop = out, how, "text_answer" if how != "unparsed" else "no_answer"
                    break
                nudged = True
                messages.append({"role": "user", "content": "Please call final_answer with your answer now."})
                continue
            final = None
            for tc in calls:
                name = tc["function"]["name"]
                raw_args = tc["function"].get("arguments") or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                except json.JSONDecodeError:
                    args = raw_args
                if name == "final_answer" and final is None:
                    final = parse_final(task, args)
                    content = "Received."
                elif name == "final_answer":
                    content = "Already received."
                elif n_tools >= calls_cap:
                    content = "Tool budget exhausted. Call final_answer now."
                else:
                    n_tools += 1
                    rec = {"n": n_tools, "tool": name, "args": args if isinstance(args, dict) else {"raw": args}}
                    try:
                        if not isinstance(args, dict):
                            raise ValueError("arguments must be a JSON object")
                        content = toolset.call(name, args)
                    except LeakError:
                        raise
                    except (ValueError, TypeError) as e:
                        content = f"Tool error: {e}"
                        rec["error"] = str(e)
                    rec["chars"] = len(content)
                    res.tool_calls.append(rec)
                    if n_tools >= calls_cap:
                        content += f"\n(That was tool call {n_tools}/{calls_cap}; call final_answer now.)"
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": content})
            if final is not None:
                res.output, res.parse = final
                res.stop = "final_answer"
                log({"event": "tool_results", "new_messages": messages[sent:]})
                break
    except Exception as e:  # BudgetExceeded / LeakError / API failure: record and re-raise for the runner
        res.error, res.stop = f"{type(e).__name__}: {e}", "error"
        log({"event": "error", "error": res.error})
        res.wall_s = time.time() - t0
        if tf:
            tf.close()
        raise
    res.wall_s = time.time() - t0
    res.cost_usd = round(res.cost_usd, 6)
    if isinstance(res.output.get(KEYS[task]), list):
        res.output[KEYS[task]] = res.output[KEYS[task]][: N_PICKS[task]]
    log({"event": "final", "output": res.output, "stop": res.stop, "usage": res.usage, "cost_usd": res.cost_usd})
    if tf:
        tf.close()
    return res

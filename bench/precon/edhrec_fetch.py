"""Polite, cached fetcher for EDHREC precon pages (EXP-007 labels).

Access policy:
  * at most 1 request / 5 s (MIN_INTERVAL), single-threaded;
  * descriptive User-Agent with contact;
  * robots.txt checked before every URL;
  * every response cached on disk under data/bench_raw/edhrec/<snapshot>/; cached URLs are
    never re-fetched;
  * exponential back-off on errors / 429 / 5xx (honours Retry-After), abort after MAX_ERRORS.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.robotparser
from datetime import datetime, timezone
from pathlib import Path

import httpx

BASE = "https://edhrec.com"
USER_AGENT = "mtg-search research benchmark (contact: kaspergrontved@proton.me; 1 req/5s)"
MIN_INTERVAL = 5.0
MAX_RETRIES = 4
MAX_ERRORS = 10
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "bench_raw" / "edhrec"


class PoliteFetcher:
    def __init__(self, snapshot: str):
        self.dir = RAW / snapshot
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.dir / "fetch_log.jsonl"
        self.client = httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30, follow_redirects=True)
        self._last = 0.0
        self.stats = {"requests": 0, "cache_hits": 0, "errors": 0, "retries": 0, "seconds_network": 0.0}
        self._robots: urllib.robotparser.RobotFileParser | None = None

    # -- robots ---------------------------------------------------------------
    def robots(self) -> urllib.robotparser.RobotFileParser:
        if self._robots is None:
            txt = self.get_text(f"{BASE}/robots.txt", check_robots=False)
            rp = urllib.robotparser.RobotFileParser()
            rp.parse(txt.splitlines())
            self._robots = rp
        return self._robots

    def allowed(self, url: str) -> bool:
        return self.robots().can_fetch(USER_AGENT, url)

    # -- cache ----------------------------------------------------------------
    def cache_path(self, url: str) -> Path:
        slug = re.sub(r"[^A-Za-z0-9]+", "_", url.removeprefix(BASE).strip("/")) or "root"
        h = hashlib.sha1(url.encode()).hexdigest()[:8]
        return self.dir / f"{slug[:80]}__{h}.html"

    def _wait(self):
        dt = time.monotonic() - self._last
        if dt < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - dt)

    def _log(self, **kw):
        kw["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.log_path.open("a") as f:
            f.write(json.dumps(kw) + "\n")

    def get_text(self, url: str, check_robots: bool = True) -> str:
        p = self.cache_path(url)
        if p.exists():
            self.stats["cache_hits"] += 1
            return p.read_text()
        if check_robots and not self.allowed(url):
            raise PermissionError(f"robots.txt disallows {url}")
        if self.stats["errors"] >= MAX_ERRORS:
            raise RuntimeError("too many errors; stopping to avoid hammering")
        backoff = 30.0
        for attempt in range(MAX_RETRIES + 1):
            self._wait()
            t0 = time.monotonic()
            try:
                r = self.client.get(url)
                status = r.status_code
            except httpx.HTTPError as e:
                r, status = None, f"exc:{type(e).__name__}"
            self._last = time.monotonic()
            self.stats["requests"] += 1
            self.stats["seconds_network"] += self._last - t0
            self._log(url=url, status=status, attempt=attempt, bytes=len(r.content) if r is not None else 0)
            if r is not None and status == 200:
                p.write_text(r.text)
                return r.text
            if r is not None and status == 404:
                self.stats["errors"] += 1
                raise FileNotFoundError(url)
            self.stats["errors"] += 1
            if attempt == MAX_RETRIES:
                break
            self.stats["retries"] += 1
            wait = backoff
            if r is not None and r.headers.get("Retry-After", "").isdigit():
                wait = max(wait, float(r.headers["Retry-After"]))
            print(f"  ! {status} on {url}; backing off {wait:.0f}s", flush=True)
            time.sleep(wait)
            backoff *= 2
        raise RuntimeError(f"giving up on {url}")


NEXT_DATA = re.compile(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.S)


def next_data(html: str) -> dict:
    m = NEXT_DATA.search(html)
    if not m:
        raise ValueError("no __NEXT_DATA__ in page")
    return json.loads(m.group(1))

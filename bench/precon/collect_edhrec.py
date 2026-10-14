"""Step 2a: fetch the EDHREC precon index + every precon page (main commander only), politely.

    uv run python bench/precon/collect_edhrec.py --snapshot 2026-10-14

Only fetches; parsing is in build_dataset.py. Re-running reuses the on-disk cache (no re-fetch).
Writes data/bench_raw/edhrec/<snapshot>/fetch_summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edhrec_fetch import BASE, MIN_INTERVAL, USER_AGENT, PoliteFetcher, next_data  # noqa: E402


def index_entries(fetcher: PoliteFetcher) -> list[dict]:
    d = next_data(fetcher.get_text(f"{BASE}/precon"))
    out = []
    for cl in d["props"]["pageProps"]["data"]["container"]["json_dict"]["cardlists"]:
        for cv in cl["cardviews"]:
            out.append({"group": cl["header"], "group_tag": cl.get("tag"), **cv})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=date.today().isoformat())
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    f = PoliteFetcher(args.snapshot)
    t0 = time.time()
    entries = index_entries(f)
    urls = sorted({e["url"] for e in entries})
    print(f"{len(entries)} index entries, {len(urls)} precon pages; ~{len(urls) * MIN_INTERVAL / 60:.0f} min max", flush=True)
    failed = []
    for i, u in enumerate(urls[: args.limit]):
        try:
            f.get_text(BASE + u)
        except (FileNotFoundError, PermissionError, RuntimeError) as e:
            failed.append({"url": u, "error": repr(e)})
            print(f"  FAILED {u}: {e!r}", flush=True)
            if isinstance(e, RuntimeError) and "too many errors" in str(e):
                break
        if i % 20 == 0:
            print(f"[{i}/{len(urls)}] {f.stats}", flush=True)
    summary = {
        "snapshot_date": args.snapshot,
        "user_agent": USER_AGENT,
        "min_interval_s": MIN_INTERVAL,
        "n_index_entries": len(entries),
        "n_pages": len(urls),
        "failed": failed,
        "stats": f.stats,
        "wall_seconds": round(time.time() - t0, 1),
    }
    (f.dir / "fetch_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

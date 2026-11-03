"""Download raw Torvik data into data/raw. Safe to re-run: files already on disk are skipped.

What we pull (all free, public endpoints on barttorvik.com):
  team_results/{year}.csv      end-of-season team ratings (one row per team)
  gamestats/{year}.csv         one row per team-game (date, opponent, result, four factors, ...)
  timemachine/{YYYYMMDD}.json.gz   Torvik ratings as they stood on that date (no hindsight)

Be polite: one request per second, plain User-Agent, no parallelism.
"""
import datetime as dt
import subprocess
import sys
import time
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
UA = "Mozilla/5.0 (cbb-model personal project)"
DELAY = 1.0
FIRST_SEASON = 2012          # 15 completed seasons: 2012..2026 (season = year it ends)
LAST_COMPLETED = 2026
SNAPSHOT_FIRST_SEASON = 2015  # daily ratings snapshots do not exist before this
SNAPSHOT_STEP_DAYS = 3        # historical backfill cadence (live job fetches daily)


def get(url: str, dest: Path, min_bytes: int = 500) -> bool:
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        r = subprocess.run(
            ["curl", "-sS", "-m", "90", "-A", UA, "-L", "-o", str(dest), "-w", "%{http_code}", url],
            capture_output=True, text=True,
        )
        time.sleep(DELAY)
        if r.stdout.strip() == "200" and dest.exists() and dest.stat().st_size >= min_bytes:
            return True
        dest.unlink(missing_ok=True)
        time.sleep(2 * (attempt + 1))
    return False


def season_days(season: int):
    d, end = dt.date(season - 1, 11, 1), dt.date(season, 4, 10)
    while d <= end:
        yield d
        d += dt.timedelta(days=SNAPSHOT_STEP_DAYS)


def main():
    seasons = range(FIRST_SEASON, LAST_COMPLETED + 1)
    for s in list(seasons) + [LAST_COMPLETED + 1]:
        ok = get(f"https://barttorvik.com/{s}_team_results.csv", RAW / "team_results" / f"{s}.csv")
        print("team_results", s, "ok" if ok else "FAILED", flush=True)
    for s in seasons:
        ok = get(f"https://barttorvik.com/getgamestats.php?year={s}&csv=1", RAW / "gamestats" / f"{s}.csv", 10_000)
        print("gamestats", s, "ok" if ok else "FAILED", flush=True)
    missing = 0
    for s in range(SNAPSHOT_FIRST_SEASON, LAST_COMPLETED + 1):
        n = 0
        for d in season_days(s):
            tag = d.strftime("%Y%m%d")
            ok = get(f"https://barttorvik.com/timemachine/team_results/{tag}_team_results.json.gz",
                     RAW / "timemachine" / f"{tag}.json.gz", 5_000)
            n += ok
            missing += (not ok)
        print("snapshots", s, n, "files", flush=True)
    print("done; snapshot misses:", missing)


if __name__ == "__main__":
    sys.exit(main())

"""Cloud-side lines refresh: pulls sportsbook lines and attaches them to the predictions already on the site.

Runs on GitHub's own servers (the Odds API is not blocked there, unlike Torvik), so the EVs keep updating even when
CJ's Mac is asleep.  It never touches ratings or results: the Mac job does that.  Games with no new line keep the last
known one (shown on the page as older lines).  Env: ODDS_API_KEY.  Optional ODDS_CACHE=<file> re-uses a saved pull
instead of spending credits again (used when a push has to be retried).
"""
import datetime as dt
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from names import load_map  # noqa: E402
from lines import attach_lines, fetch_odds, log_bets, mark_takes  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "docs" / "data"


def main():
    games = json.loads((SITE / "games.json").read_text())
    model = json.loads((SITE / "model.json").read_text())
    status = json.loads((SITE / "status.json").read_text())
    status["warnings"] = [w for w in status.get("warnings", []) if "Odds API" not in w and "Betting lines" not in w]
    now = dt.datetime.now(dt.timezone.utc)
    now_s = now.strftime("%Y-%m-%dT%H:%MZ")

    cache = os.environ.get("ODDS_CACHE", "")
    if cache and Path(cache).exists():
        odds = json.loads(Path(cache).read_text())
    else:
        odds = fetch_odds(status)
        if odds and cache:
            Path(cache).write_text(json.dumps(odds))
    if not odds:
        print("no odds pulled:", status.get("lines"), status["warnings"])
        (SITE / "status.json").write_text(json.dumps(status))
        return 1 if os.environ.get("ODDS_API_KEY") else 0

    # only games that have not started: lines for games in progress or finished are meaningless
    upcoming = [g for g in games if not g.get("done") and g.get("start") and dt.datetime.fromisoformat(g["start"].replace("Z", "+00:00")) > now]
    before = {id(g["market"]) for g in upcoming if g.get("market")}
    prev_at = status.get("lines_at")
    attach_lines(upcoming, odds, load_map(), model, status)   # edits the same dict objects that sit in `games`
    for g in upcoming:                                 # a game this pull did not cover keeps its old line, labelled as old
        m = g.get("market")
        if m and id(m) in before and not m.get("stale_at"):
            m["stale_at"] = prev_at
    mark_takes(upcoming)
    log_bets(upcoming, now_s)
    status["lines_at"] = now_s
    (SITE / "games.json").write_text(json.dumps(games))
    (SITE / "status.json").write_text(json.dumps(status))
    print(json.dumps({k: status.get(k) for k in ("lines", "lines_matched", "lines_at", "totals_bias")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

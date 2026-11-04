"""NBA picks: ratings from game results (this season + last, recent games weighted most), then EV against sportsbook lines.

Standalone so a failure here can never block the college refresh.  Writes docs/data/nba.json.
Same EV method as lines.py, with two honest differences:
  * NBA ratings come only from results (no Torvik equivalent), so they are shrunk toward the market harder (NBA_WEIGHT).
  * Early in a season the ratings lean on last year's games, which is the main weakness.
"""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lines  # noqa: E402
from names import norm as nname  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GAMES_CSV = ROOT / "data" / "nba_games.csv"
OUT = ROOT / "docs" / "data" / "nba.json"
BETS = ROOT / "data" / "nba_bets_log.csv"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
SEASON_START = dt.date(2025, 10, 20)
SIGMA, TOTAL_SIGMA = 12.5, 19.0       # typical NBA game-to-game spread of margin / total around a good prediction
NBA_WEIGHT = 0.25                      # model share vs market (lower than college: results-only ratings)
HALF_LIFE = 75.0                       # days: a game this old counts half as much
ALPHA = 6.0                            # ridge strength (in "games"): pulls thin-sample teams to average
ALIASES = {"los angeles clippers": "LAC", "la clippers": "LAC"}


def get(params):
    r = requests.get(ESPN, params=params, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.json().get("events", [])


def parse(e):
    c = e["competitions"][0]
    t = {x["homeAway"]: x for x in c["competitors"]}
    return {"game_id": e["id"], "start": e["date"], "home": t["home"]["team"]["abbreviation"],
            "away": t["away"]["team"]["abbreviation"], "home_name": t["home"]["team"]["displayName"],
            "away_name": t["away"]["team"]["displayName"], "neutral": bool(c.get("neutralSite")),
            "final": e["status"]["type"]["name"] == "STATUS_FINAL",
            "hpts": int(t["home"].get("score") or 0), "apts": int(t["away"].get("score") or 0)}


def load_games(today):
    old = pd.read_csv(GAMES_CSV, dtype={"game_id": str}) if GAMES_CSV.exists() else pd.DataFrame()
    have = set(old["game_id"]) if len(old) else set()
    since = SEASON_START
    if len(old):
        since = max(SEASON_START, pd.to_datetime(old["start"]).max().date() - dt.timedelta(days=3))
    days = [since + dt.timedelta(days=i) for i in range((today - since).days + 1)]
    from concurrent.futures import ThreadPoolExecutor

    def one(d):                                             # ESPN rejects date ranges, so one request per day
        for _ in range(3):
            try:
                return get({"dates": f"{d:%Y%m%d}"})
            except Exception:
                pass
        return []
    rows = []
    with ThreadPoolExecutor(8) as ex:
        for evs in ex.map(one, days):
            for ev in evs:
                p = parse(ev)
                if p["final"] and p["game_id"] not in have:
                    rows.append(p); have.add(p["game_id"])
    new = pd.DataFrame(rows)
    allg = pd.concat([old, new], ignore_index=True) if len(old) else new
    allg = allg.drop_duplicates("game_id").sort_values("start")
    allg.to_csv(GAMES_CSV, index=False)
    return allg


def fit(g, today):
    teams = sorted(set(g["home"]) | set(g["away"]))
    ix = {t: i for i, t in enumerate(teams)}
    n, k = len(g), len(teams)
    age = (pd.Timestamp(today, tz="UTC") - pd.to_datetime(g["start"], utc=True)).dt.total_seconds().to_numpy() / 86400
    w = 0.5 ** (age / HALF_LIFE)
    X = np.zeros((n, k + 1))
    for r, (h, a) in enumerate(zip(g["home"], g["away"])):
        X[r, ix[h]] += 1; X[r, ix[a]] -= 1
    X[:, k] = 1                                            # home-court column (unpenalised below)
    marg = (g["hpts"] - g["apts"]).to_numpy(float)
    tot = (g["hpts"] + g["apts"]).to_numpy(float)
    # totals: team "pace/scoring" effect added for both teams
    T = np.zeros((n, k + 1))
    for r, (h, a) in enumerate(zip(g["home"], g["away"])):
        T[r, ix[h]] += 1; T[r, ix[a]] += 1
    T[:, k] = 1
    P = np.eye(k + 1) * ALPHA; P[k, k] = 0
    W = np.diag(w)
    beta = np.linalg.solve(X.T @ W @ X + P, X.T @ W @ marg)
    gam = np.linalg.solve(T.T @ W @ T + P, T.T @ W @ (tot - np.average(tot, weights=w)))
    mu_t = float(np.average(tot, weights=w))
    # constant column in totals sits at 0 after centring; add the mean back
    r = beta[:k] - beta[:k].mean()
    return teams, ix, r, float(beta[k]), gam[:k], mu_t + float(gam[k])


def main():
    today = dt.datetime.now(dt.timezone.utc).date()
    status = {"warnings": []}
    g = load_games(today)
    cnt = pd.concat([g["home"], g["away"]]).value_counts()
    g = g[g["home"].map(cnt).ge(30) & g["away"].map(cnt).ge(30)]     # drops exhibitions vs non-NBA clubs
    teams, ix, r, hca, tt, base = fit(g, today)
    names = {}
    upcoming = []
    for off in range(0, 3):                                 # today, tomorrow, day after
        d = today + dt.timedelta(days=off)
        for ev in get({"dates": f"{d:%Y%m%d}"}):
            p = parse(ev)
            names[nname(p["home_name"])] = p["home"]; names[nname(p["away_name"])] = p["away"]
            if not p["final"] and p["home"] in ix and p["away"] in ix:
                upcoming.append(p)
    seen, games = set(), []
    for p in upcoming:
        if p["game_id"] in seen:
            continue
        seen.add(p["game_id"])
        h, a = ix[p["home"]], ix[p["away"]]
        games.append({"game_id": p["game_id"], "start": p["start"], "home": p["home"], "away": p["away"],
                      "home_name": p["home_name"], "away_name": p["away_name"], "neutral": p["neutral"],
                      "pred_margin": round(float(r[h] - r[a] + (0 if p["neutral"] else hca)), 2),
                      "pred_total": round(float(base + tt[h] + tt[a]), 1), "done": False})
    for k, v in ALIASES.items():
        names[k] = v
    # re-use lines.py machinery with NBA settings
    lines.URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    lines.BETS_LOG = BETS
    lines.MODEL_WEIGHT = NBA_WEIGHT
    lines.build_name_index = lambda _idmap: {"full": names, "loc": {}}
    model = {"sigma": SIGMA, "total_sigma": TOTAL_SIGMA}
    now_s = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    roi = {"n": 0}
    try:
        events = lines.fetch_odds(status)
        if events:
            # evaluate() takes w as a default bound at import time, so pass it through attach_lines' call explicitly
            import functools
            lines.evaluate = functools.partial(lines.evaluate, w=NBA_WEIGHT)
            games = lines.attach_lines(games, events, {}, model, status)
            status["lines_at"] = now_s
        log = lines.log_bets(games, now_s)
        finals = {str(x.game_id): (x.home, x.away, x.hpts, x.apts) for x in g.itertuples()}
        log = lines.grade_bets(log, finals)
        log.to_csv(BETS, index=False)
        roi = lines.roi_summary(log)
    except Exception as ex:
        status["warnings"].append(f"NBA lines/EV step failed ({type(ex).__name__}: {ex}); ratings still published.")
    table = sorted(({"team": t, "rating": round(float(r[ix[t]]), 2)} for t in teams), key=lambda x: -x["rating"])
    OUT.write_text(json.dumps({"updated": now_s, "games": games, "ratings": table, "hca": round(hca, 2),
                               "n_games": int(len(g)), "bets": roi, "status": status,
                               "weight": NBA_WEIGHT, "sigma": SIGMA}))
    print(json.dumps({"games": len(games), "with_lines": sum(1 for x in games if x.get("market")), "status": status}))


if __name__ == "__main__":
    main()

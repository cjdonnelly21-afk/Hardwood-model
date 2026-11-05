"""NFL page: ratings -> projected margin, sportsbook lines -> EV, plus the walk-forward backtest report.

Run Thursday and Saturday (see .github/workflows/nfl.yml). Standalone: a failure here never touches the college / NBA pages.
Everything about how much to trust the model comes from the backtest (src/nfl_model.py), not from a guess:
the best blend weight on 2014-2025 closing lines was ~0.04 for spreads and negative for totals, so the page is
mostly line shopping (best price across books) with a very small model nudge.
"""
import datetime as dt
import functools
import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lines  # noqa: E402
import nfl_model as M  # noqa: E402
from names import norm as nname  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "data" / "nfl.json"
BETS = ROOT / "data" / "nfl_bets_log.csv"
HP = {"half_life": 150.0, "alpha": 3.0}
M.DEFAULTS.update(HP)
NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers", "CHI": "Chicago Bears", "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys", "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs",
    "LV": "Las Vegas Raiders", "LAC": "Los Angeles Chargers", "LA": "Los Angeles Rams", "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers", "SF": "San Francisco 49ers",
    "SEA": "Seattle Seahawks", "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}


def backtest(g: pd.DataFrame) -> dict:
    r = M.walk_forward(g, 2013, int(g[g["home_score"].notna()]["season"].max()))
    r = r[r["season"] >= 2014].copy()                       # 2013 is rating warm-up
    r["d"] = r["m_model"] - r["spread_line"]
    y = r["margin"] - r["spread_line"]
    sig = float((r["margin"] - r["spread_line"]).std())
    w = float((r["d"] * y).sum() / (r["d"] ** 2).sum())
    cover = np.sign(y)
    won = np.where(r["d"] > 0, cover > 0, cover < 0)
    tab = []
    for k in (0, 2, 3, 4, 6):
        m = (r["d"].abs() >= k) & (cover != 0)
        wr = float(won[m].mean())
        tab.append({"gap": k, "n": int(m.sum()), "ats": round(wr, 3), "roi": round(wr * 100 / 110 - (1 - wr), 3)})
    t = r.dropna(subset=["total_line"])
    td = t["t_model"] - t["total_line"]
    tw = float((td * (t["tot"] - t["total_line"])).sum() / (td ** 2).sum())
    tsig = float((t["tot"] - t["total_line"]).std())
    by = r.groupby("season").apply(lambda x: pd.Series({"n": len(x), "model": (x["margin"] - x["m_model"]).abs().mean(),
                                                        "market": (x["margin"] - x["spread_line"]).abs().mean()}), include_groups=False)
    # win-probability curve: the cover sigma overrates underdogs, so fit the moneyline sigma on actual straight-up results
    from scipy.optimize import minimize_scalar
    dec = r[r["margin"] != 0]
    mk, hw = dec["spread_line"].to_numpy(float), (dec["margin"] > 0).to_numpy()   # nflverse spread_line = market home margin
    nll = lambda sg: -(np.where(hw, norm.logcdf(mk / sg), norm.logcdf(-mk / sg))).sum()
    ml_sigma = float(minimize_scalar(nll, bounds=(6, 16), method="bounded").x)
    return {"ml_sigma": round(ml_sigma, 2), "n": int(len(r)), "seasons": f"{int(r['season'].min())}-{int(r['season'].max())}",
            "mae_model": round(float((r["margin"] - r["m_model"]).abs().mean()), 2),
            "mae_market": round(float((r["margin"] - r["spread_line"]).abs().mean()), 2),
            "spread_weight": round(w, 3), "total_weight": round(tw, 3), "sigma": round(sig, 2), "total_sigma": round(tsig, 2),
            "ats_by_gap": tab, "by_season": [{"season": int(s), "n": int(x.n), "model": round(x.model, 2), "market": round(x.market, 2)}
                                             for s, x in by.iterrows()]}


def main():
    now = dt.datetime.now(dt.timezone.utc)
    status = {"warnings": []}
    g = M.load_games(refresh=True)
    bt = backtest(g)
    weight = max(0.0, bt["spread_weight"])                   # data-driven, floored at zero
    sigma = bt["sigma"]
    played = g[g["home_score"].notna()]
    asof = pd.Timestamp(now.date())
    fit = M.Fit(played, asof, HP["half_life"], HP["alpha"])
    horizon = now.date() + dt.timedelta(days=9)
    up = g[g["home_score"].isna() & (g["date"].dt.date >= now.date() - dt.timedelta(days=1)) & (g["date"].dt.date <= horizon)]
    games = []
    et = ZoneInfo("America/New_York")
    for r in up.itertuples():
        hh, mm = (str(r.gametime) if isinstance(r.gametime, str) else "13:00").split(":")
        start = dt.datetime.combine(r.date.date(), dt.time(int(hh), int(mm)), tzinfo=et).astimezone(dt.timezone.utc)
        if start < now - dt.timedelta(hours=1):
            continue
        games.append({"game_id": r.game_id, "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"), "home": r.home_team, "away": r.away_team,
                      "home_name": NAMES.get(r.home_team, r.home_team), "away_name": NAMES.get(r.away_team, r.away_team),
                      "week": int(r.week), "neutral": bool(r.neutral), "done": False,
                      "pred_margin": round(float(fit.margin(r.home_team, r.away_team, bool(r.neutral))), 2),
                      "pred_total": round(float(fit.total(r.home_team, r.away_team)), 1)})
    # reuse lines.py with NFL settings
    lines.URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
    lines.BETS_LOG = BETS
    names = {nname(v): k for k, v in NAMES.items()}
    lines.build_name_index = lambda _idmap: {"full": names, "loc": {}}
    lines.evaluate = functools.partial(lines.evaluate, w=weight, ml_sigma=bt["ml_sigma"])
    model = {"sigma": sigma, "total_sigma": bt["total_sigma"]}
    now_s = now.strftime("%Y-%m-%dT%H:%MZ")
    roi = {"n": 0}
    try:
        events = lines.fetch_odds(status)
        if events:
            games = lines.attach_lines(games, events, {}, model, status)
            status["lines_at"] = now_s
        log = lines.log_bets(games, now_s)
        finals = {str(x.game_id): (x.home_team, x.away_team, x.home_score, x.away_score) for x in played.itertuples()}
        log = lines.grade_bets(log, finals)
        log.to_csv(BETS, index=False)
        roi = lines.roi_summary(log)
    except Exception as ex:
        status["warnings"].append(f"NFL lines/EV step failed ({type(ex).__name__}: {ex}); projections still published.")
    ratings = sorted(({"team": t, "rating": round(float(fit.beta[fit.ix[t]] - fit.beta[:fit.k].mean()), 2)} for t in fit.teams),
                     key=lambda x: -x["rating"])
    OUT.write_text(json.dumps({"updated": now_s, "games": games, "ratings": ratings, "hca": round(float(fit.beta[fit.k]), 2),
                               "weight": round(weight, 3), "sigma": sigma, "backtest": bt, "bets": roi, "status": status}))
    print(json.dumps({"games": len(games), "with_lines": sum(1 for x in games if x.get("market")), "weight": weight, "status": status}))


if __name__ == "__main__":
    main()

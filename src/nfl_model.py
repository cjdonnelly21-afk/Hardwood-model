"""NFL ratings from game results (time-decayed ridge) + walk-forward backtest against closing lines.

Data: nflverse games.csv (results, closing spread/total/moneylines, QBs, rest, roof, wind) - free, no key.
margin = home_pts - away_pts.  Rating model: margin ~ hca + r_home - r_away, fit on all earlier games with
exponential time decay (half-life in days) and a ridge penalty so thin samples shrink to average.
"""
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
CACHE = ROOT / "data" / "nfl_games.csv"
FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "WSH": "WAS", "JAC": "JAX"}
DEFAULTS = {"half_life": 400.0, "alpha": 8.0}


def load_games(refresh=True) -> pd.DataFrame:
    if refresh or not CACHE.exists():
        g = pd.read_csv(CSV_URL)
        g.to_csv(CACHE, index=False)
    else:
        g = pd.read_csv(CACHE)
    g = g[g["game_type"].isin(["REG", "WC", "DIV", "CON", "SB"])].copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(FIX)
    g["date"] = pd.to_datetime(g["gameday"])
    g = g.sort_values(["date", "game_id"]).reset_index(drop=True)
    g["margin"] = g["home_score"] - g["away_score"]
    g["tot"] = g["home_score"] + g["away_score"]
    g["neutral"] = g["location"].eq("Neutral")
    return g


class Fit:
    """Ratings fit on games strictly before `asof`."""

    def __init__(self, hist: pd.DataFrame, asof, half_life, alpha):
        self.teams = sorted(set(hist["home_team"]) | set(hist["away_team"]))
        self.ix = {t: i for i, t in enumerate(self.teams)}
        k, n = len(self.teams), len(hist)
        age = (pd.Timestamp(asof) - hist["date"]).dt.days.to_numpy(float)
        w = 0.5 ** (age / half_life)
        hi = hist["home_team"].map(self.ix).to_numpy()
        ai = hist["away_team"].map(self.ix).to_numpy()
        X = np.zeros((n, k + 1)); r = np.arange(n)
        X[r, hi] += 1; X[r, ai] -= 1
        X[:, k] = (~hist["neutral"].to_numpy()).astype(float)      # home field (zero on neutral sites)
        T = np.zeros((n, k + 1)); T[r, hi] += 1; T[r, ai] += 1; T[:, k] = 1
        P = np.eye(k + 1) * alpha; P[k, k] = 0
        Xw = X * w[:, None]; Tw = T * w[:, None]
        self.beta = np.linalg.solve(X.T @ Xw + P, Xw.T @ hist["margin"].to_numpy(float))
        tot = hist["tot"].to_numpy(float); mu = np.average(tot, weights=w)
        self.gam = np.linalg.solve(T.T @ Tw + P, Tw.T @ (tot - mu))
        self.mu = mu
        self.k = k

    def margin(self, h, a, neutral=False):
        if h not in self.ix or a not in self.ix:
            return np.nan
        return self.beta[self.ix[h]] - self.beta[self.ix[a]] + (0 if neutral else self.beta[self.k])

    def total(self, h, a):
        if h not in self.ix or a not in self.ix:
            return np.nan
        return self.mu + self.gam[self.ix[h]] + self.gam[self.ix[a]] + self.gam[self.k]


def walk_forward(g: pd.DataFrame, first_season=2013, last_season=2025, **hp) -> pd.DataFrame:
    """One refit per (season, week) using only earlier games. Returns the played games with model + market columns."""
    hp = {**DEFAULTS, **hp}
    out = []
    for (s, wk), blk in g[(g["season"] >= first_season) & (g["season"] <= last_season)].groupby(["season", "week"], sort=True):
        asof = blk["date"].min()
        f = Fit(g[g["date"] < asof], asof, hp["half_life"], hp["alpha"])
        b = blk.copy()
        b["m_model"] = [f.margin(h, a, n) for h, a, n in zip(b["home_team"], b["away_team"], b["neutral"])]
        b["t_model"] = [f.total(h, a) for h, a in zip(b["home_team"], b["away_team"])]
        out.append(b)
    r = pd.concat(out)
    return r.dropna(subset=["m_model", "margin", "spread_line"])

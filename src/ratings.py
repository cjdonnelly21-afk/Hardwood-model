"""Point-in-time team ratings, fit walk-forward so no game ever sees its own future.

Model (per game, two observations, y = points per 100 possessions minus league mean):
    off_A = o_A - d_B + h * loc_A          loc_A = +1 home, 0 neutral, -1 away
    off_B = o_B - d_A - h * loc_A
  o = offensive strength, d = defensive strength (positive = good defense), h = home edge.
Tempo: poss = mu_p + t_A + t_B.
Each fit is a ridge regression shrunk toward a PRIOR (last season's final rating * carry),
with exponential time decay so recent games matter more.

Hyper-parameters (carry, lam, half_life) are what the daily job re-tunes ("learning").
"""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import scipy.sparse as sp

MU_DEFAULT = 104.4
POSS_DEFAULT = 68.0


@dataclass
class Params:
    carry: float = 0.70       # share of last season's rating that persists into the preseason prior
    lam: float = 8.0          # ridge strength (in 'weighted games' units) pulling toward the prior
    half_life: float = 75.0   # days; weight of a game halves every half_life days
    tempo_carry: float = 0.80
    tempo_lam: float = 10.0
    cap_margin: float = 30.0  # winsorise blowouts so garbage time doesn't dominate

    def dict(self):
        return asdict(self)


def _design(games: pd.DataFrame, idx: dict):
    """Sparse design for offense obs (2 per game): columns [o(T) | d(T) | h]."""
    n, T = len(games), len(idx)
    hi = games["home"].map(idx).to_numpy()
    ai = games["away"].map(idx).to_numpy()
    loc = np.where(games["neutral"].to_numpy() == 1, 0.0, 1.0)
    r = np.arange(n)
    rows = np.concatenate([r, r, r, n + r, n + r, n + r])
    cols = np.concatenate([hi, T + ai, np.full(n, 2 * T), ai, T + hi, np.full(n, 2 * T)])
    vals = np.concatenate([np.ones(n), -np.ones(n), loc, np.ones(n), -np.ones(n), -loc])
    # d enters with a minus sign: good defense (positive d) lowers the opponent's output
    return sp.csr_matrix((vals, (rows, cols)), shape=(2 * n, 2 * T + 1))


def _tempo_design(games: pd.DataFrame, idx: dict):
    n, T = len(games), len(idx)
    r = np.arange(n)
    rows = np.concatenate([r, r])
    cols = np.concatenate([games["home"].map(idx).to_numpy(), games["away"].map(idx).to_numpy()])
    return sp.csr_matrix((np.ones(2 * n), (rows, cols)), shape=(n, T))


def _ridge(X, y, w, lam_vec, prior):
    Xw = X.T.multiply(w).tocsr()
    A = (Xw @ X).toarray()
    A[np.diag_indices_from(A)] += lam_vec
    b = Xw @ y + lam_vec * prior
    return np.linalg.solve(A, b)


class SeasonFitter:
    """Walk through one season; yields pre-game features and ends with final ratings."""

    def __init__(self, games: pd.DataFrame, prior_o: dict, prior_d: dict, prior_t: dict, p: Params,
                 mu_prior=MU_DEFAULT, poss_prior=POSS_DEFAULT, prior_h=None):
        self.g = games.sort_values("date").reset_index(drop=True)
        # include last season's teams too, so teams that haven't played yet still carry a prior rating
        teams = sorted(set(self.g["home"]) | set(self.g["away"]) | set(prior_o) | set(prior_d))
        self.idx = {t: i for i, t in enumerate(teams)}
        self.teams = teams
        self.p = p
        T = len(teams)
        carry, tc = p.carry, p.tempo_carry
        self.p0 = np.concatenate([[carry * prior_o.get(t, 0.0) for t in teams],
                                  [carry * prior_d.get(t, 0.0) for t in teams], [3.0 if prior_h is None else prior_h]])
        self.pt = np.array([tc * prior_t.get(t, 0.0) for t in teams])
        self.X = _design(self.g, self.idx)
        self.XT = _tempo_design(self.g, self.idx)
        n = len(self.g)
        poss = self.g["poss"].to_numpy()
        pts_h = self.g["hpts"].to_numpy(); pts_a = self.g["apts"].to_numpy()
        # winsorise the margin around its mean so one 60-point blowout doesn't move a rating
        m = pts_h - pts_a
        shrink = np.clip(np.abs(m), 0, p.cap_margin) / np.maximum(np.abs(m), 1e-9)
        adj_h = (pts_h + pts_a) / 2 + (m * shrink) / 2
        adj_a = (pts_h + pts_a) / 2 - (m * shrink) / 2
        self.yraw = np.concatenate([adj_h / poss * 100, adj_a / poss * 100])
        self.poss = poss
        self.dates = self.g["date"].to_numpy()
        self.day = (self.g["date"] - self.g["date"].min()).dt.days.to_numpy().astype(float)
        self.mu0, self.poss0 = mu_prior, poss_prior
        self.T = T
        # home edge: weak prior in the first season, firmer prior (last season's value) afterwards
        self.lam_vec = np.concatenate([np.full(2 * T, p.lam), [5.0 if prior_h is None else 40.0]])
        self.lam_t = np.full(T, p.tempo_lam)

    def fit_before(self, cutoff_idx: int, now_day: float):
        """Fit using games [0, cutoff_idx). Returns (o, d, h, t, mu, mu_p)."""
        n = len(self.g)
        if cutoff_idx == 0:
            T = self.T
            return self.p0[:T], self.p0[T:2 * T], self.p0[-1], self.pt, self.mu0, self.poss0
        gw = 0.5 ** ((now_day - self.day[:cutoff_idx]) / self.p.half_life)
        w = np.concatenate([gw, gw])
        sel = np.concatenate([np.arange(cutoff_idx), n + np.arange(cutoff_idx)])
        X = self.X[sel]
        mu = np.average(self.yraw[sel], weights=w)
        y = self.yraw[sel] - mu
        # a prior weight of ~10 games' worth of league-average data keeps mu stable early on
        theta = _ridge(X, y, w, self.lam_vec, self.p0)
        T = self.T
        Xt = self.XT[:cutoff_idx]
        mp = np.average(self.poss[:cutoff_idx], weights=gw)
        yt = self.poss[:cutoff_idx] - mp
        t = _ridge(Xt, yt, gw, self.lam_t, self.pt)
        return theta[:T], theta[T:2 * T], theta[-1], t, mu, mp

    def walk(self):
        """Return DataFrame of pre-game features for every game + dict of final ratings."""
        g = self.g
        n = len(g)
        out = {k: np.zeros(n) for k in ["o_h", "d_h", "o_a", "d_a", "t_h", "t_a", "hca", "mu", "mu_p"]}
        gp_h = np.zeros(n, dtype=int); gp_a = np.zeros(n, dtype=int)
        played = np.zeros(self.T, dtype=int)
        hi = g["home"].map(self.idx).to_numpy(); ai = g["away"].map(self.idx).to_numpy()
        starts = np.flatnonzero(np.r_[True, self.dates[1:] != self.dates[:-1]])
        ends = np.r_[starts[1:], n]
        for s, e in zip(starts, ends):
            o, d, h, t, mu, mp = self.fit_before(s, self.day[s])
            sl = slice(s, e)
            out["o_h"][sl] = o[hi[sl]]; out["d_h"][sl] = d[hi[sl]]
            out["o_a"][sl] = o[ai[sl]]; out["d_a"][sl] = d[ai[sl]]
            out["t_h"][sl] = t[hi[sl]]; out["t_a"][sl] = t[ai[sl]]
            out["hca"][sl] = h; out["mu"][sl] = mu; out["mu_p"][sl] = mp
            gp_h[sl] = played[hi[sl]]; gp_a[sl] = played[ai[sl]]
            np.add.at(played, hi[sl], 1); np.add.at(played, ai[sl], 1)
        res = g.copy()
        for k, v in out.items():
            res[k] = v
        res["gp_h"], res["gp_a"] = gp_h, gp_a
        o, d, h, t, mu, mp = self.fit_before(n, self.day[-1] + 1)
        final = {"o": dict(zip(self.teams, o)), "d": dict(zip(self.teams, d)),
                 "t": dict(zip(self.teams, t)), "h": h, "mu": mu, "mu_p": mp, "games": dict(zip(self.teams, played))}
        return add_predictions(res), final


def add_predictions(df: pd.DataFrame) -> pd.DataFrame:
    loc = np.where(df["neutral"] == 1, 0.0, 1.0)
    per100 = (df["o_h"] - df["d_a"]) - (df["o_a"] - df["d_h"]) + 2 * df["hca"] * loc
    poss = df["mu_p"] + df["t_h"] + df["t_a"]
    df["poss_hat"] = poss
    df["margin_hat"] = per100 * poss / 100
    df["total_hat"] = (2 * df["mu"] + (df["o_h"] - df["d_a"]) + (df["o_a"] - df["d_h"])) * poss / 100
    return df


def run_all(games: pd.DataFrame, p: Params, seasons=None, verbose=False):
    """Walk every season in order. Returns (pregame_df, {season: final_ratings})."""
    seasons = seasons or sorted(games["season"].unique())
    prior_o, prior_d, prior_t = {}, {}, {}
    mu, mp, hprior = MU_DEFAULT, POSS_DEFAULT, None
    frames, finals = [], {}
    for s in seasons:
        gs = games[games["season"] == s]
        fitter = SeasonFitter(gs, prior_o, prior_d, prior_t, p, mu, mp, hprior)
        res, fin = fitter.walk()
        frames.append(res); finals[s] = fin
        prior_o, prior_d, prior_t = fin["o"], fin["d"], fin["t"]
        mu, mp, hprior = fin["mu"], fin["mu_p"], fin["h"]
        if verbose:
            r = res[res["gp_h"].add(res["gp_a"]) >= 10]
            print(s, "MAE", round((r["margin_hat"] - (r["hpts"] - r["apts"])).abs().mean(), 3), flush=True)
    return pd.concat(frames, ignore_index=True), finals


def score(df: pd.DataFrame, min_gp=0, first_season=None) -> dict:
    r = df[(df["gp_h"] >= min_gp) & (df["gp_a"] >= min_gp)]
    if first_season:
        r = r[r["season"] >= first_season]
    err = r["margin_hat"] - (r["hpts"] - r["apts"])
    return {"n": len(r), "mae": float(err.abs().mean()), "rmse": float(np.sqrt((err ** 2).mean())),
            "bias": float(err.mean()), "tot_mae": float((r["total_hat"] - (r["hpts"] + r["apts"])).abs().mean())}

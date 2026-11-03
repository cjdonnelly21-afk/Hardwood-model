"""Experiments: which extra signals / model types lower out-of-sample error?  (research script, not part of the daily job)"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_games import load_games  # noqa: E402
from ratings import Params, run_all  # noqa: E402
from train import build_frame, load_params  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache_frame.pkl"


def rest_days(g):
    long = pd.concat([pd.DataFrame({"i": g.index, "team": g["home"], "date": g["date"], "side": "h"}),
                      pd.DataFrame({"i": g.index, "team": g["away"], "date": g["date"], "side": "a"})])
    long = long.sort_values(["team", "date"])
    long["rest"] = long.groupby("team")["date"].diff().dt.days.clip(upper=10).fillna(7)
    long["last7"] = (long["date"] - long.groupby("team")["date"].shift(4)).dt.days.clip(upper=30).fillna(14)
    out = {}
    for s in ("h", "a"):
        x = long[long.side == s].set_index("i")
        out[f"rest_{s}"] = x["rest"].reindex(g.index).values
        out[f"span5_{s}"] = x["last7"].reindex(g.index).values
    return pd.DataFrame(out, index=g.index)


def get_frame():
    if CACHE.exists():
        return pickle.load(open(CACHE, "rb"))
    games = load_games()
    params = load_params()
    frame, _ = build_frame(games, params)
    short = Params(**{**params.dict(), "half_life": 30.0, "lam": 6.0, "carry": 0.8})
    pre2, _ = run_all(games, short)
    key = lambda d: d["date"].astype(str) + "|" + d["home"] + "|" + d["away"] + "|" + d.groupby(["date", "home", "away"]).cumcount().astype(str)
    sh = pd.DataFrame({"k": key(pre2).values, "own_short": pre2["margin_hat"].values, "total_short": pre2["total_hat"].values}).set_index("k")
    assert sh.index.is_unique
    frame = frame.reset_index(drop=True)
    kf = key(frame)
    frame["own_short"] = sh["own_short"].reindex(kf).values
    frame["total_short"] = sh["total_short"].reindex(kf).values
    assert frame["own_short"].notna().all()
    frame = pd.concat([frame, rest_days(frame)], axis=1)
    pickle.dump(frame, open(CACHE, "wb"))
    return frame


BASE = ["own", "tor", "own_early", "tor_early", "non_neutral", "conf_tourney", "postseason"]


def add(df):
    d = df.copy()
    gp = np.minimum(d.gp_h, d.gp_a)
    d["gpmin"] = gp
    d["short_early"] = d.own_short * (1 / (1 + gp / 5))
    d["rest_diff"] = (d.rest_h - d.rest_a).clip(-5, 5)
    d["rest_min"] = np.minimum(d.rest_h, d.rest_a)
    d["conf_game"] = (d.gtype == 1).astype(float)
    d["tempo"] = (d.tor_t_h + d.tor_t_a).fillna(d.poss_hat * 2) / 2
    d["bar_diff"] = (d.tor_bar_h - d.tor_bar_a).fillna(0)
    d["o_gap"] = d.o_h - d.d_a
    d["o_gap2"] = d.o_a - d.d_h
    d["gap_h_net"] = (d.o_h + d.d_h) - (d.o_a + d.d_a)
    d["tor_minus_own"] = d.tor - d.own
    return d


MODELS = {
    "ridge_base": (BASE, lambda: Ridge(50.0)),
    "ridge+short+rest": (BASE + ["own_short", "short_early", "rest_diff", "rest_min", "conf_game"], lambda: Ridge(50.0)),
    "huber+short+rest": (BASE + ["own_short", "short_early", "rest_diff", "rest_min", "conf_game"], lambda: HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=300)),
    "gbm_abs": (BASE + ["own_short", "rest_diff", "rest_min", "conf_game", "gpmin", "tempo", "bar_diff", "o_gap", "o_gap2", "tor_minus_own"],
                lambda: HistGradientBoostingRegressor(loss="absolute_error", max_iter=250, learning_rate=0.05, max_leaf_nodes=12,
                                                      min_samples_leaf=150, l2_regularization=5.0, random_state=0)),
}


def main():
    f = add(get_frame())
    f = f[f.has_tor]
    res = {k: [] for k in MODELS}
    for s in range(2018, 2027):
        tr, te = f[f.season < s], f[(f.season == s) & (f.gp_h >= 3) & (f.gp_a >= 3)]
        y, yt = tr.hpts - tr.apts, te.hpts - te.apts
        for name, (cols, mk) in MODELS.items():
            m = mk().fit(tr[cols].values, y.values)
            p = m.predict(te[cols].values)
            res[name].append(((p - yt).abs().mean(), (((p > 0) == (yt > 0)).mean())))
        print(s, {k: round(v[-1][0], 3) for k, v in res.items()}, flush=True)
    print("\nMEAN MAE / win acc, 2018-2026 and 2023-2026 only")
    for k, v in res.items():
        a = np.array(v)
        print(f"{k:20s} {a[:,0].mean():.3f}  {a[:,1].mean():.4f}   | last4 {a[-4:,0].mean():.3f}")


if __name__ == "__main__":
    main()

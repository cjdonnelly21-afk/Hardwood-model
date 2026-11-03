"""Stage 2: blend my own point-in-time ratings with Torvik's, learn the weights, backtest walk-forward.

    margin = b0 + b1*own + b2*torvik + b3*own*early + b4*torvik*early + b5*non_neutral + b6*conf_tourney + b7*postseason
    total  = linear in [own_total, torvik_total, poss_hat]
    win%   = Phi(margin / sigma)      sigma is estimated from out-of-sample residuals

"early" decays with games played, so the model can lean on Torvik's preseason ratings in November
and on the in-season evidence later - the weights are learned from 10+ seasons, not hand-set.
Linear on purpose: the website evaluates the identical formula in the browser.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_games import load_games  # noqa: E402
from ratings import Params, run_all  # noqa: E402
from torvik_features import attach, load_snapshots, torvik_margin  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MARGIN_FEATS = ["own", "tor", "own_early", "tor_early", "non_neutral", "conf_tourney", "postseason"]
TOTAL_FEATS = ["own_total", "tor_total", "poss_hat"]


def load_params() -> Params:
    f = ROOT / "data" / "params.json"
    return Params(**json.loads(f.read_text())["params"]) if f.exists() else Params()


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """df needs own-rating columns (margin_hat, total_hat, poss_hat, gp_*) and tor_* columns."""
    out = df.copy()
    out["tor_m"] = torvik_margin(out)
    out["has_tor"] = out["tor_m"].notna()
    out["own"] = out["margin_hat"]
    out["tor"] = out["tor_m"].fillna(out["margin_hat"])
    gp = np.minimum(out["gp_h"], out["gp_a"])
    early = 1.0 / (1.0 + gp / 5.0)
    out["own_early"] = out["own"] * early
    out["tor_early"] = out["tor"] * early
    out["non_neutral"] = (out["neutral"] == 0).astype(float)
    out["conf_tourney"] = (out["gtype"] == 2).astype(float)
    out["postseason"] = (out["gtype"] == 3).astype(float)
    out["own_total"] = out["total_hat"]
    tempo = (out["tor_t_h"] + out["tor_t_a"]) / 2
    tsum = (out["tor_oe_h"] - out["tor_de_a"] + out["tor_oe_a"] - out["tor_de_h"] + 2 * 104.0) * tempo / 100
    out["tor_total"] = (tsum - 0).fillna(out["total_hat"])
    return out


def fit(train: pd.DataFrame, alpha: float = 50.0) -> dict:
    t = train[train["has_tor"]]
    m = Ridge(alpha=alpha).fit(t[MARGIN_FEATS], t["hpts"] - t["apts"])
    resid = (t["hpts"] - t["apts"]) - m.predict(t[MARGIN_FEATS])
    tm = Ridge(alpha=alpha).fit(t[TOTAL_FEATS], t["hpts"] + t["apts"])
    tres = (t["hpts"] + t["apts"]) - tm.predict(t[TOTAL_FEATS])
    return {"margin": {"intercept": float(m.intercept_), **dict(zip(MARGIN_FEATS, map(float, m.coef_)))},
            "total": {"intercept": float(tm.intercept_), **dict(zip(TOTAL_FEATS, map(float, tm.coef_)))},
            "sigma": float(resid.std()), "total_sigma": float(tres.std()), "n_train": int(len(t))}


def predict(model: dict, feats: pd.DataFrame) -> pd.DataFrame:
    mg, tt = model["margin"], model["total"]
    p = pd.DataFrame(index=feats.index)
    p["pred_margin"] = mg["intercept"] + sum(mg[k] * feats[k] for k in MARGIN_FEATS)
    p["pred_total"] = tt["intercept"] + sum(tt[k] * feats[k] for k in TOTAL_FEATS)
    p["win_prob_home"] = norm.cdf(p["pred_margin"] / model["sigma"])
    return p


def build_frame(games: pd.DataFrame, params: Params):
    pre, finals = run_all(games, params)
    snap = load_snapshots()
    return make_features(attach(pre, snap)), finals


def logloss(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def walk_forward(frame: pd.DataFrame, first_test=2018, min_gp=3) -> pd.DataFrame:
    rows, pooled = [], []
    for s in sorted(frame["season"].unique()):
        if s < first_test:
            continue
        model = fit(frame[frame["season"] < s])
        te = frame[(frame["season"] == s) & frame["has_tor"] & (frame["gp_h"] >= min_gp) & (frame["gp_a"] >= min_gp)]
        if te.empty:
            continue
        pr = predict(model, te)
        act = te["hpts"] - te["apts"]
        y = (act > 0).astype(float)
        pooled.append(pd.DataFrame({"p": pr["win_prob_home"].to_numpy(), "y": y.to_numpy()}))
        rows.append({
            "season": s, "n": len(te),
            "mae_blend": (pr["pred_margin"] - act).abs().mean(),
            "mae_own": (te["own"] - act).abs().mean(),
            "mae_torvik": (te["tor_m"] - act).abs().mean(),
            "tot_mae": (pr["pred_total"] - (te["hpts"] + te["apts"])).abs().mean(),
            "win_acc": ((pr["win_prob_home"] > 0.5) == (y > 0.5)).mean(),
            "logloss": logloss(pr["win_prob_home"].to_numpy(), y.to_numpy()),
            "brier": float(((pr["win_prob_home"] - y) ** 2).mean()),
        })
    out = pd.DataFrame(rows)
    out.attrs["pooled"] = pd.concat(pooled, ignore_index=True)
    return out


def calibration_bins(pooled: pd.DataFrame) -> list:
    b = pd.cut(pooled["p"], [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    g = pooled.groupby(b, observed=True).agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size")).reset_index(drop=True)
    return g.round(4).to_dict(orient="records")


def main():
    games = load_games()
    params = load_params()
    print("params", params.dict())
    frame, finals = build_frame(games, params)
    print("games with Torvik snapshot:", int(frame["has_tor"].sum()), "of", len(frame))
    bt = walk_forward(frame)
    pd.set_option("display.width", 200)
    print(bt.round(3).to_string(index=False))
    print("MEAN", bt.drop(columns="season").mean().round(3).to_dict())
    model = fit(frame)
    model["backtest"] = bt.round(4).to_dict(orient="records")
    (ROOT / "data" / "model.json").write_text(json.dumps(model, indent=2))
    print(json.dumps({k: model[k] for k in ("margin", "total", "sigma", "total_sigma", "n_train")}, indent=1))


if __name__ == "__main__":
    main()

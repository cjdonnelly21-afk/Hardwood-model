"""Daily job: refresh data -> grade yesterday -> relearn -> predict upcoming games -> export site data.

    python src/update.py            # normal daily run
    python src/update.py --tune     # also re-tune hyper-parameters (the job does this weekly on Sundays)

Every step degrades gracefully: if Torvik or ESPN is unreachable the site keeps its last data and
status.json says so, instead of publishing garbage.
"""
import argparse
import datetime as dt
import gzip
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_games  # noqa: E402
from names import load_map  # noqa: E402
from build_games import load_games  # noqa: E402
from ratings import Params, run_all  # noqa: E402
from train import fit, load_params, make_features, predict, walk_forward, build_frame, calibration_bins  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
SITE = ROOT / "docs" / "data"
UA = "Mozilla/5.0 (cbb-model personal project)"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
HORIZON_DAYS = 3


def current_season(today: dt.date) -> int:
    return today.year + 1 if today.month >= 10 else today.year


FETCH_LOG = {}


def fetch(url: str, dest: Path, min_bytes=500) -> bool:
    import os
    dest.parent.mkdir(parents=True, exist_ok=True)
    if os.environ.get("CBB_FORCE_FETCH_FAIL"):          # test hook: simulate Torvik blocking us
        FETCH_LOG[url.split("barttorvik.com/")[-1]] = "forced failure (test)"
        return False
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    r = subprocess.run(["curl", "-sS", "-m", "90", "-A", UA, "-L", "-o", str(tmp), "-w", "%{http_code}", url],
                       capture_output=True, text=True)
    time.sleep(1)
    code = r.stdout.strip()
    head = tmp.read_bytes()[:160] if tmp.exists() else b""
    FETCH_LOG[url.split("barttorvik.com/")[-1]] = f"http {code}, {tmp.stat().st_size if tmp.exists() else 0} bytes, {r.stderr.strip()[:80]} {head[:120]!r}"
    ok = code == "200" and tmp.exists() and tmp.stat().st_size >= min_bytes
    if ok and head.lstrip().lower().startswith((b"<!doctype", b"<html")):
        ok = False  # Torvik's "Verifying Browser" gate, not data
    if ok:
        tmp.replace(dest)
    elif tmp.exists():
        tmp.unlink()
    return ok


LATEST = ROOT / "data" / "torvik_latest.csv"
LATEST_DATE = ROOT / "data" / "torvik_latest.date"


def refresh_torvik(season: int, today: dt.date, status: dict):
    dest = RAW / "team_results" / f"{season}.csv"
    ok1 = fetch(f"https://barttorvik.com/{season}_team_results.csv", dest)
    ok2 = fetch(f"https://barttorvik.com/getgamestats.php?year={season}&csv=1", RAW / "gamestats" / f"{season}.csv", 200)
    status["torvik_ratings_ok"], status["torvik_games_ok"] = ok1, ok2
    status["fetch_log"] = dict(FETCH_LOG)
    if ok1:   # remember the last good download, and keep our own point-in-time snapshot (same layout as Torvik's time machine)
        LATEST.write_bytes(dest.read_bytes())
        LATEST_DATE.write_text(today.isoformat())
        df = pd.read_csv(dest)
        rows = df.astype(object).where(df.notna(), None).values.tolist()
        snap = RAW / "timemachine" / f"{today:%Y%m%d}.json.gz"
        snap.parent.mkdir(parents=True, exist_ok=True)
        if not snap.exists():
            with gzip.open(snap, "wt") as f:
                json.dump(rows, f)
    elif LATEST.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(LATEST.read_bytes())
        status["warnings"].append(f"Torvik blocked today's download; showing Torvik ratings from {LATEST_DATE.read_text().strip()}.")
    else:
        raise SystemExit("No Torvik ratings available (download failed and no saved copy).")
    if not ok2:
        status["warnings"].append("Torvik game results unavailable today, so ratings were not updated with the newest games.")


def espn_day(d: dt.date):
    r = requests.get(ESPN, params={"dates": f"{d:%Y%m%d}", "groups": 50, "limit": 400}, timeout=30, headers={"User-Agent": UA})
    r.raise_for_status()
    return r.json().get("events", [])


def parse_events(events, idmap):
    out, unknown = [], set()
    for e in events:
        c = e["competitions"][0]
        h = next(x for x in c["competitors"] if x["homeAway"] == "home")
        a = next(x for x in c["competitors"] if x["homeAway"] == "away")
        hn, an = idmap.get(str(h["team"]["id"])), idmap.get(str(a["team"]["id"]))
        if hn is None or an is None:
            unknown.add(h["team"]["displayName"] if hn is None else a["team"]["displayName"])
            continue
        done = e["status"]["type"]["completed"]
        out.append({"game_id": e["id"], "start": e["date"], "home": hn, "away": an,
                    "neutral": int(bool(c.get("neutralSite"))), "state": e["status"]["type"]["name"], "done": done,
                    "hpts": float(h["score"]) if done else np.nan, "apts": float(a["score"]) if done else np.nan,
                    "conf_game": int(bool(c.get("conferenceCompetition"))),
                    "note": (c.get("notes") or [{}])[0].get("headline", "")})
    return out, unknown


def current_state(games: pd.DataFrame, params: Params, season: int):
    """Final-as-of-now ratings for the live season (priors only if no games have been played yet)."""
    from ratings import SeasonFitter, MU_DEFAULT, POSS_DEFAULT
    prev = games[games["season"] < season]
    _, finals = run_all(prev, params)
    last = finals[max(finals)]
    cur = games[games["season"] == season]
    if len(cur):
        f = SeasonFitter(cur, last["o"], last["d"], last["t"], params, last["mu"], last["mu_p"], last["h"])
        _, fin = f.walk()
        return fin
    c, tc = params.carry, params.tempo_carry
    return {"o": {k: c * v for k, v in last["o"].items()}, "d": {k: c * v for k, v in last["d"].items()},
            "t": {k: tc * v for k, v in last["t"].items()}, "h": last["h"], "mu": last["mu"], "mu_p": last["mu_p"],
            "games": {}}


def build_upcoming(sched: pd.DataFrame, fin: dict, tor: pd.DataFrame, model: dict) -> pd.DataFrame:
    t = tor.set_index("team")
    df = sched[sched["home"].isin(t.index) & sched["away"].isin(t.index)].copy()
    df["o_h"] = df["home"].map(fin["o"]).fillna(0.0); df["d_h"] = df["home"].map(fin["d"]).fillna(0.0)
    df["o_a"] = df["away"].map(fin["o"]).fillna(0.0); df["d_a"] = df["away"].map(fin["d"]).fillna(0.0)
    df["t_h"] = df["home"].map(fin["t"]).fillna(0.0); df["t_a"] = df["away"].map(fin["t"]).fillna(0.0)
    df["hca"], df["mu"], df["mu_p"] = fin["h"], fin["mu"], fin["mu_p"]
    df["gp_h"] = df["home"].map(fin["games"]).fillna(0).astype(int); df["gp_a"] = df["away"].map(fin["games"]).fillna(0).astype(int)
    loc = np.where(df["neutral"] == 1, 0.0, 1.0)
    df["poss_hat"] = df["mu_p"] + df["t_h"] + df["t_a"]
    df["margin_hat"] = ((df.o_h - df.d_a) - (df.o_a - df.d_h) + 2 * df.hca * loc) * df.poss_hat / 100
    df["total_hat"] = (2 * df.mu + (df.o_h - df.d_a) + (df.o_a - df.d_h)) * df.poss_hat / 100
    for side, p in (("home", "h"), ("away", "a")):
        df[f"tor_oe_{p}"] = df[side].map(t["adjoe"]); df[f"tor_de_{p}"] = df[side].map(t["adjde"])
        df[f"tor_t_{p}"] = df[side].map(t["adjt"])
    df["gtype"] = np.where(df["conf_game"] == 1, 1, 0)
    df.loc[df["note"].str.contains("Conference Tournament|Championship", case=False, na=False) & (df["neutral"] == 1), "gtype"] = 2
    df.loc[df["note"].str.contains("NCAA|NIT|First Four|Sweet|Elite|Final Four", case=False, na=False), "gtype"] = 3
    feats = make_features(df)
    return pd.concat([feats, predict(model, feats)], axis=1)


def sigma_to_prob(x):
    from scipy.stats import norm
    return norm.cdf(x)


def update_log(pred: pd.DataFrame, finished: list, today: dt.date) -> pd.DataFrame:
    path = ROOT / "data" / "predictions_log.csv"
    log = pd.read_csv(path) if path.exists() else pd.DataFrame(
        columns=["game_id", "date", "home", "away", "neutral", "pred_margin", "pred_total", "win_prob_home", "made_at", "hpts", "apts"])
    log["game_id"] = log["game_id"].astype(str)
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M")
    for _, r in pred[~pred["done"] & (pred["state"] == "STATUS_SCHEDULED")].iterrows():
        row = {"game_id": str(r.game_id), "date": r.start[:10], "home": r.home, "away": r.away, "neutral": r.neutral,
               "pred_margin": round(r.pred_margin, 2), "pred_total": round(r.pred_total, 1),
               "win_prob_home": round(r.win_prob_home, 4), "made_at": now, "hpts": np.nan, "apts": np.nan}
        m = log["game_id"] == row["game_id"]
        if m.any():  # refresh only while the game hasn't started: last pre-tip number is what gets graded
            log.loc[m, list(row)] = list(row.values())
        else:
            log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    for g in finished:
        m = log["game_id"] == str(g["game_id"])
        if m.any():
            log.loc[m, ["hpts", "apts"]] = [g["hpts"], g["apts"]]
    log.to_csv(path, index=False)
    return log


def live_accuracy(log: pd.DataFrame) -> dict:
    d = log.dropna(subset=["hpts"]).copy()
    if d.empty:
        return {"n": 0}
    d["act"] = d.hpts - d.apts
    d["win"] = (d.act > 0).astype(float)
    err = d.pred_margin - d.act
    bins = pd.cut(d.win_prob_home, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    cal = d.groupby(bins, observed=True).agg(pred=("win_prob_home", "mean"), actual=("win", "mean"), n=("win", "size")).reset_index(drop=True)
    return {"n": int(len(d)), "mae": round(float(err.abs().mean()), 3), "bias": round(float(err.mean()), 3),
            "win_acc": round(float(((d.win_prob_home > .5) == (d.win > .5)).mean()), 4),
            "brier": round(float(((d.win_prob_home - d.win) ** 2).mean()), 4),
            "calibration": cal.round(4).to_dict(orient="records"),
            "daily": d.assign(day=d.date).groupby("day").apply(
                lambda x: pd.Series({"n": len(x), "mae": (x.pred_margin - x.act).abs().mean(),
                                     "acc": ((x.win_prob_home > .5) == (x.win > .5)).mean()}), include_groups=False
            ).round(3).reset_index().tail(120).to_dict(orient="records")}


def blended_power(model, fin, t, name):
    """Blended rating: expected margin vs an average D1 team on a neutral court (same formula as the predictions)."""
    o, d, tm = fin["o"][name], fin["d"][name], fin["t"][name]
    poss = fin["mu_p"] + tm
    own = (o + d) * poss / 100
    avg_oe, avg_de, avg_t = t["adjoe"].mean(), t["adjde"].mean(), t["adjt"].mean()
    tempo = (t.loc[name, "adjt"] + avg_t) / 2
    tor = ((t.loc[name, "adjoe"] - t.loc[name, "adjde"]) - (avg_oe - avg_de)) * tempo / 100
    gp = fin["games"].get(name, 0)
    early = 1.0 / (1.0 + gp / 5.0)
    m = model["margin"]
    return m["intercept"] + m["own"] * own + m["tor"] * tor + m["own_early"] * own * early + m["tor_early"] * tor * early


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args()
    today = dt.datetime.now(dt.timezone(dt.timedelta(hours=-5))).date()  # US Eastern-ish calendar day
    season = current_season(today)
    status = {"run_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "season": season, "warnings": []}
    SITE.mkdir(parents=True, exist_ok=True)

    if not args.no_fetch:
        refresh_torvik(season, today, status)
    build_games.main()
    games = load_games()

    if args.tune or today.weekday() == 6 or not (ROOT / "data" / "params.json").exists():
        subprocess.run([sys.executable, "-I", str(ROOT / "src" / "tune.py")], check=False)
    params = load_params()

    # relearn: rebuild pre-game features for all history (incl. the live season so far) and refit the blend
    frame, _ = build_frame(games, params)
    model = fit(frame)
    bt = walk_forward(frame)
    model["backtest"] = bt.round(4).to_dict(orient="records")
    model["calibration"] = calibration_bins(bt.attrs["pooled"])
    (ROOT / "data" / "model.json").write_text(json.dumps(model, indent=2))

    fin = current_state(games, params, season)
    tor = pd.read_csv(RAW / "team_results" / f"{season}.csv")
    tor = tor.rename(columns={"adjt": "adjt"})

    idmap = load_map()
    events, unknown, finished = [], set(), []
    for k in range(-2, HORIZON_DAYS + 1):
        try:
            ev = espn_day(today + dt.timedelta(days=k))
        except Exception as ex:
            status["warnings"].append(f"ESPN {today + dt.timedelta(days=k)}: {ex}")
            continue
        parsed, unk = parse_events(ev, idmap)
        unknown |= unk
        if k < 0:
            finished += [p for p in parsed if p["done"]]
        else:
            events += parsed
    if unknown:
        status["skipped_non_d1"] = sorted(unknown)  # exhibition opponents etc.; informational, not a warning
    sched = pd.DataFrame(events)
    pred = build_upcoming(sched, fin, tor, model) if len(sched) else pd.DataFrame()
    log = update_log(pred if len(pred) else pd.DataFrame(columns=["done", "state"]), finished, today)

    # ---- exports for the site
    t = tor.set_index("team")
    teams = []
    for name in fin["o"]:
        if name not in t.index:
            continue
        o, d = fin["o"][name], fin["d"][name]
        power = blended_power(model, fin, t, name)
        teams.append({"team": name, "power": round(power, 2), "conf": t.loc[name, "conf"], "record": t.loc[name, "record"],
                      "o": round(o, 2), "d": round(d, 2), "net": round(o + d, 2), "t": round(fin["t"][name], 2),
                      "gp": int(fin["games"].get(name, 0)),
                      "tor_oe": round(float(t.loc[name, "adjoe"]), 2), "tor_de": round(float(t.loc[name, "adjde"]), 2),
                      "tor_t": round(float(t.loc[name, "adjt"]), 2), "tor_rank": int(t.loc[name, "rank"]),
                      "tor_barthag": round(float(t.loc[name, "barthag"]), 4)})
    teams.sort(key=lambda r: -r["power"])
    for i, r in enumerate(teams, 1):
        r["rank"] = i
    cols = ["game_id", "start", "home", "away", "neutral", "state", "done", "hpts", "apts", "pred_margin", "pred_total", "win_prob_home"]
    games_out = pred[cols].replace({np.nan: None}).to_dict(orient="records") if len(pred) else []
    (SITE / "teams.json").write_text(json.dumps({"season": season, "mu": fin["mu"], "mu_p": fin["mu_p"], "hca": fin["h"], "teams": teams}))
    (SITE / "games.json").write_text(json.dumps(games_out))
    (SITE / "model.json").write_text(json.dumps({k: model[k] for k in ("margin", "total", "sigma", "total_sigma", "n_train", "backtest", "calibration")}))
    (SITE / "accuracy.json").write_text(json.dumps(live_accuracy(log)))
    (SITE / "params.json").write_text((ROOT / "data" / "params.json").read_text() if (ROOT / "data" / "params.json").exists() else "{}")
    status["games_listed"], status["teams"] = len(games_out), len(teams)
    (SITE / "status.json").write_text(json.dumps(status))
    print(json.dumps(status))


if __name__ == "__main__":
    main()

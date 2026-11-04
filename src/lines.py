"""Betting lines from The Odds API (free plan) -> expected value (EV) for every spread / moneyline / total.

Needs env ODDS_API_KEY.  Without it the site simply shows model picks with no EV.
One call returns every NCAAB game with all US books for 3 markets = 3 credits; twice a day is ~180 of the 500 free monthly credits.

How EV is computed (all per $1 staked):
  1. Market consensus = median line across books.
  2. The model's margin / total is shrunk toward the market:  adj = W*model + (1-W)*market.
     W (MODEL_WEIGHT) is a guess, not something fitted: the free plan has no historical lines to backtest against,
     and the closing line is usually sharper than any public model, so trusting the raw model would overstate every edge.
  3. P(win bet) comes from the normal model used everywhere else (sigma from the backtest).
  4. EV = p * (decimal_odds - 1) - (1 - p), using the best price across books (line shopping).
Bets are logged (data/bets_log.csv) and graded after the games finish, so a real ROI builds up over the season.
"""
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from names import norm as nname  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "data" / "ref"
BETS_LOG = ROOT / "data" / "bets_log.csv"
URL = "https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds"
MODEL_WEIGHT = 0.3      # share of the model vs the market when estimating the "true" margin / total
MAX_POINT_GAP = 2.0     # a spread/total more than this far from the median across books is treated as stale or erroneous
MAX_PROB_GAP = 0.04     # a moneyline price whose implied chance is this far "too good" vs the median is treated as an outlier
MAX_ML_DECIMAL = 4.0    # no moneyline longshots beyond +300: normal-curve tails are least trustworthy there
MIN_EV = 0.0            # only bets above this are called plays
SPREAD_BONUS_NOTE = "Win probability uses the same normal model as the rest of the site."


def dec(american: float) -> float:
    return 1 + american / 100 if american > 0 else 1 + 100 / -american


def ev_of(p: float, american: float) -> float:
    return p * (dec(american) - 1) - (1 - p)


def build_name_index(idmap: dict) -> dict:
    """normalised ESPN displayName / school -> Torvik name."""
    d = json.loads((REF / "espn_teams.json").read_text())
    idx, locs = {}, {}
    for t in (x["team"] for x in d["sports"][0]["leagues"][0]["teams"]):
        tv = idmap.get(str(t["id"]))
        if tv:
            idx[nname(t["displayName"])] = tv
            locs[nname(t["location"])] = tv
    return {"full": idx, "loc": locs}


def to_torvik(name: str, ix: dict):
    n = nname(name)
    if n in ix["full"]:
        return ix["full"][n]
    best = None
    for loc, tv in ix["loc"].items():          # longest school name that prefixes the odds name
        if n.startswith(loc + " ") or n == loc:
            if best is None or len(loc) > len(best[0]):
                best = (loc, tv)
    return best[1] if best else None


def fetch_odds(status: dict):
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        status["lines"] = "off: no ODDS_API_KEY set"
        return None
    try:
        r = requests.get(URL, params={"regions": "us", "markets": "h2h,spreads,totals", "oddsFormat": "american",
                                      "dateFormat": "iso", "apiKey": key}, timeout=40)
    except Exception as ex:  # never put the URL (it holds the key) in a message
        status["warnings"].append(f"Odds API request failed: {type(ex).__name__}")
        return None
    if r.status_code != 200:
        status["warnings"].append(f"Odds API returned HTTP {r.status_code}")
        return None
    status["lines"] = f"on: {len(r.json())} events, credits left {r.headers.get('x-requests-remaining', '?')}"
    return r.json()


def _med(xs):
    return float(statistics.median(xs)) if xs else None


def parse_event(ev: dict, ix: dict):
    h, a = to_torvik(ev["home_team"], ix), to_torvik(ev["away_team"], ix)
    if not h or not a:
        return None
    sp_h, tot, ml_h, ml_a = [], [], [], []      # (book, point/price..) tuples
    for b in ev.get("bookmakers", []):
        for m in b.get("markets", []):
            o = {x["name"]: x for x in m["outcomes"]}
            if m["key"] == "spreads" and ev["home_team"] in o and ev["away_team"] in o:
                sp_h.append((b["title"], o[ev["home_team"]]["point"], o[ev["home_team"]]["price"], o[ev["away_team"]]["price"]))
            elif m["key"] == "totals" and "Over" in o and "Under" in o:
                tot.append((b["title"], o["Over"]["point"], o["Over"]["price"], o["Under"]["price"]))
            elif m["key"] == "h2h" and ev["home_team"] in o and ev["away_team"] in o:
                ml_h.append((b["title"], o[ev["home_team"]]["price"]))
                ml_a.append((b["title"], o[ev["away_team"]]["price"]))
    return {"home": h, "away": a, "start": ev["commence_time"], "spread": sp_h, "total": tot, "ml_h": ml_h, "ml_a": ml_a}


def evaluate(g: dict, ln: dict, sigma: float, tsigma: float, w: float = MODEL_WEIGHT, total_bias: float = 0.0) -> dict:
    """g: a prediction row (home, away, pred_margin, pred_total); ln: parsed odds for the same game."""
    mk_margin = _med([-pt for _, pt, _, _ in ln["spread"]])       # home margin implied by the market
    mk_total = _med([pt for _, pt, _, _ in ln["total"]])
    m_adj = w * g["pred_margin"] + (1 - w) * mk_margin if mk_margin is not None else g["pred_margin"]
    pt_cal = g["pred_total"] - total_bias                           # model total after removing its average gap to the market
    t_adj = w * pt_cal + (1 - w) * mk_total if mk_total is not None else pt_cal
    h, a = g["home"], g["away"]
    cands = []
    # drop outlier quotes (stale / mistaken lines) so "best price" means shopping, not chasing errors
    sp_ok = [x for x in ln["spread"] if mk_margin is None or abs(-x[1] - mk_margin) <= MAX_POINT_GAP]
    tot_ok = [x for x in ln["total"] if mk_total is None or abs(x[1] - mk_total) <= MAX_POINT_GAP]

    def ml_ok(rows):
        ips = [1 / dec(pr) for _, pr in rows]
        med = _med(ips)
        return [r for r, ip in zip(rows, ips) if ip >= med - MAX_PROB_GAP and dec(r[1]) <= MAX_ML_DECIMAL] if med else []
    mlh_ok, mla_ok = ml_ok(ln["ml_h"]), ml_ok(ln["ml_a"])

    def add(market, side, label, book, price, p, line):
        cands.append({"market": market, "side": side, "label": label, "book": book, "price": int(price), "line": line,
                      "p": round(float(p), 4), "ev": round(ev_of(float(p), price), 4)})

    for book, pt, ph, pa in sp_ok:
        pc = norm.cdf((m_adj + pt) / sigma)                       # home covers if margin + point > 0
        add("spread", "home", f"{h} {pt:+g}", book, ph, pc, pt)
        add("spread", "away", f"{a} {-pt:+g}", book, pa, 1 - pc, -pt)
    for book, pt, po, pu in tot_ok:
        po_p = norm.cdf((t_adj - pt) / tsigma)
        add("total", "over", f"Over {pt:g}", book, po, po_p, pt)
        add("total", "under", f"Under {pt:g}", book, pu, 1 - po_p, pt)
    ph_win = norm.cdf(m_adj / sigma)
    for book, pr in mlh_ok:
        add("moneyline", "home", f"{h} to win", book, pr, ph_win, None)
    for book, pr in mla_ok:
        add("moneyline", "away", f"{a} to win", book, pr, 1 - ph_win, None)

    # best price per distinct bet (market, side, line): that is the "shop the line" step
    best = {}
    for c in cands:
        k = (c["market"], c["side"], c["line"])
        if k not in best or c["ev"] > best[k]["ev"]:
            best[k] = c
    ranked = sorted(best.values(), key=lambda c: -c["ev"])
    # keep the single best bet per market so the card is not four versions of the same pick
    per_market, seen = [], set()
    for c in ranked:
        if c["market"] not in seen:
            per_market.append(c); seen.add(c["market"])
    return {"market_margin": None if mk_margin is None else round(mk_margin, 1),
            "market_total": None if mk_total is None else round(mk_total, 1),
            "adj_margin": round(m_adj, 2), "adj_total": round(t_adj, 1), "books": len({c["book"] for c in cands}),
            "bets": per_market, "best": per_market[0] if per_market else None}


def attach_lines(games_out: list, events: list, idmap: dict, model: dict, status: dict) -> list:
    ix = build_name_index(idmap)
    parsed = [p for p in (parse_event(e, ix) for e in events) if p]
    status["lines_matched"] = 0
    by = {}
    for p in parsed:
        by.setdefault((p["home"], p["away"]), []).append(p)
    matched = []
    for g in games_out:
        if g.get("done"):
            continue
        cand = by.get((g["home"], g["away"]), [])
        # neutral-site games are sometimes listed with the teams flipped
        flip = by.get((g["away"], g["home"]), [])
        ln = None
        for p in cand:
            if abs((pd.Timestamp(p["start"]) - pd.Timestamp(g["start"])).total_seconds()) < 36 * 3600:
                ln = p; break
        if ln is None and flip and g.get("neutral"):
            p = flip[0]
            ln = {"home": g["home"], "away": g["away"], "start": p["start"],
                  "spread": [(b, -pt, pa, ph) for b, pt, ph, pa in p["spread"]], "total": p["total"],
                  "ml_h": p["ml_a"], "ml_a": p["ml_h"]}
        if ln is None:
            continue
        matched.append((g, ln))
    # the model's totals can sit systematically above/below the market (early season especially). Measure that gap across
    # every matched game and remove it, so only game-to-game differences count as edges instead of a blanket Over/Under.
    gaps = [g["pred_total"] - _med([pt for _, pt, _, _ in ln["total"]]) for g, ln in matched if ln["total"]]
    bias = float(np.median(gaps)) if len(gaps) >= 8 else 0.0
    status["totals_bias"] = round(bias, 2)
    for g, ln in matched:
        g["market"] = evaluate(g, ln, model["sigma"], model["total_sigma"], total_bias=bias)
        g["market"]["total_bias"] = round(bias, 1)
        status["lines_matched"] += 1
    return games_out


# ------------------------------------------------------------------ track record
def log_bets(games_out: list, now: str):
    cols = ["game_id", "made_at", "market", "label", "side", "line", "price", "book", "p", "ev", "result", "units"]
    old = pd.read_csv(BETS_LOG, dtype={"game_id": str, "result": object}) if BETS_LOG.exists() else pd.DataFrame(columns=cols)
    done_keys = set(zip(old["game_id"], old["market"]))
    rows = []
    for g in games_out:
        m = g.get("market")
        if not m or g.get("done"):
            continue
        for b in m["bets"]:
            if b["ev"] > MIN_EV and (str(g["game_id"]), b["market"]) not in done_keys:
                rows.append({"game_id": str(g["game_id"]), "made_at": now, "market": b["market"], "label": b["label"],
                             "side": b["side"], "line": b["line"], "price": b["price"], "book": b["book"], "p": b["p"],
                             "ev": b["ev"], "result": np.nan, "units": np.nan})
    if rows:
        old = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    old.to_csv(BETS_LOG, index=False)
    return old


def grade_bets(log: pd.DataFrame, finals: dict) -> pd.DataFrame:
    """finals: game_id -> (home, away, hpts, apts). Fills result (W/L/P) and units for ungraded bets."""
    log = log.copy()
    log["result"] = log["result"].astype(object)      # an all-empty column is read back as float; it must hold W/L/P
    log["units"] = pd.to_numeric(log["units"], errors="coerce")
    for i, r in log[log["result"].isna()].iterrows():
        f = finals.get(str(r.game_id))
        if not f:
            continue
        _, _, hp, ap = f
        margin, total = hp - ap, hp + ap
        if r.market == "moneyline":
            won = margin > 0 if r.side == "home" else margin < 0
            push = False
        elif r.market == "spread":
            x = (margin + r.line) if r.side == "home" else (-margin + r.line)
            won, push = x > 0, x == 0
        else:
            x = total - r.line if r.side == "over" else r.line - total
            won, push = x > 0, x == 0
        log.at[i, "result"] = "P" if push else ("W" if won else "L")
        log.at[i, "units"] = 0.0 if push else ((dec(r.price) - 1) if won else -1.0)
    return log


def roi_summary(log: pd.DataFrame) -> dict:
    g = log[log["result"].notna()]
    if g.empty:
        return {"n": 0}
    out = {"n": int(len(g)), "units": round(float(g["units"].sum()), 2), "roi": round(float(g["units"].mean()), 4),
           "win_rate": round(float((g["result"] == "W").sum() / max(1, (g["result"] != "P").sum())), 4),
           "avg_ev_claimed": round(float(g["ev"].mean()), 4)}
    out["by_market"] = {m: {"n": int(len(x)), "roi": round(float(x["units"].mean()), 4)} for m, x in g.groupby("market")}
    return out

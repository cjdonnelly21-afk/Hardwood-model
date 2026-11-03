"""Torvik ratings as they stood BEFORE each game (from the time-machine snapshots).

Snapshots are lists of rows shaped like team_results.csv:
  [rank, team, conf, record, adjoe, oe_rank, adjde, de_rank, barthag, ..., adjt]
For a game on date D we use the latest snapshot dated strictly before D (never the same day),
so a rating can never include the game it is being used to predict.
"""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SNAP_DIR = ROOT / "data" / "raw" / "timemachine"
MAX_STALE_DAYS = 12


CACHE_DIR = ROOT / "data" / "snapshots"
COLS = ["snap_date", "team", "adjoe", "adjde", "barthag", "adjt", "trank"]
GZ = {"method": "gzip", "mtime": 0}   # byte-stable, so unchanged seasons never show up in git diffs


def season_of(d: pd.Series) -> pd.Series:
    return d.dt.year + (d.dt.month >= 10).astype(int)


def _read_raw(files) -> pd.DataFrame:
    rows = []
    for f in files:
        try:
            data = json.loads(gzip.open(f).read().decode())
        except Exception:
            continue
        sd = pd.to_datetime(f.name[:8], format="%Y%m%d")
        for r in data:
            rows.append((sd, r[1], r[4], r[6], r[8], r[-1], r[0]))
    return pd.DataFrame(rows, columns=COLS)


def load_snapshots() -> pd.DataFrame:
    """Per-season cache files (data/snapshots/{season}.csv.gz) + any raw snapshots newer than the cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    parts = [pd.read_csv(f, parse_dates=["snap_date"]) for f in sorted(CACHE_DIR.glob("*.csv.gz"))]
    cached = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame({c: pd.Series(dtype="datetime64[ns]" if c == "snap_date" else "object") for c in COLS})
    last = cached["snap_date"].max() if len(cached) else pd.Timestamp("1900-01-01")
    new = [f for f in sorted(SNAP_DIR.glob("*.json.gz")) if pd.to_datetime(f.name[:8], format="%Y%m%d") > last]
    if new:
        add = _read_raw(new)
        cached = add if cached.empty else pd.concat([cached, add], ignore_index=True)
        cached["snap_date"] = pd.to_datetime(cached["snap_date"])
        add_seasons = set(season_of(add["snap_date"]))
        cs = season_of(cached["snap_date"])
        for sn in add_seasons:
            cached[cs == sn].sort_values(["snap_date", "team"]).round(4).to_csv(CACHE_DIR / f"{sn}.csv.gz", index=False, compression=GZ)
    return cached.sort_values("snap_date").reset_index(drop=True)


def attach(games: pd.DataFrame, snap: pd.DataFrame) -> pd.DataFrame:
    """Add tor_* columns for home and away. NaN where no recent-enough snapshot exists."""
    g = games.sort_values("date").copy()
    g["_row"] = np.arange(len(g))
    tol = pd.Timedelta(days=MAX_STALE_DAYS)
    for side in ("home", "away"):
        pre = "h" if side == "home" else "a"
        s = snap.rename(columns={"team": side, "adjoe": f"tor_oe_{pre}", "adjde": f"tor_de_{pre}",
                                 "barthag": f"tor_bar_{pre}", "adjt": f"tor_t_{pre}", "trank": f"tor_rk_{pre}",
                                 "snap_date": f"snap_{pre}"})
        g = pd.merge_asof(g.sort_values("date"), s.sort_values(f"snap_{pre}"),
                          left_on="date", right_on=f"snap_{pre}", by=side,
                          direction="backward", allow_exact_matches=False, tolerance=tol)
    return g.sort_values("_row").drop(columns="_row").reset_index(drop=True)


def torvik_margin(df: pd.DataFrame, hca_per100: float = 2.7) -> pd.Series:
    """Torvik-style predicted margin: efficiency gap per 100 possessions, scaled by expected tempo."""
    loc = np.where(df["neutral"] == 1, 0.0, 1.0)
    gap = (df["tor_oe_h"] - df["tor_de_h"]) - (df["tor_oe_a"] - df["tor_de_a"]) + 2 * hca_per100 * loc
    tempo = (df["tor_t_h"] + df["tor_t_a"]) / 2
    return gap * tempo / 100

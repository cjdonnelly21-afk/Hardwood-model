"""Turn Torvik's per-team-game rows (data/raw/gamestats/*.csv) into one row per game.

Output: data/games.parquet-like CSV (data/games.csv.gz) with columns
  season, date, gtype (0 nonconf, 1 conf, 2 conf tourney, 3 postseason),
  home, away, neutral, hconf, aconf, hpts, apts, poss,
  h_efg h_to h_or h_ftr a_efg a_to a_or a_ftr   (offense of each side),

Torvik lists every game twice (once per team). 'A' rows are the mirror of an 'H' row;
neutral games are kept once, oriented by alphabetical order so nothing leaks from the file order.
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "gamestats"

COLS = {0: "date", 1: "gtype", 2: "team", 3: "conf", 4: "opp", 5: "loc", 6: "result",
        9: "off_ppp", 10: "efg", 11: "to", 12: "or", 13: "ftr",
        14: "def_ppp", 15: "d_efg", 16: "d_to", 17: "d_or", 18: "d_ftr",
        20: "oconf", 21: "gnum", 22: "season", 23: "poss", 27: "tor27"}

RES = re.compile(r"([WL]),\s*(\d+)-(\d+)")


def load_season(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, usecols=list(COLS), dtype={5: str})
    df = df.rename(columns=COLS)
    m = df["result"].str.extract(RES)
    win = m[0] == "W"
    hi, lo = m[1].astype(float), m[2].astype(float)
    df["pts"] = hi.where(win, lo)
    df["opp_pts"] = lo.where(win, hi)
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%y")
    return df.drop(columns="result")


def to_games(t: pd.DataFrame) -> pd.DataFrame:
    # orient: take H rows as-is; for N rows keep the alphabetically-first team as "home" slot
    h = t[t["loc"] == "H"].copy()
    n = t[(t["loc"] == "N") & (t["team"] < t["opp"])].copy()
    # a game whose H row is missing but A row exists (rare): mirror it
    a = t[t["loc"] == "A"].copy()
    key = lambda d: set(zip(d["date"], d["team"], d["opp"]))
    have = key(h)
    a = a.loc[[(d, o, tm) not in have for d, tm, o in zip(a["date"], a["team"], a["opp"])]] if len(a) else a

    def build(d, neutral, mirror=False):
        if mirror:
            return pd.DataFrame({
                "season": d["season"], "date": d["date"], "gtype": d["gtype"],
                "home": d["opp"], "away": d["team"], "neutral": neutral,
                "hconf": d["oconf"], "aconf": d["conf"], "hpts": d["opp_pts"], "apts": d["pts"],
                "poss": d["poss"],
                "h_efg": d["d_efg"], "h_to": d["d_to"], "h_or": d["d_or"], "h_ftr": d["d_ftr"],
                "a_efg": d["efg"], "a_to": d["to"], "a_or": d["or"], "a_ftr": d["ftr"],
                "tor27": -pd.to_numeric(d["tor27"], errors="coerce")})
        return pd.DataFrame({
            "season": d["season"], "date": d["date"], "gtype": d["gtype"],
            "home": d["team"], "away": d["opp"], "neutral": neutral,
            "hconf": d["conf"], "aconf": d["oconf"], "hpts": d["pts"], "apts": d["opp_pts"],
            "poss": d["poss"],
            "h_efg": d["efg"], "h_to": d["to"], "h_or": d["or"], "h_ftr": d["ftr"],
            "a_efg": d["d_efg"], "a_to": d["d_to"], "a_or": d["d_or"], "a_ftr": d["d_ftr"],
            "tor27": d["tor27"]})

    g = pd.concat([build(h, 0), build(n, 1), build(a, 0, mirror=True)], ignore_index=True)
    # col 27 is a POST-game performance number (corr .85 with the final margin): hindsight, never a feature.
    g = g.drop(columns="tor27")
    return g.sort_values(["date", "home"]).reset_index(drop=True)


GAMES_DIR = ROOT / "data" / "games"
GZ = {"method": "gzip", "mtime": 0}


def load_games() -> pd.DataFrame:
    parts = [pd.read_csv(f, parse_dates=["date"]) for f in sorted(GAMES_DIR.glob("*.csv.gz"))]
    return pd.concat(parts, ignore_index=True).sort_values(["season", "date", "home"]).reset_index(drop=True)


def main():
    """Rebuild a season's file from its raw Torvik download. Seasons with no raw file here are left as they are
    (the cloud job only downloads the live season)."""
    GAMES_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(RAW.glob("*.csv")):
        g = to_games(load_season(p))
        g = g[~g[["hpts", "apts", "poss"]].isna().any(axis=1)]
        out = GAMES_DIR / f"{int(p.stem)}.csv.gz"
        g.to_csv(out, index=False, compression=GZ)
        n += len(g)
        print(p.stem, len(g), "games", int(g["neutral"].sum()), "neutral")
    print("wrote", n, "games;", len(load_games()), "total")


if __name__ == "__main__":
    sys.exit(main())

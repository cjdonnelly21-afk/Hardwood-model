"""ESPN team id -> Torvik team name.  Auto-match on normalised school name, plus manual overrides.

Run directly to rebuild data/ref/espn_to_torvik.json and list anything left unmatched.
The daily job also reports unmatched teams it sees on the schedule, so a new/renamed school
never silently drops out of the predictions.
"""
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "data" / "ref"

# ESPN "location" -> Torvik name, where normalisation alone can't bridge the gap
OVERRIDES = {
    "American University": "American", "App State": "Appalachian St.", "California Baptist": "Cal Baptist",
    "Florida International": "FIU", "Grambling": "Grambling St.", "IU Indianapolis": "IU Indy",
    "Kansas City": "UMKC", "LSU New Orleans": "New Orleans", "Long Island University": "LIU",
    "Loyola Maryland": "Loyola MD", "McNeese": "McNeese St.", "Miami (OH)": "Miami OH", "Miami": "Miami FL",
    "Nicholls": "Nicholls St.", "Ole Miss": "Mississippi", "Omaha": "Nebraska Omaha",
    "Pennsylvania": "Penn", "SE Louisiana": "Southeastern Louisiana", "Sam Houston": "Sam Houston St.",
    "Seattle U": "Seattle", "South Carolina Upstate": "USC Upstate",
    "Texas A&M-Corpus Christi": "Texas A&M Corpus Chris", "UAlbany": "Albany", "UConn": "Connecticut",
    "UIC": "Illinois Chicago", "UL Monroe": "Louisiana Monroe", "UT Martin": "Tennessee Martin",
    "Queens University": "Queens", "Queens": "Queens", "Southern Indiana": "Southern Indiana",
    "Lindenwood": "Lindenwood",
}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = s.replace("&", "and").replace("'", "").replace(".", "").replace("-", " ")
    s = re.sub(r"\bstate\b", "st", s)
    s = re.sub(r"\bsaint\b", "st", s)
    return re.sub(r"\s+", " ", s).strip()


def build(espn_teams: list, torvik_names: list) -> tuple[dict, list]:
    tv = {norm(t): t for t in torvik_names}
    tvset = set(torvik_names)
    mapping, unmatched = {}, []
    for t in espn_teams:
        loc = t["location"]
        name = OVERRIDES.get(loc)
        if name is None:
            name = tv.get(norm(loc))
        if name is not None and name in tvset:
            mapping[str(t["id"])] = name
        else:
            unmatched.append(loc)
    return mapping, unmatched


def load_map() -> dict:
    return json.loads((REF / "espn_to_torvik.json").read_text())


def main():
    import pandas as pd
    d = json.loads((REF / "espn_teams.json").read_text())
    teams = [t["team"] for t in d["sports"][0]["leagues"][0]["teams"]]
    tv = pd.read_csv(ROOT / "data" / "raw" / "team_results" / "2027.csv")["team"].tolist()
    mapping, unmatched = build(teams, tv)
    (REF / "espn_to_torvik.json").write_text(json.dumps(mapping, indent=0))
    print("mapped", len(mapping), "of", len(teams), "ESPN teams")
    print("ESPN teams with no Torvik match (non-D1 / unknown):", unmatched)
    print("Torvik teams with no ESPN id:", sorted(set(tv) - set(mapping.values())))


if __name__ == "__main__":
    main()

"""One-off probe: does the Odds API plan include player props, and what do they cost? Prints no key."""
import os, requests
key = os.environ["ODDS_API_KEY"].strip()
B = "https://api.the-odds-api.com/v4/sports"
def call(path, **p):
    r = requests.get(B + path, params={"apiKey": key, **p}, timeout=40)
    body = r.text.replace(key, "***")
    print(f"{path.split('/')[-1]} {p.get('markets','')} -> HTTP {r.status_code} cost={r.headers.get('x-requests-last')} remaining={r.headers.get('x-requests-remaining')}")
    return r, body
for sport, mk in (("americanfootball_nfl", "player_pass_yds,player_anytime_td"), ("basketball_nba", "player_points")):
    r, body = call(f"/{sport}/events")
    ev = r.json() if r.status_code == 200 else []
    print(sport, "events listed:", len(ev))
    if not ev:
        print(body[:300]); continue
    r, body = call(f"/{sport}/events/{ev[0]['id']}/odds", regions="us", markets=mk, oddsFormat="american")
    if r.status_code != 200:
        print(body[:300]); continue
    d = r.json()
    print(ev[0]["away_team"], "@", ev[0]["home_team"], "books:", len(d.get("bookmakers", [])))
    for b in d.get("bookmakers", [])[:3]:
        for m in b["markets"][:2]:
            print(" ", b["title"], m["key"], len(m["outcomes"]), "outcomes; sample:", [(o["name"], o.get("description"), o.get("point"), o["price"]) for o in m["outcomes"][:2]])

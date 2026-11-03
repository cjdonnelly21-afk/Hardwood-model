# Hardwood Model

A college basketball prediction model with an interactive website that refreshes itself every day.

## How it works
1. **Data** – 15 seasons (2011-12 to 2025-26) of game results from Bart Torvik, plus Torvik's team ratings as they stood on each date (no hindsight).
2. **Own ratings** (`src/ratings.py`) – for every game date, a ridge regression fits adjusted offense, defense, tempo and home-court edge using only earlier games, shrunk toward last season's ratings.
3. **Blend** (`src/train.py`) – a linear model combines my ratings with Torvik's pre-game ratings. Weights depend on how many games a team has played, so it leans on Torvik in November and on in-season results later.
4. **Daily job** (`src/update.py`) – pulls new results, grades yesterday's predictions, refits the blend, predicts the next three days of games, and writes `docs/data/*.json`. Hyper-parameters are re-tuned every Sunday (`src/tune.py`).
5. **Site** (`docs/`) – static page: Today's picks, Rankings, Matchup tool, Model report.

## Run it
    pip install -r requirements.txt
    python src/update.py            # daily refresh
    python src/update.py --tune     # also re-tune hyper-parameters
    cd docs && python -m http.server 8000   # view the site

First-time history download (only needed locally, ~15 min): `python src/fetch_torvik.py`, then `python src/build_games.py`.

## Put it online (free)
1. Create a GitHub repo and push this folder.
2. Settings > Pages > Source: *Deploy from a branch*, branch `main`, folder `/docs`.
3. Settings > Actions > General > Workflow permissions: *Read and write*.
4. Actions > daily-update > *Run workflow* once to test. After that it runs twice a day on its own.

## Honest limits
- No betting lines are used, so this does not measure edge against the market.
- Rosters, transfers and injuries are not modelled directly; Torvik's preseason ratings are the only proxy.
- If Torvik blocks the cloud runner, the job keeps the last good ratings and shows a banner on the site.

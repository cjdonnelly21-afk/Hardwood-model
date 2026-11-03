"""Coordinate-descent search over the rating hyper-parameters.

Tuned on seasons <= TUNE_END only; later seasons are a true hold-out.
Writes data/params.json (the daily job reads it and re-runs this weekly).
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_games import load_games  # noqa: E402
from ratings import Params, run_all, score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TUNE_END = 2022
GRID = {
    "carry": [0.7, 0.8, 0.85, 0.9, 0.95],
    "lam": [1.0, 2.0, 3.0, 4.0, 6.0],
    "half_life": [150.0, 250.0, 400.0, 1e6],
    "cap_margin": [20.0, 30.0, 45.0],
    "tempo_carry": [0.6, 0.8, 0.95],
}
_G = None


def _init():
    global _G
    _G = load_games()


def _eval(args):
    pd_, = args
    p = Params(**pd_)
    seasons = [s for s in sorted(_G["season"].unique()) if s <= TUNE_END]
    df, _ = run_all(_G, p, seasons=seasons)
    sc = score(df, min_gp=5, first_season=2013)
    return pd_, sc


def main(rounds=2):
    pf = ROOT / "data" / "params.json"
    best = json.loads(pf.read_text())["params"] if pf.exists() else Params().dict()
    with Pool(2, initializer=_init) as pool:
        base = pool.map(_eval, [(best,)])[0][1]
        print("start", base, flush=True)
        best_mae = base["mae"]
        for r in range(rounds):
            improved = False
            for k, vals in GRID.items():
                cands = [dict(best, **{k: v}) for v in vals if v != best[k]]
                for pd_, sc in pool.map(_eval, [(c,) for c in cands]):
                    print(f"  {k}={pd_[k]}: mae {sc['mae']:.4f} rmse {sc['rmse']:.3f}", flush=True)
                    if sc["mae"] < best_mae - 1e-4:
                        best_mae, best, improved = sc["mae"], pd_, True
                print(f"round {r} after {k}: best {best_mae:.4f} {best}", flush=True)
            if not improved:
                break
    (ROOT / "data" / "params.json").write_text(json.dumps({"params": best, "tune_mae": best_mae, "tune_end": TUNE_END}, indent=2))
    print("FINAL", best_mae, best)


if __name__ == "__main__":
    main()

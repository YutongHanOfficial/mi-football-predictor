"""
Walk-forward backtest for the rating engine.

    python backtest.py           # report accuracy for every season with the current EngineConfig
    python backtest.py --save    # also write backtest_summary.csv for the app's season history

Every game day is predicted using only games played before it, the same way
the Model Accuracy tab works. The oldest season only seeds the ratings.
"""
import sys

import pandas as pd

from engine import Engine, EngineConfig, config_hash, discover_seasons

SEASON_FILES = discover_seasons()
SUMMARY_FILE = "backtest_summary.csv"
SCOPES = ["In-State Only", "Michigan vs. Out-of-State", "All Games"]


def add_weeks(bt):
    """Week 1 = first week of Michigan games each season (weeks start Wednesday)."""
    bt = bt.copy()
    bt["week_num"] = (bt["date_obj"] - pd.Timedelta(days=2)).dt.isocalendar().week.astype(int)
    mi_start = bt[bt["matchup_scope"] == "In-State Only"].groupby("season")["week_num"].min()
    bt["rel_week"] = bt["week_num"] - bt["season"].map(mi_start).fillna(bt["week_num"].min()).astype(int) + 1
    return bt


def summarize_seasons(bt, cfg):
    rows = []
    for scope in SCOPES:
        d = bt if scope == "All Games" else bt[bt["matchup_scope"] == scope]
        for season, x in d.groupby("season"):
            late = x[x["rel_week"] >= 4]
            rows.append({
                "season": season, "scope": scope,
                "games": len(x), "acc": 100 * x["correct"].mean(),
                "spread": x["spread_err"].mean(), "total": x["total_err"].mean(), "brier": x["brier_score"].mean(),
                "games_wk4": len(late), "acc_wk4": 100 * late["correct"].mean() if len(late) else float("nan"),
                "spread_wk4": late["spread_err"].mean(), "total_wk4": late["total_err"].mean(),
            })
    out = pd.DataFrame(rows).round(4)
    out["config_hash"] = config_hash(cfg)
    return out


def main(save=False):
    cfg = EngineConfig()
    bt = add_weeks(Engine(SEASON_FILES, cfg).backtest())
    summary = summarize_seasons(bt, cfg)
    for scope in SCOPES:
        s = summary[summary["scope"] == scope]
        print(f"\n== {scope} ==")
        print(f"{'season':>8} {'games':>6} {'acc %':>7} {'spread MAE':>11} {'total MAE':>10} {'brier':>7} {'wk4+ acc':>9}")
        for r in s.itertuples():
            print(f"{r.season:>8} {r.games:>6} {r.acc:>7.1f} {r.spread:>11.2f} {r.total:>10.2f} {r.brier:>7.3f} {r.acc_wk4:>9.1f}")
        d = bt if scope == "All Games" else bt[bt["matchup_scope"] == scope]
        print(f"{'all':>8} {len(d):>6} {100 * d['correct'].mean():>7.1f} {d['spread_err'].mean():>11.2f} "
              f"{d['total_err'].mean():>10.2f} {d['brier_score'].mean():>7.3f}")
    if save:
        summary.to_csv(SUMMARY_FILE, index=False)
        print(f"\nSaved {SUMMARY_FILE}")


if __name__ == "__main__":
    sys.exit(main(save="--save" in sys.argv[1:]))

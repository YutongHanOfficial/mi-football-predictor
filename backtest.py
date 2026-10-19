"""
Walk-forward backtest for the rating engine.

    python backtest.py            # report accuracy with the current EngineConfig

Every game day is predicted using only games played before it, the same way
the Model Accuracy tab works.
"""
import sys

from engine import Engine, EngineConfig, summarize

SEASON_FILES = [("2025", "games_2025.csv"), ("2026", "games_2026.csv")]


def main():
    bt = Engine(SEASON_FILES, EngineConfig()).backtest()
    for scope in ["In-State Only", "Michigan vs. Out-of-State", None]:
        print(f"\n== {scope or 'All Games'} ==")
        print(f"{'season':>8} {'games':>6} {'acc %':>7} {'spread MAE':>11} {'total MAE':>10} {'brier':>7}")
        for season, m in summarize(bt, scope).items():
            print(f"{season:>8} {m['n']:>6} {m['acc']:>7.1f} {m['spread_mae']:>11.2f} {m['total_mae']:>10.2f} {m['brier']:>7.3f}")


if __name__ == "__main__":
    sys.exit(main())

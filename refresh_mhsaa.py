"""
Refresh mhsaa_<season>.csv from MHSAA's Playoff Point Summary feed.

    python refresh_mhsaa.py

MHSAA updates its official playoff-point averages less often than scores come in,
so `pop_through` records which games the snapshot covers: the cutoff date at which
the records in games_<season>.csv best match MHSAA's official win-loss records.
The playoff predictor compares only games played before that date.
"""
import json
import urllib.request
from datetime import date, timedelta

import pandas as pd

URL = "https://my.mhsaa.com//DesktopModules/MHSAA-Endpoint/handlers/Tournaments.ashx?Method=GetPop&YearTypeCode="


def snapshot_cutoff(official, games_path):
    """First date after the last game the official records include."""
    try:
        g = pd.read_csv(games_path).dropna(subset=["home_score", "away_score"])
    except FileNotFoundError:
        return date.today().isoformat()
    g = g[g["home_score"] != g["away_score"]]
    target = (official["official_wins"] + official["official_losses"]).to_dict()
    best, best_hits = None, -1
    for day in sorted(g["date"].unique()):
        cutoff = (pd.Timestamp(day) + timedelta(days=1)).date().isoformat()
        played = pd.concat([g.loc[g["date"] < cutoff, "home"], g.loc[g["date"] < cutoff, "away"]]).value_counts()
        hits = sum(int(played.get(t, 0)) == n for t, n in target.items())
        if hits >= best_hits:
            best, best_hits = cutoff, hits
    return best or date.today().isoformat()


def main():
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (mi-football-predictor)"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        rows = json.load(resp)
    d = pd.DataFrame(rows)
    if d.empty:
        print("MHSAA returned no playoff-point data; nothing changed")
        return
    season = int(d["YearTypeCode"].max())
    d = d[(d["IsEightPlayer"].astype(str) == "0") & (d["YearTypeCode"] == season)]
    out = pd.DataFrame({
        "school": d["School"].str.strip(),
        "division": d["Class"].str.strip().str.rstrip("$").astype(int),
        "opted_up": d["Class"].str.contains(r"\$").astype(int),
        "enrollment": d["Enrollment"],
        "games_scheduled": d["GamesInSchedule"],
        "official_pop": d["PlayoffPoints"],
        "official_wins": d["Wins"],
        "official_losses": d["Losses"],
        "official_rank": d["Rank"],
    }).sort_values("school")
    out["pop_through"] = snapshot_cutoff(out.set_index("school"), f"games_{season}.csv")
    path = f"mhsaa_{season}.csv"
    out.to_csv(path, index=False)
    print(f"Saved {len(out)} teams to {path} (official numbers cover games before {out['pop_through'].iloc[0]})")


if __name__ == "__main__":
    main()

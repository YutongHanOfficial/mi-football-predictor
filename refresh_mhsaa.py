"""
Refresh mhsaa_2026.csv from MHSAA's Playoff Point Summary feed.

    python refresh_mhsaa.py

Run it after adding a week's scores to games_2026.csv, so the official playoff-point
averages and the game data cover the same weeks. `pop_through` records the date the
snapshot was taken; the playoff predictor compares only games played before it.
"""
import json
import urllib.request
from datetime import date

import pandas as pd

URL = "https://my.mhsaa.com//DesktopModules/MHSAA-Endpoint/handlers/Tournaments.ashx?Method=GetPop&YearTypeCode="


def main():
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        rows = json.load(resp)
    d = pd.DataFrame(rows)
    d = d[d["IsEightPlayer"].astype(str) == "0"]
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
        "pop_through": date.today().isoformat(),
    }).sort_values("school")
    out.to_csv("mhsaa_2026.csv", index=False)
    print(f"Saved {len(out)} teams to mhsaa_2026.csv")


if __name__ == "__main__":
    main()

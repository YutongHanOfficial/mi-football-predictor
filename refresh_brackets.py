"""
Build brackets_<season>.csv from MHSAA's official 11-player football brackets.

    python refresh_brackets.py                     # current season, after Selection Sunday
    python refresh_brackets.py --json saved.json   # parse a saved TournamentData response

This is the feed behind mhsaa.com/sports/football → Brackets:
  1. SportSeason/Search  {"Year": "2026", "Sport": "FB", "Gender": "M"}  → tournament id
  2. Tournament/TournamentData?id=<id>  → every division's rounds, games, teams and POP
Until MHSAA publishes the brackets, TournamentData comes back empty and nothing is written.
The update-data GitHub Action runs this after fetching games.

Once brackets_<season>.csv exists, the Playoffs tab uses the official field and Regions
instead of projecting them. Delete the file to go back to projections.

CSV format (one row per qualifier; also fine to type in by hand if the feed changes):
    division  1-8
    region    1-4 within the division. Regions 1 and 2 meet in one Semifinal, 3 and 4 in the other.
    seed      1-8 within the Region (by POP). District 1 = 1v8 & 4v5, District 2 = 2v7 & 3v6.
    school    exactly as in mhsaa_<season>.csv / games_<season>.csv
    pop       official playoff-point average at selection (optional; decides hosts)
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import date

import pandas as pd

API = "https://my.mhsaa.com/DesktopModules/MHSAA-Endpoint/API/"
YEAR = str(date.today().year)
# First-round pairings by seed, grouped by District
DISTRICTS = [{(1, 8), (4, 5)}, {(2, 7), (3, 6)}]


def fetch(url, body=None):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if body else "GET")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def fetch_tournament(year=YEAR):
    season_id = fetch(API + "SportSeason/Search", {"Year": year, "Sport": "FB", "Gender": "M"})
    if not season_id:
        return {}  # no tournament set up for this year yet
    try:
        return fetch(f"{API}Tournament/TournamentData?id={season_id}")
    except urllib.error.HTTPError as e:
        if e.code == 404:  # what the feed returns before the brackets are drawn
            return {}
        raise


def parse(data):
    """TournamentData JSON → (bracket rows, list of problems)."""
    rows, problems = [], []
    for div in (data or {}).get("Divisions") or []:
        d = int(div["DivisionId"])
        first = next((r for r in div["Rounds"] if r["RoundNumber"] == 1), None)
        if first is None:
            continue
        games = sorted(first["Brackets"], key=lambda b: (b["Contest"]["TournamentSortOrder"], b["Contest"]["BracketPosition"]))
        # Each District (BracketGroup) has two first-round games; Districts 2k-1 and 2k form Region k
        teams = []
        for b in games:
            c = b["Contest"]
            district = int(b["BracketGroup"])
            for side in ("HomeTeam", "AwayTeam"):
                t = c.get(side)
                if t and t.get("Name"):
                    teams.append({"division": d, "region": (district - 1) // 2 + 1, "district": district,
                                  "game": c["TournamentContestId"], "school": t["Name"].strip(),
                                  "pop": round(float(t.get("Pop") or 0), 3), "rank": int(t.get("Rank") or 99)})
        df = pd.DataFrame(teams)
        if df.empty:
            continue
        for region, g in df.groupby("region"):
            if len(g) != 8:
                problems.append(f"Division {d} Region {region}: {len(g)} teams")
                continue
            g = g.sort_values(["pop", "rank"], ascending=[False, True]).copy()
            g["seed"] = range(1, 9)
            # The seeds should reproduce MHSAA's first-round games and Districts
            pairs = g.groupby("game")["seed"].apply(lambda s: tuple(sorted(s)))
            by_district = g.drop_duplicates("game").set_index("game")["district"]
            got = sorted([{pairs[gm] for gm in by_district.index[by_district == k]} for k in by_district.unique()], key=min)
            if got != sorted(DISTRICTS, key=min):
                problems.append(f"Division {d} Region {region}: pairings {sorted(pairs)} don't match seeds by POP")
            rows.append(g)
    if not rows:
        return pd.DataFrame(columns=["division", "region", "seed", "school", "pop"]), problems
    out = pd.concat(rows)[["division", "region", "seed", "school", "pop"]].sort_values(["division", "region", "seed"])
    for d, g in out.groupby("division"):
        if len(g) != 32:
            problems.append(f"Division {d}: {len(g)} teams")
    return out, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", help="parse a saved TournamentData response instead of downloading")
    ap.add_argument("--year", default=YEAR)
    ap.add_argument("--out", help="output file (default: brackets_<year>.csv)")
    ap.add_argument("--force", action="store_true", help="write the file even if checks fail")
    args = ap.parse_args()

    args.out = args.out or f"brackets_{args.year}.csv"
    divisions_file = f"mhsaa_{args.year}.csv"

    data = json.load(open(args.json)) if args.json else fetch_tournament(args.year)
    out, problems = parse(data)
    if out.empty:
        print("MHSAA has not published the brackets yet (the tournament feed is empty). Nothing written.")
        return

    known = pd.read_csv(divisions_file).set_index("school")["division"]
    unknown = sorted(set(out["school"]) - set(known.index))
    if unknown:
        problems.append(f"Schools not in {divisions_file} (rename them in the CSV): {unknown}")
    moved = out[out["school"].isin(known.index) & (out["school"].map(known) != out["division"])]
    for _, r in moved.iterrows():
        print(f"Note: {r['school']} plays in Division {r['division']} (listed as {known[r['school']]} in {divisions_file})")

    for p in problems:
        print("Problem:", p)
    if problems and not args.force:
        sys.exit(f"Not writing {args.out}; fix the problems above or rerun with --force.")
    out.to_csv(args.out, index=False)
    print(f"Saved {len(out)} teams in {out['division'].nunique()} divisions to {args.out}")


if __name__ == "__main__":
    main()

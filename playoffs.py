"""
MHSAA 11-player playoff predictor (2026 rules).

Playoff-point average (POP), per the 2026 MHSAA bulletin:
  * Win points by opponent division: D1 60, D2 55, D3 50, D4 45, D5 40, D6 35, D7 30, D8 25.
    Base = sum of win points / games played.
  * Bonus: for every opponent, multiplier by the opponent's division
    (D1-2: 6, D3-4: 5, D5-6: 4, D7-8: 3) x opponent's wins / opponent's games.
    An opponent's win over you does not count toward your bonus (validated against
    the official MHSAA numbers), and a school gets no bonus for a game it forfeited.
  * POP = base + bonus, rounded to the thousandth.

Selection and brackets:
  * Divisions are fixed in advance by enrollment. The top 32 by POP in each division
    qualify (minimum 7 games). No automatic qualifiers.
  * Each division is split into four geographic Regions of 8. Within a Region, seeds by
    POP: District 1 = 1v8, 4v5; District 2 = 2v7, 3v6. Higher POP hosts through the
    Regional final. Semifinals and Ford Field finals are treated as neutral sites.

Regions are built from school locations (school_locations.csv: each school's town,
from US Census gazetteer coordinates): each division's 32 qualifiers are split into
four balanced geographic groups of 8. MHSAA draws the official Regions by hand, so
they can differ at the edges.

Once MHSAA publishes the brackets (Selection Sunday), brackets_2026.csv (see
refresh_brackets.py) replaces all of that: the official field, Regions and seeds are
used as-is and only the bracket is simulated. Playoff results added to the game data
fix those winners, so the odds update week by week.
"""

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from engine import is_oos

WIN_POINTS = {1: 60, 2: 55, 3: 50, 4: 45, 5: 40, 6: 35, 7: 30, 8: 25}
BONUS_MULT = {1: 6, 2: 6, 3: 5, 4: 5, 5: 4, 6: 4, 7: 3, 8: 3}
FIELD_PER_DIVISION = 32
MIN_GAMES = 7
ROUNDS = ["District Semifinal", "District Final", "Regional Final", "Semifinal", "Final"]
# Official bracket: Regions 1 and 2 meet in one Semifinal, 3 and 4 in the other
OFFICIAL_SEMIS = ((0, 1), (2, 3))


def load_divisions(path):
    df = pd.read_csv(path)
    return df.set_index("school")


def load_brackets(path):
    """Official brackets (division, region, seed, school[, pop]); checks every Region is seeded 1-8."""
    df = pd.read_csv(path)
    df["school"] = df["school"].str.strip()
    problems = []
    for d, g in df.groupby("division"):
        for r in range(1, 5):
            seeds = sorted(g.loc[g["region"] == r, "seed"])
            if seeds != list(range(1, 9)):
                problems.append(f"Division {d} Region {r} has seeds {seeds}")
    dupes = df["school"][df["school"].duplicated()].tolist()
    if dupes:
        problems.append(f"listed twice: {dupes}")
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return df


# ------------------------------------------------------------------
# Playoff points
# ------------------------------------------------------------------

class PlayoffPoints:
    """Vectorized POP calculator over a fixed set of teams."""

    def __init__(self, teams, divisions):
        self.teams = list(teams)
        self.index = {t: i for i, t in enumerate(self.teams)}
        self.div = np.array([divisions.get(t, 0) for t in self.teams], dtype=int)
        self._refresh_tables()

    def _refresh_tables(self):
        self.win_pts = np.array([WIN_POINTS.get(d, 0) for d in self.div], dtype=float)
        self.mult = np.array([BONUS_MULT.get(d, 0) for d in self.div], dtype=float)

    def set_division(self, team, division):
        self.div[self.index[team]] = division
        self._refresh_tables()

    def compute(self, win_idx, lose_idx, forfeit):
        """Return (pop, games, wins) arrays given winner/loser index arrays."""
        T = len(self.teams)
        games = np.bincount(win_idx, minlength=T) + np.bincount(lose_idx, minlength=T)
        wins = np.bincount(win_idx, minlength=T).astype(float)
        g = np.maximum(games, 1).astype(float)
        base = np.bincount(win_idx, weights=self.win_pts[lose_idx], minlength=T) / g
        # Winner's bonus from the loser: loser never beat the winner in this game
        w_bonus = self.mult[lose_idx] * wins[lose_idx] / g[lose_idx]
        # Loser's bonus from the winner: drop the winner's win over us; none if we forfeited
        l_bonus = np.where(forfeit, 0.0, self.mult[win_idx] * (wins[win_idx] - 1) / g[win_idx])
        bonus = (np.bincount(win_idx, weights=w_bonus, minlength=T)
                 + np.bincount(lose_idx, weights=l_bonus, minlength=T))
        pop = np.round(base + bonus, 3)
        return pop, games, wins


def counted_games(games, index, last_date):
    """Regular-season results that count for playoff points, as index arrays."""
    g = games[(games["date_obj"] <= last_date) & games["completed"]]
    g = g[g["home"].isin(index) & g["away"].isin(index)]
    g = g[g["home_score"] != g["away_score"]]  # 0-0 placeholders are unknown results
    home_won = (g["home_score"] > g["away_score"]).to_numpy()
    w = np.where(home_won, g["home"].map(index), g["away"].map(index)).astype(int)
    l = np.where(home_won, g["away"].map(index), g["home"].map(index)).astype(int)
    return w, l, g["is_forfeit"].to_numpy(bool), g["date_obj"].to_numpy()


# ------------------------------------------------------------------
# Approximate geography from the schedule graph
# ------------------------------------------------------------------

def load_locations(path):
    """School -> (x, y) in miles on a flat local projection of Michigan."""
    g = pd.read_csv(path)
    x = (g["lon"] + 85.0) * 69.0 * np.cos(np.radians(44.0))
    y = (g["lat"] - 44.0) * 69.0
    return {s: np.array([xx, yy]) for s, xx, yy in zip(g["school"], x, y)}


def split_regions(points, n_regions=4, size=8, iters=8):
    """Balanced k-means: split points into n_regions groups of exactly `size`."""
    n = len(points)
    # Deterministic farthest-point init
    centers = [points[np.argmax(np.linalg.norm(points - points.mean(0), axis=1))]]
    while len(centers) < n_regions:
        d = np.min([np.linalg.norm(points - c, axis=1) for c in centers], axis=0)
        centers.append(points[np.argmax(d)])
    centers = np.array(centers)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        cost = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        cost = np.repeat(cost, size, axis=1)[:, :n]
        rows, cols = linear_sum_assignment(cost)
        new = np.empty(n, dtype=int)
        new[rows] = cols // size
        if np.array_equal(new, labels):
            break
        labels = new
        centers = np.array([points[labels == k].mean(0) for k in range(n_regions)])
    return labels, centers


# ------------------------------------------------------------------
# Monte Carlo playoff simulation
# ------------------------------------------------------------------

class PlayoffSimulator:
    def __init__(self, engine, divisions_df, locations, season="2026", last_regular_date="2026-10-24", brackets=None):
        self.engine = engine
        self.ratings = engine.current()
        self.season = season
        self.last_date = pd.Timestamp(last_regular_date)
        div_map = divisions_df["division"].to_dict()
        self.official = divisions_df

        season_games = engine.games[engine.games["season"] == season]
        names = pd.unique(pd.concat([season_games["home"], season_games["away"]]))
        # Michigan 11-player members plus out-of-state opponents count for points
        self.teams = sorted(t for t in names if t in div_map or is_oos(t))
        self.pp = PlayoffPoints(self.teams, div_map)
        self.mi_mask = np.array([t in div_map for t in self.teams])

        self.w, self.l, self.ff, dates = counted_games(season_games, self.pp.index, self.last_date)
        # The official snapshot covers games before `pop_through`; compare like with like
        through = pd.Timestamp(divisions_df["pop_through"].iloc[0])
        snap = dates < np.datetime64(through)
        self._infer_oos_divisions(snap)
        snap_pop, _, _ = self.pp.compute(self.w[snap], self.l[snap], self.ff[snap])
        # Remaining gap to the official number (unreported/0-0 games etc.) carried forward
        off = np.array([divisions_df["official_pop"].get(t, np.nan) for t in self.teams])
        self.offset = np.where(self.mi_mask & ~np.isnan(off), np.nan_to_num(off) - snap_pop, 0.0)
        self.match_rate = float(np.mean(np.abs(self.offset[self.mi_mask]) < 0.0015))
        pop, _, _ = self.pp.compute(self.w, self.l, self.ff)
        self.current_pop = np.round(pop + self.offset, 3)

        # Schools without a known location sit at the center of the Lower Peninsula
        self.coords = {t: locations.get(t, np.array([0.0, -50.0])) for t in self.teams if t in div_map}

        # Unplayed regular-season games between known teams
        rem = season_games[~season_games["completed"] & (season_games["date_obj"] <= self.last_date)]
        rem = rem[rem["home"].isin(self.pp.index) & rem["away"].isin(self.pp.index)]
        rem = pd.concat([rem[["date", "home", "away"]], self._crossover_games(divisions_df, rem)], ignore_index=True)
        self.remaining = rem.reset_index(drop=True)
        self.rem_h = rem["home"].map(self.pp.index).to_numpy(int)
        self.rem_a = rem["away"].map(self.pp.index).to_numpy(int)
        self.rem_p = np.array([self.ratings.predict(h, a)["prob_home"] for h, a in zip(rem["home"], rem["away"])])
        self._region_cache = {}
        self._prob_cache = {}

        # Completed playoff games (after the regular season) fix their winners
        self.results = {}
        self.scores = {}  # same keys: (winner's points, loser's points)
        played = season_games[(season_games["date_obj"] > self.last_date) & season_games["completed"]]
        for _, g in played.iterrows():
            if g["home"] in self.pp.index and g["away"] in self.pp.index and g["home_score"] != g["away_score"]:
                h, a = self.pp.index[g["home"]], self.pp.index[g["away"]]
                key = frozenset((h, a))
                self.results[key] = h if g["home_score"] > g["away_score"] else a
                self.scores[key] = (int(max(g["home_score"], g["away_score"])), int(min(g["home_score"], g["away_score"])))

        self.official_bracket = brackets is not None
        if self.official_bracket:
            self._load_official(brackets)

    def _load_official(self, brackets):
        missing = sorted(set(brackets["school"]) - set(self.pp.index))
        if missing:
            raise ValueError(f"Bracket schools not found in the game data: {missing}")
        # {division: [Region 1..4 as team indices in seed order]}
        self.field = {}
        for d, g in brackets.groupby("division"):
            g = g.sort_values(["region", "seed"])
            self.field[int(d)] = [g.loc[g["region"] == r, "school"].map(self.pp.index).to_numpy(int) for r in range(1, 5)]
            for t in g["school"]:
                self.pp.div[self.pp.index[t]] = int(d)  # the bracket is the final word on divisions
        # Hosting goes to the higher POP at selection
        self.bracket_pop = self.current_pop.copy()
        if "pop" in brackets:
            for t, p in zip(brackets["school"], brackets["pop"]):
                if pd.notna(p):
                    self.bracket_pop[self.pp.index[t]] = p
        # Seeds are POP order within a Region; make sure hosting agrees with them
        for regions in self.field.values():
            for seeds in regions:
                self.bracket_pop[seeds] = np.maximum.accumulate(self.bracket_pop[seeds][::-1])[::-1] + np.linspace(8e-6, 1e-6, 8)

    def _crossover_games(self, divisions_df, rem):
        """League crossover games whose opponents are not set yet (e.g. "KLAA crossover").

        MHSAA lists how many games each school has scheduled. Teams short of that count
        are paired with each other the way crossovers work: a nearby school (by schedule
        proximity) of similar strength that they have not already played.
        """
        played = pd.concat([pd.Series(self.w), pd.Series(self.l)]).value_counts()
        upcoming = pd.concat([rem["home"], rem["away"]]).map(self.pp.index).value_counts()
        need = []
        for t, row in divisions_df.iterrows():
            if t not in self.pp.index:
                continue
            i = self.pp.index[t]
            short = int(row["games_scheduled"]) - int(played.get(i, 0)) - int(upcoming.get(i, 0))
            need += [t] * max(short, 0)
        met = set()
        for w, l in zip(self.w, self.l):
            met.add((self.teams[w], self.teams[l]))
            met.add((self.teams[l], self.teams[w]))
        power = self.ratings.team_power()
        strength = {t: power[self.ratings.index[t]] for t in set(need)}
        rows, pool = [], sorted(need, key=lambda t: -strength[t])
        while len(pool) > 1:
            t = pool.pop(0)
            cands = [c for c in pool if c != t and (t, c) not in met]
            if not cands:
                continue
            spread = np.std(list(strength.values())) or 1.0
            c = min(cands, key=lambda c: np.linalg.norm(self.coords[t] - self.coords[c])
                    + abs(strength[t] - strength[c]) / spread)
            pool.remove(c)
            met.update({(t, c), (c, t)})
            rows.append({"date": self.last_date.strftime("%Y-%m-%d") + " (TBD crossover)", "home": t, "away": c})
        self.crossovers = pd.DataFrame(rows, columns=["date", "home", "away"])
        return self.crossovers

    def _infer_oos_divisions(self, snap):
        """Pick each out-of-state opponent's Michigan division to best match official POPs."""
        off = self.official["official_pop"]
        mi = [i for i, t in enumerate(self.teams) if t in off.index]
        target = off.reindex([self.teams[i] for i in mi]).to_numpy()
        oos = [t for t in self.teams if is_oos(t)]
        for t in oos:
            self.pp.set_division(t, 3)
        w_s, l_s, ff_s = self.w[snap], self.l[snap], self.ff[snap]
        involved = defaultdict(set)
        for w, l in zip(w_s, l_s):
            involved[w].add(l)
            involved[l].add(w)
        for _ in range(2):
            for t in oos:
                i = self.pp.index[t]
                nbrs = [j for j in involved[i] if self.mi_mask[j]]
                if not nbrs:
                    continue
                pos = [mi.index(j) for j in nbrs]
                best, best_err = None, None
                for d in range(1, 9):
                    self.pp.set_division(t, d)
                    pop, _, _ = self.pp.compute(w_s, l_s, ff_s)
                    err = np.abs(pop[nbrs] - target[pos]).sum()
                    if best_err is None or err < best_err - 1e-9:
                        best, best_err = d, err
                self.pp.set_division(t, best)

    def _prob(self, home, away, neutral):
        key = (home, away, neutral)
        if key not in self._prob_cache:
            self._prob_cache[key] = self.ratings.predict(self.teams[home], self.teams[away], neutral=neutral)["prob_home"]
        return self._prob_cache[key]

    def _regions(self, division, qualifiers):
        """Split qualifiers into 4 geographic Regions. Returns ({team index: region}, semifinal pairs)."""
        key = (division, tuple(sorted(int(q) for q in qualifiers)))
        if key not in self._region_cache:
            teams = list(key[1])
            pts = np.array([self.coords[self.teams[q]] for q in teams])
            labels, centers = split_regions(pts)
            # Pair regions for the semifinals: the split with the closest centers
            pairings = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
            best = min(pairings, key=lambda p: sum(np.linalg.norm(centers[a] - centers[b]) for a, b in p))
            self._region_cache[key] = ({t: int(l) for t, l in zip(teams, labels)}, best)
        return self._region_cache[key]

    def _game(self, a, b, pop, rng, neutral=False):
        """Higher POP hosts unless neutral. Returns winner index; games already played keep their result."""
        done = self.results.get(frozenset((int(a), int(b))))
        if done is not None:
            return done
        home, away = (a, b) if pop[a] >= pop[b] else (b, a)
        return home if rng.random() < self._prob(home, away, neutral) else away

    def _play_division(self, regions, semis, pop, rng, reach, top_seed):
        """Play one division's bracket. `regions` = 4 arrays of team indices in seed order."""
        champs = {}
        for reg, seeds in enumerate(regions):
            top_seed[seeds[0]] += 1
            s = {k + 1: seeds[k] for k in range(8)}
            reach[seeds, 0] += 1
            semis_won = [self._game(s[a], s[b], pop, rng) for a, b in ((1, 8), (4, 5), (2, 7), (3, 6))]
            reach[semis_won, 1] += 1
            d1 = self._game(semis_won[0], semis_won[1], pop, rng)
            d2 = self._game(semis_won[2], semis_won[3], pop, rng)
            reach[[d1, d2], 2] += 1
            champs[reg] = self._game(d1, d2, pop, rng)
            reach[champs[reg], 3] += 1
        finalists = [self._game(champs[a], champs[b], pop, rng, neutral=True) for a, b in semis]
        for t in finalists:
            reach[t, 4] += 1
        champion = self._game(finalists[0], finalists[1], pop, rng, neutral=True)
        reach[champion, 5] += 1

    def run(self, n_sims=1000, seed=7):
        rng = np.random.default_rng(seed)
        T = len(self.teams)
        qualify = np.zeros(T)
        pop_sum = np.zeros(T)
        top_seed = np.zeros(T)
        reach = np.zeros((T, len(ROUNDS) + 1))  # reach[:, k] = won k playoff games
        divisions = self.pp.div
        if self.official_bracket:
            # Field, Regions and seeds are set: play out only the bracket
            for d, regions in self.field.items():
                qualify[np.concatenate(regions)] = n_sims
            pop_sum[:] = self.bracket_pop * n_sims
            for _ in range(n_sims):
                for regions in self.field.values():
                    self._play_division(regions, OFFICIAL_SEMIS, self.bracket_pop, rng, reach, top_seed)
            self._results = (qualify, pop_sum, top_seed, reach, n_sims)
            return self.summary()
        for _ in range(n_sims):
            home_wins = rng.random(len(self.rem_p)) < self.rem_p
            w = np.concatenate([self.w, np.where(home_wins, self.rem_h, self.rem_a)])
            l = np.concatenate([self.l, np.where(home_wins, self.rem_a, self.rem_h)])
            ff = np.concatenate([self.ff, np.zeros(len(home_wins), dtype=bool)])
            pop, games, _ = self.pp.compute(w, l, ff)
            pop = pop + self.offset
            pop_sum += pop
            noise = rng.random(T) * 1e-6  # random tiebreak
            for d in range(1, 9):
                elig = np.where(self.mi_mask & (divisions == d) & (games >= MIN_GAMES))[0]
                order = elig[np.argsort(-(pop[elig] + noise[elig]))]
                field = order[:FIELD_PER_DIVISION]
                qualify[field] += 1
                if len(field) < FIELD_PER_DIVISION:
                    continue
                region_of, semis = self._regions(d, field)
                labels = np.array([region_of[int(t)] for t in field])
                regions = []
                for reg in range(4):
                    teams = field[labels == reg]
                    regions.append(teams[np.argsort(-(pop[teams] + noise[teams]))])
                self._play_division(regions, semis, pop, rng, reach, top_seed)
        self._results = (qualify, pop_sum, top_seed, reach, n_sims)
        return self.summary()

    def projected_bracket(self, division):
        """The official bracket once published; before that the most likely field (top 32
        by projected POP), split into Regions and seeded.

        Returns a list of 4 Regions, each a list of (seed, team, POP), plus the Region
        pairings for the semifinals."""
        if self.official_bracket:
            if division not in self.field:
                return [], []
            return ([[(k + 1, self.teams[t], self.bracket_pop[t]) for k, t in enumerate(seeds)]
                     for seeds in self.field[division]], OFFICIAL_SEMIS)
        qualify, pop_sum, _, _, n = self._results
        proj = pop_sum / n
        elig = np.where(self.mi_mask & (self.pp.div == division))[0]
        field = elig[np.argsort(-proj[elig])][:FIELD_PER_DIVISION]
        if len(field) < FIELD_PER_DIVISION:
            return [], []
        region_of, semis = self._regions(division, field)
        labels = np.array([region_of[int(t)] for t in field])
        regions = []
        for reg in range(4):
            teams = field[labels == reg]
            seeds = teams[np.argsort(-proj[teams])]
            regions.append([(k + 1, self.teams[t], proj[t]) for k, t in enumerate(seeds)])
        return regions, semis

    def eliminated(self):
        """Teams that have lost a completed playoff game."""
        return {self.teams[t] for pair, w in self.results.items() for t in pair if t != w}

    def result(self, team_a, team_b):
        """Winner of a completed playoff game between the two teams, else None."""
        w = self.results.get(frozenset((self.pp.index[team_a], self.pp.index[team_b])))
        return None if w is None else self.teams[w]

    def score(self, team_a, team_b):
        """(winner's points, loser's points) of a completed playoff game, else None."""
        return self.scores.get(frozenset((self.pp.index[team_a], self.pp.index[team_b])))

    def matchup_prob(self, team_a, team_b, pop_a, pop_b, neutral=False):
        """Win probability for team_a, with the higher POP hosting unless neutral."""
        a, b = self.pp.index[team_a], self.pp.index[team_b]
        done = self.results.get(frozenset((a, b)))
        if done is not None:
            return float(done == a)
        if neutral or pop_a >= pop_b:
            return self._prob(a, b, neutral)
        return 1 - self._prob(b, a, neutral)

    def summary(self):
        qualify, pop_sum, top_seed, reach, n = self._results
        rows = []
        for i, t in enumerate(self.teams):
            if not self.mi_mask[i]:
                continue
            rows.append({
                "Team": t,
                "Div": int(self.pp.div[i]),
                "Current POP": self.current_pop[i],
                "Proj POP": pop_sum[i] / n,
                "Make Playoffs": qualify[i] / n,
                "Region #1 Seed": top_seed[i] / n,
                "Win 1st Round": reach[i, 1] / n,
                "Win District": reach[i, 2] / n,
                "Win Regional": reach[i, 3] / n,
                "Reach Final": reach[i, 4] / n,
                "Win Title": reach[i, 5] / n,
            })
        return pd.DataFrame(rows)

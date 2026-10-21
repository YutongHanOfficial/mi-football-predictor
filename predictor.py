import importlib
import os
import re
import sys
import pandas as pd
import altair as alt
from datetime import datetime, timedelta
from collections import deque
import streamlit as st

# Streamlit Cloud re-runs this script after a deploy without restarting Python, so
# a stale copy of our own modules can linger in memory. Reload them on every run
# (they only define functions and classes, so this is cheap).
for _module in ("engine", "playoffs"):
    if _module in sys.modules:
        importlib.reload(sys.modules[_module])

from engine import Engine, EngineConfig, discover_seasons, is_oos
from playoffs import PlayoffSimulator, load_divisions, load_locations, FIELD_PER_DIVISION

# ==========================================
# ⚙️ ENGINE CONFIGURATION
# ==========================================
# All model parameters live in engine.EngineConfig and were tuned by
# walk-forward backtesting (see backtest.py). Override any of them here.
ENGINE_CONFIG = EngineConfig()

# Every games_<season>.csv on disk (see fetch_games.py). The newest is the current
# season; the oldest only seeds the ratings, every other one is browsable as an archive.
SEASON_FILES = discover_seasons()
CURRENT_SEASON = SEASON_FILES[-1][0] if SEASON_FILES else ""
# Official MHSAA divisions, enrollments and playoff points (refresh_mhsaa.py)
DIVISIONS_FILE = f"mhsaa_{CURRENT_SEASON}.csv"
# Last day of regular-season games that count for playoff points (MHSAA bulletin).
# Add next season's date here when its bulletin comes out.
LAST_REGULAR_DAY = {"2026": "2026-10-24"}
# Town coordinates for each school (US Census gazetteer), used to draw Regions
LOCATIONS_FILE = "school_locations.csv"
PLAYOFF_SIMS = 2000

# ==========================================
# 🧮 HELPER FUNCTIONS
# ==========================================

def convert_to_moneyline(win_prob):
    if win_prob <= 0.001: return "+99900"
    if win_prob >= 0.999: return "-99900"

    if win_prob > 0.5:
        ml = -1 * (win_prob / (1 - win_prob)) * 100
        return f"{int(ml)}"
    elif win_prob < 0.5:
        ml = ((1 - win_prob) / win_prob) * 100
        return f"+{int(ml)}"
    else:
        return "+100"

# ==========================================
# 🧮 PREDICTOR (wraps the rating engine for the UI)
# ==========================================

EMPTY_STATS = {"W": 0, "L": 0, "PF": 0, "PA": 0, "GP": 0, "GP_stats": 0}


def team_split(ratings):
    """Per-team (offense, -defense) whose difference is the team's contribution to
    the projected margin, so Power = Offense + Defense in the UI tables."""
    power = ratings.team_power()
    adj = power - (ratings.off - ratings.deff)
    return ratings.off + adj / 2, ratings.deff - adj / 2


def calc_stats(games):
    stats = {}
    for g in games:
        if g.get("home_score") is None:
            continue
        h, a = g["home"], g["away"]
        hs, as_ = int(g["home_score"]), int(g["away_score"])
        for t in (h, a):
            stats.setdefault(t, dict(EMPTY_STATS))
        stats[h]["GP"] += 1
        stats[a]["GP"] += 1
        if hs > as_:
            stats[h]["W"] += 1
            stats[a]["L"] += 1
        elif as_ > hs:
            stats[a]["W"] += 1
            stats[h]["L"] += 1
        if not g.get("is_forfeit", False):
            stats[h]["GP_stats"] += 1
            stats[a]["GP_stats"] += 1
            stats[h]["PF"] += hs
            stats[h]["PA"] += as_
            stats[a]["PF"] += as_
            stats[a]["PA"] += hs
    return stats


def calc_ranks(stats_dict):
    rows = []
    for t, s in stats_dict.items():
        gp, gp_stats = max(1, s["GP"]), max(1, s["GP_stats"])
        rows.append({"team": t, "win_pct": s["W"] / gp, "pf": s["PF"], "pa": s["PA"],
                     "diff": s["PF"] - s["PA"], "ppg": s["PF"] / gp_stats, "papg": s["PA"] / gp_stats})

    def get_ranks(key, reverse=True):
        ranks = {}
        for oos in (False, True):
            pool = sorted((r for r in rows if is_oos(r["team"]) == oos), key=lambda r: r[key], reverse=reverse)
            for i, r in enumerate(pool):
                ranks[r["team"]] = f"{i + 1}/{len(pool)}" + (" (OOS)" if oos else "")
        return ranks

    return {"win_pct": get_ranks("win_pct"), "pf": get_ranks("pf"), "pa": get_ranks("pa", False),
            "diff": get_ranks("diff"), "ppg": get_ranks("ppg"), "papg": get_ranks("papg", False)}


class SeasonView:
    """Everything the UI shows for one season: ratings, games, stats, ranks, projections."""

    def __init__(self, predictor, season):
        engine = predictor.engine
        self.predictor = predictor
        self.engine = engine
        self.season = season
        self.is_current = season == predictor.current_season
        self.games = predictor.games_as_dicts(season)
        self.ratings = engine._final[season]
        self.preseason = engine.preseason(season)

        played = self.ratings.games_played > 0
        scheduled = {g["home"] for g in self.games} | {g["away"] for g in self.games}
        o, d = team_split(self.ratings)
        self.teams = {}
        for t, i in self.ratings.index.items():
            if played[i] or t in scheduled:
                self.teams[t] = {"OSRS": float(o[i]), "DSRS": float(d[i]), "games": int(self.ratings.games_played[i])}
        self.team_list = sorted(self.teams)

        order = sorted(self.teams, key=lambda t: self.power(t), reverse=True)
        self._in_state = [t for t in order if not is_oos(t)]
        self._oos = [t for t in order if is_oos(t)]

        self.basic_stats = calc_stats(self.games)
        self.ranks = calc_ranks(self.basic_stats)
        # Unplayed games in a finished season were canceled, not upcoming
        self.upcoming = [g for g in self.games if g.get("home_score") is None] if self.is_current else []

    def power(self, team):
        t = self.teams.get(team)
        return t["OSRS"] - t["DSRS"] if t else 0.0

    def stats(self, team):
        return self.basic_stats.get(team, EMPTY_STATS)

    def rank_display(self, team):
        try:
            if is_oos(team):
                return f"{self._oos.index(team) + 1}/{len(self._oos)} (OOS)"
            return f"{self._in_state.index(team) + 1}/{len(self._in_state)}"
        except ValueError:
            return "N/A"

    def find_connection_path(self, team_a, team_b):
        if team_a not in self.teams or team_b not in self.teams:
            return None
        graph = {}
        games = self.games + self.predictor.games_as_dicts(self.predictor.previous_season(self.season))
        for g in games:
            if g.get("home_score") is not None:
                graph.setdefault(g["home"], set()).add(g["away"])
                graph.setdefault(g["away"], set()).add(g["home"])
        queue, visited = deque([(team_a, [team_a])]), {team_a}
        while queue:
            current_team, path = queue.popleft()
            if current_team == team_b:
                return path
            for neighbor in graph.get(current_team, ()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def predict_matchup(self, away_team, home_team, neutral=False):
        p = self.ratings.predict(home_team, away_team, neutral=neutral)
        prob_h = p["prob_home"]
        prob_a = 1 - prob_h

        spread_val_raw = round(p["margin"] * 2) / 2
        ou_val = round(p["total"] * 2) / 2
        if spread_val_raw > 0:
            spread_val, spread_str = -spread_val_raw, f"{home_team} -{spread_val_raw:g}"
        elif spread_val_raw < 0:
            spread_val, spread_str = abs(spread_val_raw), f"{away_team} -{abs(spread_val_raw):g}"
        else:
            spread_val, spread_str = 0, "PK"

        score_h, score_a = round(p["score_home"]), round(p["score_away"])
        # Keep the projected score consistent with the projected winner
        if score_h == score_a:
            if prob_h >= 0.5: score_h += 1
            else: score_a += 1

        return {
            "away_team": away_team, "home_team": home_team,
            "prob_a": prob_a, "prob_h": prob_h,
            "spread_str": spread_str, "spread_val": spread_val, "median_total": ou_val,
            "avg_score_a": score_a, "avg_score_h": score_h,
            "path": self.find_connection_path(away_team, home_team),
        }

    def rating_history(self, team_name):
        """Daily power / offense / defense / rank for one team through this season."""
        dates = sorted({g["date_obj"] for g in self.games if g.get("home_score") is not None and g.get("date_obj")})
        if not dates or team_name not in self.ratings.index:
            return []
        start_dt, last_game_dt = dates[0], dates[-1]
        now = datetime.now()
        end_dt = max(last_game_dt, now) if (self.is_current and now.year == last_game_dt.year) else last_game_dt

        # The Wednesday strictly before the first game
        days_to_wed = (start_dt.weekday() - 2) % 7 or 7
        preseason_dt = start_dt - timedelta(days=days_to_wed)

        def snapshot(ratings):
            o, d = team_split(ratings)
            i = ratings.index[team_name]
            pool = sorted(((t, o[j] - d[j]) for t, j in ratings.index.items()
                           if t in self.teams and is_oos(t) == is_oos(team_name)), key=lambda x: -x[1])
            rank_num = next((k + 1 for k, (t, _) in enumerate(pool) if t == team_name), None)
            suffix = " (OOS)" if is_oos(team_name) else ""
            rank = f"{rank_num}/{len(pool)}{suffix}" if rank_num else "N/A"
            return round(o[i] - d[i], 2), round(o[i], 2), round(-d[i], 2), rank, rank_num

        history = []
        last = snapshot(self.preseason)
        history.append({"Date": preseason_dt, "Label": f"{preseason_dt.month}/{preseason_dt.day} (Pre)",
                        "Power": last[0], "Offense": last[1], "Defense": last[2], "Rank": last[3], "Rank_Num": last[4]})
        game_dates, current_dt = set(dates), start_dt
        while current_dt <= end_dt:
            if current_dt in game_dates:
                last = snapshot(self.engine.ratings(self.season, as_of=pd.Timestamp(current_dt) + timedelta(days=1)))
            history.append({"Date": current_dt, "Label": f"{current_dt.month}/{current_dt.day}",
                            "Power": last[0], "Offense": last[1], "Defense": last[2], "Rank": last[3], "Rank_Num": last[4]})
            current_dt += timedelta(days=1)
        return history


class SeasonPredictor:
    """All seasons on disk; the newest is the current season, the oldest only seeds ratings."""

    def __init__(self, season_files, cfg=None):
        season_files = [(s, f) for s, f in season_files if f and os.path.exists(f)]
        self.engine = Engine(season_files, cfg or ENGINE_CONFIG)
        self.cfg = self.engine.cfg
        self.seasons = list(self.engine.seasons)
        self.current_season = self.seasons[-1]
        self.warmup_season = self.seasons[0] if len(self.seasons) > 1 else None
        self._games = {}
        self._views = {}

    def previous_season(self, season):
        i = self.seasons.index(season)
        return self.seasons[i - 1] if i > 0 else None

    def games_as_dicts(self, season):
        if season is None:
            return []
        if season not in self._games:
            g = self.engine.games[self.engine.games["season"] == season]
            self._games[season] = [{
                "date": r.date, "date_obj": r.date_obj.to_pydatetime() if pd.notna(r.date_obj) else None,
                "home": r.home, "away": r.away,
                "home_score": int(r.home_score) if r.completed else None,
                "away_score": int(r.away_score) if r.completed else None,
                "is_forfeit": bool(r.is_forfeit),
            } for r in g.itertuples(index=False)]
        return self._games[season]

    def view(self, season):
        if season not in self._views:
            self._views[season] = SeasonView(self, season)
        return self._views[season]

    def backtest(self, season):
        """Walk-forward predictions for one season, with week numbers and baselines."""
        if season == self.warmup_season:
            return None
        df = self.engine.backtest(seasons=[season])
        if df.empty:
            return None
        # Shift date back 2 days so Wednesday acts as the start of the week
        df["week_num"] = (df["date_obj"] - pd.Timedelta(days=2)).dt.isocalendar().week.astype(int)
        # Week 1 = first week of Michigan games (out-of-state teams often start a week earlier)
        mi_start = df[df["matchup_scope"] == "In-State Only"].groupby("season")["week_num"].min()
        df["rel_week"] = df["week_num"] - df["season"].map(mi_start).fillna(df["week_num"].min()).astype(int) + 1

        # Baselines: always pick home team / pick team with better record so far
        df["base_home_correct"] = (df["act_winner"] == df["home"]).astype(int)
        records, base_rec = {}, []
        for _, day in df.groupby("date", sort=True):
            for r in day.itertuples():
                h_rec = records.get(r.home, [0, 0])
                a_rec = records.get(r.away, [0, 0])
                pick = r.home if h_rec[0] / max(1, sum(h_rec)) >= a_rec[0] / max(1, sum(a_rec)) else r.away
                base_rec.append((r.Index, int(pick == r.act_winner)))
            for r in day.itertuples():
                loser = r.away if r.act_winner == r.home else r.home
                records.setdefault(r.act_winner, [0, 0])[0] += 1
                records.setdefault(loser, [0, 0])[1] += 1
        df["base_rec_correct"] = pd.Series(dict(base_rec))
        return df


# ==========================================
# ⚡ STREAMLIT CACHING WRAPPERS
# ==========================================

def code_version():
    """Changes whenever the model code or data changes, so cached results are rebuilt."""
    parts = []
    for f in ["engine.py", "predictor.py", "playoffs.py", DIVISIONS_FILE, LOCATIONS_FILE] + [f for _, f in SEASON_FILES]:
        parts.append(f"{f}:{os.path.getmtime(f) if os.path.exists(f) else 0}:{os.path.getsize(f) if os.path.exists(f) else 0}")
    return "|".join(parts)

@st.cache_resource(max_entries=1)
def load_predictor(version):
    if not any(os.path.exists(f) for _, f in SEASON_FILES):
        return None
    return SeasonPredictor(SEASON_FILES)

@st.cache_data(max_entries=5000)
def get_cached_prediction(_predictor, season, away_team, home_team, neutral=False, version=None):
    return _predictor.view(season).predict_matchup(away_team, home_team, neutral=neutral)

@st.cache_data(max_entries=500)
def get_cached_history(_predictor, season, team_name, version=None):
    return _predictor.view(season).rating_history(team_name)

@st.cache_data(max_entries=20)
def get_cached_backtest(_predictor, season, version=None):
    return _predictor.backtest(season)

@st.cache_resource(max_entries=1)
def load_playoff_sim(_predictor, version):
    if not os.path.exists(DIVISIONS_FILE) or CURRENT_SEASON not in LAST_REGULAR_DAY:
        return None
    sim = PlayoffSimulator(_predictor.engine, load_divisions(DIVISIONS_FILE), load_locations(LOCATIONS_FILE),
                           season=CURRENT_SEASON, last_regular_date=LAST_REGULAR_DAY[CURRENT_SEASON])
    sim.run(PLAYOFF_SIMS)
    return sim


def compute_filtered_accuracy_metrics(full_df, scope_selection, skip_early_weeks=False, by_season=False):
    if full_df is None or full_df.empty:
        return None
        
    if scope_selection == "In-State Only":
        df = full_df[full_df["matchup_scope"] == "In-State Only"].copy()
    elif scope_selection == "Michigan vs. Out-of-State":
        df = full_df[full_df["matchup_scope"] == "Michigan vs. Out-of-State"].copy()
    else:
        df = full_df.copy()

    if skip_early_weeks:
        df = df[df["rel_week"] >= 4]

    if df.empty:
        return None
        
    df['conf_bin'] = pd.cut(df['confidence'], bins=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    calib_table = df.groupby('conf_bin', observed=False).agg(
        Count=('correct', 'count'),
        Pred_Prob=('confidence', 'mean'),
        Act_Win_Pct=('correct', 'mean')
    ).reset_index()
    calib_table['Calib_Error'] = calib_table['Pred_Prob'] - calib_table['Act_Win_Pct']

    df = df.sort_values(["season", "week_num"])
    if by_season:
        # One point per season for the all-seasons view
        df["rel_week"] = 0
        df["week_label"] = df["season"]
    else:
        df["week_label"] = df["season"] + " Wk " + df["rel_week"].astype(str)

    weekly = df.groupby(["season", "rel_week", "week_label"]).agg(
        games=("correct", "count"),
        correct=("correct", "sum"),
        spread_err=("spread_err", "mean"),
        total_err=("total_err", "mean"),
        confidence=("confidence", "mean")
    ).reset_index()

    weekly = weekly.sort_values(["season", "rel_week"])
    weekly["cum_games"] = weekly["games"].cumsum()
    weekly["cum_correct"] = weekly["correct"].cumsum()
    weekly["cum_accuracy"] = weekly["cum_correct"] / weekly["cum_games"]
    weekly["accuracy"] = weekly["correct"] / weekly["games"]

    return {
        "df": df,
        "calib_table": calib_table,
        "weekly_df": weekly,
        "total_games": len(df),
        "total_correct": int(df["correct"].sum()),
        "win_acc": df["correct"].mean(),
        "base_home_acc": df["base_home_correct"].mean(),
        "base_rec_acc": df["base_rec_correct"].mean(),
        "avg_spread_err": df["spread_err"].mean(),
        "avg_total_err": df["total_err"].mean(),
        "avg_confidence": df["confidence"].mean(),
        "brier_score": df["brier_score"].mean()
    }


# ==========================================
# 🌐 STREAMLIT UI & MODERN DESIGN SYSTEM
# ==========================================

st.set_page_config(page_title="High School Football Predictor", page_icon="🏈", layout="wide")

# Custom Modern CSS Styling (Glassmorphism Cards, Dark Elevation, Vibrant Accents)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Smooth Container Cards */
    .stCard {
        background-color: #121318;
        border: 1px solid #262833;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    
    /* Header Polish */
    .main-header {
        font-size: 2.25rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    
    /* Metric Card Fixes */
    div[data-testid="stMetric"] {
        background-color: #181920;
        border: 1px solid #2a2d3d;
        padding: 14px 18px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    div[data-testid="stMetricValue"] > div {
        font-size: 1.65rem !important;
        font-weight: 700 !important;
        color: #f3f4f6 !important;
    }

    div[data-testid="stMetricLabel"] > div {
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #9ca3af !important;
    }

    /* Tab Polish */
    button[data-baseweb="tab"] {
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 10px 18px !important;
        border-radius: 8px !important;
    }
    
    .stMarkdown a.header-anchor,
    .stMarkdown a.anchor,
    h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {
        display: none !important;
        pointer-events: none !important;
    }
    </style>
""", unsafe_allow_html=True)

VERSION = code_version()
predictor = load_predictor(VERSION)

st.markdown("<h1 class='main-header'>🏈 High School Football Predictor Engine</h1>", unsafe_allow_html=True)
st.caption("Bayesian Power Ratings, Calibrated Matchup Projections & Walk-Forward Accuracy Tracking")
st.write("")

if predictor is None:
    st.error("⚠️ No game data found! Run `python fetch_games.py` to download `games_<season>.csv` files from MHSAA.")
else:
    # ----------------------------------------------------
    # SEASON PICKER: every tab shows the selected season
    # ----------------------------------------------------
    season_options = list(reversed(predictor.seasons))
    col_season, col_season_note = st.columns([1, 3])
    with col_season:
        season = st.selectbox(
            "Season", season_options, key="season",
            format_func=lambda s: f"{s} (current)" if s == predictor.current_season else f"{s} (archive)",
        )
    view = predictor.view(season)
    with col_season_note:
        st.write("")
        if not view.is_current:
            st.caption(f"📚 Viewing the archived **{season}** season: ratings, rankings and matchup projections use each team's end-of-season rating.")
        elif season != season_options[-1]:
            st.caption(f"Switch seasons to browse archived ratings, schedules and leaderboards back to {season_options[-1]}.")
    st.write("")

    tab1, tab2, tab3, tab4, tab_playoffs, tab5, tab6 = st.tabs([
        "🎮 Matchup Simulator", 
        "🏆 Power Rankings", 
        "📅 Team Schedules", 
        "📆 Upcoming Matchups", 
        "🏟️ Playoff Predictor",
        "📈 Season Leaderboards",
        "🎯 Model Accuracy"
    ], key="main_tab", on_change="rerun")  # lazy: only the open tab's code runs

    get_rank_display = view.rank_display
    all_teams = view.team_list

    def remembered_index(state_key, default):
        """Keep a team picked on one season selected when switching to another season."""
        team = st.session_state.get(state_key, default)
        if team in all_teams:
            return all_teams.index(team)
        return all_teams.index(default) if default in all_teams else 0

    def remember(state_key, team):
        if team:
            st.session_state[state_key] = team

    # ----------------------------------------------------
    # TAB 1: MATCHUP SIMULATOR
    # ----------------------------------------------------
    with tab1:
        if tab1.open:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("### ✈️ Away Team")
                away = st.selectbox("Away Team Select", all_teams, index=remembered_index("pick_away", "Grand Blanc"), label_visibility="collapsed")
                remember("pick_away", away)
                if away:
                    a_stats = view.stats(away)
                    a_pwr = round(view.power(away), 2)
                    st.caption(f"🏆 **Rank:** #{get_rank_display(away)} | ⚡ **Power Rating:** {a_pwr}")
                    a_gp_stats = max(1, a_stats.get("GP_stats", 0))
                    st.caption(f"📊 **Record:** {a_stats['W']}-{a_stats['L']} | 🟢 **PPG:** {a_stats['PF']/a_gp_stats:.1f} | 🔴 **PA/G:** {a_stats['PA']/a_gp_stats:.1f}")

            with col_b:
                st.markdown("### 🏠 Home Team")
                home = st.selectbox("Home Team Select", all_teams, index=remembered_index("pick_home", "Davison"), label_visibility="collapsed")
                remember("pick_home", home)
                if home:
                    h_stats = view.stats(home)
                    h_pwr = round(view.power(home), 2)
                    st.caption(f"🏆 **Rank:** #{get_rank_display(home)} | ⚡ **Power Rating:** {h_pwr}")
                    h_gp_stats = max(1, h_stats.get("GP_stats", 0))
                    st.caption(f"📊 **Record:** {h_stats['W']}-{h_stats['L']} | 🟢 **PPG:** {h_stats['PF']/h_gp_stats:.1f} | 🔴 **PA/G:** {h_stats['PA']/h_gp_stats:.1f}")

            neutral_site = st.checkbox("Neutral site (no home-field advantage)", value=False)

            st.write("")
            if st.button("🚀 Run Projection", width="stretch", type="primary"):
                if away == home:
                    st.warning("Please select two different teams.")
                else:
                    res = get_cached_prediction(predictor, season, away, home, neutral_site, version=VERSION)
                    if not view.is_current:
                        st.caption(f"Projection uses both teams' end-of-{season} ratings.")

                    if res["path"]:
                        hops = len(res["path"]) - 1
                        st.info(f"**Network Path Found ({hops} hop{'s' if hops > 1 else ''}):** " + " ➔ ".join(res["path"]))

                    st.markdown("---")
                    st.markdown("<h3 style='text-align: center; color: #a1a1aa;'>📊 Projected Final Score</h3>", unsafe_allow_html=True)
                    st.markdown(f"<h1 style='text-align: center; margin-bottom: 30px;'>{away} {res['avg_score_a']} — {res['avg_score_h']} {home}</h1>", unsafe_allow_html=True)
                
                    m1, m2, m3, m4 = st.columns([1, 1.5, 1, 1])
                    m1.metric(f"{away} Win Prob", f"{res['prob_a']*100:.1f}%", convert_to_moneyline(res['prob_a']))
                    m2.metric("True Spread", res["spread_str"])
                    m3.metric("Over / Under", f"{res['median_total']:g} pts")
                    m4.metric(f"{home} Win Prob", f"{res['prob_h']*100:.1f}%", convert_to_moneyline(res['prob_h']))

    # ----------------------------------------------------
    # TAB 2: POWER Rankings
    # ----------------------------------------------------
    with tab2:
        if tab2.open:
            col_title, col_filter = st.columns([2, 1])
            with col_title: st.subheader(f"{season} Power Rankings", anchor=False)
            with col_filter:
                st.write("") 
                view_filter = st.radio("Region Filter", ["Michigan (In-State)", "Out of State (OOS)"], horizontal=True, label_visibility="collapsed")
            
            show_oos = (view_filter == "Out of State (OOS)")

            rankings = []
            for t_name, t_data in view.teams.items():
                if show_oos != is_oos(t_name): continue

                o_rating, d_rating = t_data["OSRS"], t_data["DSRS"]
                s_rec = view.stats(t_name)
                rankings.append({
                    "Team": t_name,
                    "Record": f"{s_rec['W']}-{s_rec['L']}",
                    "Power Rating": round(o_rating - d_rating, 2),
                    "Offense": round(o_rating, 2),
                    "Defense": round(-d_rating, 2),
                })
            
            rankings.sort(key=lambda x: x["Power Rating"], reverse=True)
            for idx, r in enumerate(rankings):
                r["Rank"] = f"{idx + 1}/{len(rankings)}{' (OOS)' if show_oos else ''}"
            
            st.dataframe(rankings, column_order=["Rank", "Team", "Record", "Power Rating", "Offense", "Defense"], width="stretch", hide_index=True)

    # ----------------------------------------------------
    # TAB 3: TEAM SCHEDULES & HUB
    # ----------------------------------------------------
    with tab3:
        if tab3.open:
            st.subheader("Team Schedule", anchor=False)
        
            selected_team = st.selectbox("Select Team Hub:", all_teams, index=remembered_index("pick_hub", "Grand Blanc"))
            remember("pick_hub", selected_team)

            if selected_team:
                team_rank = get_rank_display(selected_team)
                p_rating = round(view.power(selected_team), 2)
                t_basic = view.stats(selected_team)
                r_win_pct, r_ppg, r_papg, r_pf, r_diff = (view.ranks[k] for k in ("win_pct", "ppg", "papg", "pf", "diff"))
                target_games, display_year = view.games, season

                gp, gp_stats = max(1, t_basic["GP"]), max(1, t_basic.get("GP_stats", 0))
                pf, pa = t_basic["PF"], t_basic["PA"]
            
                st.markdown("### 📊 Team Dashboard")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Overall Rank", f"#{team_rank}")
                m2.metric("Power Rating", f"{p_rating}")
                m3.metric(f"{display_year} Record", f"{t_basic['W']}-{t_basic['L']}")
                m4.metric(f"Win % (#{r_win_pct.get(selected_team, 'N/A')})", f"{t_basic['W'] / gp:.3f}")
            
                st.write("") 
                m5, m6, m7, m8 = st.columns(4)
                m5.metric(f"Points Per Game (#{r_ppg.get(selected_team, 'N/A')})", f"{pf / gp_stats if t_basic.get('GP_stats', 0) > 0 else 0.0:.1f}")
                m6.metric(f"Pts Against / Gm (#{r_papg.get(selected_team, 'N/A')})", f"{pa / gp_stats if t_basic.get('GP_stats', 0) > 0 else 0.0:.1f}")
                m7.metric(f"Total Points For (#{r_pf.get(selected_team, 'N/A')})", f"{pf}")
                m8.metric(f"Point Diff (#{r_diff.get(selected_team, 'N/A')})", f"{'+' if (pf-pa) > 0 else ''}{pf-pa}")
            
                st.markdown("---")
            
                completed_schedule, upcoming_schedule = [], []
                for g in target_games:
                    if g["home"] == selected_team or g["away"] == selected_team:
                        is_home = (g["home"] == selected_team)
                        opp = g["away"] if is_home else g["home"]
                        location_prefix = "vs" if is_home else "@"
                    
                        if g.get("home_score") not in [None, ""]:
                            team_score = int(g["home_score"]) if is_home else int(g["away_score"])
                            opp_score = int(g["away_score"]) if is_home else int(g["home_score"])
                            mov = team_score - opp_score
                        
                            completed_schedule.append({
                                "Date": g.get("date", "-"),
                                "Opponent": f"{location_prefix} {opp}",
                                "Score": "FORFEIT" if g.get("is_forfeit") else f"{team_score}-{opp_score}",
                                "MOV": f"+{mov}" if mov > 0 else str(mov),
                                "Result": "W" if mov > 0 else "L"
                            })
                        elif view.is_current:
                            upcoming_schedule.append({"Date": g.get("date", "-"), "is_home": is_home, "opp": opp, "location_prefix": location_prefix})

                with st.spinner(f"Building {selected_team}'s {season} rating history..."):
                    history_data = get_cached_history(predictor, season, selected_team, version=VERSION)
                if len(history_data) > 1:
                    df_hist = pd.DataFrame(history_data)
                    df_hist["Date"] = pd.to_datetime(df_hist["Date"])
                
                    col_title, col_freq = st.columns([3, 1])
                    with col_title: st.markdown("### 📈 Season Progression")
                    with col_freq: graph_freq = st.radio("Update Frequency", ["Weekly", "Daily"], horizontal=True)
                
                    # Filtering for Weekly View
                    if graph_freq == "Weekly":
                        df_hist = df_hist.sort_values("Date")
                    
                        # Monday=0, Tuesday=1, Wednesday=2
                        # This strictly filters the dataframe to only keep Wednesdays 
                        df_hist = df_hist[df_hist['Date'].dt.weekday == 2].reset_index(drop=True)

                    col_chart1, col_chart2 = st.columns(2)
                
                    hover = alt.selection_point(fields=['Date'], nearest=True, on='mouseover', empty=False)
                    base_ratings = alt.Chart(df_hist).encode(x=alt.X('Date:T', axis=alt.Axis(format='%m/%d', title=None)))
                
                    with col_chart1:
                        st.markdown("**Team Ratings over Time**")
                        lines_ratings = base_ratings.transform_fold(['Power', 'Offense', 'Defense'], as_=['Metric', 'Rating']).mark_line().encode(
                            y=alt.Y('Rating:Q', title=None), color=alt.Color('Metric:N', legend=alt.Legend(orient="bottom", title=None))
                        )
                        selectors_ratings = base_ratings.mark_rule(opacity=0, size=30).encode(
                            tooltip=[alt.Tooltip('Label:N', title='Date'), alt.Tooltip('Power:Q'), alt.Tooltip('Offense:Q'), alt.Tooltip('Defense:Q')]
                        ).add_params(hover)
                        rules_ratings = base_ratings.mark_rule(color='gray', strokeDash=[3, 3]).encode(opacity=alt.condition(hover, alt.value(0.5), alt.value(0)))
                        points_ratings = lines_ratings.mark_point(size=70, filled=True).encode(opacity=alt.condition(hover, alt.value(1), alt.value(0)))
                        st.altair_chart((lines_ratings + rules_ratings + selectors_ratings + points_ratings), width="stretch")
                    
                    with col_chart2:
                        st.markdown("**Rank**")
                        line_rank = base_ratings.mark_line(color='#66b3ff').encode(y=alt.Y('Rank_Num:Q', title=None, scale=alt.Scale(reverse=True)))
                        selectors_rank = base_ratings.mark_rule(opacity=0, size=30).encode(
                            tooltip=[alt.Tooltip('Label:N', title='Date'), alt.Tooltip('Rank:N', title='Rank')]
                        ).add_params(hover)
                        rules_rank = base_ratings.mark_rule(color='gray', strokeDash=[3, 3]).encode(opacity=alt.condition(hover, alt.value(0.5), alt.value(0)))
                        points_rank = line_rank.mark_point(size=70, filled=True, color='#66b3ff').encode(opacity=alt.condition(hover, alt.value(1), alt.value(0)))
                        st.altair_chart((line_rank + rules_rank + selectors_rank + points_rank), width="stretch")
                
                    df_table = df_hist[[c for c in ['Label', 'Power', 'Offense', 'Defense', 'Rank'] if c in df_hist.columns]].copy()
                    if 'Label' in df_table.columns: df_table = df_table.rename(columns={'Label': 'Date'})
                    st.dataframe(df_table, width="stretch", hide_index=True)
                st.markdown("---")
            
                st.markdown(f"### 📜 {display_year} Season Schedule")
                if completed_schedule: st.dataframe(completed_schedule, width="stretch", hide_index=True)
                else: st.info(f"No completed games recorded yet for the {display_year} season.")
                
                st.markdown("---")
            
                if upcoming_schedule:
                    st.markdown("### 🔮 Upcoming Game Projections")
                    if upcoming_schedule:
                        for match in upcoming_schedule:
                            away_t = selected_team if not match["is_home"] else match["opp"]
                            home_t = match["opp"] if not match["is_home"] else selected_team
                        
                            proj = get_cached_prediction(predictor, season, away_t, home_t, version=VERSION)
                            win_p = proj["prob_h"] if match["is_home"] else proj["prob_a"]
                            proj_team_pts = proj["avg_score_h"] if match["is_home"] else proj["avg_score_a"]
                            proj_opp_pts = proj["avg_score_a"] if match["is_home"] else proj["avg_score_h"]
                        
                            with st.container():
                                st.markdown(f"#### **{match['Date']}** {match['location_prefix']} **{match['opp']}**")
                                col1, col2, col3, col4 = st.columns([1, 2.5, 1, 1])
                                col1.metric("Win Prob", f"{win_p*100:.1f}%")
                                col2.metric("Spread", proj["spread_str"])
                                col3.metric("Over/Under", f"{proj['median_total']:g}")
                                col4.metric("Proj Score", f"{proj_team_pts}-{proj_opp_pts}")
                                st.divider()
                    else:
                        st.info("No upcoming unplayed games found in the schedule.")

    # ----------------------------------------------------
    # TAB 4: UPCOMING & TOP MATCHUPS (UN-NESTED & FULL WIDTH)
    # ----------------------------------------------------
    with tab4:
        if tab4.open:
            st.subheader("Upcoming Game Projections", anchor=False)
            st.caption("Displays all unplayed games on the current schedule categorized by quality and match date.")
        
            upcoming_raw = view.upcoming

            if not upcoming_raw:
                if view.is_current:
                    st.info("No upcoming games found on the schedule. All games have recorded scores.")
                else:
                    st.info(f"The {season} season is complete. Switch to the {predictor.current_season} season to see upcoming games.")
            else:
                upcoming_processed = []
            
                for g in upcoming_raw:
                    h, a = g["home"], g["away"]
                
                    # Same projection path as the Team Schedule tab
                    proj = get_cached_prediction(predictor, season, a, h, version=VERSION)
                    quality = view.power(h) + view.power(a)
                
                    abs_margin = abs(proj["spread_val"])

                    upcoming_processed.append({
                        "Date": g.get("date", "Unknown"),
                        "Away": a,
                        "Home": h,
                        "Spread": proj["spread_str"],
                        "Total": proj["median_total"],
                        "Proj Score": f"{proj['avg_score_a']}-{proj['avg_score_h']}",
                        "_abs_margin": abs_margin,
                        "_quality": quality
                    })

                st.markdown("### 🔥 Top Tier Matchups")
                st.caption("Highest combined power ratings across upcoming games")
                top_quality = sorted(upcoming_processed, key=lambda x: x["_quality"], reverse=True)[:10]
                st.dataframe(pd.DataFrame(top_quality).drop(columns=["_abs_margin", "_quality"]), hide_index=True, width="stretch")
            
                st.markdown("---")
                st.markdown("### ⚔️ Closest Projections")
                st.caption("Tightest predicted point spreads")
                top_close = sorted(upcoming_processed, key=lambda x: x["_abs_margin"])[:10]
                st.dataframe(pd.DataFrame(top_close).drop(columns=["_abs_margin", "_quality"]), hide_index=True, width="stretch")

                st.markdown("---")
                st.markdown("### 🗓️ Master Calendar")
            
                dates = sorted(list(set([x["Date"] for x in upcoming_processed])))
                for d in dates:
                    with st.expander(f"📅 Games on {d}", expanded=(d == dates[0])):
                        day_games = [x for x in upcoming_processed if x["Date"] == d]
                        st.dataframe(pd.DataFrame(day_games).drop(columns=["Date", "_abs_margin", "_quality"]), hide_index=True, width="stretch")

    # ----------------------------------------------------
    # TAB 5: SEASON LEADERBOARDS
    # ----------------------------------------------------
    with tab5:
        if tab5.open:
            col_lb_title, col_lb_filter = st.columns([2, 1])
            with col_lb_title: st.subheader(f"{season} Season Leaderboards", anchor=False)
            with col_lb_filter:
                st.write("") 
                show_oos_lb = (st.radio("Region Filter", ["Michigan (In-State)", "Out of State (OOS)"], horizontal=True, label_visibility="collapsed", key="lb_filter") == "Out of State (OOS)")
            
            stat_rows = []
            for t in all_teams:
                if show_oos_lb != is_oos(t): continue
                
                s = view.stats(t)
                gp, gp_stats = max(1, s["GP"]), max(1, s.get("GP_stats", 0))
                pf, pa = s["PF"], s["PA"]
            
                stat_rows.append({
                    "Rank": get_rank_display(t),
                    "Team": t,
                    "GP": s["GP"],
                    "Record": f"{s['W']}-{s['L']}",
                    "Win %": round(s['W'] / gp, 3) if s["GP"] > 0 else 0.000,
                    "PF": pf,
                    "PA": pa,
                    "Diff": pf - pa,
                    "PPG": round(pf / gp_stats, 1) if s.get("GP_stats", 0) > 0 else 0.0,
                    "PA/G": round(pa / gp_stats, 1) if s.get("GP_stats", 0) > 0 else 0.0
                })
            
            def safe_rank_sort(rank_str):
                try: return int(rank_str.split('/')[0])
                except: return 9999
                
            stat_rows.sort(key=lambda x: safe_rank_sort(x["Rank"]))
            st.dataframe(stat_rows, column_order=["Rank", "Team", "Record", "Win %", "GP", "PF", "PA", "Diff", "PPG", "PA/G"], width="stretch", hide_index=True)

    # ----------------------------------------------------
    # TAB 6: MODEL ACCURACY (WITH DYNAMIC SCOPE TOGGLE)
    # ----------------------------------------------------
    with tab6:
        if tab6.open:
            st.subheader("Model Accuracy", anchor=False)
            st.caption("Tests the model's accuracy by predicting past games using only the information that was available right before kickoff. This proves how well the model works in real-world scenarios.")

            scored_seasons = [s_ for s_ in reversed(predictor.seasons) if s_ != predictor.warmup_season]
            col_period, col_scope_toggle = st.columns([1, 2])
            with col_period:
                acc_period = st.selectbox(
                    "Seasons", ["All seasons"] + scored_seasons, key="acc_period",
                    help=f"{predictor.warmup_season} only seeds the ratings (there's no earlier season to predict it from), so accuracy starts in {scored_seasons[-1]}." if predictor.warmup_season else None,
                )
            with col_scope_toggle:
                scope_selection = st.radio(
                    "Filter Matchup Scope:",
                    ["In-State Only", "Michigan vs. Out-of-State", "All Games"],
                    horizontal=True,
                    index=0
                )
                skip_early = st.toggle(
                    "Skip weeks 1–3",
                    value=False,
                    help="Most public models (e.g. goosepoop.io) report accuracy from week 4 on, once teams have played a few games."
                )

            all_time = acc_period == "All seasons"
            if all_time:
                with st.spinner(f"Replaying every season since {scored_seasons[-1]} game by game (only the first time)..."):
                    frames = [get_cached_backtest(predictor, s_, version=VERSION) for s_ in reversed(scored_seasons)]
                    frames = [f for f in frames if f is not None]
                    full_bt = pd.concat(frames, ignore_index=True) if frames else None
                period_note = f"{scored_seasons[-1]}–{scored_seasons[0]}"
            else:
                with st.spinner(f"Replaying the {acc_period} season game by game..."):
                    full_bt = get_cached_backtest(predictor, acc_period, version=VERSION)
                period_note = acc_period
            bd = compute_filtered_accuracy_metrics(full_bt, scope_selection, skip_early, by_season=all_time)
            weeks_note = ", Weeks 4+" if skip_early else ""
            period_word = "Season" if all_time else "Week"

            if bd is None or bd["total_games"] == 0:
                st.warning(f"No historical prediction data available for the selected scope: **{scope_selection}**.")
            else:
                win_acc = bd["win_acc"] * 100
            
                st.markdown(f"### 🎯 Accuracy · {bd['total_games']:,} Games ({period_note}, {scope_selection}{weeks_note})")
            
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Accuracy", f"{win_acc:.1f}%")
                c2.metric("Correct Picks", f"{bd['total_correct']}/{bd['total_games']}")
                c3.metric(
                    "Avg Spread Error", 
                    f"{bd['avg_spread_err']:.1f}", 
                    help="Average point difference between the predicted spread and the actual game margin. Lower is better."
                )
                c4.metric(
                    "Avg Total Error", 
                    f"{bd['avg_total_err']:.1f}", 
                    help="Average point difference between the predicted total points and the actual total points scored. Lower is better."
                )
                c5.metric(
                    "Brier Score", 
                    f"{bd['brier_score']:.3f}", 
                    help="Measures how accurate the win probabilities are. A score closer to 0 is better. If the model says a team has an 80% chance to win, they should actually win 80% of the time."
                )

                st.markdown("---")
                st.markdown("### 📈 Baseline Comparisons")
                st.caption("Compares the model's accuracy against simple guessing strategies to show its true value.")
                b1, b2, b3 = st.columns(3)
                b1.metric("Always Pick Home Team", f"{bd['base_home_acc']*100:.1f}%")
                b2.metric("Always Pick Better Record", f"{bd['base_rec_acc']*100:.1f}%")
                b3.metric("Rating Engine (This Model)", f"{win_acc:.1f}%")

                st.markdown("---")
                st.markdown("### 📊 Probability Calibration Table")
                st.caption("Checks if the model's confidence is realistic. For example, when the model predicts a 70% chance of winning, do those teams actually win about 70% of the time?")
            
                calib = bd["calib_table"].copy()
                calib["conf_bin"] = calib["conf_bin"].astype(str)
                st.dataframe(
                    calib,
                    column_config={
                        "conf_bin": st.column_config.TextColumn("Confidence Range"),
                        "Count": st.column_config.NumberColumn("Predictions"),
                        "Pred_Prob": st.column_config.NumberColumn("Avg Predicted Prob", format="%.3f"),
                        "Act_Win_Pct": st.column_config.NumberColumn("Actual Win Rate", format="%.3f"),
                        "Calib_Error": st.column_config.NumberColumn("Calibration Error", format="%.3f")
                    },
                    hide_index=True, width="stretch"
                )

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"### 📅 {'Season-by-Season' if all_time else 'Weekly'} Win Accuracy ({scope_selection})")
            
                weekly_df = bd["weekly_df"]
            
                # Interactive hover selector for combined tooltip
                hover_acc = alt.selection_point(fields=['week_label'], nearest=True, on='mouseover', empty=False)

                base_chart = alt.Chart(weekly_df).encode(
                    x=alt.X('week_label:N', sort=None, title=None, axis=alt.Axis(labelAngle=-45))
                )
            
                line_weekly = base_chart.mark_line(color='#38bdf8', size=2).encode(
                    y=alt.Y('accuracy:Q', scale=alt.Scale(domain=[0.4, 1.0]), axis=alt.Axis(format='%', title=None))
                )
                points_weekly = line_weekly.mark_point(color='#38bdf8', filled=True, size=50).encode(
                    opacity=alt.condition(hover_acc, alt.value(1), alt.value(0))
                )
            
                line_cum = base_chart.mark_line(color='#f59e0b', size=2).encode(
                    y=alt.Y('cum_accuracy:Q')
                )
                points_cum = line_cum.mark_point(color='#f59e0b', filled=True, size=50).encode(
                    opacity=alt.condition(hover_acc, alt.value(1), alt.value(0))
                )

                selectors_acc = base_chart.mark_rule(opacity=0, size=30).encode(
                    tooltip=[
                        alt.Tooltip('week_label:N', title=period_word),
                        alt.Tooltip('accuracy:Q', title=f'{period_word} Accuracy', format='.1%'),
                        alt.Tooltip('cum_accuracy:Q', title='Cumulative Acc', format='.1%')
                    ]
                ).add_params(hover_acc)
            
                rules_acc = base_chart.mark_rule(color='gray', strokeDash=[3, 3]).encode(
                    opacity=alt.condition(hover_acc, alt.value(0.5), alt.value(0))
                )
            
                chart = alt.layer(line_weekly, points_weekly, line_cum, points_cum, rules_acc, selectors_acc).properties(height=350)
                st.altair_chart(chart, width="stretch")
            
                legend_label = "Season" if all_time else "Weekly"
                st.markdown(
                    f"""<div style="display: flex; justify-content: center; gap: 20px; font-size: 14px; font-weight: bold; color: var(--text-color); margin-top: -15px;">
                        <div style="display: flex; align-items: center; gap: 5px;"><div style="width: 14px; height: 14px; background-color: #38bdf8; border-radius: 3px;"></div> {legend_label} Accuracy</div>
                        <div style="display: flex; align-items: center; gap: 5px;"><div style="width: 14px; height: 14px; background-color: #f59e0b; border-radius: 3px;"></div> Cumulative</div>
                    </div><br>""", unsafe_allow_html=True
                )

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"### 📋 {'Season' if all_time else 'Weekly'} Breakdown Log ({scope_selection})")
            
                display_df = weekly_df[["week_label", "games", "correct", "accuracy", "spread_err", "total_err", "cum_accuracy"]].copy()
                display_df.columns = [period_word, "Games", "Correct", "Accuracy", "Spread Error", "Total Error", "Cumul. Acc"]
            
                # Formatting correction: Convert to literal percentages before rendering
                display_df["Accuracy"] = display_df["Accuracy"] * 100
                display_df["Cumul. Acc"] = display_df["Cumul. Acc"] * 100
            
                st.dataframe(
                    display_df,
                    column_config={
                        period_word: st.column_config.TextColumn(width="medium"),
                        "Games": st.column_config.NumberColumn(width="small"),
                        "Correct": st.column_config.NumberColumn(width="small"),
                        "Accuracy": st.column_config.NumberColumn(format="%.1f%%", width="small"),
                        "Spread Error": st.column_config.NumberColumn(format="%.1f pts", width="small"),
                        "Total Error": st.column_config.NumberColumn(format="%.1f pts", width="small"),
                        "Cumul. Acc": st.column_config.NumberColumn(format="%.1f%%", width="small")
                    },
                    hide_index=True,
                    width="stretch"
                )

                with st.expander("🔍 View Raw Out-Of-Sample Predictions Log"):
                    if all_time:
                        st.caption("Pick a single season above to see its game-by-game predictions.")
                    else:
                        st.dataframe(bd["df"])

    # ----------------------------------------------------
    # TAB: PLAYOFF PREDICTOR
    # ----------------------------------------------------
    with tab_playoffs:
        if tab_playoffs.open:
            st.subheader("Playoff Predictor", anchor=False)
            st.caption(f"Simulates the rest of the regular season {PLAYOFF_SIMS:,} times, computes official MHSAA playoff-point averages (POP), selects the top {FIELD_PER_DIVISION} in each division, then plays out the bracket with the rating engine.")

            if view.is_current:
                with st.spinner(f"Simulating the rest of the season {PLAYOFF_SIMS:,} times..."):
                    sim = load_playoff_sim(predictor, VERSION)
            else:
                sim = None
            if not view.is_current:
                st.info(f"The playoff predictor covers the current season. Switch to {predictor.current_season} to see playoff odds.")
            elif sim is None:
                st.warning(f"Playoff data for {CURRENT_SEASON} isn't set up yet: run `python refresh_mhsaa.py` and add the season's last regular-season day to `LAST_REGULAR_DAY` in predictor.py.")
            else:
                summary = sim.summary()
                official = sim.official
                summary["Record"] = summary["Team"].map(lambda t: f"{official.loc[t, 'official_wins']}-{official.loc[t, 'official_losses']}" if t in official.index else "")

                col_div, col_view = st.columns([1, 2])
                with col_div:
                    division = st.selectbox("Division", list(range(1, 9)), format_func=lambda d: f"Division {d}", key="playoff_div")
                with col_view:
                    st.write("")
                    view = st.radio("View", ["Playoff Odds", "Projected Bracket"], horizontal=True, label_visibility="collapsed", key="playoff_view")

                div_df = summary[summary["Div"] == division].sort_values(["Make Playoffs", "Proj POP"], ascending=False).reset_index(drop=True)
                locks = int((div_df["Make Playoffs"] >= 0.995).sum())
                bubble = div_df[(div_df["Make Playoffs"] > 0.02) & (div_df["Make Playoffs"] < 0.98)]
                fav = div_df.sort_values("Win Title", ascending=False).iloc[0]

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Teams in Division", len(div_df))
                m2.metric("Locked In (>99%)", locks)
                m3.metric("On the Bubble", len(bubble))
                m4.metric("Title Favorite", f"{fav['Win Title']*100:.1f}%", fav["Team"], delta_color="off", delta_arrow="off")

                if view == "Playoff Odds":
                    pct_cols = ["Make Playoffs", "Region #1 Seed", "Win 1st Round", "Win District", "Win Regional", "Reach Final", "Win Title"]
                    show = div_df[["Team", "Record", "Current POP", "Proj POP"] + pct_cols].copy()
                    show.insert(0, "Rank", range(1, len(show) + 1))
                    for c in pct_cols:
                        show[c] = show[c] * 100
                    st.dataframe(
                        show,
                        column_config={
                            "Current POP": st.column_config.NumberColumn(format="%.3f", help="Official MHSAA playoff-point average right now"),
                            "Proj POP": st.column_config.NumberColumn(format="%.2f", help="Average POP after the remaining regular-season games"),
                            **{c: st.column_config.ProgressColumn(c, format="%.1f%%", min_value=0, max_value=100) for c in pct_cols},
                        },
                        hide_index=True, width="stretch", height=min(38 * len(show) + 40, 900)
                    )
                    if not bubble.empty:
                        st.markdown("#### 🫧 Bubble Watch")
                        st.caption("Teams whose playoff fate depends on the remaining games")
                        b = bubble[["Team", "Record", "Current POP", "Proj POP", "Make Playoffs"]].copy()
                        b["Make Playoffs"] = b["Make Playoffs"] * 100
                        st.dataframe(b, column_config={"Make Playoffs": st.column_config.ProgressColumn("Make Playoffs", format="%.1f%%", min_value=0, max_value=100)}, hide_index=True, width="stretch")
                else:
                    regions, semis = sim.projected_bracket(division)
                    if not regions:
                        st.info("Not enough eligible teams to build a bracket for this division.")
                    else:
                        st.caption("Most likely field (top 32 by projected POP). Seeds 1-8 by POP within each Region; District 1 = 1v8 & 4v5, District 2 = 2v7 & 3v6. The higher seed has the higher POP and hosts.")
                        for r_i, region in enumerate(regions):
                            seeds = {seed: (team, pop) for seed, team, pop in region}
                            st.markdown(f"##### Region {r_i + 1}")
                            rows = []
                            for district, pairs in (("District 1", ((1, 8), (4, 5))), ("District 2", ((2, 7), (3, 6)))):
                                for a, b in pairs:
                                    ta, pa = seeds[a]
                                    tb, pb = seeds[b]
                                    rows.append({
                                        "District": district,
                                        "Higher Seed": f"#{a} {ta}",
                                        "Lower Seed": f"#{b} {tb}",
                                        "Higher Seed Win %": sim.matchup_prob(ta, tb, pa, pb) * 100,
                                    })
                            st.dataframe(pd.DataFrame(rows), column_config={"Higher Seed Win %": st.column_config.ProgressColumn("Higher Seed Win %", format="%.0f%%", min_value=0, max_value=100)}, hide_index=True, width="stretch")
                        st.caption(f"Semifinals pair Region {semis[0][0] + 1} with Region {semis[0][1] + 1}, and Region {semis[1][0] + 1} with Region {semis[1][1] + 1}.")

                with st.expander("ℹ️ How the playoff predictor works"):
                    st.markdown(f"""
**Playoff-point average (2026 MHSAA rules)**
- *Base:* win points by opponent division (D1 60, D2 55, D3 50, D4 45, D5 40, D6 35, D7 30, D8 25) divided by games played.
- *Bonus:* for every opponent, a multiplier by its division (D1–2: 6, D3–4: 5, D5–6: 4, D7–8: 3) × that opponent's wins ÷ its games. An opponent's win over you doesn't count, and a school gets no bonus for a game it forfeited.
- This reproduces MHSAA's published averages exactly for {sim.match_rate*100:.0f}% of teams; the rest differ because of unreported or forfeited games in the score data, so each team's current gap to the official number is carried forward.

**Selection and bracket**
- Divisions are set by enrollment before the season. The top {FIELD_PER_DIVISION} by POP in each division qualify (minimum 7 games); there are no automatic qualifiers.
- Each division is split into four geographic Regions of 8 using each school's town location. MHSAA draws the official Regions by hand, so they can differ at the edges.
- Out-of-state opponents get the Michigan division that best reproduces the official averages. League crossover games without a set opponent (e.g. KLAA crossovers) are filled with a nearby school of similar strength.
- Every game is decided with the rating engine's win probability. Higher POP hosts through the Regional final; Semifinals and Ford Field Finals are treated as neutral sites.
""")

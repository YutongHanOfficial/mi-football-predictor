import os
import re
import pandas as pd
import altair as alt
from datetime import datetime, timedelta
from collections import deque
import streamlit as st

from engine import Engine, EngineConfig, is_oos

# ==========================================
# ⚙️ ENGINE CONFIGURATION
# ==========================================
# All model parameters live in engine.EngineConfig and were tuned by
# walk-forward backtesting (see backtest.py). Override any of them here.
ENGINE_CONFIG = EngineConfig()

SEASON_FILES = [("2025", "games_2025.csv"), ("2026", "games_2026.csv")]

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

class SeasonPredictor:
    def __init__(self, season_files, cfg=None):
        season_files = [(s, f) for s, f in season_files if f and os.path.exists(f)]
        self.engine = Engine(season_files, cfg or ENGINE_CONFIG)
        self.cfg = self.engine.cfg
        self.past_season = self.engine.seasons[-2] if len(self.engine.seasons) > 1 else None
        self.current_season = self.engine.seasons[-1]

        self.historical_games = self._games_as_dicts(self.past_season) if self.past_season else []
        self.current_games = self._games_as_dicts(self.current_season)

        self.teams = {t: {} for t in self.engine.teams}
        self._store_ratings(self.engine.current(), "active_")
        if self.past_season:
            self._store_ratings(self.engine._final[self.past_season], "hist_")
        self._store_ratings(self.engine.preseason(self.current_season), "preseason_")

        self._calculate_basic_stats()
        self.backtest_data = self._run_strict_walk_forward()

    def _games_as_dicts(self, season):
        g = self.engine.games[self.engine.games["season"] == season]
        out = []
        for r in g.itertuples(index=False):
            out.append({
                "date": r.date, "date_obj": r.date_obj.to_pydatetime() if pd.notna(r.date_obj) else None,
                "home": r.home, "away": r.away,
                "home_score": int(r.home_score) if r.completed else None,
                "away_score": int(r.away_score) if r.completed else None,
                "is_forfeit": bool(r.is_forfeit),
            })
        return out

    def _team_split(self, ratings):
        """Per-team (offense, -defense) whose difference is the team's contribution to
        the projected margin, so Power = Offense + Defense in the UI tables."""
        power = ratings.team_power()
        adj = power - (ratings.off - ratings.deff)
        return ratings.off + adj / 2, ratings.deff - adj / 2

    def _store_ratings(self, ratings, prefix):
        o, d = self._team_split(ratings)
        for t, i in ratings.index.items():
            self.teams[t][f"{prefix}OSRS"] = float(o[i])
            self.teams[t][f"{prefix}DSRS"] = float(d[i])
            self.teams[t][f"{prefix}games"] = int(ratings.games_played[i])

    def _calc_stats(self, games):
        stats = {t: {"W": 0, "L": 0, "PF": 0, "PA": 0, "GP": 0, "GP_stats": 0} for t in self.teams}
        for g in games:
            if g.get("home_score") not in [None, ""]:
                h, a = g["home"], g["away"]
                hs, as_ = int(g["home_score"]), int(g["away_score"])
                is_forfeit = g.get("is_forfeit", False)

                if h not in stats: stats[h] = {"W": 0, "L": 0, "PF": 0, "PA": 0, "GP": 0, "GP_stats": 0}
                if a not in stats: stats[a] = {"W": 0, "L": 0, "PF": 0, "PA": 0, "GP": 0, "GP_stats": 0}

                stats[h]["GP"] += 1
                stats[a]["GP"] += 1

                if hs > as_:
                    stats[h]["W"] += 1
                    stats[a]["L"] += 1
                elif as_ > hs:
                    stats[a]["W"] += 1
                    stats[h]["L"] += 1

                if not is_forfeit:
                    stats[h]["GP_stats"] += 1
                    stats[a]["GP_stats"] += 1
                    stats[h]["PF"] += hs
                    stats[h]["PA"] += as_
                    stats[a]["PF"] += as_
                    stats[a]["PA"] += hs
        return stats

    def _calc_ranks(self, stats_dict):
        all_teams_stats = []
        for t, s in stats_dict.items():
            gp = max(1, s["GP"])
            gp_stats = max(1, s.get("GP_stats", 0))
            pf = s["PF"]
            pa = s["PA"]
            all_teams_stats.append({
                "team": t,
                "win_pct": s["W"] / gp,
                "pf": pf,
                "pa": pa,
                "diff": pf - pa,
                "ppg": pf / gp_stats if gp_stats > 0 else 0,
                "papg": pa / gp_stats if gp_stats > 0 else 0
            })

        def get_ranks(sort_key, reverse=True):
            in_state = [x for x in all_teams_stats if not is_oos(x["team"])]
            oos = [x for x in all_teams_stats if is_oos(x["team"])]
            in_state_sorted = sorted(in_state, key=lambda x: x[sort_key], reverse=reverse)
            oos_sorted = sorted(oos, key=lambda x: x[sort_key], reverse=reverse)

            ranks = {}
            for i, item in enumerate(in_state_sorted):
                ranks[item["team"]] = f"{i + 1}/{len(in_state_sorted)}"
            for i, item in enumerate(oos_sorted):
                ranks[item["team"]] = f"{i + 1}/{len(oos_sorted)} (OOS)"
            return ranks

        return {
            "win_pct": get_ranks("win_pct", True),
            "pf": get_ranks("pf", True),
            "pa": get_ranks("pa", False),
            "diff": get_ranks("diff", True),
            "ppg": get_ranks("ppg", True),
            "papg": get_ranks("papg", False)
        }

    def _calculate_basic_stats(self):
        self.basic_stats = self._calc_stats(self.current_games)
        self.hist_basic_stats = self._calc_stats(self.historical_games)

        curr_ranks = self._calc_ranks(self.basic_stats)
        self.ranks_win_pct = curr_ranks["win_pct"]
        self.ranks_pf = curr_ranks["pf"]
        self.ranks_pa = curr_ranks["pa"]
        self.ranks_diff = curr_ranks["diff"]
        self.ranks_ppg = curr_ranks["ppg"]
        self.ranks_papg = curr_ranks["papg"]

        hist_ranks = self._calc_ranks(self.hist_basic_stats)
        self.hist_ranks_win_pct = hist_ranks["win_pct"]
        self.hist_ranks_pf = hist_ranks["pf"]
        self.hist_ranks_pa = hist_ranks["pa"]
        self.hist_ranks_diff = hist_ranks["diff"]
        self.hist_ranks_ppg = hist_ranks["ppg"]
        self.hist_ranks_papg = hist_ranks["papg"]

    # ==========================================
    # 🔒 WALK-FORWARD BACKTEST (zero data leakage)
    # ==========================================

    def _run_strict_walk_forward(self):
        df = self.engine.backtest()
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
        for (season, date), day in df.groupby(["season", "date"], sort=True):
            for r in day.itertuples():
                h_rec = records.get((season, r.home), [0, 0])
                a_rec = records.get((season, r.away), [0, 0])
                h_wp = h_rec[0] / max(1, sum(h_rec))
                a_wp = a_rec[0] / max(1, sum(a_rec))
                pick = r.home if h_wp >= a_wp else r.away
                base_rec.append((r.Index, int(pick == r.act_winner)))
            for r in day.itertuples():
                loser = r.away if r.act_winner == r.home else r.home
                records.setdefault((season, r.act_winner), [0, 0])[0] += 1
                records.setdefault((season, loser), [0, 0])[1] += 1
        df["base_rec_correct"] = pd.Series(dict(base_rec))
        return df

    def _find_connection_path(self, team_a, team_b):
        if team_a not in self.teams or team_b not in self.teams:
            return None

        graph = {t: set() for t in self.teams}
        for g in self.historical_games + self.current_games:
            if g.get("home_score") is not None:
                graph[g["home"]].add(g["away"])
                graph[g["away"]].add(g["home"])

        queue = deque([(team_a, [team_a])])
        visited = set([team_a])

        while queue:
            current_team, path = queue.popleft()
            if current_team == team_b:
                return path
            for neighbor in graph.get(current_team, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def predict_matchup(self, away_team, home_team, neutral=False):
        p = self.engine.current().predict(home_team, away_team, neutral=neutral)
        prob_h = p["prob_home"]
        prob_a = 1 - prob_h

        spread_val_raw = round(p["margin"] * 2) / 2
        ou_val = round(p["total"] * 2) / 2

        if spread_val_raw > 0:
            spread_val = -spread_val_raw
            spread_str = f"{home_team} -{spread_val_raw:g}"
        elif spread_val_raw < 0:
            spread_val = abs(spread_val_raw)
            spread_str = f"{away_team} -{abs(spread_val_raw):g}"
        else:
            spread_val = 0
            spread_str = "PK"

        score_h, score_a = round(p["score_home"]), round(p["score_away"])
        # Keep the projected score consistent with the projected winner
        if score_h == score_a:
            if prob_h >= 0.5: score_h += 1
            else: score_a += 1

        return {
            "away_team": away_team,
            "home_team": home_team,
            "prob_a": prob_a,
            "prob_h": prob_h,
            "spread_str": spread_str,
            "spread_val": spread_val,
            "median_total": ou_val,
            "avg_score_a": score_a,
            "avg_score_h": score_h,
            "path": self._find_connection_path(away_team, home_team)
        }

    def _ranked(self, ratings_by_team, team_name):
        pool = [(t, v) for t, v in ratings_by_team.items() if is_oos(t) == is_oos(team_name)]
        pool.sort(key=lambda x: x[1], reverse=True)
        suffix = " (OOS)" if is_oos(team_name) else ""
        rank_num = next((i + 1 for i, v in enumerate(pool) if v[0] == team_name), None)
        return (f"{rank_num}/{len(pool)}{suffix}" if rank_num else "N/A"), rank_num

    def get_team_rating_history(self, team_name):
        history = []
        season = self.current_season
        dates = sorted({g["date_obj"] for g in self.current_games if g.get("home_score") is not None and g.get("date_obj")})
        if not dates: return []

        start_dt, last_game_dt = dates[0], dates[-1]
        current_real_dt = datetime.now()
        end_dt = max(last_game_dt, current_real_dt) if current_real_dt.year == last_game_dt.year else last_game_dt

        # Calculate the Wednesday strictly before the first game
        days_to_wed = (start_dt.weekday() - 2) % 7
        if days_to_wed == 0:
            days_to_wed = 7
        preseason_dt = start_dt - timedelta(days=days_to_wed)

        def snapshot(ratings):
            o, d = self._team_split(ratings)
            power = {t: o[i] - d[i] for t, i in ratings.index.items()}
            i = ratings.index[team_name]
            rank, rank_num = self._ranked(power, team_name)
            return round(power[team_name], 2), round(o[i], 2), round(-d[i], 2), rank, rank_num

        last = snapshot(self.engine.preseason(season))
        history.append({
            "Date": preseason_dt, "Label": f"{preseason_dt.month}/{preseason_dt.day} (Pre)",
            "Power": last[0], "Offense": last[1], "Defense": last[2], "Rank": last[3], "Rank_Num": last[4]
        })

        game_dates = set(dates)
        current_dt = start_dt
        while current_dt <= end_dt:
            if current_dt in game_dates:
                last = snapshot(self.engine.ratings(season, as_of=pd.Timestamp(current_dt) + timedelta(days=1)))
            history.append({
                "Date": current_dt, "Label": f"{current_dt.month}/{current_dt.day}",
                "Power": last[0], "Offense": last[1], "Defense": last[2], "Rank": last[3], "Rank_Num": last[4]
            })
            current_dt += timedelta(days=1)

        return history


# ==========================================
# ⚡ STREAMLIT CACHING WRAPPERS
# ==========================================

def code_version():
    """Changes whenever the model code or data changes, so cached results are rebuilt."""
    parts = []
    for f in ["engine.py", "predictor.py"] + [f for _, f in SEASON_FILES]:
        parts.append(f"{f}:{os.path.getmtime(f) if os.path.exists(f) else 0}:{os.path.getsize(f) if os.path.exists(f) else 0}")
    return "|".join(parts)

@st.cache_resource(max_entries=1)
def load_predictor(version):
    if not any(os.path.exists(f) for _, f in SEASON_FILES):
        return None
    return SeasonPredictor(SEASON_FILES)

@st.cache_data(max_entries=5000)
def get_cached_prediction(_predictor, away_team, home_team, neutral=False, version=None):
    return _predictor.predict_matchup(away_team, home_team, neutral=neutral)

@st.cache_data(max_entries=500)
def get_cached_history(_predictor, team_name, version=None):
    return _predictor.get_team_rating_history(team_name)


def compute_filtered_accuracy_metrics(full_df, scope_selection, skip_early_weeks=False):
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
    st.error("⚠️ No game data found! Please upload `games_2025.csv` or `games_2026.csv` to your repository.")
else:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎮 Matchup Simulator", 
        "🏆 Power Rankings", 
        "📅 Team Schedules", 
        "📆 Upcoming Matchups", 
        "📈 Season Leaderboards",
        "🎯 Model Accuracy"
    ])

    sorted_teams = sorted(predictor.teams.items(), key=lambda x: (x[1].get("active_OSRS", 0) - x[1].get("active_DSRS", 0)), reverse=True)
    in_state_teams = [t for t, _ in sorted_teams if not is_oos(t)]
    oos_teams = [t for t, _ in sorted_teams if is_oos(t)]
    total_in_state, total_oos = len(in_state_teams), len(oos_teams)
    
    def get_rank_display(t_name):
        try:
            return f"{oos_teams.index(t_name) + 1}/{total_oos} (OOS)" if is_oos(t_name) else f"{in_state_teams.index(t_name) + 1}/{total_in_state}"
        except ValueError:
            return "N/A"

    all_teams = sorted(list(predictor.teams.keys()))
    gb_idx = all_teams.index("Grand Blanc") if "Grand Blanc" in all_teams else 0
    dav_idx = all_teams.index("Davison") if "Davison" in all_teams else (1 if len(all_teams) > 1 else 0)

    # ----------------------------------------------------
    # TAB 1: MATCHUP SIMULATOR
    # ----------------------------------------------------
    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### ✈️ Away Team")
            away = st.selectbox("Away Team Select", all_teams, index=gb_idx, label_visibility="collapsed")
            if away:
                a_stats = predictor.basic_stats.get(away, {"W":0, "L":0, "PF":0, "PA":0, "GP":0, "GP_stats":0})
                a_pwr = round(predictor.teams[away].get("active_OSRS",0) - predictor.teams[away].get("active_DSRS",0), 2)
                st.caption(f"🏆 **Rank:** #{get_rank_display(away)} | ⚡ **Power Rating:** {a_pwr}")
                a_gp_stats = max(1, a_stats.get("GP_stats", 0))
                st.caption(f"📊 **Record:** {a_stats['W']}-{a_stats['L']} | 🟢 **PPG:** {a_stats['PF']/a_gp_stats:.1f} | 🔴 **PA/G:** {a_stats['PA']/a_gp_stats:.1f}")

        with col_b:
            st.markdown("### 🏠 Home Team")
            home = st.selectbox("Home Team Select", all_teams, index=dav_idx, label_visibility="collapsed")
            if home:
                h_stats = predictor.basic_stats.get(home, {"W":0, "L":0, "PF":0, "PA":0, "GP":0, "GP_stats":0})
                h_pwr = round(predictor.teams[home].get("active_OSRS",0) - predictor.teams[home].get("active_DSRS",0), 2)
                st.caption(f"🏆 **Rank:** #{get_rank_display(home)} | ⚡ **Power Rating:** {h_pwr}")
                h_gp_stats = max(1, h_stats.get("GP_stats", 0))
                st.caption(f"📊 **Record:** {h_stats['W']}-{h_stats['L']} | 🟢 **PPG:** {h_stats['PF']/h_gp_stats:.1f} | 🔴 **PA/G:** {h_stats['PA']/h_gp_stats:.1f}")

        neutral_site = st.checkbox("Neutral site (no home-field advantage)", value=False)

        st.write("")
        if st.button("🚀 Run Projection", width="stretch", type="primary"):
            if away == home:
                st.warning("Please select two different teams.")
            else:
                res = get_cached_prediction(predictor, away, home, neutral_site, version=VERSION)

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
        col_title, col_filter = st.columns([2, 1])
        with col_title: st.subheader("Power Rankings", anchor=False)
        with col_filter:
            st.write("") 
            view_filter = st.radio("Region Filter", ["Michigan (In-State)", "Out of State (OOS)"], horizontal=True, label_visibility="collapsed")
            
        show_oos = (view_filter == "Out of State (OOS)")
        
        rankings = []
        for t_name, t_data in predictor.teams.items():
            if show_oos != is_oos(t_name): continue
            
            o_rating, d_rating = t_data.get("active_OSRS", 0.0), t_data.get("active_DSRS", 0.0)
            rankings.append({
                "Team": t_name,
                "Power Rating": round(o_rating - d_rating, 2),
                "Offense": round(o_rating, 2),
                "Defense": round(-d_rating, 2),
            })
            
        rankings.sort(key=lambda x: x["Power Rating"], reverse=True)
        for idx, r in enumerate(rankings):
            r["Rank"] = f"{idx + 1}/{len(rankings)}{' (OOS)' if show_oos else ''}"
            
        st.dataframe(rankings, column_order=["Rank", "Team", "Power Rating", "Offense", "Defense"], width="stretch", hide_index=True)

    # ----------------------------------------------------
    # TAB 3: TEAM SCHEDULES & HUB
    # ----------------------------------------------------
    with tab3:
        st.subheader("Team Schedule", anchor=False)
        
        col_hub_team, col_hub_season = st.columns([2, 1])
        with col_hub_team: selected_team = st.selectbox("Select Team Hub:", all_teams, index=gb_idx, key="hub_team_select")
        with col_hub_season:
            st.write("") 
            is_archive = (st.radio("Season View", ["2026 Current", "2025 Archive"], horizontal=True, label_visibility="collapsed") == "2025 Archive")
            
        if selected_team:
            if is_archive:
                hist_sorted_teams = sorted(predictor.teams.items(), key=lambda x: (x[1].get("hist_OSRS", 0) - x[1].get("hist_DSRS", 0)), reverse=True)
                hist_in_state = [t for t, _ in hist_sorted_teams if not is_oos(t)]
                hist_oos = [t for t, _ in hist_sorted_teams if is_oos(t)]
                
                try:
                    team_rank = f"{hist_oos.index(selected_team) + 1}/{len(hist_oos)} (OOS)" if is_oos(selected_team) else f"{hist_in_state.index(selected_team) + 1}/{len(hist_in_state)}"
                except ValueError:
                    team_rank = "N/A"
                    
                t_stats = predictor.teams[selected_team]
                p_rating = round(t_stats.get("hist_OSRS", 0) - t_stats.get("hist_DSRS", 0), 2)
                t_basic = predictor.hist_basic_stats.get(selected_team, {"W":0, "L":0, "PF":0, "PA":0, "GP":0, "GP_stats":0})
                
                r_win_pct, r_ppg, r_papg, r_pf, r_diff = predictor.hist_ranks_win_pct, predictor.hist_ranks_ppg, predictor.hist_ranks_papg, predictor.hist_ranks_pf, predictor.hist_ranks_diff
                target_games, display_year = predictor.historical_games, "2025"
            else:
                team_rank = get_rank_display(selected_team)
                t_stats = predictor.teams[selected_team]
                p_rating = round(t_stats.get("active_OSRS", 0) - t_stats.get("active_DSRS", 0), 2)
                t_basic = predictor.basic_stats.get(selected_team, {"W":0, "L":0, "PF":0, "PA":0, "GP":0, "GP_stats":0})
                
                r_win_pct, r_ppg, r_papg, r_pf, r_diff = predictor.ranks_win_pct, predictor.ranks_ppg, predictor.ranks_papg, predictor.ranks_pf, predictor.ranks_diff
                target_games, display_year = predictor.current_games, "2026"

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
                    else:
                        upcoming_schedule.append({"Date": g.get("date", "-"), "is_home": is_home, "opp": opp, "location_prefix": location_prefix})

            if not is_archive:
                history_data = get_cached_history(predictor, selected_team, version=VERSION)
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
            
            if not is_archive:
                st.markdown("### 🔮 Upcoming Game Projections")
                if upcoming_schedule:
                    for match in upcoming_schedule:
                        away_t = selected_team if not match["is_home"] else match["opp"]
                        home_t = match["opp"] if not match["is_home"] else selected_team
                        
                        proj = get_cached_prediction(predictor, away_t, home_t, version=VERSION)
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
            else:
                st.info("ℹ Season Progression and Future Projections are hidden while viewing archived seasons.")

    # ----------------------------------------------------
    # TAB 4: UPCOMING & TOP MATCHUPS (UN-NESTED & FULL WIDTH)
    # ----------------------------------------------------
    with tab4:
        st.subheader("Upcoming Game Projections", anchor=False)
        st.caption("Displays all unplayed games on the current schedule categorized by quality and match date.")
        
        upcoming_raw = [g for g in predictor.current_games if g.get("home_score") in [None, ""]]
        
        if not upcoming_raw:
            st.info("No upcoming games found on the schedule. All games have recorded scores.")
        else:
            upcoming_processed = []
            
            for g in upcoming_raw:
                h, a = g["home"], g["away"]
                
                # Same projection path as the Team Schedule tab
                proj = get_cached_prediction(predictor, a, h, version=VERSION)
                
                h_off = predictor.teams.get(h, {}).get("active_OSRS", 0.0)
                h_def = predictor.teams.get(h, {}).get("active_DSRS", 0.0)
                a_off = predictor.teams.get(a, {}).get("active_OSRS", 0.0)
                a_def = predictor.teams.get(a, {}).get("active_DSRS", 0.0)
                quality = (h_off - h_def) + (a_off - a_def)
                
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
        col_lb_title, col_lb_filter = st.columns([2, 1])
        with col_lb_title: st.subheader("Season Leaderboards", anchor=False)
        with col_lb_filter:
            st.write("") 
            show_oos_lb = (st.radio("Region Filter", ["Michigan (In-State)", "Out of State (OOS)"], horizontal=True, label_visibility="collapsed", key="lb_filter") == "Out of State (OOS)")
            
        stat_rows = []
        for t in all_teams:
            if show_oos_lb != is_oos(t): continue
                
            s = predictor.basic_stats.get(t, {"W":0, "L":0, "PF":0, "PA":0, "GP":0, "GP_stats":0})
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
        st.subheader("Model Accuracy", anchor=False)
        st.caption("Tests the model's accuracy by predicting past games using only the information that was available right before kickoff. This proves how well the model works in real-world scenarios.")
        
        col_scope_title, col_scope_toggle = st.columns([1, 2])
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

        bd = compute_filtered_accuracy_metrics(predictor.backtest_data, scope_selection, skip_early)
        weeks_note = ", Weeks 4+" if skip_early else ""
        
        if bd is None or bd["total_games"] == 0:
            st.warning(f"No historical prediction data available for the selected scope: **{scope_selection}**.")
        else:
            win_acc = bd["win_acc"] * 100
            
            st.markdown(f"### 🎯 Accuracy · {bd['total_games']} Games ({scope_selection}{weeks_note})")
            
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
            st.markdown(f"### 📅 Weekly Win Accuracy ({scope_selection})")
            
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
                    alt.Tooltip('week_label:N', title='Week'),
                    alt.Tooltip('accuracy:Q', title='Weekly Accuracy', format='.1%'),
                    alt.Tooltip('cum_accuracy:Q', title='Cumulative Acc', format='.1%')
                ]
            ).add_params(hover_acc)
            
            rules_acc = base_chart.mark_rule(color='gray', strokeDash=[3, 3]).encode(
                opacity=alt.condition(hover_acc, alt.value(0.5), alt.value(0))
            )
            
            chart = alt.layer(line_weekly, points_weekly, line_cum, points_cum, rules_acc, selectors_acc).properties(height=350)
            st.altair_chart(chart, width="stretch")
            
            st.markdown(
                """<div style="display: flex; justify-content: center; gap: 20px; font-size: 14px; font-weight: bold; color: var(--text-color); margin-top: -15px;">
                    <div style="display: flex; align-items: center; gap: 5px;"><div style="width: 14px; height: 14px; background-color: #38bdf8; border-radius: 3px;"></div> Weekly Accuracy</div>
                    <div style="display: flex; align-items: center; gap: 5px;"><div style="width: 14px; height: 14px; background-color: #f59e0b; border-radius: 3px;"></div> Cumulative</div>
                </div><br>""", unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"### 📋 Weekly Breakdown Log ({scope_selection})")
            
            display_df = weekly_df[["week_label", "games", "correct", "accuracy", "spread_err", "total_err", "cum_accuracy"]].copy()
            display_df.columns = ["Week", "Games", "Correct", "Accuracy", "Spread Error", "Total Error", "Cumul. Acc"]
            
            # Formatting correction: Convert to literal percentages before rendering
            display_df["Accuracy"] = display_df["Accuracy"] * 100
            display_df["Cumul. Acc"] = display_df["Cumul. Acc"] * 100
            
            st.dataframe(
                display_df,
                column_config={
                    "Week": st.column_config.TextColumn(width="medium"),
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
                st.dataframe(bd["df"])

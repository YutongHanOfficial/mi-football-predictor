import math
import random
import csv
import os
import statistics
import pandas as pd
import altair as alt
from datetime import datetime, timedelta
from collections import deque
import streamlit as st

# ==========================================
# 🧮 MATHEMATICAL ENGINE & SIMULATOR
# ==========================================

def generate_poisson(lam):
    if lam <= 0: return 0
    L = math.exp(-lam)
    k, p = 0, 1.0
    while p > L:
        k += 1
        p *= random.random()
    return k - 1

def generate_football_score(expected_points):
    expected_events = expected_points / 6.0
    num_events = generate_poisson(expected_events)
    score = 0
    for _ in range(num_events):
        roll = random.random()
        if roll < 0.75: score += 7
        elif roll < 0.95: score += 3
        else: score += 6
    return score

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

class SeasonPredictor:
    def __init__(self, past_csv, current_csv=None, regression_factor=0.25, prior_weight=4):
        self.teams = {}
        self.league_avg_points = 24.0
        self.regression_factor = regression_factor 
        self.prior_weight = prior_weight
        
        self.historical_games = self._load_and_dedupe_csv(past_csv)
        if self.historical_games:
            completed_hist = [g for g in self.historical_games if g.get("home_score") not in [None, ""]]
            self._build_srs_model(completed_hist, prefix="hist_")
            self._regress_to_preseason()

        self.current_games = self._load_and_dedupe_csv(current_csv) if current_csv else []
        if self.current_games:
            completed_curr = [g for g in self.current_games if g.get("home_score") not in [None, ""]]
            self._build_srs_model(completed_curr, prefix="curr_")
        
        self._blend_ratings()
        self._calculate_basic_stats()

    def _load_and_dedupe_csv(self, filename):
        games = []
        unique_games = set()
        
        if not filename or not os.path.exists(filename):
            return games
            
        with open(filename, mode='r', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            for row in reader:
                try:
                    date = row.get("date", "").strip()
                    home = row["home"].strip()
                    away = row["away"].strip()
                    hs_raw = row.get("home_score", "").strip()
                    as_raw = row.get("away_score", "").strip()
                    
                    team_a, team_b = sorted([home, away])
                    game_signature = (date, team_a, team_b)
                    
                    if game_signature in unique_games:
                        continue
                    unique_games.add(game_signature)

                    if hs_raw != "" and as_raw != "":
                        hs = int(hs_raw)
                        as_ = int(as_raw)
                        
                        if (hs == 1 and as_ == 0) or (hs == 0 and as_ == 1):
                            continue
                            
                        games.append({
                            "date": date,
                            "home": home, "away": away, 
                            "home_score": hs, "away_score": as_
                        })
                    else:
                        games.append({
                            "date": date,
                            "home": home, "away": away, 
                            "home_score": None, "away_score": None
                        })
                except (KeyError, ValueError):
                    pass
                    
        return games

    def _build_srs_model(self, games, prefix, iterations=100):
        temp_teams = {}
        total_points = 0
        
        for game in games:
            home, away = game["home"], game["away"]
            hs, as_ = game["home_score"], game["away_score"]
            
            for team in (home, away):
                if team not in temp_teams:
                    temp_teams[team] = {"OSRS": 0.0, "DSRS": 0.0, "game_log": []}
            
            temp_teams[home]["game_log"].append({"opponent": away, "points_scored": hs, "points_allowed": as_})
            temp_teams[away]["game_log"].append({"opponent": home, "points_scored": as_, "points_allowed": hs})
            total_points += (hs + as_)
            
        league_avg = total_points / (len(games) * 2) if games else 24.0

        for _ in range(iterations):
            new_ratings = {}
            for team, data in temp_teams.items():
                sum_adj_off = league_avg
                sum_adj_def = league_avg
                num_games = len(data["game_log"]) + 1 
                
                for game in data["game_log"]:
                    opp = game["opponent"]
                    sum_adj_off += (game["points_scored"] - temp_teams[opp]["DSRS"])
                    sum_adj_def += (game["points_allowed"] - temp_teams[opp]["OSRS"])
                
                new_ratings[team] = {
                    "OSRS": (sum_adj_off / num_games) - league_avg,
                    "DSRS": (sum_adj_def / num_games) - league_avg
                }
                
            for team in temp_teams:
                temp_teams[team]["OSRS"] = new_ratings[team]["OSRS"]
                temp_teams[team]["DSRS"] = new_ratings[team]["DSRS"]

        for team, data in temp_teams.items():
            if team not in self.teams: self.teams[team] = {}
            self.teams[team][f"{prefix}OSRS"] = data["OSRS"]
            self.teams[team][f"{prefix}DSRS"] = data["DSRS"]
            self.teams[team][f"{prefix}games"] = len(data["game_log"])
            self.teams[team][f"{prefix}game_log"] = data["game_log"]

        if prefix == "hist_":
            self.league_avg_points = league_avg

    def _regress_to_preseason(self):
        for team in self.teams:
            h_osrs = self.teams[team].get("hist_OSRS", 0.0)
            h_dsrs = self.teams[team].get("hist_DSRS", 0.0)
            self.teams[team]["preseason_OSRS"] = h_osrs * (1 - self.regression_factor)
            self.teams[team]["preseason_DSRS"] = h_dsrs * (1 - self.regression_factor)

    def _blend_ratings(self):
        for team in self.teams:
            pre_osrs = self.teams[team].get("preseason_OSRS", 0.0)
            pre_dsrs = self.teams[team].get("preseason_DSRS", 0.0)
            
            curr_osrs = self.teams[team].get("curr_OSRS", pre_osrs)
            curr_dsrs = self.teams[team].get("curr_DSRS", pre_dsrs)
            curr_games = self.teams[team].get("curr_games", 0)
            
            self.teams[team]["active_OSRS"] = ((self.prior_weight * pre_osrs) + (curr_games * curr_osrs)) / (self.prior_weight + curr_games)
            self.teams[team]["active_DSRS"] = ((self.prior_weight * pre_dsrs) + (curr_games * curr_dsrs)) / (self.prior_weight + curr_games)

    def _calculate_basic_stats(self):
        self.basic_stats = {t: {"W": 0, "L": 0, "PF": 0, "PA": 0, "GP": 0} for t in self.teams}
        for g in self.current_games:
            if g.get("home_score") not in [None, ""]:
                h, a = g["home"], g["away"]
                hs, as_ = int(g["home_score"]), int(g["away_score"])
                
                if h not in self.basic_stats: self.basic_stats[h] = {"W": 0, "L": 0, "PF": 0, "PA": 0, "GP": 0}
                if a not in self.basic_stats: self.basic_stats[a] = {"W": 0, "L": 0, "PF": 0, "PA": 0, "GP": 0}

                self.basic_stats[h]["GP"] += 1
                self.basic_stats[a]["GP"] += 1
                self.basic_stats[h]["PF"] += hs
                self.basic_stats[h]["PA"] += as_
                self.basic_stats[a]["PF"] += as_
                self.basic_stats[a]["PA"] += hs

                if hs > as_:
                    self.basic_stats[h]["W"] += 1
                    self.basic_stats[a]["L"] += 1
                elif as_ > hs:
                    self.basic_stats[a]["W"] += 1
                    self.basic_stats[h]["L"] += 1

    def _find_connection_path(self, team_a, team_b):
        if team_a not in self.teams or team_b not in self.teams: return None
        
        graph = {}
        for team in self.teams:
            graph[team] = set()
            for game in self.teams[team].get("hist_game_log", []): graph[team].add(game["opponent"])
            for game in self.teams[team].get("curr_game_log", []): graph[team].add(game["opponent"])
            
        queue = deque([(team_a, [team_a])])
        visited = set([team_a])
        
        while queue:
            current_team, path = queue.popleft()
            if current_team == team_b: return path 
            for neighbor in graph.get(current_team, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None 

    def predict_matchup(self, away_team, home_team, num_simulations=10000):
        a_off = self.teams[away_team]["active_OSRS"] if away_team in self.teams else 0.0
        a_def = self.teams[away_team]["active_DSRS"] if away_team in self.teams else 0.0
        h_off = self.teams[home_team]["active_OSRS"] if home_team in self.teams else 0.0
        h_def = self.teams[home_team]["active_DSRS"] if home_team in self.teams else 0.0
        
        exp_pts_a = max(0.1, self.league_avg_points + a_off + h_def)
        exp_pts_h = max(0.1, self.league_avg_points + h_off + a_def)
        
        a_wins, h_wins, a_total_pts, h_total_pts = 0, 0, 0, 0
        all_home_margins, all_totals = [], []
        
        for _ in range(num_simulations):
            score_a = generate_football_score(exp_pts_a)
            score_h = generate_football_score(exp_pts_h)
            
            if score_a == score_h:
                if random.random() > 0.5: score_a += 7
                else: score_h += 7
            
            all_home_margins.append(score_h - score_a)
            all_totals.append(score_h + score_a)
            
            if score_a > score_h: a_wins += 1
            else: h_wins += 1
            
            a_total_pts += score_a
            h_total_pts += score_h
                
        prob_a = a_wins / num_simulations
        prob_h = h_wins / num_simulations
        median_home_margin = statistics.median(all_home_margins)
        median_total = statistics.median(all_totals)
        
        if median_home_margin > 0:
            spread_val = -median_home_margin
            spread_str = f"{home_team} -{median_home_margin:g}"
        elif median_home_margin < 0:
            spread_val = abs(median_home_margin)
            spread_str = f"{away_team} -{abs(median_home_margin):g}"
        else:
            spread_val = 0
            spread_str = "PK"
            
        path = self._find_connection_path(away_team, home_team)

        return {
            "away_team": away_team, "home_team": home_team,
            "prob_a": prob_a, "prob_h": prob_h,
            "spread_str": spread_str, "spread_val": spread_val,
            "median_total": median_total,
            "avg_score_a": round(a_total_pts / num_simulations),
            "avg_score_h": round(h_total_pts / num_simulations),
            "path": path
        }
        
    def get_team_rating_history(self, team_name):
        history = []
        
        all_dates = [g["date"] for g in self.current_games if g.get("home_score") not in [None, ""] and g.get("date")]
        if not all_dates:
            return []
            
        start_date_str = min(all_dates)
        end_date_str = max(all_dates)
        
        start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
        last_game_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
        
        current_real_dt = datetime.now()
        end_dt = max(last_game_dt, current_real_dt) if current_real_dt.year == last_game_dt.year else last_game_dt
        
        preseason_dt = start_dt - timedelta(days=1)
        
        pre_ratings = []
        for t in self.teams:
            p_osrs = self.teams[t].get("preseason_OSRS", 0.0)
            p_dsrs = self.teams[t].get("preseason_DSRS", 0.0)
            pre_ratings.append((t, p_osrs - p_dsrs))
        pre_ratings.sort(key=lambda x: x[1], reverse=True)
        preseason_rank = next((i + 1 for i, v in enumerate(pre_ratings) if v[0] == team_name), "N/A")
        
        pre_osrs = self.teams.get(team_name, {}).get("preseason_OSRS", 0.0)
        pre_dsrs_raw = self.teams.get(team_name, {}).get("preseason_DSRS", 0.0)
        
        history.append({
            "Date": preseason_dt, 
            "Label": f"{preseason_dt.month}/{preseason_dt.day} (Pre)", 
            "Power": round(pre_osrs - pre_dsrs_raw, 2),
            "Offense": round(pre_osrs, 2),
            "Defense": round(-pre_dsrs_raw, 2),
            "Rank": preseason_rank
        })
        
        games_by_date = {}
        for g in self.current_games:
            if g.get("home_score") not in [None, ""]:
                d = g["date"]
                if d not in games_by_date: games_by_date[d] = []
                games_by_date[d].append(g)
                
        cumulative_games = []
        current_dt = start_dt
        
        last_power = round(pre_osrs - pre_dsrs_raw, 2)
        last_off = round(pre_osrs, 2)
        last_def = round(-pre_dsrs_raw, 2)
        last_rank = preseason_rank
        
        while current_dt <= end_dt:
            date_str = current_dt.strftime("%Y-%m-%d")
            display_label = f"{current_dt.month}/{current_dt.day}"
            
            if date_str in games_by_date:
                cumulative_games.extend(games_by_date[date_str])
                
                temp_teams = {}
                total_points = 0
                for g in cumulative_games:
                    home, away = g["home"], g["away"]
                    hs, as_ = g["home_score"], g["away_score"]
                    for t in (home, away):
                        if t not in temp_teams:
                            temp_teams[t] = {"OSRS": 0.0, "DSRS": 0.0, "game_log": []}
                    temp_teams[home]["game_log"].append({"opponent": away, "points_scored": hs, "points_allowed": as_})
                    temp_teams[away]["game_log"].append({"opponent": home, "points_scored": as_, "points_allowed": hs})
                    total_points += (hs + as_)
                    
                league_avg = total_points / (len(cumulative_games) * 2) if cumulative_games else 24.0
                
                for _ in range(40): 
                    new_ratings = {}
                    for t, data in temp_teams.items():
                        sum_adj_off = league_avg
                        sum_adj_def = league_avg
                        num_games = len(data["game_log"]) + 1 
                        for game in data["game_log"]:
                            opp = game["opponent"]
                            sum_adj_off += (game["points_scored"] - temp_teams.get(opp, {"DSRS":0})["DSRS"])
                            sum_adj_def += (game["points_allowed"] - temp_teams.get(opp, {"OSRS":0})["OSRS"])
                        new_ratings[t] = {
                            "OSRS": (sum_adj_off / num_games) - league_avg,
                            "DSRS": (sum_adj_def / num_games) - league_avg
                        }
                    for t in temp_teams:
                        temp_teams[t]["OSRS"] = new_ratings[t]["OSRS"]
                        temp_teams[t]["DSRS"] = new_ratings[t]["DSRS"]
                        
                active_ratings = []
                for t in self.teams:
                    t_pre_osrs = self.teams[t].get("preseason_OSRS", 0.0)
                    t_pre_dsrs = self.teams[t].get("preseason_DSRS", 0.0)
                    t_data = temp_teams.get(t, {"OSRS": 0.0, "DSRS": 0.0, "game_log": []})
                    t_act_osrs = ((self.prior_weight * t_pre_osrs) + (len(t_data["game_log"]) * t_data["OSRS"])) / (self.prior_weight + len(t_data["game_log"]))
                    t_act_dsrs = ((self.prior_weight * t_pre_dsrs) + (len(t_data["game_log"]) * t_data["DSRS"])) / (self.prior_weight + len(t_data["game_log"]))
                    
                    t_power = t_act_osrs - t_act_dsrs
                    active_ratings.append((t, t_power))
                    
                    if t == team_name:
                        last_power = round(t_power, 2)
                        last_off = round(t_act_osrs, 2)
                        last_def = round(-t_act_dsrs, 2) 
                        
                active_ratings.sort(key=lambda x: x[1], reverse=True)
                last_rank = next((i + 1 for i, v in enumerate(active_ratings) if v[0] == team_name), "N/A")
                
            history.append({
                "Date": current_dt,
                "Label": display_label,
                "Power": last_power,
                "Offense": last_off,
                "Defense": last_def,
                "Rank": last_rank
            })
            
            current_dt += timedelta(days=1)
            
        return history


# ==========================================
# 🌐 STREAMLIT WEB APP USER INTERFACE
# ==========================================

st.set_page_config(page_title="High School Football Predictor", page_icon="🏈", layout="wide")

st.markdown("""
    <style>
    div[data-testid="stMetricValue"] > div {
        white-space: normal !important;
        word-wrap: break-word !important;
        line-height: 1.2 !important;
        font-size: 1.75rem !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    past_file = "games_2025.csv" if os.path.exists("games_2025.csv") else None
    curr_file = "games_2026.csv" if os.path.exists("games_2026.csv") else None
    if not past_file and not curr_file:
        return None
    return SeasonPredictor(past_file, curr_file, regression_factor=0.25, prior_weight=4)

predictor = load_predictor()

st.title("🏈 High School Football Predictor Engine")

if predictor is None:
    st.error("⚠️ No game data found! Please upload `games_2025.csv` or `games_2026.csv` to your GitHub repository.")
else:
    # Added a new 4th Tab for Season Stats
    tab1, tab2, tab3, tab4 = st.tabs(["🎮 Matchup Simulator", "🏆 Power Rankings", "📅 Team Schedules & Hub", "📈 Season Leaderboards"])

    # Pre-calculate statewide rankings for quick reference in tabs
    sorted_teams = sorted(predictor.teams.items(), key=lambda x: (x[1].get("active_OSRS", 0) - x[1].get("active_DSRS", 0)), reverse=True)
    def get_rank(t_name):
        return next((i + 1 for i, (t, _) in enumerate(sorted_teams) if t == t_name), "N/A")

    # ----------------------------------------------------
    # TAB 1: MATCHUP SIMULATOR (Redesigned)
    # ----------------------------------------------------
    with tab1:
        all_teams = sorted(list(predictor.teams.keys()))
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("### ✈️ Away Team")
            away = st.selectbox("Away Team Select", all_teams, index=0 if all_teams else None, label_visibility="collapsed")
            if away:
                a_stats = predictor.basic_stats.get(away, {"W":0, "L":0, "PF":0, "PA":0, "GP":0})
                a_pwr = round(predictor.teams[away].get("active_OSRS",0) - predictor.teams[away].get("active_DSRS",0), 2)
                a_gp = max(1, a_stats["GP"])
                
                # Dynamic Team Stats Dashboard
                st.caption(f"🏆 **State Rank:** #{get_rank(away)} | ⚡ **Power Rating:** {a_pwr}")
                st.caption(f"📊 **Record:** {a_stats['W']}-{a_stats['L']} | 🟢 **PPG:** {a_stats['PF']/a_gp:.1f} | 🔴 **PA/G:** {a_stats['PA']/a_gp:.1f}")

        with col_b:
            st.markdown("### 🏠 Home Team")
            default_h_idx = 1 if len(all_teams) > 1 else 0
            home = st.selectbox("Home Team Select", all_teams, index=default_h_idx, label_visibility="collapsed")
            if home:
                h_stats = predictor.basic_stats.get(home, {"W":0, "L":0, "PF":0, "PA":0, "GP":0})
                h_pwr = round(predictor.teams[home].get("active_OSRS",0) - predictor.teams[home].get("active_DSRS",0), 2)
                h_gp = max(1, h_stats["GP"])
                
                # Dynamic Team Stats Dashboard
                st.caption(f"🏆 **State Rank:** #{get_rank(home)} | ⚡ **Power Rating:** {h_pwr}")
                st.caption(f"📊 **Record:** {h_stats['W']}-{h_stats['L']} | 🟢 **PPG:** {h_stats['PF']/h_gp:.1f} | 🔴 **PA/G:** {h_stats['PA']/h_gp:.1f}")

        # Moved simulation settings into an expander to clean up the primary view
        with st.expander("⚙️ Advanced Simulation Settings"):
            sims = st.select_slider("Monte Carlo Iterations", options=[1, 10, 100, 1000, 5000, 10000, 50000, 100000], value=10000)

        st.write("") # Add a little vertical breathing room
        if st.button("🚀 Run Vegas Simulation", use_container_width=True, type="primary"):
            if away == home:
                st.warning("Please select two different teams.")
            else:
                res = predictor.predict_matchup(away, home, num_simulations=sims)
                
                if res["path"]:
                    hops = len(res["path"]) - 1
                    st.info(f"**Network Path Found ({hops} hop{'s' if hops > 1 else ''}):** " + " ➔ ".join(res["path"]))

                st.markdown("---")
                
                # Centered, scoreboard-style final output
                st.markdown("<h3 style='text-align: center; color: #a1a1aa;'>📊 Projected Final Score</h3>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='text-align: center; margin-bottom: 30px;'>{away} {res['avg_score_a']} — {res['avg_score_h']} {home}</h1>", unsafe_allow_html=True)
                
                m1, m2, m3, m4 = st.columns([1, 1.5, 1, 1])
                m1.metric(f"{away} Win Prob", f"{res['prob_a']*100:.1f}%", convert_to_moneyline(res['prob_a']))
                m2.metric("True Spread", res["spread_str"])
                m3.metric("Over / Under", f"{res['median_total']:g} pts")
                m4.metric(f"{home} Win Prob", f"{res['prob_h']*100:.1f}%", convert_to_moneyline(res['prob_h']))

    # ----------------------------------------------------
    # TAB 2: POWER RANKINGS
    # ----------------------------------------------------
    with tab2:
        st.subheader("Statewide Power Rankings")
        
        rankings = []
        for t_name, t_data in predictor.teams.items():
            o_rating = t_data.get("active_OSRS", 0.0)
            d_rating = t_data.get("active_DSRS", 0.0)
            net_power = o_rating - d_rating
            
            rankings.append({
                "Team": t_name,
                "Power Rating": round(net_power, 2),
                "Offense": round(o_rating, 2),
                "Defense": round(-d_rating, 2),
            })
            
        rankings.sort(key=lambda x: x["Power Rating"], reverse=True)
        
        for idx, r in enumerate(rankings):
            r["Rank"] = idx + 1
            
        st.dataframe(
            rankings, 
            column_order=["Rank", "Team", "Power Rating", "Offense", "Defense"],
            use_container_width=True, 
            hide_index=True
        )

    # ----------------------------------------------------
    # TAB 3: TEAM SCHEDULES & HUB
    # ----------------------------------------------------
    with tab3:
        st.subheader("Team Schedule & Live Projections")
        
        selected_team = st.selectbox("Select Team Hub:", all_teams, key="hub_team_select")
        
        if selected_team:
            team_rank = get_rank(selected_team)
            t_stats = predictor.teams[selected_team]
            p_rating = round(t_stats.get("active_OSRS", 0) - t_stats.get("active_DSRS", 0), 2)
            
            completed_schedule = []
            upcoming_schedule = []
            wins, losses = 0, 0
            
            for g in predictor.current_games:
                if g["home"] == selected_team or g["away"] == selected_team:
                    is_home = (g["home"] == selected_team)
                    opp = g["away"] if is_home else g["home"]
                    location_prefix = "vs" if is_home else "@"
                    
                    if g.get("home_score") not in [None, ""]:
                        team_score = int(g["home_score"]) if is_home else int(g["away_score"])
                        opp_score = int(g["away_score"]) if is_home else int(g["home_score"])
                        mov = team_score - opp_score
                        result = "W" if mov > 0 else "L"
                        if mov > 0: wins += 1
                        else: losses += 1
                        
                        completed_schedule.append({
                            "Date": g.get("date", "-"),
                            "Opponent": f"{location_prefix} {opp}",
                            "Score": f"{team_score}-{opp_score}",
                            "MOV": f"+{mov}" if mov > 0 else str(mov),
                            "Result": result
                        })
                    else:
                        upcoming_schedule.append({
                            "Date": g.get("date", "-"),
                            "is_home": is_home,
                            "opp": opp,
                            "location_prefix": location_prefix
                        })

            c1, c2, c3 = st.columns(3)
            c1.metric("Power Rating", f"{p_rating}")
            c2.metric("State Rank", f"#{team_rank}")
            c3.metric("2026 Record", f"{wins}-{losses}")
            
            history_data = predictor.get_team_rating_history(selected_team)
            if len(history_data) > 1:
                st.markdown("### 📈 Season Progression")
                
                df_hist = pd.DataFrame(history_data)
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    st.markdown("**Team Ratings over Time**")
                    
                    hover = alt.selection_point(
                        fields=['Date'],
                        nearest=True,
                        on='mouseover',
                        empty=False
                    )
                    
                    base_ratings = alt.Chart(df_hist).encode(
                        x=alt.X('Date:T', axis=alt.Axis(format='%m/%d', labelAngle=0, title=None))
                    )
                    
                    lines_ratings = base_ratings.transform_fold(
                        ['Power', 'Offense', 'Defense'],
                        as_=['Metric', 'Rating']
                    ).mark_line().encode(
                        y=alt.Y('Rating:Q', title=None),
                        color=alt.Color('Metric:N', legend=alt.Legend(orient="bottom", title=None))
                    )
                    
                    selectors_ratings = base_ratings.mark_rule(opacity=0, size=30).encode(
                        tooltip=[
                            alt.Tooltip('Label:N', title='Date'),
                            alt.Tooltip('Power:Q', title='Power'),
                            alt.Tooltip('Offense:Q', title='Offense'),
                            alt.Tooltip('Defense:Q', title='Defense')
                        ]
                    ).add_params(hover)
                    
                    rules_ratings = base_ratings.mark_rule(color='gray', strokeDash=[3, 3]).encode(
                        opacity=alt.condition(hover, alt.value(0.5), alt.value(0))
                    )
                    
                    points_ratings = lines_ratings.mark_point(size=70, filled=True).encode(
                        opacity=alt.condition(hover, alt.value(1), alt.value(0))
                    )
                    
                    st.altair_chart((lines_ratings + rules_ratings + selectors_ratings + points_ratings).interactive(), use_container_width=True)
                    
                with col_chart2:
                    st.markdown("**Statewide Rank (Lower is Better)**")
                    
                    base_rank = alt.Chart(df_hist).encode(
                        x=alt.X('Date:T', axis=alt.Axis(format='%m/%d', labelAngle=0, title=None))
                    )
                    
                    line_rank = base_rank.mark_line(color='#66b3ff').encode(
                        y=alt.Y('Rank:Q', title=None, scale=alt.Scale(reverse=True))
                    )
                    
                    selectors_rank = base_rank.mark_rule(opacity=0, size=30).encode(
                        tooltip=[
                            alt.Tooltip('Label:N', title='Date'),
                            alt.Tooltip('Rank:Q', title='State Rank')
                        ]
                    ).add_params(hover)
                    
                    rules_rank = base_rank.mark_rule(color='gray', strokeDash=[3, 3]).encode(
                        opacity=alt.condition(hover, alt.value(0.5), alt.value(0))
                    )
                    
                    points_rank = line_rank.mark_point(size=70, filled=True, color='#66b3ff').encode(
                        opacity=alt.condition(hover, alt.value(1), alt.value(0))
                    )
                    
                    st.altair_chart((line_rank + rules_rank + selectors_rank + points_rank).interactive(), use_container_width=True)
                
                df_table = df_hist[['Label', 'Power', 'Offense', 'Defense', 'Rank']].rename(columns={'Label': 'Date'})
                st.dataframe(df_table, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            st.markdown("### 📜 2026 Season Schedule")
            if completed_schedule:
                st.dataframe(completed_schedule, use_container_width=True, hide_index=True)
            else:
                st.info("No completed games recorded yet for the 2026 season.")
                
            st.markdown("---")
            
            st.markdown("### 🔮 Upcoming Game Projections")
            if upcoming_schedule:
                for match in upcoming_schedule:
                    away_t = selected_team if not match["is_home"] else match["opp"]
                    home_t = match["opp"] if not match["is_home"] else selected_team
                    
                    proj = predictor.predict_matchup(away_t, home_t, num_simulations=2000)
                    
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
    # TAB 4: SEASON LEADERBOARDS & STATS (New)
    # ----------------------------------------------------
    with tab4:
        st.subheader("Season Leaderboards & Statistical Aggregates")
        
        stat_rows = []
        for t in all_teams:
            s = predictor.basic_stats.get(t, {"W":0, "L":0, "PF":0, "PA":0, "GP":0})
            gp = s["GP"]
            pf = s["PF"]
            pa = s["PA"]
            
            stat_rows.append({
                "Rank": get_rank(t),
                "Team": t,
                "GP": gp,
                "Record": f"{s['W']}-{s['L']}",
                "Win %": round(s['W'] / gp, 3) if gp > 0 else 0.000,
                "PF": pf,
                "PA": pa,
                "Diff": pf - pa,
                "PPG": round(pf / gp, 1) if gp > 0 else 0.0,
                "PA/G": round(pa / gp, 1) if gp > 0 else 0.0
            })
            
        # Ensure the table sorts by state rank by default
        stat_rows.sort(key=lambda x: (isinstance(x["Rank"], str), x["Rank"]))
        
        st.dataframe(
            stat_rows, 
            column_order=["Rank", "Team", "Record", "Win %", "GP", "PF", "PA", "Diff", "PPG", "PA/G"],
            use_container_width=True, 
            hide_index=True
        )

import math
import random
import csv
import os
import statistics
import pandas as pd
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
        
        # 1. Grab all valid dates in the current season
        all_dates = [g["date"] for g in self.current_games if g.get("home_score") not in [None, ""] and g.get("date")]
        if not all_dates:
            return []
            
        start_date_str = min(all_dates)
        end_date_str = max(all_dates)
        
        start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
        
        # Plot Preseason 1 day before the first state-wide game to maintain chronological order
        preseason_dt = start_dt - timedelta(days=1)
        
        pre_ratings = []
        for t in self.teams:
            p_osrs = self.teams[t].get("preseason_OSRS", 0.0)
            p_dsrs = self.teams[t].get("preseason_DSRS", 0.0)
            pre_ratings.append((t, p_osrs - p_dsrs))
        pre_ratings.sort(key=lambda x: x[1], reverse=True)
        preseason_rank = next((i + 1 for i, v in enumerate(pre_ratings) if v[0] == team_name), "N/A")
        
        pre_osrs = self.teams.get(team_name, {}).get("preseason_OSRS", 0.0)
        pre_dsrs = self.teams.get(team_name, {}).get("preseason_DSRS", 0.0)
        
        history.append({
            "Date": preseason_dt,
            "Label": "Preseason",
            "Power": round(pre_osrs - pre_dsrs, 2),
            "Offense": round(pre_osrs, 2),
            "Defense": round(pre_dsrs, 2),
            "Rank": preseason_rank
        })
        
        # Map games by date to optimize the loop
        games_by_date = {}
        for g in self.current_games:
            if g.get("home_score") not in [None, ""]:
                d = g["date"]
                if d not in games_by_date: games_by_date[d] = []
                games_by_date[d].append(g)
                
        cumulative_games = []
        current_dt = start_dt
        
        last_power = round(pre_osrs - pre_dsrs, 2)
        last_off = round(pre_osrs, 2)
        last_def = round(pre_dsrs, 2)
        last_rank = preseason_rank
        
        # Iterate day-by-day
        while current_dt <= end_dt:
            date_str = current_dt.strftime("%Y-%m-%d")
            
            # If games were played anywhere in the state today, recalculate SRS
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
                        last_def = round(t_act_dsrs, 2)
                        
                active_ratings.sort(key=lambda x: x[1], reverse=True)
                last_rank = next((i + 1 for i, v in enumerate(active_ratings) if v[0] == team_name), "N/A")
                
            history.append({
                "Date": current_dt,
                "Label": date_str,
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

# UI Fix for extremely long team names AND hiding Streamlit branding
st.markdown("""
    <style>
    /* 1. Prevent long team names from being cut off */
    div[data-testid="stMetricValue"] > div {
        white-space: normal !important;
        word-wrap: break-word !important;
        line-height: 1.2 !important;
        font-size: 1.75rem !important;
    }
    
    /* 2. Your Forum Snippet: Hides standard Streamlit UI */
    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"],
    div[data-testid="stStatusWidget"],
    #MainMenu,
    header,
    footer {
        visibility: hidden !important;
        height: 0% !important;
        position: fixed !important;
    }

    /* 3. Aggressive wildcard targets for the Cloud Badge */
    iframe[title*="Streamlit Cloud"],
    div[class^="viewerBadge"], 
    div[class^="styles_viewerBadge"],
    div[class*="_profileContainer_"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
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
    tab1, tab2, tab3 = st.tabs(["🎮 Matchup Simulator", "🏆 Power Rankings", "📅 Team Schedules & Hub"])

    # ----------------------------------------------------
    # TAB 1: MATCHUP SIMULATOR
    # ----------------------------------------------------
    with tab1:
        st.subheader("Simulate Any Matchup")
        all_teams = sorted(list(predictor.teams.keys()))
        
        col_a, col_b, col_c = st.columns([2, 2, 1])
        with col_a:
            away = st.selectbox("Away Team", all_teams, index=0 if all_teams else None)
        with col_b:
            default_h_idx = 1 if len(all_teams) > 1 else 0
            home = st.selectbox("Home Team", all_teams, index=default_h_idx)
        with col_c:
            sims = st.select_slider("Simulations", options=[1, 10, 100, 1000, 5000, 10000, 50000, 100000], value=10000)

        if st.button("🚀 Run Vegas Simulation", use_container_width=True):
            if away == home:
                st.warning("Please select two different teams.")
            else:
                res = predictor.predict_matchup(away, home, num_simulations=sims)
                
                if res["path"]:
                    hops = len(res["path"]) - 1
                    st.info(f"**Network Path Found ({hops} hop{'s' if hops > 1 else ''}):** " + " ➔ ".join(res["path"]))

                st.markdown("---")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("True Spread", res["spread_str"])
                m2.metric("Over / Under", f"{res['median_total']:g} pts")
                m3.metric(f"{away} Win Prob", f"{res['prob_a']*100:.1f}%", convert_to_moneyline(res['prob_a']))
                m4.metric(f"{home} Win Prob", f"{res['prob_h']*100:.1f}%", convert_to_moneyline(res['prob_h']))

                st.markdown("---")
                st.subheader("📊 Average Score Projection")
                st.markdown(f"### **{away} {res['avg_score_a']}** — **{res['avg_score_h']} {home}**")

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
                "Offense (OSRS)": round(o_rating, 2),
                "Defense (DSRS)": round(d_rating, 2),
            })
            
        rankings.sort(key=lambda x: x["Power Rating"], reverse=True)
        
        for idx, r in enumerate(rankings):
            r["Rank"] = idx + 1
            
        st.dataframe(
            rankings, 
            column_order=["Rank", "Team", "Power Rating", "Offense (OSRS)", "Defense (DSRS)"],
            use_container_width=True, 
            hide_index=True
        )

    # ----------------------------------------------------
    # TAB 3: TEAM SCHEDULES & HUB
    # ----------------------------------------------------
    with tab3:
        st.subheader("Team Schedule & Live Projections")
        
        all_teams = sorted(list(predictor.teams.keys()))
        selected_team = st.selectbox("Select Team Hub:", all_teams, key="hub_team_select")
        
        if selected_team:
            sorted_teams = sorted(predictor.teams.items(), key=lambda x: (x[1].get("active_OSRS", 0) - x[1].get("active_DSRS", 0)), reverse=True)
            team_rank = next((i + 1 for i, (t, _) in enumerate(sorted_teams) if t == selected_team), "N/A")
            
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
            
            # --- DAILY RATING PROGRESSION GRAPHS ---
            history_data = predictor.get_team_rating_history(selected_team)
            if len(history_data) > 1:
                st.markdown("### 📈 Season Progression")
                
                # Assign true datetime index to fix Streamlit sorting
                df_hist = pd.DataFrame(history_data).set_index("Date")
                
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    st.markdown("**Team Ratings over Time**")
                    st.line_chart(df_hist[["Power", "Offense", "Defense"]], use_container_width=True)
                    
                with col_chart2:
                    st.markdown("**Statewide Rank (Lower is Better)**")
                    st.line_chart(df_hist[["Rank"]], use_container_width=True)
                
                # Expose the table data using the string labels (so "Preseason" prints nicely)
                df_table = df_hist.reset_index()[["Label", "Power", "Offense", "Defense", "Rank"]].rename(columns={"Label": "Date"})
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
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Win Prob", f"{win_p*100:.1f}%")
                        col2.metric("Spread", proj["spread_str"])
                        col3.metric("Over/Under", f"{proj['median_total']:g}")
                        col4.metric("Proj Score", f"{proj_team_pts}-{proj_opp_pts}")
                        st.divider()
            else:
                st.info("No upcoming unplayed games found in the schedule.")

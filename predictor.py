import math
import random
import csv
import os
import statistics
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
    """Converts a win probability (0.0 to 1.0) into major market American Odds."""
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
        self.prior_weight = prior_weight # Treats last year's data as worth '4 games'
        
        # 1. Build the baseline from last year
        self.historical_games = self._load_and_dedupe_csv(past_csv)
        if self.historical_games:
            self._build_srs_model(self.historical_games, prefix="hist_")
            self._regress_to_preseason()

        # 2. Add this year's games and blend them
        self.current_games = self._load_and_dedupe_csv(current_csv) if current_csv else []
        if self.current_games:
            self._build_srs_model(self.current_games, prefix="curr_")
        
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
                    date = row["date"].strip()
                    home = row["home"].strip()
                    away = row["away"].strip()
                    hs = int(row["home_score"])
                    as_ = int(row["away_score"])
                    
                    # --- FILTER OUT FORFEITS (1-0 / 0-1) ---
                    if (hs == 1 and as_ == 0) or (hs == 0 and as_ == 1):
                        continue 
                    # ---------------------------------------
                    
                    team_a, team_b = sorted([home, away])
                    game_signature = (date, team_a, team_b)
                    
                    if game_signature not in unique_games:
                        unique_games.add(game_signature)
                        games.append({
                            "date": date,
                            "home": home, "away": away, 
                            "home_score": hs, "away_score": as_
                        })
                except (KeyError, ValueError):
                    pass # Skip empty or invalid rows
                    
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
        max_a_margin, max_h_margin = 0, 0
        best_a_blowout = f"{away_team} 0, {home_team} 0"
        best_h_blowout = f"{home_team} 0, {away_team} 0"
        
        all_home_margins, all_totals = [], []
        
        for _ in range(num_simulations):
            score_a = generate_football_score(exp_pts_a)
            score_h = generate_football_score(exp_pts_h)
            
            if score_a == score_h:
                if random.random() > 0.5: score_a += 7
                else: score_h += 7
            
            all_home_margins.append(score_h - score_a)
            all_totals.append(score_h + score_a)
                    
            if score_a > score_h:
                score_str = f"{away_team} {score_a}, {home_team} {score_h}"
                a_wins += 1
                if (score_a - score_h) > max_a_margin:
                    max_a_margin = score_a - score_h
                    best_a_blowout = score_str
            else:
                score_str = f"{home_team} {score_h}, {away_team} {score_a}"
                h_wins += 1
                if (score_h - score_a) > max_h_margin:
                    max_h_margin = score_h - score_a
                    best_h_blowout = score_str
            
            a_total_pts += score_a
            h_total_pts += score_h
                
        prob_a = a_wins / num_simulations
        prob_h = h_wins / num_simulations
        median_home_margin = statistics.median(all_home_margins)
        median_total = statistics.median(all_totals)
        
        if median_home_margin > 0:
            spread_str = f"{home_team} -{median_home_margin:g}"
        elif median_home_margin < 0:
            spread_str = f"{away_team} -{abs(median_home_margin):g}"
        else:
            spread_str = "PK (Pick 'em)"
            
        path = self._find_connection_path(away_team, home_team)

        return {
            "away_team": away_team, "home_team": home_team,
            "prob_a": prob_a, "prob_h": prob_h,
            "ml_a": convert_to_moneyline(prob_a), "ml_h": convert_to_moneyline(prob_h),
            "spread_str": spread_str,
            "median_total": median_total,
            "avg_score_a": round(a_total_pts / num_simulations),
            "avg_score_h": round(h_total_pts / num_simulations),
            "best_a_blowout": best_a_blowout, "max_a_margin": max_a_margin,
            "best_h_blowout": best_h_blowout, "max_h_margin": max_h_margin,
            "path": path
        }


# ==========================================
# 🌐 STREAMLIT WEB APP USER INTERFACE
# ==========================================

st.set_page_config(page_title="High School Football Predictor", page_icon="🏈", layout="wide")

@st.cache_resource
def load_predictor():
    past_file = "games_2025.csv" if os.path.exists("games_2025.csv") else None
    curr_file = "games_2026.csv" if os.path.exists("games_2026.csv") else None
    if not past_file and not curr_file:
        return None
    return SeasonPredictor(past_file, curr_file, regression_factor=0.25, prior_weight=4)

predictor = load_predictor()

st.title("🏈 High School Football Predictor Engine")
st.caption("Vegas-style simulation engine using Simple Rating System (SRS) analytics.")

if predictor is None:
    st.error("⚠️ No game data found! Please upload `games_2025.csv` or `games_2026.csv` to your GitHub repository.")
else:
    tab1, tab2, tab3 = st.tabs(["🎮 Matchup Simulator", "🏆 Power Rankings", "📜 Game Logs"])

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
            # Default home team selection to a different index if possible
            default_h_idx = 1 if len(all_teams) > 1 else 0
            home = st.selectbox("Home Team", all_teams, index=default_h_idx)
        with col_c:
            sims = st.select_slider("Simulations", options=[1000, 5000, 10000, 25000], value=10000)

        if st.button("🚀 Run Vegas Simulation", use_container_width=True):
            if away == home:
                st.warning("Please select two different teams.")
            else:
                res = predictor.predict_matchup(away, home, num_simulations=sims)
                
                # Network Connection Breadcrumb
                if res["path"]:
                    hops = len(res["path"]) - 1
                    st.info(f"**Network Path Found ({hops} hop{'s' if hops > 1 else ''}):** " + " ➔ ".join(res["path"]))
                else:
                    st.warning("⚠️ No direct schedule bridge found between these two teams. Projections rely purely on neutral baseline ratings.")

                st.markdown("---")
                
                # Key Vegas Lines Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("True Spread", res["spread_str"])
                m2.metric("Over / Under", f"{res['median_total']:g} pts")
                m3.metric(f"{away} Win Prob / ML", f"{res['prob_a']*100:.1f}%", res['ml_a'])
                m4.metric(f"{home} Win Prob / ML", f"{res['prob_h']*100:.1f}%", res['ml_h'])

                st.markdown("---")
                
                # Score Projection
                col_left, col_right = st.columns(2)
                with col_left:
                    st.subheader("📊 Average Score Projection")
                    st.markdown(f"### **{away} {res['avg_score_a']}** — **{res['avg_score_h']} {home}**")
                
                with col_right:
                    st.subheader("🔥 Extreme Best-Case Outliers")
                    st.write(f"**{away} Max Blowout:** {res['best_a_blowout']} (+{res['max_a_margin']} pts)")
                    st.write(f"**{home} Max Blowout:** {res['best_h_blowout']} (+{res['max_h_margin']} pts)")

    # ----------------------------------------------------
    # TAB 2: POWER RANKINGS
    # ----------------------------------------------------
    with tab2:
        st.subheader("Statewide Power Rankings")
        
        rankings = []
        for t_name, t_data in predictor.teams.items():
            o_rating = t_data.get("active_OSRS", 0.0)
            d_rating = t_data.get("active_DSRS", 0.0)
            net_power = o_rating + d_rating
            games_p = t_data.get("curr_games", 0) + t_data.get("hist_games", 0)
            
            rankings.append({
                "Team": t_name,
                "Power Rating": round(net_power, 2),
                "Offense (OSRS)": round(o_rating, 2),
                "Defense (DSRS)": round(d_rating, 2),
                "Total Games Evaluated": games_p
            })
            
        rankings.sort(key=lambda x: x["Power Rating"], reverse=True)
        
        search_query = st.text_input("🔍 Search for a team:", "")
        if search_query:
            rankings = [r for r in rankings if search_query.lower() in r["Team"].lower()]
            
        st.dataframe(rankings, use_container_width=True, hide_index=False)

    # ----------------------------------------------------
    # TAB 3: GAME LOGS
    # ----------------------------------------------------
    with tab3:
        st.subheader("Loaded Game History")
        all_logs = predictor.historical_games + predictor.current_games
        if all_logs:
            st.write(f"Displaying **{len(all_logs)}** verified games from CSV records:")
            st.dataframe(all_logs, use_container_width=True)
        else:
            st.write("No active games loaded.")

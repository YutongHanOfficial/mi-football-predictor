import math
import random
import csv
import os
import re
import statistics
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
from collections import deque
import streamlit as st

# ==========================================
# ⚙️ ENGINE CONFIGURATION
# ==========================================
ENABLE_MOV_ADJUSTMENT = True

# Select your mathematical curve: "tanh", "log", "piecewise", or "soft_piecewise"
MOV_METHOD = "soft_piecewise" 

# The threshold where diminishing returns begin (35 = Michigan HS Running Clock)
MAX_MARGIN = 35.0  

ENABLE_BAYESIAN_ANCHORS = True
PRIOR_WEIGHT_GAMES = 4.0  # How strongly to pull teams toward their prior (4 is standard)
HOME_FIELD_ADVANTAGE = 2.0  # Point advantage given to home teams
TIME_DECAY_PER_WEEK = 0.975 # Weight multiplier per week elapsed

# MHSAA Empirical Baselines (Point values relative to state average)
DIVISION_BASELINES = {
    1: 14.0, 
    2: 10.0, 
    3: 6.0, 
    4: 2.0, 
    5: -2.0, 
    6: -6.0, 
    7: -10.0, 
    8: -14.0
}

# 🏛️ GRAVITY PILLARS
TEAM_DIVISIONS = {
    "Detroit Catholic Central": 1, 
    "Detroit Cass Tech": 1,
    "Clarkston": 1,
    "East Kentwood": 1,
    "Saline": 1,
    "Grand Blanc": 1,
    "Davison": 1,
    "Portage Central": 1,
    "Oxford": 1,
    "Howell": 1,
    "Hudsonville": 1,
    "Rochester Hills Stoney Creek": 1,
    "Romeo": 1,
    "Belleville": 1,
    "Hartland": 1,
    "Grandville": 1,
    "Orchard Lake St Mary's": 2,
    "Muskegon Mona Shores": 2,
    "South Lyon": 2,
    "Byron Center": 2,
    "Gibraltar Carlson": 2,
    "Caledonia": 2,
    "Bloomfield Hills Brother Rice": 2,
    "Milford": 2,
    "Sterling Heights": 2,
    "South Lyon East": 2,
    "Harper Woods": 3,
    "DeWitt": 3,
    "Chelsea": 3,
    "Detroit Martin Luther King": 3,
    "Coopersville": 3,
    "Zeeland West": 3,
    "Mount Pleasant": 3,
    "St Joseph": 3,
    "Lowell": 3,
    "Port Huron": 3,
    "Dearborn Divine Child": 4,
    "Goodrich": 4,
    "Hudsonville Unity Christian": 4,
    "Portland": 4,
    "Edwardsburg": 4,
    "Paw Paw": 4,
    "Grand Rapids Kenowa Hills": 4,
    "Flint Southwestern": 4,
    "Ogemaw Heights": 5,
    "Frankenmuth": 5,
    "Grand Rapids Catholic Central": 5,
    "Hopkins": 5,
    "Bridgeport": 5,
    "Jackson Lumen Christi": 6,
    "Kingsley": 6,
    "Clinton": 6,
    "Almont": 6,
    "Ecorse": 6,
    "Menominee": 7,
    "Pewamo-Westphalia": 7,
    "Ithaca": 7,
    "Millington": 7,
    "Beal City": 8,
    "Hudson": 8,
    "Detroit Douglass": 8,
    "Madison Heights Madison": 8,
    "Allen Park Cabrini": 8,
}

# ==========================================
# 🧮 HELPER FUNCTIONS
# ==========================================

def is_oos(team_name):
    if not isinstance(team_name, str): 
        return False
    match = re.search(r'\(([A-Za-z]{2,4})\)$', team_name.strip())
    if match:
        state = match.group(1).upper()
        if state != "MI": 
            return True
    return False

def apply_blowout_diminishing_returns(home_score, away_score, mov_method=MOV_METHOD, max_margin=MAX_MARGIN):
    if not ENABLE_MOV_ADJUSTMENT:
        return home_score, away_score
        
    margin = home_score - away_score
    raw_mov = abs(margin)
    
    if raw_mov == 0:
        return home_score, away_score
        
    if mov_method == "tanh":
        adj_mov = max_margin * math.tanh(raw_mov / max_margin)
    elif mov_method == "log":
        adj_mov = 3.365 * math.log(raw_mov + 1)
    elif mov_method == "piecewise":
        if raw_mov <= max_margin:
            adj_mov = raw_mov
        else:
            adj_mov = max_margin + (math.sqrt(raw_mov - max_margin) * 2.0)
    elif mov_method == "soft_piecewise":
        if raw_mov <= max_margin:
            adj_mov = raw_mov
        else:
            adj_mov = max_margin + ((raw_mov - max_margin) * 0.5)
    else:
        adj_mov = raw_mov
        
    if margin > 0:
        return away_score + adj_mov, away_score
    else:
        return home_score, home_score + adj_mov

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

def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def metric_card(title, value, theme="green"):
    if theme == "green":
        bg = "#ecfdf5"
        border = "#a7f3d0"
        text = "#059669"
    elif theme == "orange":
        bg = "#fffbeb"
        border = "#fde68a"
        text = "#d97706"
    else:
        bg = "#f3f4f6"
        border = "#e5e7eb"
        text = "#4b5563"
    return f"""
    <div style="background-color: {bg}; border: 1px solid {border}; padding: 15px; border-radius: 8px; text-align: center; margin-bottom: 15px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
        <h2 style="color: {text}; margin: 0; font-size: 26px; font-weight: 800;">{value}</h2>
        <p style="color: {text}; margin: 0; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">{title}</p>
    </div>
    """

# ==========================================
# 🧮 MATHEMATICAL ENGINE & SIMULATOR
# ==========================================

class SeasonPredictor:
    def __init__(self, past_csv, current_csv=None, regression_factor=0.25):
        self.teams = {}
        self.league_avg_points = 24.0
        self.regression_factor = regression_factor 
        
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
        
        # Runs the strict walk-forward backtest (zero data leakage)
        self.backtest_data = self._run_strict_walk_forward()

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
                    
                    date_obj = None
                    if date:
                        try:
                            date_obj = datetime.strptime(date, "%Y-%m-%d")
                        except ValueError:
                            pass

                    if hs_raw != "" and as_raw != "":
                        hs = int(hs_raw)
                        as_ = int(as_raw)
                        is_forfeit = (hs == 1 and as_ == 0) or (hs == 0 and as_ == 1)
                        games.append({"date": date, "date_obj": date_obj, "home": home, "away": away, "home_score": hs, "away_score": as_, "is_forfeit": is_forfeit})
                    else:
                        games.append({"date": date, "date_obj": date_obj, "home": home, "away": away, "home_score": None, "away_score": None, "is_forfeit": False})
                except (KeyError, ValueError):
                    pass
                    
        return games

    def _build_srs_model(self, games, prefix, iterations=100):
        temp_teams = {}
        total_points = 0
        valid_games_count = 0
        
        target_date = datetime.now() if prefix == "curr_" else max([g["date_obj"] for g in games if g.get("date_obj")], default=datetime.now())
        
        for game in games:
            home, away = game["home"], game["away"]
            for team in (home, away):
                if team not in temp_teams:
                    temp_teams[team] = {"OSRS": 0.0, "DSRS": 0.0, "game_log": []}
            
            if game.get("is_forfeit"): continue

            hs, as_ = game["home_score"], game["away_score"]
            adj_hs, adj_as = apply_blowout_diminishing_returns(hs, as_)
            
            # Mathematical Upgrade: Time Decay
            days_ago = (target_date - game["date_obj"]).days if game.get("date_obj") else 0
            game_weight = max(0.1, TIME_DECAY_PER_WEEK ** (days_ago / 7.0))
            
            # Mathematical Upgrade: Neutralize Home Field Advantage
            neut_hs = adj_hs - (HOME_FIELD_ADVANTAGE / 2.0)
            neut_as = adj_as + (HOME_FIELD_ADVANTAGE / 2.0)
            
            temp_teams[home]["game_log"].append({"opponent": away, "points_scored": neut_hs, "points_allowed": neut_as, "weight": game_weight})
            temp_teams[away]["game_log"].append({"opponent": home, "points_scored": neut_as, "points_allowed": neut_hs, "weight": game_weight})
            total_points += (adj_hs + adj_as)
            valid_games_count += 1
            
        league_avg = total_points / (valid_games_count * 2) if valid_games_count else 24.0

        for team in temp_teams:
            if prefix == "hist_":
                div = TEAM_DIVISIONS.get(team, 4)
                prior_power = DIVISION_BASELINES.get(div, 0.0)
                prior_osrs = prior_power / 2.0
                prior_dsrs = -prior_power / 2.0
            else:
                if team in self.teams and "preseason_OSRS" in self.teams[team]:
                    prior_osrs = self.teams[team]["preseason_OSRS"]
                    prior_dsrs = self.teams[team]["preseason_DSRS"]
                else:
                    div = TEAM_DIVISIONS.get(team, 4)
                    prior_power = DIVISION_BASELINES.get(div, 0.0)
                    prior_osrs = prior_power / 2.0
                    prior_dsrs = -prior_power / 2.0
            
            temp_teams[team]["prior_OSRS"] = prior_osrs
            temp_teams[team]["prior_DSRS"] = prior_dsrs

        for _ in range(iterations):
            new_ratings = {}
            for team, data in temp_teams.items():
                sum_weights = sum(g["weight"] for g in data["game_log"])
                weight_prior = PRIOR_WEIGHT_GAMES if ENABLE_BAYESIAN_ANCHORS else 0.0
                total_games = sum_weights + weight_prior
                
                sum_adj_off = (temp_teams[team]["prior_OSRS"] + league_avg) * weight_prior
                sum_adj_def = (temp_teams[team]["prior_DSRS"] + league_avg) * weight_prior
                
                for game in data["game_log"]:
                    opp = game["opponent"]
                    sum_adj_off += (game["points_scored"] - temp_teams.get(opp, {"DSRS": 0.0})["DSRS"]) * game["weight"]
                    sum_adj_def += (game["points_allowed"] - temp_teams.get(opp, {"OSRS": 0.0})["OSRS"]) * game["weight"]
                
                new_ratings[team] = {
                    "OSRS": (sum_adj_off / total_games) - league_avg if total_games > 0 else 0.0,
                    "DSRS": (sum_adj_def / total_games) - league_avg if total_games > 0 else 0.0
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
            div = TEAM_DIVISIONS.get(team, 4)
            div_prior = DIVISION_BASELINES.get(div, 0.0)
            off_prior = div_prior / 2.0
            def_prior = -div_prior / 2.0
            
            self.teams[team]["preseason_OSRS"] = (h_osrs * (1 - self.regression_factor)) + (off_prior * self.regression_factor)
            self.teams[team]["preseason_DSRS"] = (h_dsrs * (1 - self.regression_factor)) + (def_prior * self.regression_factor)

    def _blend_ratings(self):
        for team in self.teams:
            curr_games = self.teams[team].get("curr_games", 0)
            if curr_games > 0:
                self.teams[team]["active_OSRS"] = self.teams[team].get("curr_OSRS", 0.0)
                self.teams[team]["active_DSRS"] = self.teams[team].get("curr_DSRS", 0.0)
            else:
                self.teams[team]["active_OSRS"] = self.teams[team].get("preseason_OSRS", 0.0)
                self.teams[team]["active_DSRS"] = self.teams[team].get("preseason_DSRS", 0.0)

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
    # 🔒 SECURE WALK-FORWARD ENGINE
    # ==========================================
    
    def _calculate_point_in_time_ratings(self, training_games, is_hist, target_date_obj):
        """Calculates temporary, sealed ratings using ONLY games played strictly before prediction day."""
        temp_teams = {}
        total_points = 0
        valid_games_count = 0
        
        for t in self.teams.keys():
            if is_hist: 
                p = DIVISION_BASELINES.get(TEAM_DIVISIONS.get(t, 4), 0.0)
                temp_teams[t] = {"prior_OSRS": p/2.0, "prior_DSRS": -p/2.0, "OSRS": p/2.0, "DSRS": -p/2.0, "game_log": []}
            else: 
                prior_o = self.teams.get(t, {}).get("preseason_OSRS", 0.0)
                prior_d = self.teams.get(t, {}).get("preseason_DSRS", 0.0)
                temp_teams[t] = {
                    "prior_OSRS": prior_o, 
                    "prior_DSRS": prior_d, 
                    "OSRS": prior_o,
                    "DSRS": prior_d,
                    "game_log": []
                }
                
        for g in training_games:
            h, a = g["home"], g["away"]
            if h not in temp_teams: temp_teams[h] = {"prior_OSRS": 0.0, "prior_DSRS": 0.0, "OSRS": 0.0, "DSRS": 0.0, "game_log": []}
            if a not in temp_teams: temp_teams[a] = {"prior_OSRS": 0.0, "prior_DSRS": 0.0, "OSRS": 0.0, "DSRS": 0.0, "game_log": []}
            
            adj_hs, adj_as = apply_blowout_diminishing_returns(g["home_score"], g["away_score"])
            
            days_ago = (target_date_obj - g["date_obj"]).days if g.get("date_obj") else 0
            game_weight = max(0.1, TIME_DECAY_PER_WEEK ** (days_ago / 7.0))
            
            neut_hs = adj_hs - (HOME_FIELD_ADVANTAGE / 2.0)
            neut_as = adj_as + (HOME_FIELD_ADVANTAGE / 2.0)
            
            temp_teams[h]["game_log"].append({"opponent": a, "points_scored": neut_hs, "points_allowed": neut_as, "weight": game_weight})
            temp_teams[a]["game_log"].append({"opponent": h, "points_scored": neut_as, "points_allowed": neut_hs, "weight": game_weight})
            total_points += (adj_hs + adj_as)
            valid_games_count += 1
            
        league_avg = total_points / (valid_games_count * 2) if valid_games_count else 24.0

        for _ in range(30):
            new_ratings = {}
            for t, data in temp_teams.items():
                sum_weights = sum(g["weight"] for g in data["game_log"])
                weight_prior = PRIOR_WEIGHT_GAMES if ENABLE_BAYESIAN_ANCHORS else 0.0
                total_games = sum_weights + weight_prior
                
                sum_adj_off = (data["prior_OSRS"] + league_avg) * weight_prior
                sum_adj_def = (data["prior_DSRS"] + league_avg) * weight_prior
                
                for g in data["game_log"]:
                    opp = g["opponent"]
                    sum_adj_off += (g["points_scored"] - temp_teams.get(opp, {"DSRS": 0.0})["DSRS"]) * g["weight"]
                    sum_adj_def += (g["points_allowed"] - temp_teams.get(opp, {"OSRS": 0.0})["OSRS"]) * g["weight"]
                    
                new_ratings[t] = {
                    "OSRS": (sum_adj_off / total_games) - league_avg if total_games > 0 else data["prior_OSRS"],
                    "DSRS": (sum_adj_def / total_games) - league_avg if total_games > 0 else data["prior_DSRS"]
                }
            for t in temp_teams: temp_teams[t].update(new_ratings[t])
            
        return temp_teams, league_avg

    def _run_strict_walk_forward(self):
        results = []
        for season, games, is_hist in [("2025", self.historical_games, True), ("2026", self.current_games, False)]:
            if not games: continue
            
            # Sort chronologically and extract valid scored games
            valid = []
            for g in games:
                if g.get("home_score") not in [None, ""] and not g.get("is_forfeit") and g.get("date_obj"):
                    try:
                        week_num = g["date_obj"].isocalendar()[1]
                        valid.append({**g, "week_num": week_num})
                    except ValueError: pass
                    
            valid.sort(key=lambda x: x["date_obj"])
            
            # Batch strictly by date
            games_by_date = {}
            for g in valid: games_by_date.setdefault(g["date"], []).append(g)
                
            training_history = []
            point_in_time_records = {} # {team: [wins, losses]}
            
            for date_str in sorted(games_by_date.keys()):
                todays_games = games_by_date[date_str]
                target_date_obj = todays_games[0]["date_obj"]
                
                # A. Train SRS purely on history BEFORE today
                pit_ratings, pit_league_avg = self._calculate_point_in_time_ratings(training_history, is_hist, target_date_obj)
                
                # B. Generate out-of-sample predictions
                for g in todays_games:
                    h, a = g["home"], g["away"]
                    act_h, act_a = int(g["home_score"]), int(g["away_score"])
                    act_margin, act_total = act_h - act_a, act_h + act_a
                    act_winner = h if act_margin > 0 else (a if act_margin < 0 else "Tie")
                    if act_winner == "Tie": continue
                    
                    # Point-in-Time OSRS/DSRS
                    h_off = pit_ratings.get(h, {}).get("OSRS", pit_ratings.get(h, {}).get("prior_OSRS", 0.0))
                    h_def = pit_ratings.get(h, {}).get("DSRS", pit_ratings.get(h, {}).get("prior_DSRS", 0.0))
                    a_off = pit_ratings.get(a, {}).get("OSRS", pit_ratings.get(a, {}).get("prior_OSRS", 0.0))
                    a_def = pit_ratings.get(a, {}).get("DSRS", pit_ratings.get(a, {}).get("prior_DSRS", 0.0))
                    
                    exp_h = max(0.1, pit_league_avg + h_off + a_def + (HOME_FIELD_ADVANTAGE / 2.0))
                    exp_a = max(0.1, pit_league_avg + a_off + h_def - (HOME_FIELD_ADVANTAGE / 2.0))
                    pred_margin = exp_h - exp_a
                    pred_winner = h if pred_margin > 0 else a
                    
                    # Mathematical Upgrade: Volatility curve flattened to 16.5
                    prob_h_win = norm_cdf(pred_margin / 16.5)
                    confidence = prob_h_win if pred_winner == h else (1 - prob_h_win)
                    
                    # Point-In-Time Records Baseline
                    h_rec, a_rec = point_in_time_records.get(h, [0,0]), point_in_time_records.get(a, [0,0])
                    h_wp = h_rec[0]/max(1, h_rec[0]+h_rec[1])
                    a_wp = a_rec[0]/max(1, a_rec[0]+a_rec[1])
                    base_rec_winner = h if h_wp >= a_wp else a
                    
                    results.append({
                        "season": season, "date": date_str, "week_num": g["week_num"],
                        "home": h, "away": a,
                        "pred_margin": pred_margin, "pred_total": exp_h + exp_a, 
                        "act_margin": act_margin, "act_total": act_total,
                        "pred_winner": pred_winner, "act_winner": act_winner, 
                        "prob_h_win": prob_h_win, "confidence": confidence,
                        "correct": 1 if pred_winner == act_winner else 0,
                        "base_home_correct": 1 if act_winner == h else 0,
                        "base_rec_correct": 1 if base_rec_winner == act_winner else 0,
                        "spread_err": abs(act_margin - pred_margin),
                        "total_err": abs(act_total - (exp_h + exp_a)),
                        "brier_score": ( (1 if act_winner == h else 0) - prob_h_win ) ** 2,
                        "home_rating_before": round(h_off - h_def, 2),
                        "away_rating_before": round(a_off - a_def, 2)
                    })
                    
                # C. Reveal results and update training history for tomorrow
                training_history.extend(todays_games)
                for g in todays_games:
                    gw = g["home"] if int(g["home_score"]) > int(g["away_score"]) else g["away"]
                    gl = g["away"] if gw == g["home"] else g["home"]
                    if gw not in point_in_time_records: point_in_time_records[gw] = [0,0]
                    if gl not in point_in_time_records: point_in_time_records[gl] = [0,0]
                    point_in_time_records[gw][0] += 1
                    point_in_time_records[gl][1] += 1

        df = pd.DataFrame(results)
        if df.empty: return None

        # Build Calibration Table
        df['conf_bin'] = pd.cut(df['confidence'], bins=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
        calib_table = df.groupby('conf_bin', observed=False).agg(
            Count=('correct', 'count'),
            Pred_Prob=('confidence', 'mean'),
            Act_Win_Pct=('correct', 'mean')
        ).reset_index()
        calib_table['Calib_Error'] = calib_table['Pred_Prob'] - calib_table['Act_Win_Pct']

        # Weekly Grouping
        df = df.sort_values(["season", "week_num"])
        min_weeks = df.groupby("season")["week_num"].min().to_dict()
        df["rel_week"] = df.apply(lambda row: row["week_num"] - min_weeks[row["season"]] + 1, axis=1)
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

    def _find_connection_path(self, team_a, team_b):
        if team_a not in self.teams or team_b not in self.teams: 
            return None
        
        graph = {}
        for team in self.teams:
            graph[team] = set()
            for game in self.teams[team].get("hist_game_log", []): 
                graph[team].add(game["opponent"])
            for game in self.teams[team].get("curr_game_log", []): 
                graph[team].add(game["opponent"])
            
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

    def predict_matchup(self, away_team, home_team, num_simulations=10000, mode="median"):
        a_off = self.teams[away_team]["active_OSRS"] if away_team in self.teams else 0.0
        a_def = self.teams[away_team]["active_DSRS"] if away_team in self.teams else 0.0
        h_off = self.teams[home_team]["active_OSRS"] if home_team in self.teams else 0.0
        h_def = self.teams[home_team]["active_DSRS"] if home_team in self.teams else 0.0
        
        # Mathematical Upgrade: Neutralize Home Field Advantage
        exp_pts_a = max(0.1, self.league_avg_points + a_off + h_def - (HOME_FIELD_ADVANTAGE / 2.0))
        exp_pts_h = max(0.1, self.league_avg_points + h_off + a_def + (HOME_FIELD_ADVANTAGE / 2.0))
        
        a_wins, h_wins = 0, 0
        all_score_a, all_score_h = [], []
        all_home_margins, all_totals = [], []
        
        for _ in range(num_simulations):
            score_a = generate_football_score(exp_pts_a)
            score_h = generate_football_score(exp_pts_h)
            
            if score_a == score_h:
                if random.random() > 0.5: score_a += 7
                else: score_h += 7
            
            all_score_a.append(score_a)
            all_score_h.append(score_h)
            all_home_margins.append(score_h - score_a)
            all_totals.append(score_h + score_a)
            
            if score_a > score_h: a_wins += 1
            else: h_wins += 1
                
        prob_a = a_wins / num_simulations
        prob_h = h_wins / num_simulations

        if mode == "median":
            final_score_a = statistics.median(all_score_a)
            final_score_h = statistics.median(all_score_h)
            calc_margin = statistics.median(all_home_margins)
            calc_total = statistics.median(all_totals)
        else:
            final_score_a = sum(all_score_a) / num_simulations
            final_score_h = sum(all_score_h) / num_simulations
            calc_margin = final_score_h - final_score_a
            calc_total = final_score_h + final_score_a
        
        spread_val_raw = round(calc_margin * 2) / 2
        ou_val = round(calc_total * 2) / 2

        if spread_val_raw > 0:
            spread_val = -spread_val_raw
            spread_str = f"{home_team} -{spread_val_raw:g}"
        elif spread_val_raw < 0:
            spread_val = abs(spread_val_raw)
            spread_str = f"{away_team} -{abs(spread_val_raw):g}"
        else:
            spread_val = 0
            spread_str = "PK"
            
        path = self._find_connection_path(away_team, home_team)

        return {
            "away_team": away_team, 
            "home_team": home_team,
            "prob_a": prob_a, 
            "prob_h": prob_h,
            "spread_str": spread_str, 
            "spread_val": spread_val,
            "median_total": ou_val,
            "avg_score_a": round(final_score_a),
            "avg_score_h": round(final_score_h),
            "path": path
        }
        
    def get_team_rating_history(self, team_name):
        history = []
        
        all_dates = [g["date"] for g in self.current_games if g.get("home_score") not in [None, ""] and g.get("date")]
        if not all_dates: return []
            
        start_date_str, end_date_str = min(all_dates), max(all_dates)
        start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
        last_game_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
        
        current_real_dt = datetime.now()
        end_dt = max(last_game_dt, current_real_dt) if current_real_dt.year == last_game_dt.year else last_game_dt
        preseason_dt = start_dt - timedelta(days=1)
        
        pre_in_state, pre_oos = [], []
        for t in self.teams:
            p_osrs = self.teams[t].get("preseason_OSRS", 0.0)
            p_dsrs = self.teams[t].get("preseason_DSRS", 0.0)
            if is_oos(t): pre_oos.append((t, p_osrs - p_dsrs))
            else: pre_in_state.append((t, p_osrs - p_dsrs))
                
        pre_in_state.sort(key=lambda x: x[1], reverse=True)
        pre_oos.sort(key=lambda x: x[1], reverse=True)
        
        if is_oos(team_name):
            target_list, suffix = pre_oos, " (OOS)"
        else:
            target_list, suffix = pre_in_state, ""
            
        total_pool = len(target_list)
        rank_num = next((i + 1 for i, v in enumerate(target_list) if v[0] == team_name), "N/A")
        preseason_rank = f"{rank_num}/{total_pool}{suffix}" if rank_num != "N/A" else "N/A"
        
        pre_osrs = self.teams.get(team_name, {}).get("preseason_OSRS", 0.0)
        pre_dsrs_raw = self.teams.get(team_name, {}).get("preseason_DSRS", 0.0)
        
        history.append({
            "Date": preseason_dt, 
            "Label": f"{preseason_dt.month}/{preseason_dt.day} (Pre)", 
            "Power": round(pre_osrs - pre_dsrs_raw, 2),
            "Offense": round(pre_osrs, 2),
            "Defense": round(-pre_dsrs_raw, 2),
            "Rank": preseason_rank,
            "Rank_Num": rank_num if rank_num != "N/A" else None
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
        last_off, last_def = round(pre_osrs, 2), round(-pre_dsrs_raw, 2)
        last_rank = preseason_rank
        
        while current_dt <= end_dt:
            date_str = current_dt.strftime("%Y-%m-%d")
            display_label = f"{current_dt.month}/{current_dt.day}"
            
            if date_str in games_by_date:
                cumulative_games.extend(games_by_date[date_str])
                
                temp_teams = {}
                total_points = 0
                valid_games_count = 0
                
                for g in cumulative_games:
                    home, away = g["home"], g["away"]
                    for t in (home, away):
                        if t not in temp_teams:
                            # State fix applied here for OSRS/DSRS
                            temp_teams[t] = {"prior_OSRS": 0.0, "prior_DSRS": 0.0, "OSRS": 0.0, "DSRS": 0.0, "game_log": []}
                    
                    if g.get("is_forfeit"): continue
                        
                    hs, as_ = g["home_score"], g["away_score"]
                    adj_hs, adj_as = apply_blowout_diminishing_returns(hs, as_)
                    
                    days_ago = (current_dt - g["date_obj"]).days if g.get("date_obj") else 0
                    game_weight = max(0.1, TIME_DECAY_PER_WEEK ** (days_ago / 7.0))
                    
                    neut_hs = adj_hs - (HOME_FIELD_ADVANTAGE / 2.0)
                    neut_as = adj_as + (HOME_FIELD_ADVANTAGE / 2.0)
                    
                    temp_teams[home]["game_log"].append({"opponent": away, "points_scored": neut_hs, "points_allowed": neut_as, "weight": game_weight})
                    temp_teams[away]["game_log"].append({"opponent": home, "points_scored": neut_as, "points_allowed": neut_hs, "weight": game_weight})
                    total_points += (adj_hs + adj_as)
                    valid_games_count += 1
                    
                league_avg = total_points / (valid_games_count * 2) if valid_games_count else 24.0
                
                for t in temp_teams:
                    pr_o = self.teams.get(t, {}).get("preseason_OSRS", 0.0)
                    pr_d = self.teams.get(t, {}).get("preseason_DSRS", 0.0)
                    temp_teams[t]["prior_OSRS"] = pr_o
                    temp_teams[t]["prior_DSRS"] = pr_d
                    temp_teams[t]["OSRS"] = pr_o
                    temp_teams[t]["DSRS"] = pr_d

                for _ in range(40): 
                    new_ratings = {}
                    for t, data in temp_teams.items():
                        sum_weights = sum(gm["weight"] for gm in data["game_log"])
                        weight_prior = PRIOR_WEIGHT_GAMES if ENABLE_BAYESIAN_ANCHORS else 0.0
                        total_games = sum_weights + weight_prior
                        
                        sum_adj_off = (temp_teams[t]["prior_OSRS"] + league_avg) * weight_prior
                        sum_adj_def = (temp_teams[t]["prior_DSRS"] + league_avg) * weight_prior
                        
                        for game in data["game_log"]:
                            opp = game["opponent"]
                            sum_adj_off += (game["points_scored"] - temp_teams.get(opp, {"DSRS":0})["DSRS"]) * game["weight"]
                            sum_adj_def += (game["points_allowed"] - temp_teams.get(opp, {"OSRS":0})["OSRS"]) * game["weight"]
                            
                        new_ratings[t] = {
                            "OSRS": (sum_adj_off / total_games) - league_avg if total_games > 0 else temp_teams[t]["prior_OSRS"],
                            "DSRS": (sum_adj_def / total_games) - league_avg if total_games > 0 else temp_teams[t]["prior_DSRS"]
                        }
                        
                    for t in temp_teams:
                        temp_teams[t]["OSRS"] = new_ratings[t]["OSRS"]
                        temp_teams[t]["DSRS"] = new_ratings[t]["DSRS"]
                        
                act_in_state, act_oos = [], []
                for t in self.teams:
                    t_data = temp_teams.get(t, {"OSRS": self.teams[t].get("preseason_OSRS", 0.0), "DSRS": self.teams[t].get("preseason_DSRS", 0.0)})
                    t_power = t_data["OSRS"] - t_data["DSRS"]
                    if is_oos(t): act_oos.append((t, t_power))
                    else: act_in_state.append((t, t_power))
                    
                    if t == team_name:
                        last_power = round(t_power, 2)
                        last_off, last_def = round(t_data["OSRS"], 2), round(-t_data["DSRS"], 2) 
                        
                act_in_state.sort(key=lambda x: x[1], reverse=True)
                act_oos.sort(key=lambda x: x[1], reverse=True)
                
                target_list, suffix = (act_oos, " (OOS)") if is_oos(team_name) else (act_in_state, "")
                total_pool = len(target_list)
                rank_num = next((i + 1 for i, v in enumerate(target_list) if v[0] == team_name), "N/A")
                last_rank = f"{rank_num}/{total_pool}{suffix}" if rank_num != "N/A" else "N/A"
                
            history.append({
                "Date": current_dt,
                "Label": display_label,
                "Power": last_power,
                "Offense": last_off,
                "Defense": last_def,
                "Rank": last_rank,
                "Rank_Num": rank_num if rank_num != "N/A" else None
            })
            
            current_dt += timedelta(days=1)
            
        return history


# ==========================================
# ⚡ STREAMLIT CACHING WRAPPERS 
# ==========================================

@st.cache_resource
def load_predictor():
    past_file = "games_2025.csv" if os.path.exists("games_2025.csv") else None
    curr_file = "games_2026.csv" if os.path.exists("games_2026.csv") else None
    if not past_file and not curr_file:
        return None
    return SeasonPredictor(past_file, curr_file, regression_factor=0.25)

@st.cache_data
def get_cached_prediction(_predictor, away_team, home_team, num_simulations, mode="median"):
    return _predictor.predict_matchup(away_team, home_team, num_simulations=num_simulations, mode=mode)

@st.cache_data
def get_cached_history(_predictor, team_name):
    return _predictor.get_team_rating_history(team_name)


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
    .stMarkdown a.header-anchor,
    .stMarkdown a.anchor,
    h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {
        display: none !important;
        pointer-events: none !important;
    }
    </style>
""", unsafe_allow_html=True)

predictor = load_predictor()

st.title("🏈 High School Football Predictor Engine", anchor=False)

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

        with st.expander("⚙️ Advanced Simulation Settings"):
            sims = st.select_slider("Monte Carlo Iterations", options=[1, 10, 100, 1000, 5000, 10000, 50000, 100000], value=10000)
            sim_mode_ui = st.radio("Projection Math Method", ["Median (Recommended)", "Mean (Average)"], horizontal=True)
            sim_mode = "median" if "Median" in sim_mode_ui else "mean"

        st.write("") 
        if st.button("🚀 Run Simulation", use_container_width=True, type="primary"):
            if away == home:
                st.warning("Please select two different teams.")
            else:
                if sims >= 10000:
                    res = get_cached_prediction(predictor, away, home, sims, sim_mode)
                    st.caption(f"🔒 *Displaying stable, cached projection ({sim_mode.title()} Mode).*")
                else:
                    res = predictor.predict_matchup(away, home, num_simulations=sims, mode=sim_mode)
                    st.caption(f"🎲 *Live simulation complete ({sim_mode.title()} Mode). Expect variance at lower iteration counts!*")
                
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
    # TAB 2: POWER RANKINGS
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
                history_data = get_cached_history(predictor, selected_team)
                if len(history_data) > 1:
                    st.markdown("### 📈 Season Progression")
                    df_hist = pd.DataFrame(history_data)
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
                        st.altair_chart((lines_ratings + rules_ratings + selectors_ratings + points_ratings), use_container_width=True)
                        
                    with col_chart2:
                        st.markdown("**Rank**")
                        line_rank = base_ratings.mark_line(color='#66b3ff').encode(y=alt.Y('Rank_Num:Q', title=None, scale=alt.Scale(reverse=True)))
                        selectors_rank = base_ratings.mark_rule(opacity=0, size=30).encode(
                            tooltip=[alt.Tooltip('Label:N', title='Date'), alt.Tooltip('Rank:N', title='Rank')]
                        ).add_params(hover)
                        rules_rank = base_ratings.mark_rule(color='gray', strokeDash=[3, 3]).encode(opacity=alt.condition(hover, alt.value(0.5), alt.value(0)))
                        points_rank = line_rank.mark_point(size=70, filled=True, color='#66b3ff').encode(opacity=alt.condition(hover, alt.value(1), alt.value(0)))
                        st.altair_chart((line_rank + rules_rank + selectors_rank + points_rank), use_container_width=True)
                    
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
                        
                        proj = get_cached_prediction(predictor, away_t, home_t, 2000, "median")
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
    # TAB 4: UPCOMING & TOP MATCHUPS
    # ----------------------------------------------------
    with tab4:
        st.subheader("Upcoming Game Projections", anchor=False)
        st.caption("Displays all unplayed games on the current schedule categorized by date and quality.")
        
        upcoming_raw = [g for g in predictor.current_games if g.get("home_score") in [None, ""]]
        
        if not upcoming_raw:
            st.info("No upcoming games found on the schedule. All games have recorded scores.")
        else:
            upcoming_processed = []
            
            for g in upcoming_raw:
                h, a = g["home"], g["away"]
                h_off = predictor.teams.get(h, {}).get("active_OSRS", 0.0)
                h_def = predictor.teams.get(h, {}).get("active_DSRS", 0.0)
                a_off = predictor.teams.get(a, {}).get("active_OSRS", 0.0)
                a_def = predictor.teams.get(a, {}).get("active_DSRS", 0.0)
                
                exp_h = predictor.league_avg_points + h_off + a_def + (HOME_FIELD_ADVANTAGE / 2.0)
                exp_a = predictor.league_avg_points + a_off + h_def - (HOME_FIELD_ADVANTAGE / 2.0)
                
                margin = exp_h - exp_a
                abs_margin = abs(margin)
                quality = (h_off - h_def) + (a_off - a_def)
                
                if margin > 0:
                    spread_str = f"{h} -{round(margin*2)/2:g}"
                elif margin < 0:
                    spread_str = f"{a} -{round(abs_margin*2)/2:g}"
                else:
                    spread_str = "PK"

                upcoming_processed.append({
                    "Date": g.get("date", "Unknown"),
                    "Away": a,
                    "Home": h,
                    "Spread": spread_str,
                    "Total": round((exp_h + exp_a)*2)/2,
                    "Proj Score": f"{max(0, round(exp_a))}-{max(0, round(exp_h))}",
                    "_abs_margin": abs_margin,
                    "_quality": quality
                })

            col_best1, col_best2 = st.columns(2)
            
            with col_best1:
                st.markdown("### 🔥 Top Tier Matchups")
                st.caption("Highest combined power ratings")
                top_quality = sorted(upcoming_processed, key=lambda x: x["_quality"], reverse=True)[:10]
                st.dataframe(pd.DataFrame(top_quality).drop(columns=["_abs_margin", "_quality"]), hide_index=True, width="stretch")
                
            with col_best2:
                st.markdown("### ⚔️ Closest Projections")
                st.caption("Tightest mathematical spreads")
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
    # TAB 6: MODEL ACCURACY AUDIT
    # ----------------------------------------------------
    with tab6:
        st.subheader("Model Accuracy Audit", anchor=False)
        st.caption("Validates predictive power using a strict chronological walk-forward system. Every game is predicted using ONLY data available before kickoff.")
        
        bd = predictor.backtest_data
        
        if bd is None or bd["total_games"] == 0:
            st.warning("Not enough scored historical data to generate accuracy backtest.")
        else:
            win_acc = bd["win_acc"] * 100
            
            st.markdown(f"<div style='margin-bottom: 25px; font-weight: bold; font-size: 14px; color: #78350f;'>ALL-TIME OUT-OF-SAMPLE · {bd['total_games']} GAMES RATED</div>", unsafe_allow_html=True)
            
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1: st.markdown(metric_card("WALK-FORWARD ACC", f"{win_acc:.1f}%", "green"), unsafe_allow_html=True)
            with c2: st.markdown(metric_card("CORRECT PICKS", f"{bd['total_correct']}/{bd['total_games']}", "green"), unsafe_allow_html=True)
            with c3: st.markdown(metric_card("SPREAD MAE", f"{bd['avg_spread_err']:.1f}", "orange"), unsafe_allow_html=True)
            with c4: st.markdown(metric_card("TOTAL SCORE MAE", f"{bd['avg_total_err']:.1f}", "orange"), unsafe_allow_html=True)
            with c5: st.markdown(metric_card("BRIER SCORE", f"{bd['brier_score']:.3f}", "gray"), unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 📈 Baseline Comparisons")
            st.caption("Measures model value added against simple naive algorithms.")
            b1, b2, b3 = st.columns(3)
            b1.metric("Always Pick Home Team", f"{bd['base_home_acc']*100:.1f}%")
            b2.metric("Always Pick Better Record", f"{bd['base_rec_acc']*100:.1f}%")
            b3.metric("Your SRS Engine", f"{win_acc:.1f}%")

            st.markdown("---")
            st.markdown("### 📊 Probability Calibration Table")
            st.caption("Verifies if predicted confidence matches actual real-world win rates.")
            
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
            st.markdown(f"<div style='margin-bottom: 10px; font-weight: bold; font-size: 14px; color: #78350f;'>ALL YEARS — WIN ACCURACY BY WEEK</div>", unsafe_allow_html=True)
            
            weekly_df = bd["weekly_df"]
            base_chart = alt.Chart(weekly_df).encode(
                x=alt.X('week_label:N', sort=None, title=None, axis=alt.Axis(labelAngle=-45))
            )
            
            line_weekly = base_chart.mark_line(color='#38bdf8', size=2, point=alt.OverlayMarkDef(color='#38bdf8', filled=True, size=50)).encode(
                y=alt.Y('accuracy:Q', scale=alt.Scale(domain=[0.5, 1.0]), axis=alt.Axis(format='%', title=None))
            )
            
            line_cum = base_chart.mark_line(color='#f59e0b', size=2, point=alt.OverlayMarkDef(color='#f59e0b', filled=True, size=50)).encode(
                y=alt.Y('cum_accuracy:Q')
            )
            
            chart = alt.layer(line_weekly, line_cum).properties(height=350)
            st.altair_chart(chart, use_container_width=True)
            
            st.markdown(
                """<div style="display: flex; justify-content: center; gap: 20px; font-size: 14px; font-weight: bold; color: #4b5563; margin-top: -15px;">
                    <div style="display: flex; align-items: center; gap: 5px;"><div style="width: 14px; height: 14px; background-color: #38bdf8; border-radius: 3px;"></div> Weekly Accuracy</div>
                    <div style="display: flex; align-items: center; gap: 5px;"><div style="width: 14px; height: 14px; background-color: #f59e0b; border-radius: 3px;"></div> Cumulative</div>
                </div><br>""", unsafe_allow_html=True
            )

            st.markdown(f"<div style='margin-bottom: 15px; margin-top: 25px; font-weight: bold; font-size: 14px; color: #78350f;'>ALL YEARS — WEEKLY BREAKDOWN</div>", unsafe_allow_html=True)
            
            display_df = weekly_df[["week_label", "games", "correct", "accuracy", "spread_err", "total_err", "cum_accuracy"]].copy()
            display_df.columns = ["WEEK", "GAMES", "CORRECT", "ACCURACY", "SPREAD ERR", "TOTAL ERR", "CUMUL. ACC"]
            
            st.dataframe(
                display_df,
                column_config={
                    "WEEK": st.column_config.TextColumn(width="medium"),
                    "GAMES": st.column_config.NumberColumn(width="small"),
                    "CORRECT": st.column_config.NumberColumn(width="small"),
                    "ACCURACY": st.column_config.NumberColumn(format="%.1f%%", width="small"),
                    "SPREAD ERR": st.column_config.NumberColumn(format="%.1f pts", width="small"),
                    "TOTAL ERR": st.column_config.NumberColumn(format="%.1f pts", width="small"),
                    "CUMUL. ACC": st.column_config.NumberColumn(format="%.1f%%", width="small")
                },
                hide_index=True,
                width="stretch"
            )

            with st.expander("🔍 View Raw Out-Of-Sample Predictions Log"):
                st.dataframe(bd["df"])

    # ----------------------------------------------------
    # ADMIN TOOLS
    # ----------------------------------------------------
    st.markdown("---")
    with st.expander("⚙️ Admin & Developer Tools"):
        st.caption("Streamlit Cloud hides the native 'Clear Cache' menu for users who aren't logged in as the app author. Use this button to manually refresh data if needed.")
        if st.button("🧹 Force Clear Cache", use_container_width=True):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()

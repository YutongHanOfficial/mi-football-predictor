"""
Rating engine for the high school football predictor.

Two Bayesian ridge models are solved jointly over the whole schedule graph:

  * Margin model   : compressed_margin = HFA + R_home - R_away
  * Points model   : pts_home = MU + HFA_p/2 + O_home + D_away
                     pts_away = MU - HFA_p/2 + O_away + D_home

Every team rating is shrunk toward a prior (last season's rating, regressed
toward the mean). The prior strength is a number of "pseudo-games", so once a
team has played enough real games the data dominates. This replaces the old
hard-coded division baselines: a team's prior comes from how it actually
performed, not from the division it plays in.

Predictions from both models are blended, then calibrated (spread scale and
win-probability sigma) using parameters tuned by walk-forward backtesting.
"""

import math
import re
from dataclasses import dataclass, field, asdict
from datetime import timedelta

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class EngineConfig:
    # Margin compression: margins past CAP count at SLOPE (running clock / garbage time)
    mov_cap: float = 99.0
    mov_slope: float = 0.25
    # Weekly time decay applied to games within a season
    decay_per_week: float = 0.98
    # Prior strength in pseudo-games, for teams with/without last-season data
    prior_games_known: float = 0.5
    prior_games_new: float = 0.1
    # Fraction of last season's rating carried into the new preseason prior
    carryover: float = 0.8
    # Prior strength for the points (offense/defense) model
    pts_prior_games_known: float = 5.0
    pts_prior_games_new: float = 1.0
    pts_carryover: float = 0.85
    pts_cap: float = 49.0
    # How much of the final margin comes from the margin model (vs points model)
    margin_blend: float = 0.9
    # Multiply the compressed-margin prediction to project raw margins
    spread_scale: float = 1.0
    # Std. dev. of actual margin around predicted margin (for win probability)
    win_sigma: float = 18.0
    # Totals calibration: total = 2*MU + TOTAL_SHIFT + TOTAL_SCALE * (matchup deviation)
    total_shift: float = 2.0
    total_scale: float = 0.6
    # Prior mean for teams never seen before (MI vs. out-of-state)
    new_team_prior: float = -15.0
    new_oos_prior: float = 6.0

    def to_dict(self):
        return asdict(self)


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

STATE_RE = re.compile(r"\(([A-Za-z]{2,4})\)$")


def is_oos(team_name):
    if not isinstance(team_name, str):
        return False
    m = STATE_RE.search(team_name.strip())
    return bool(m) and m.group(1).upper() != "MI"


def load_games(path, season):
    """Load a season CSV into a DataFrame, deduplicating repeated rows."""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    for c in ("date", "home", "away", "home_score", "away_score"):
        df[c] = df[c].str.strip()
    key = df.apply(lambda r: (r["date"],) + tuple(sorted([r["home"], r["away"]])), axis=1)
    df = df[~key.duplicated()].copy()
    df["season"] = str(season)
    df["date_obj"] = pd.to_datetime(df["date"], errors="coerce")
    df["completed"] = (df["home_score"] != "") & (df["away_score"] != "")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    hs, as_ = df["home_score"], df["away_score"]
    df["is_forfeit"] = df["completed"] & (((hs == 1) & (as_ == 0)) | ((hs == 0) & (as_ == 1)))
    return df.reset_index(drop=True)


# ------------------------------------------------------------------
# Core model
# ------------------------------------------------------------------

def compress_margin(m, cap, slope):
    a = np.abs(m)
    return np.sign(m) * np.where(a <= cap, a, cap + (a - cap) * slope)


@dataclass
class Ratings:
    teams: list
    index: dict
    power: np.ndarray      # margin-model rating (compressed-margin points vs average team)
    off: np.ndarray        # points scored above average
    deff: np.ndarray       # points allowed above average (lower is better)
    hfa: float
    mu: float
    hfa_pts: float
    games_played: np.ndarray
    cfg: EngineConfig = field(repr=False, default=None)

    def predict(self, home, away, neutral=False, unknown_prior=0.0):
        """Return dict with expected margin (home - away), total, scores and win prob."""
        c = self.cfg
        ih, ia = self.index.get(home), self.index.get(away)
        r_h = self.power[ih] if ih is not None else unknown_prior
        r_a = self.power[ia] if ia is not None else unknown_prior
        o_h = self.off[ih] if ih is not None else 0.0
        d_h = self.deff[ih] if ih is not None else 0.0
        o_a = self.off[ia] if ia is not None else 0.0
        d_a = self.deff[ia] if ia is not None else 0.0
        hfa = 0.0 if neutral else self.hfa
        hfa_p = 0.0 if neutral else self.hfa_pts

        m_margin = (hfa + r_h - r_a) * c.spread_scale
        exp_h = self.mu + hfa_p / 2 + o_h + d_a
        exp_a = self.mu - hfa_p / 2 + o_a + d_h
        p_margin = exp_h - exp_a
        margin = c.margin_blend * m_margin + (1 - c.margin_blend) * p_margin
        total = max(7.0, 2 * self.mu + c.total_shift + c.total_scale * (exp_h + exp_a - 2 * self.mu))
        # Split the total so the projected score agrees with the blended margin
        score_h = max(0.0, (total + margin) / 2)
        score_a = max(0.0, (total - margin) / 2)
        p_home = 0.5 * (1 + math.erf(margin / (c.win_sigma * math.sqrt(2))))
        return {
            "margin": margin,
            "total": score_h + score_a,
            "score_home": score_h,
            "score_away": score_a,
            "prob_home": p_home,
        }


def _ridge(X, y, w, prior_prec, prior_mean):
    """Solve min sum w(y - Xb)^2 + sum prec_j (b_j - mean_j)^2 with sparse matrices."""
    Xt_w = X.T.multiply(w).tocsr()
    A = (Xt_w @ X + sp.diags(prior_prec)).tocsc()
    rhs = Xt_w @ y + prior_prec * prior_mean
    return spla.spsolve(A, rhs)


def _solve_margin(nT, hi, ai, y, w, prior, lam):
    """Ridge: min sum w(y - hfa - r_h + r_a)^2 + sum lam_i (r_i - prior_i)^2."""
    n_g = len(hi)
    rows = np.repeat(np.arange(n_g), 3)
    cols = np.column_stack([hi, ai, np.full(n_g, nT)]).ravel()
    vals = np.tile([1.0, -1.0, 1.0], n_g)
    X = sp.csr_matrix((vals, (rows, cols)), shape=(n_g, nT + 1))
    # Weak prior on HFA (~2.5 pts) so it is stable early in the season
    sol = _ridge(X, y, w, np.append(lam, 20.0), np.append(prior, 2.5))
    return sol[:nT], sol[nT]


def _solve_points(nT, hi, ai, ph, pa, w, prior_o, prior_d, lam):
    """Ridge on points scored. Unknowns: O (nT), D (nT), MU, HFA_p."""
    n_g = len(hi)
    iMU, iH = 2 * nT, 2 * nT + 1
    # One row per team-game: pts = MU + O_scorer + D_opponent +/- HFA_p/2
    off_idx = np.concatenate([hi, ai])
    def_idx = nT + np.concatenate([ai, hi])
    side = np.concatenate([np.full(n_g, 0.5), np.full(n_g, -0.5)])
    n_r = 2 * n_g
    rows = np.repeat(np.arange(n_r), 4)
    cols = np.column_stack([off_idx, def_idx, np.full(n_r, iMU), np.full(n_r, iH)]).ravel()
    vals = np.column_stack([np.ones(n_r), np.ones(n_r), np.ones(n_r), side]).ravel()
    X = sp.csr_matrix((vals, (rows, cols)), shape=(n_r, 2 * nT + 2))
    prec = np.concatenate([lam, lam, [1.0, 20.0]])
    mean = np.concatenate([prior_o, prior_d, [25.0, 2.5]])
    sol = _ridge(X, np.concatenate([ph, pa]), np.concatenate([w, w]), prec, mean)
    return sol[:nT], sol[nT:2 * nT], sol[iMU], sol[iH]


def fit_ratings(teams, index, games, as_of, prior_power, prior_off, prior_def,
                has_history, cfg: EngineConfig):
    """Fit ratings using completed games strictly before `as_of`."""
    nT = len(teams)
    lam_m = np.where(has_history, cfg.prior_games_known, cfg.prior_games_new)
    lam_p = np.where(has_history, cfg.pts_prior_games_known, cfg.pts_prior_games_new)
    g = games
    if len(g):
        hi = g["home"].map(index).to_numpy()
        ai = g["away"].map(index).to_numpy()
        hs = g["home_score"].to_numpy(float)
        as_ = g["away_score"].to_numpy(float)
        weeks_ago = (as_of - g["date_obj"]).dt.days.to_numpy() / 7.0
        w = np.power(cfg.decay_per_week, weeks_ago)
        y = compress_margin(hs - as_, cfg.mov_cap, cfg.mov_slope)
        power, hfa = _solve_margin(nT, hi, ai, y, w, prior_power, lam_m)
        off, deff, mu, hfa_p = _solve_points(
            nT, hi, ai, np.minimum(hs, cfg.pts_cap), np.minimum(as_, cfg.pts_cap),
            w, prior_off, prior_def, lam_p)
        gp = np.bincount(hi, minlength=nT) + np.bincount(ai, minlength=nT)
    else:
        power, hfa = prior_power.copy(), 2.5
        off, deff, mu, hfa_p = prior_off.copy(), prior_def.copy(), 25.0, 2.5
        gp = np.zeros(nT, dtype=int)
    return Ratings(list(teams), index, power, off, deff, hfa, mu, hfa_p, gp, cfg)


class Engine:
    """Holds every season's games and produces point-in-time ratings."""

    def __init__(self, season_files, cfg: EngineConfig = None):
        self.cfg = cfg or EngineConfig()
        frames = [load_games(path, season) for season, path in season_files]
        self.games = pd.concat(frames, ignore_index=True)
        self.seasons = [str(s) for s, _ in season_files]
        names = pd.unique(pd.concat([self.games["home"], self.games["away"]]))
        self.teams = sorted(names)
        self.index = {t: i for i, t in enumerate(self.teams)}
        self.oos_mask = np.array([is_oos(t) for t in self.teams])
        self._final = {}
        self._priors = {}
        for s in self.seasons:
            self._priors[s] = self._make_priors(s)
            self._final[s] = self.ratings(s)

    def season_games(self, season, completed=True):
        g = self.games[self.games["season"] == str(season)]
        if completed:
            g = g[g["completed"] & ~g["is_forfeit"]]
        return g

    def _make_priors(self, season):
        c = self.cfg
        nT = len(self.teams)
        new_prior = np.where(self.oos_mask, c.new_oos_prior, c.new_team_prior)
        i = self.seasons.index(season)
        if i == 0:
            return new_prior.copy(), np.zeros(nT), np.zeros(nT), np.zeros(nT, dtype=bool)
        prev = self._final[self.seasons[i - 1]]
        played = prev.games_played > 0
        power = np.where(played, c.carryover * prev.power, new_prior)
        off = np.where(played, c.pts_carryover * prev.off, 0.0)
        deff = np.where(played, c.pts_carryover * prev.deff, 0.0)
        return power, off, deff, played

    def ratings(self, season, as_of=None):
        """Ratings for `season` using games before `as_of` (default: all completed)."""
        season = str(season)
        g = self.season_games(season)
        if as_of is None:
            as_of = g["date_obj"].max() + timedelta(days=1) if len(g) else pd.Timestamp.now()
        g = g[g["date_obj"] < as_of]
        pp, po, pd_, hist = self._priors[season]
        return fit_ratings(self.teams, self.index, g, as_of, pp, po, pd_, hist, self.cfg)

    def preseason(self, season):
        pp, po, pd_, hist = self._priors[str(season)]
        return Ratings(list(self.teams), self.index, pp, po, pd_, 2.5, 25.0, 2.5,
                       np.zeros(len(self.teams), dtype=int), self.cfg)

    def current(self):
        return self._final[self.seasons[-1]]

    def backtest(self):
        """Strict walk-forward: each game day is predicted using only earlier games."""
        rows = []
        for s in self.seasons:
            g = self.season_games(s)
            for date, day in g.groupby("date_obj", sort=True):
                r = self.ratings(s, as_of=date)
                for _, row in day.iterrows():
                    h, a = row["home"], row["away"]
                    hs, as_ = row["home_score"], row["away_score"]
                    if hs == as_:
                        continue
                    p = r.predict(h, a)
                    act_m, act_t = hs - as_, hs + as_
                    h_o, a_o = is_oos(h), is_oos(a)
                    scope = ("In-State Only" if not (h_o or a_o)
                             else "Michigan vs. Out-of-State" if h_o != a_o else "OOS Only")
                    pred_winner = h if p["margin"] > 0 else a
                    act_winner = h if act_m > 0 else a
                    conf = p["prob_home"] if pred_winner == h else 1 - p["prob_home"]
                    rows.append({
                        "season": s, "date": row["date"], "date_obj": date,
                        "home": h, "away": a, "matchup_scope": scope,
                        "pred_margin": p["margin"], "pred_total": p["total"],
                        "act_margin": act_m, "act_total": act_t,
                        "pred_winner": pred_winner, "act_winner": act_winner,
                        "prob_h_win": p["prob_home"], "confidence": conf,
                        "correct": int(pred_winner == act_winner),
                        "spread_err": abs(act_m - p["margin"]),
                        "total_err": abs(act_t - p["total"]),
                        "brier_score": (int(act_m > 0) - p["prob_home"]) ** 2,
                        "home_rating_before": r.power[r.index[h]],
                        "away_rating_before": r.power[r.index[a]],
                    })
        return pd.DataFrame(rows)


def summarize(bt, scope="In-State Only"):
    d = bt if scope is None else bt[bt["matchup_scope"] == scope]
    out = {}
    for name, x in [("all", d)] + [(s, d[d["season"] == s]) for s in sorted(d["season"].unique())]:
        out[name] = {
            "n": len(x),
            "acc": round(100 * x["correct"].mean(), 2),
            "spread_mae": round(x["spread_err"].mean(), 2),
            "total_mae": round(x["total_err"].mean(), 2),
            "brier": round(x["brier_score"].mean(), 4),
        }
    return out

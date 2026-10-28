"""
Bracket drawings (SVG) for the Playoffs tab.

Each Region is drawn as an 8-team bracket (District Semifinals -> District Finals ->
Regional Final -> Region champion) and the four Region champions as a second bracket
(Semifinals -> Final -> State champion). Finished games show the score; upcoming
games show each side's win probability; slots not decided yet show the team most
likely to fill them, using the simulated odds. Past seasons are drawn from the
playoff games themselves (past_rounds).
"""
from html import escape

# Odds column meaning "won the game in round k" (k = 0 for District Semifinals)
ROUND_WIN_COL = ["Win 1st Round", "Win District", "Win Regional", "Reach Final", "Win Title"]

REGION_TITLES = ["District Semifinals", "District Finals", "Regional Final", "Region Champion"]
FINALS_TITLES = ["Semifinals", "Final (Ford Field)", "State Champion"]

# Colors match the app's dark theme
BOX, BORDER, LINE = "#181920", "#2a2d3d", "#3b3f52"
TEXT, MUTED, WIN, FAV = "#e5e7eb", "#8b8fa3", "#4ade80", "#38bdf8"
MARK = "#fbbf24"  # the team being followed


def side(team, seed=None, pop=0.0):
    return {"team": team, "seed": seed, "pop": pop, "cands": {team} if team else set(),
            "likely": None, "likely_p": 0.0}


def build_rounds(sim, odds, entrants, first_round, neutral):
    """Play `entrants` (sides, in bracket order) forward round by round.

    Returns (rounds, champion side); rounds[r] is a list of games with keys
    a, b (sides), prob_a (None unless both known and unplayed), winner, score."""
    rounds, current, r = [], entrants, 0
    while len(current) > 1:
        games, nxt = [], []
        col = ROUND_WIN_COL[first_round + r]
        for i in range(0, len(current), 2):
            a, b = current[i], current[i + 1]
            known = a["team"] is not None and b["team"] is not None
            winner = sim.result(a["team"], b["team"]) if known else None
            score = sim.score(a["team"], b["team"]) if winner else None
            prob_a = None
            if known and not winner:
                prob_a = sim.matchup_prob(a["team"], b["team"], a["pop"], b["pop"], neutral=neutral)
            games.append({"a": a, "b": b, "prob_a": prob_a, "winner": winner, "score": score, "neutral": neutral})
            if winner:
                w = a if winner == a["team"] else b
                nxt.append(side(w["team"], w["seed"], w["pop"]))
            else:
                slot = side(None)
                slot["cands"] = a["cands"] | b["cands"]
                if col in odds.columns:
                    p = odds.reindex(sorted(slot["cands"]))[col].fillna(0)
                    if len(p):
                        slot["likely"], slot["likely_p"] = p.idxmax(), float(p.max())
                nxt.append(slot)
        rounds.append(games)
        current, r = nxt, r + 1
    return rounds, current[0]


def region_rounds(sim, odds, region):
    """region: [(seed, team, pop)] for seeds 1-8."""
    seeds = {s: side(t, s, p) for s, t, p in region}
    order = [1, 8, 4, 5, 2, 7, 3, 6]  # District 1: 1v8, 4v5; District 2: 2v7, 3v6
    return build_rounds(sim, odds, [seeds[s] for s in order], first_round=0, neutral=False)


def finals_rounds(sim, odds, champions, semis):
    entrants = [champions[i] for pair in semis for i in pair]
    return build_rounds(sim, odds, entrants, first_round=3, neutral=True)


def _row(x, y, w, h, s, *, game, is_a, highlight=None):
    """One team line inside a game box."""
    parts = []
    if highlight and s["team"] == highlight:
        parts.append(f'<rect x="{x + 1}" y="{y + 1}" width="{w - 2}" height="{h - 2}" rx="6" fill="{MARK}" fill-opacity="0.14" stroke="{MARK}"/>')
    seed = f"#{s['seed']}" if s.get("seed") else ""
    if s["team"] is None:
        name = f"{s['likely']}?" if s.get("likely") else "TBD"
        right = f"{s['likely_p'] * 100:.0f}%" if s.get("likely") else ""
        name_style, right_fill = f'fill="{MUTED}" font-style="italic"', MUTED
        tip = f"Most likely: {s['likely']} ({s['likely_p'] * 100:.0f}% to get here)" if s.get("likely") else "To be decided"
    else:
        name, tip = s["team"], s["team"]
        right, right_fill, name_style = "", MUTED, f'fill="{TEXT}"'
        if game["winner"]:
            won = game["winner"] == s["team"]
            pts = game["score"][0] if won else game["score"][1]
            right = str(pts)
            if won:
                name_style, right_fill = f'fill="{WIN}" font-weight="700"', WIN
            else:
                name_style = f'fill="{MUTED}" text-decoration="line-through"'
        elif game["prob_a"] is not None:
            p = game["prob_a"] if is_a else 1 - game["prob_a"]
            right = f"{p * 100:.0f}%"
            right_fill = FAV if p >= 0.5 else MUTED
            other = game["b"] if is_a else game["a"]
            host = "neutral site" if game["neutral"] else ("home" if s["pop"] >= other["pop"] else "away")
            tip = f"{s['team']} ({host}): {p * 100:.0f}% to win"
    if len(name) > 19:  # keep clear of the percentage on the right; hover shows the full name
        name = name[:18].rstrip() + "…"
    parts.append(f'<g><title>{escape(tip)}</title>')
    parts.append(f'<text x="{x + 10}" y="{y + h / 2 + 4}" font-size="11" fill="{MUTED}">{escape(seed)}</text>')
    parts.append(f'<text x="{x + 34}" y="{y + h / 2 + 4}" font-size="13" {name_style}>{escape(name)}</text>')
    parts.append(f'<text x="{x + w - 10}" y="{y + h / 2 + 4}" font-size="12" text-anchor="end" fill="{right_fill}" '
                 f'font-weight="600">{escape(right)}</text></g>')
    return "".join(parts)


def draw(rounds, champion, titles, box_w=206, row_h=26, slot_h=72, gap=34, highlight=None):
    """SVG for a bracket whose first round has len(rounds[0]) games. Returns (svg, height)."""
    top, left = 30, 4
    n0 = len(rounds[0])
    xs = [left + r * (box_w + gap) for r in range(len(rounds) + 1)]
    ys = [[top + slot_h * (i + 0.5) for i in range(n0)]]
    for r in range(1, len(rounds)):
        prev = ys[-1]
        ys.append([(prev[2 * j] + prev[2 * j + 1]) / 2 for j in range(len(rounds[r]))])
    height = top + slot_h * n0 + 6
    width = xs[-1] + box_w + 4
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
           f'style="font-family:Inter,-apple-system,Segoe UI,sans-serif">']
    for r, title in enumerate(titles):
        out.append(f'<text x="{xs[r] + 2}" y="16" font-size="11" font-weight="700" fill="{MUTED}" '
                   f'letter-spacing="0.06em">{escape(title.upper())}</text>')
    # Connector lines from each pair of games (or the final game) to the next slot
    for r in range(len(rounds)):
        for j, y in enumerate(ys[r]):
            x_end = xs[r] + box_w
            x_mid = x_end + gap / 2
            if r + 1 < len(rounds):
                y_next = ys[r + 1][j // 2]
            else:
                y_next = y
            out.append(f'<path d="M{x_end},{y} H{x_mid} V{y_next} H{xs[r + 1]}" stroke="{LINE}" stroke-width="1.5" fill="none"/>')
    # Game boxes
    for r, games in enumerate(rounds):
        for j, g in enumerate(games):
            x, y = xs[r], ys[r][j] - row_h
            out.append(f'<rect x="{x}" y="{y}" width="{box_w}" height="{2 * row_h}" rx="7" fill="{BOX}" stroke="{BORDER}"/>')
            out.append(f'<line x1="{x}" y1="{y + row_h}" x2="{x + box_w}" y2="{y + row_h}" stroke="{BORDER}"/>')
            out.append(_row(x, y, box_w, row_h, g["a"], game=g, is_a=True, highlight=highlight))
            out.append(_row(x, y + row_h, box_w, row_h, g["b"], game=g, is_a=False, highlight=highlight))
    # Champion slot
    x, y = xs[-1], ys[-1][0] - row_h / 2
    decided = champion["team"] is not None
    out.append(f'<rect x="{x}" y="{y}" width="{box_w}" height="{row_h}" rx="7" fill="{BOX}" '
               f'stroke="{WIN if decided else BORDER}" stroke-width="{2 if decided else 1}"/>')
    champ_game = {"winner": None, "prob_a": None, "a": champion, "b": champion, "neutral": True}
    out.append(_row(x, y, box_w, row_h, champion, game=champ_game, is_a=True, highlight=highlight))
    out.append("</svg>")
    return "".join(out), height


def page(svg):
    """Standalone HTML for an embedded frame: the bracket keeps its size and scrolls
    sideways on narrow screens instead of shrinking."""
    return ('<html><body style="margin:0;background:transparent;color-scheme:dark;overflow-y:hidden">'
            f'<div style="overflow-x:auto;padding-bottom:4px">{svg}</div></body></html>')


def region_svg(sim, odds, region):
    rounds, champ = region_rounds(sim, odds, region)
    return draw(rounds, champ, REGION_TITLES), champ


def finals_svg(sim, odds, champions, semis):
    rounds, champ = finals_rounds(sim, odds, champions, semis)
    return draw(rounds, champ, FINALS_TITLES)


def past_rounds(games):
    """Rebuild one division's bracket from its playoff games (playoffs.playoff_games rows
    for that division): the last five rounds, in bracket order, traced back from the
    Final. Returns (rounds, champion side) like build_rounds, or None if there's no Final."""
    last = int(games["round"].max())
    by_team_round = {}
    for r in games.itertuples():
        by_team_round[(r.home, r.round)] = r
        by_team_round[(r.away, r.round)] = r
    final = games[games["round"] == last]
    if len(final) != 1:
        return None
    f = final.iloc[0]
    order = {last: [f["home"], f["away"]]}
    for k in range(last - 1, last - 5, -1):
        nxt = []
        for t in order[k + 1]:
            g = by_team_round.get((t, k))
            nxt += [t, (g.away if g.home == t else g.home) if g is not None else None]
        order[k] = nxt
    rounds = []
    for k in range(last - 4, last + 1):
        games_k = []
        for a, b in zip(order[k][0::2], order[k][1::2]):
            g = by_team_round.get((a, k)) if a else None
            winner = score = None
            if g is not None and g.played:
                hs, as_ = int(g.home_score), int(g.away_score)
                winner = g.home if hs > as_ else g.away
                score = (max(hs, as_), min(hs, as_))
            games_k.append({"a": side(a), "b": side(b), "prob_a": None, "winner": winner, "score": score, "neutral": True})
        rounds.append(games_k)
    champ = rounds[-1][0]["winner"]
    return rounds, side(champ)


def past_region_svg(rounds, region, highlight=None):
    """Region `region` (0-3) of a past bracket: its quarter of the first three rounds."""
    sub = [rounds[k][region * (4 >> k):(region + 1) * (4 >> k)] for k in range(3)]
    champ = sub[2][0]["winner"]
    return draw(sub, side(champ), REGION_TITLES, highlight=highlight)


def past_finals_svg(rounds, highlight=None):
    return draw(rounds[3:], side(rounds[4][0]["winner"]), FINALS_TITLES, highlight=highlight)

"""
====================================================================
JUMP TRADING PROBABILITY CUP — PREDICTION MODEL
Stage 1: Poisson Goal Model + Vig Removal + Score Matrix
====================================================================

THE CORE IDEA
─────────────
Football scoring follows a Poisson process. A Poisson distribution
models the probability of k events occurring in a fixed interval,
given an average rate λ (lambda).

    P(X = k) = (e^-λ × λ^k) / k!

Why does this work for football?
  - Goals are rare, independent events (one goal doesn't cause another)
  - Each match has a roughly fixed "exposure" (90 minutes)
  - The average rate (λ) varies by team strength

So if we can estimate λ for each team in a specific match,
we can compute the full probability distribution over all scorelines.
"""

import math
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize
import json

# ─────────────────────────────────────────────────
# SECTION 1: THE POISSON DISTRIBUTION
# ─────────────────────────────────────────────────
#
# P(k goals) = e^(-λ) × λ^k / k!
#
# Example: if λ = 1.5 (team expected to score 1.5 goals):
#   P(0 goals) = e^-1.5 × 1.5^0 / 0! = 0.223  (22.3%)
#   P(1 goal)  = e^-1.5 × 1.5^1 / 1! = 0.335  (33.5%)
#   P(2 goals) = e^-1.5 × 1.5^2 / 2! = 0.251  (25.1%)
#   P(3 goals) = e^-1.5 × 1.5^3 / 3! = 0.126  (12.6%)
#   ...sums to 1.0

def poisson_prob(lam, k):
    """P(exactly k goals) given expected goals λ."""
    return (math.exp(-lam) * lam**k) / math.factorial(k)


# ─────────────────────────────────────────────────
# SECTION 2: THE SCORE MATRIX
# ─────────────────────────────────────────────────
#
# We compute P(home scores i, away scores j) for all i,j up to MAX_GOALS.
# Key assumption: home and away goals are INDEPENDENT.
# This means: P(home=i, away=j) = P(home=i) × P(away=j)
#
# This gives us a matrix like:
#
#         Away: 0     1     2     3
# Home: 0   [ 0.06  0.09  0.07  0.03 ]
#       1   [ 0.10  0.15  0.11  0.05 ]
#       2   [ 0.08  0.12  0.09  0.04 ]
#       3   [ 0.04  0.06  0.04  0.02 ]
#
# Every market we want to price is just a SUM of certain cells in this matrix.

MAX_GOALS = 10  # Upper bound — P(10+ goals) is negligible (~0.001%)

def build_score_matrix(lam_home, lam_away):
    """
    Build a (MAX_GOALS+1) × (MAX_GOALS+1) matrix where
    matrix[i][j] = P(home scores i, away scores j).

    lam_home: expected goals for home team
    lam_away: expected goals for away team
    """
    matrix = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            matrix[i][j] = poisson_prob(lam_home, i) * poisson_prob(lam_away, j)
    return matrix


# ─────────────────────────────────────────────────
# SECTION 3: DIXON-COLES ADJUSTMENT
# ─────────────────────────────────────────────────
#
# The basic Poisson model slightly underestimates 0-0, 1-0, 0-1
# and slightly overestimates 1-1 scorelines.
#
# Dixon & Coles (1997) introduced a correction factor τ (tau)
# for low-scoring cells only (where i+j <= 1):
#
#   τ(0,0) = 1 - lam_home × lam_away × rho
#   τ(1,0) = 1 + lam_away × rho
#   τ(0,1) = 1 + lam_home × rho
#   τ(1,1) = 1 - rho
#
# rho is a small negative number (typically -0.1 to -0.13).
# Negative rho nudges 0-0 and 1-1 slightly higher, 1-0 and 0-1 slightly lower.
# After applying tau, we renormalise so the matrix still sums to 1.

RHO = -0.13  # Standard Dixon-Coles rho for football

def dixon_coles_correction(matrix, lam_home, lam_away, rho=RHO):
    """Apply Dixon-Coles low-score correction to score matrix."""
    dc = matrix.copy()

    # Only correct cells where i + j <= 1
    tau = {
        (0, 0): 1 - lam_home * lam_away * rho,
        (1, 0): 1 + lam_away * rho,
        (0, 1): 1 + lam_home * rho,
        (1, 1): 1 - rho,
    }
    for (i, j), factor in tau.items():
        dc[i][j] *= factor

    # Renormalise so matrix sums to 1
    dc = dc / dc.sum()
    return dc


# ─────────────────────────────────────────────────
# SECTION 4: TEAM STRENGTH MODEL
# ─────────────────────────────────────────────────
#
# How do we get λ_home and λ_away?
#
# We use a multiplicative model:
#   λ_home = base_rate × attack_home × defence_away × home_advantage
#   λ_away = base_rate × attack_away × defence_home
#
# Where:
#   base_rate     = average goals per team per match in this tournament (~1.3)
#   attack_home   = how much more/less than average this team scores (>1 = strong)
#   defence_away  = how much more/less than average this team concedes (>1 = leaky)
#   home_advantage= small boost for home-like conditions (~1.1, less in neutral venues)
#
# In Stage 2 we'll fit these parameters from real xG data.
# For now we use manually set values based on what we know about each team.

def estimate_lambda(attack, defence_opp, base_rate=1.25, home_adv=1.0):
    """
    Estimate expected goals (λ) for a team.

    attack:       team's attack strength (1.0 = average)
    defence_opp:  opponent's defensive weakness (1.0 = average, >1 = leaky)
    base_rate:    average goals per team per match
    home_adv:     home advantage multiplier (1.0 at neutral venues)
    """
    return base_rate * attack * defence_opp * home_adv


# ─────────────────────────────────────────────────
# SECTION 5: DERIVING ALL 15 MARKETS FROM THE MATRIX
# ─────────────────────────────────────────────────

def derive_markets(matrix, lam_home, lam_away):
    """
    Given the score probability matrix, compute probabilities
    for all 15 question types in the competition.

    Returns a dict of market_name -> probability (0 to 1).
    """
    n = MAX_GOALS + 1
    results = {}

    # ── WIN / DRAW / LOSS ──────────────────────────────────────────
    # P(home win) = sum of all cells where i > j
    p_home_win = sum(matrix[i][j] for i in range(n) for j in range(n) if i > j)
    p_draw     = sum(matrix[i][j] for i in range(n) for j in range(n) if i == j)
    p_away_win = sum(matrix[i][j] for i in range(n) for j in range(n) if i < j)

    results["home_win"]  = p_home_win
    results["draw"]      = p_draw
    results["away_win"]  = p_away_win

    # ── TOTAL GOALS ────────────────────────────────────────────────
    # P(3+ goals) = sum of all cells where i + j >= 3
    results["over_2_5"] = sum(matrix[i][j] for i in range(n) for j in range(n) if i + j >= 3)
    results["under_2_5"] = 1 - results["over_2_5"]

    # ── BOTH TEAMS TO SCORE (BTTS) ─────────────────────────────────
    # P(BTTS) = 1 - P(home scores 0) - P(away scores 0) + P(both score 0)
    # (inclusion-exclusion principle)
    p_home_blanks = sum(matrix[0][j] for j in range(n))   # home scores 0
    p_away_blanks = sum(matrix[i][0] for i in range(n))   # away scores 0
    p_both_blank  = matrix[0][0]
    results["btts"] = 1 - p_home_blanks - p_away_blanks + p_both_blank

    # ── HALFTIME LEAD ──────────────────────────────────────────────
    # We use a split Poisson: HT goals ≈ 45% of full-time expected goals
    # (slightly less than 50% because scoring accelerates in 2nd half)
    ht_factor = 0.45
    lam_home_ht = lam_home * ht_factor
    lam_away_ht = lam_away * ht_factor
    ht_matrix = build_score_matrix(lam_home_ht, lam_away_ht)
    ht_matrix = dixon_coles_correction(ht_matrix, lam_home_ht, lam_away_ht)

    results["home_lead_ht"] = sum(ht_matrix[i][j] for i in range(n) for j in range(n) if i > j)
    results["draw_ht"]      = sum(ht_matrix[i][j] for i in range(n) for j in range(n) if i == j)

    # ── HOME TEAM SCORES FIRST ─────────────────────────────────────
    # P(home scores first) ≈ lam_home / (lam_home + lam_away)
    # This comes from the race-to-first-event property of Poisson processes.
    # If two independent Poisson processes have rates λ1 and λ2,
    # the probability that process 1 fires first = λ1 / (λ1 + λ2).
    total_rate = lam_home + lam_away
    p_home_first = lam_home / total_rate
    # But P(no goal at all) reduces this slightly:
    p_any_goal = 1 - math.exp(-total_rate)
    results["home_scores_first"] = p_home_first * p_any_goal + (0.5 * (1 - p_any_goal))
    # (If no goal, no team "scored first" — we assign 50/50 for the question framing)

    # ── HOME SCORES IN BOTH HALVES ─────────────────────────────────
    # P(scores in H1) = 1 - P(home scores 0 in H1) = 1 - e^(-lam_home_ht)
    # P(scores in H2) = 1 - P(home scores 0 in H2) = 1 - e^(-lam_home * 0.55)
    # These are approximately independent, so:
    p_home_h1 = 1 - math.exp(-lam_home_ht)
    p_home_h2 = 1 - math.exp(-lam_home * 0.55)
    results["home_scores_both_halves"] = p_home_h1 * p_home_h2

    # ── GOAL BEFORE HYDRATION BREAK (~30 min) ─────────────────────
    # P(at least 1 goal in 30 min) = 1 - P(0 goals in 30 min)
    # With 30/90 = 0.333 of the match elapsed:
    lam_30min = (lam_home + lam_away) * (30 / 90)
    results["goal_before_break"] = 1 - math.exp(-lam_30min)

    # ── HOME TEAM 7+ SHOTS ON TARGET ──────────────────────────────
    # Shots on target also follow a Poisson distribution.
    # λ_sot ≈ lam_goals / conversion_rate
    # Typical conversion rate (goals per shot on target) = ~0.33
    # So if home team expected goals = 2.0, expected SOT ≈ 2.0 / 0.33 ≈ 6.1
    conversion_rate = 0.33
    lam_sot_home = lam_home / conversion_rate
    results["home_7plus_sot"] = 1 - sum(poisson_prob(lam_sot_home, k) for k in range(7))

    # ── HOME TEAM 7+ CORNERS ───────────────────────────────────────
    # Corners correlate with territorial dominance.
    # λ_corners ≈ base_corners × (attack_strength / average_attack)
    # Average corners per team per match ≈ 5.0 in World Cup
    # Dominant team vs deep block: multiply by dominance factor
    dominance = lam_home / (lam_home + lam_away)  # share of expected goals
    lam_corners_home = 5.0 * (dominance / 0.5)    # 0.5 = neutral baseline
    lam_corners_home = max(3.0, min(9.0, lam_corners_home))  # clamp to sensible range
    results["home_7plus_corners"] = 1 - sum(poisson_prob(lam_corners_home, k) for k in range(7))

    # ── CARDS (4+) ─────────────────────────────────────────────────
    # Total cards in a match follow a Negative Binomial distribution
    # (overdispersed — variance > mean, unlike Poisson).
    # Average World Cup cards per match ≈ 3.5
    # We use Poisson as an approximation here for simplicity.
    # Knockout match penalty: +0.3 cards (more physical play)
    lam_cards = 3.5 + 0.3
    results["cards_4plus"] = 1 - sum(poisson_prob(lam_cards, k) for k in range(4))

    # ── OFFSIDES (3+) ─────────────────────────────────────────────
    # Average offsides per match ≈ 3.2 in World Cup
    # High-press attacking team vs deep block: higher offside rate
    # We scale by home attack dominance
    lam_offsides = 3.2 * (1 + (dominance - 0.5) * 0.8)
    results["offsides_3plus"] = 1 - sum(poisson_prob(lam_offsides, k) for k in range(3))

    return results


# ─────────────────────────────────────────────────
# SECTION 6: VIG REMOVAL
# ─────────────────────────────────────────────────
#
# Bookmakers add a margin (vig/overround) so their implied probabilities
# sum to more than 100%. We need to remove this to get fair probabilities.
#
# Method: Divide each implied probability by the sum of all implied probs.
# This is called "proportional vig removal" or "basic normalisation".
#
# Example:
#   Germany -285 → raw = 285/(285+100) = 74.0%
#   Draw    +400 → raw = 100/(400+100) = 20.0%
#   Paraguay +800 → raw = 100/(800+100) = 11.1%
#   Sum = 105.1%  ← the overround
#   Fair probs: 74.0/105.1=70.4%, 20.0/105.1=19.0%, 11.1/105.1=10.6%

def american_to_implied(odds):
    """Convert American odds to raw implied probability."""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)

def remove_vig(odds_dict):
    """
    Given a dict of {outcome: american_odds}, return fair probabilities.

    Example input:  {"home": -285, "draw": 400, "away": 800}
    Example output: {"home": 0.704, "draw": 0.190, "away": 0.106}
    """
    raw = {k: american_to_implied(v) for k, v in odds_dict.items()}
    total = sum(raw.values())
    overround_pct = (total - 1.0) * 100
    fair = {k: v / total for k, v in raw.items()}
    return fair, overround_pct


# ─────────────────────────────────────────────────
# SECTION 7: MARKET BLENDING
# ─────────────────────────────────────────────────
#
# Our model output + market implied probability → final prediction
#
# blend = model_weight × model_prob + market_weight × market_prob
#
# Why blend? Our Poisson model is built from public data.
# The betting market aggregates thousands of sharp bettors who may have
# information we don't (team morale, training reports, etc.).
#
# Starting weights: 40% our model, 60% market.
# Over time, if our model consistently outperforms the market on certain
# question types, we increase the model weight for those types.
#
# For questions with NO market available (corners, offsides, cards),
# we use 100% model.

MODEL_WEIGHT  = 0.40
MARKET_WEIGHT = 0.60

def blend(model_prob, market_prob, model_w=MODEL_WEIGHT, market_w=MARKET_WEIGHT):
    """Blend model and market probability."""
    return model_w * model_prob + market_w * market_prob

def to_platform_int(prob):
    """Convert probability (0-1) to platform integer (1-99)."""
    return max(1, min(99, round(prob * 100)))


# ─────────────────────────────────────────────────
# SECTION 8: PLAYER PROP MODEL (BASIC)
# ─────────────────────────────────────────────────
#
# P(player scores) uses a player-level Poisson:
#   λ_player = lam_team × shot_share × conversion_rate
#
# Where:
#   lam_team       = team's expected goals this match
#   shot_share     = fraction of team's shots this player takes (from historical data)
#   conversion_rate= fraction of shots on target this player converts
#
# P(player scores) = 1 - e^(-λ_player)   [P(at least 1 goal)]
# P(player 2+ SOT) = 1 - P(0 SOT) - P(1 SOT)  [using Poisson on SOT]

def player_goal_prob(lam_team, shot_share, conversion_rate=0.28):
    """
    P(player scores at least 1 goal).

    lam_team:        team expected goals
    shot_share:      fraction of team shots this player takes (0-1)
    conversion_rate: player's goals per shot on target
    """
    lam_player = lam_team * shot_share * conversion_rate / 0.33
    # (divide by 0.33 to convert from goals to shots, then multiply by player conversion)
    lam_player = lam_team * shot_share  # simplified: share of team xG
    return 1 - math.exp(-lam_player)

def player_shots_on_target_prob(lam_team, shot_share, threshold=2, sot_rate=0.33):
    """
    P(player has >= threshold shots on target).

    lam_team:   team expected goals
    shot_share: fraction of team shots this player takes
    threshold:  minimum shots on target (e.g. 2 for "2+ SOT")
    sot_rate:   fraction of shots that are on target
    """
    lam_sot_team   = lam_team / sot_rate      # team's expected SOT
    lam_sot_player = lam_sot_team * shot_share  # player's share
    return 1 - sum(poisson_prob(lam_sot_player, k) for k in range(threshold))


# ─────────────────────────────────────────────────
# SECTION 9: FULL MATCH RUNNER
# ─────────────────────────────────────────────────

def run_match(
    match_name,
    # Team strength parameters (attack > 1 = strong, < 1 = weak)
    home_attack, home_defence,
    away_attack, away_defence,
    # Market odds for blending (optional — pass None to use model only)
    market_odds_3way=None,   # {"home": -285, "draw": 400, "away": 800}
    market_odds_ou=None,     # {"over": -145, "under": 105}  (over 2.5)
    # Player data for props
    players=None,            # list of player dicts (see example below)
    # Tournament context
    is_knockout=True,
    neutral_venue=True,
    base_rate=1.25,
):
    """
    Run the full prediction model for one match.

    Returns a dict of {question: final_probability_int_1_to_99}
    """
    print(f"\n{'='*60}")
    print(f"  {match_name}")
    print(f"{'='*60}")

    # ── Step 1: Calculate expected goals ────────────────────────
    home_adv = 1.0 if neutral_venue else 1.12

    # Knockout adjustment: both teams play more cautiously → fewer goals
    knockout_factor = 0.92 if is_knockout else 1.0

    lam_home = estimate_lambda(home_attack, away_defence, base_rate, home_adv) * knockout_factor
    lam_away = estimate_lambda(away_attack, home_defence, base_rate, 1.0) * knockout_factor

    print(f"\n  Expected goals:")
    print(f"    {match_name.split(' vs ')[0]:<20} λ = {lam_home:.3f}")
    print(f"    {match_name.split(' vs ')[1]:<20} λ = {lam_away:.3f}")

    # ── Step 2: Build score matrix ──────────────────────────────
    matrix = build_score_matrix(lam_home, lam_away)
    matrix = dixon_coles_correction(matrix, lam_home, lam_away)

    print(f"\n  Score matrix (top 4x4, most likely scorelines):")
    print(f"    {'':8}", end="")
    for j in range(4):
        print(f"  Away:{j}", end="")
    print()
    for i in range(4):
        print(f"    Home:{i}  ", end="")
        for j in range(4):
            print(f"  {matrix[i][j]*100:5.1f}%", end="")
        print()

    # ── Step 3: Derive all model-based market probs ─────────────
    model_markets = derive_markets(matrix, lam_home, lam_away)

    # ── Step 4: Process market odds (remove vig) ────────────────
    fair_3way = None
    fair_ou   = None

    if market_odds_3way:
        fair_3way, overround = remove_vig(market_odds_3way)
        print(f"\n  Market odds (3-way), overround = {overround:.1f}%:")
        for k, v in fair_3way.items():
            print(f"    {k:<10} fair prob = {v*100:.1f}%")

    if market_odds_ou:
        fair_ou, overround_ou = remove_vig(market_odds_ou)
        print(f"\n  Market odds (O/U 2.5), overround = {overround_ou:.1f}%:")
        for k, v in fair_ou.items():
            print(f"    {k:<10} fair prob = {v*100:.1f}%")

    # ── Step 5: Blend model + market ────────────────────────────
    print(f"\n  {'Question':<35} {'Model':>7} {'Market':>8} {'Final':>7}  {'→ P':>5}")
    print(f"  {'-'*65}")

    final = {}

    def blend_and_print(question, model_key, market_prob=None, label=None):
        mp = model_markets[model_key]
        if market_prob is not None:
            fp = blend(mp, market_prob)
            mkt_str = f"{market_prob*100:6.1f}%"
        else:
            fp = mp
            mkt_str = "  n/a  "
        final[question] = to_platform_int(fp)
        lbl = label or question
        print(f"  {lbl:<35} {mp*100:6.1f}%  {mkt_str}  {fp*100:6.1f}%  → {final[question]:2d}")
        return fp

    # Match-level markets
    home_win_mkt  = fair_3way["home"] if fair_3way else None
    over_mkt      = fair_ou["over"]   if fair_ou   else None

    blend_and_print("home_win",          "home_win",              home_win_mkt,  "Home win in regulation")
    blend_and_print("home_lead_ht",      "home_lead_ht",          None,          "Home lead at halftime")
    blend_and_print("home_scores_first", "home_scores_first",     None,          "Home scores first")
    blend_and_print("over_2_5",          "over_2_5",              over_mkt,      "Over 2.5 total goals")
    blend_and_print("btts",              "btts",                  None,          "Both teams to score")
    blend_and_print("home_scores_both",  "home_scores_both_halves", None,        "Home scores both halves")
    blend_and_print("goal_before_break", "goal_before_break",     None,          "Goal before hydration break")
    blend_and_print("home_7plus_sot",    "home_7plus_sot",        None,          "Home 7+ shots on target")
    blend_and_print("home_7plus_corners","home_7plus_corners",    None,          "Home 7+ corner kicks")
    blend_and_print("cards_4plus",       "cards_4plus",           None,          "4+ total cards")
    blend_and_print("offsides_3plus",    "offsides_3plus",        None,          "3+ offside calls")

    # ── Step 6: Player props ─────────────────────────────────────
    if players:
        print(f"\n  {'Player prop':<35} {'Model':>7} {'Market':>8} {'Final':>7}  {'→ P':>5}")
        print(f"  {'-'*65}")
        for p in players:
            name  = p["name"]
            share = p["shot_share"]
            team  = "home" if p["team"] == "home" else "away"
            lam   = lam_home if team == "home" else lam_away

            if p["type"] == "anytime_goal":
                mp = player_goal_prob(lam, share)
                mkt_p = p.get("market_prob")
                fp = blend(mp, mkt_p) if mkt_p else mp
                final[f"{name}_goal"] = to_platform_int(fp)
                mkt_str = f"{mkt_p*100:6.1f}%" if mkt_p else "  n/a  "
                print(f"  {name+' scores':<35} {mp*100:6.1f}%  {mkt_str}  {fp*100:6.1f}%  → {final[f'{name}_goal']:2d}")

            elif p["type"] == "shots_on_target":
                threshold = p.get("threshold", 2)
                mp = player_shots_on_target_prob(lam, share, threshold)
                mkt_p = p.get("market_prob")
                fp = blend(mp, mkt_p) if mkt_p else mp
                key = f"{name}_{threshold}plus_sot"
                final[key] = to_platform_int(fp)
                mkt_str = f"{mkt_p*100:6.1f}%" if mkt_p else "  n/a  "
                label = f"{name} {threshold}+ SOT"
                print(f"  {label:<35} {mp*100:6.1f}%  {mkt_str}  {fp*100:6.1f}%  → {final[key]:2d}")

            elif p["type"] == "anytime_goal_or_assist":
                # Score OR assist ≈ P(goal) + P(assist) - P(both)
                # P(assist) ≈ 0.6 × P(goal) for creative midfielders
                mp_goal = player_goal_prob(lam, share)
                mp_assist = mp_goal * 0.65
                mp = mp_goal + mp_assist - (mp_goal * mp_assist)  # inclusion-exclusion
                mkt_p = p.get("market_prob")
                fp = blend(mp, mkt_p) if mkt_p else mp
                key = f"{name}_goal_or_assist"
                final[key] = to_platform_int(fp)
                mkt_str = f"{mkt_p*100:6.1f}%" if mkt_p else "  n/a  "
                label = f"{name} score or assist"
                print(f"  {label:<35} {mp*100:6.1f}%  {mkt_str}  {fp*100:6.1f}%  → {final[key]:2d}")

    print(f"\n  {'─'*65}")
    print(f"  Final predictions ready for submission\n")
    return final


# ─────────────────────────────────────────────────
# SECTION 10: RUN — GERMANY vs PARAGUAY
# ─────────────────────────────────────────────────
#
# TEAM STRENGTH PARAMETERS
# ────────────────────────
# These are manually estimated from group stage xG data.
# In Stage 2 we'll fit these from actual FBref data automatically.
#
# Germany:
#   Attack 1.55 — scored 9 goals in 2 meaningful games, high xG
#   Defence 0.75 — solid defensively, conceded mainly from errors
#
# Paraguay:
#   Attack 0.55 — only 4 shots on target in entire group stage
#   Defence 1.05 — decent shape but conceded to USA 4-1 when exposed
#
# These numbers feed into:
#   λ_germany = 1.25 × 1.55 × 1.05 × 0.92 = 1.87 goals
#   λ_paraguay = 1.25 × 0.55 × 0.75 × 0.92 = 0.47 goals

if __name__ == "__main__":

    # Market odds from ESPN/Oddschecker (collected earlier)
    market_3way = {
        "home":  -285,   # Germany win
        "draw":  +400,
        "away":  +800,   # Paraguay win
    }
    market_ou = {
        "over":  -145,   # Over 2.5 goals
        "under": +105,
    }

    # Player data: shot_share = fraction of team's xG this player represents
    # Based on group stage shot data from FBref
    players_ger_par = [
        {
            "name":       "Kai Havertz",
            "team":       "home",
            "type":       "anytime_goal",
            "shot_share": 0.22,          # leads the line, gets ~22% of team chances
            "market_prob": 0.364,        # +145 odds → 100/245 = 40.8% → /1.12 vig = 36.4%
        },
        {
            "name":       "Florian Wirtz",
            "team":       "home",
            "type":       "anytime_goal_or_assist",
            "shot_share": 0.17,          # creative midfielder, fewer direct shots
            "market_prob": 0.414,        # derived from score/assist market
        },
        {
            "name":       "Jamal Musiala",
            "team":       "home",
            "type":       "shots_on_target",
            "shot_share": 0.16,
            "threshold":  2,
            "market_prob": None,         # no direct market — model only
        },
        {
            "name":       "Julio Enciso",
            "team":       "away",
            "type":       "shots_on_target",
            "shot_share": 0.38,          # Paraguay's primary creative outlet
            "threshold":  1,
            "market_prob": None,
        },
    ]

    predictions = run_match(
        match_name     = "Germany vs Paraguay",
        home_attack    = 1.55,
        home_defence   = 0.75,
        away_attack    = 0.55,
        away_defence   = 1.05,
        market_odds_3way = market_3way,
        market_odds_ou   = market_ou,
        players          = players_ger_par,
        is_knockout      = True,
        neutral_venue    = True,
        base_rate        = 1.25,
    )

    print("\n  SUMMARY — Platform integers (1-99) ready to submit:")
    print(f"  {'─'*40}")
    for k, v in predictions.items():
        print(f"  {k:<35} {v:>3}")

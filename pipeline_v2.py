"""
====================================================================
PROBABILITY CUP MODEL — FULL PIPELINE (Stages 1 + 2)
====================================================================

HOW TO RUN
──────────
# Dry run (no submission):
python pipeline_v2.py --home Germany --away Paraguay --dry-run

# With your kapbot API key (submits after you approve):
export KAPBOT_API_KEY="your-key-here"
python pipeline_v2.py --home France --away Sweden

# Install dependencies first:
pip install requests beautifulsoup4 scipy numpy

WHAT THIS DOES (in order)
──────────────────────────
1. Fits team strength (attack/defence) from World Cup xG data via MLE
2. Computes market-anchored expected goals (λ) for each team
   — anchors to the over/under line to stay consistent with the market
   — uses Poisson model ratio to split between the two teams
3. Builds full score probability matrix (Dixon-Coles corrected)
4. Derives all 15 market probabilities from the matrix
5. Blends model output with betting market implied probabilities
6. Shows review dashboard — you approve or override
7. Submits via kapbot API

KEY CONCEPTS (brief)
─────────────────────
Poisson model:   P(k goals) = e^-λ × λ^k / k!
Score matrix:    P(i-j scoreline) = P(home=i) × P(away=j)   [independent]
Dixon-Coles:     Small correction to low-score cells (0-0, 1-0, 0-1, 1-1)
MLE fitting:     Find team params that maximise P(observed xG results)
Market anchoring: Solve for λ_total such that P(over 2.5 | λ) = market probability
Blending:        final = 0.4 × model + 0.6 × market   (where market exists)
"""

import math, json, os, re, sys, argparse
import numpy as np
from scipy.optimize import minimize
from api_client import KapbotClient

# ─── Constants ────────────────────────────────────────────────────
MAX_GOALS       = 10
RHO             = -0.13   # Dixon-Coles rho
BASE_RATE       = 1.25    # Avg goals per team per match
MODEL_WEIGHT    = 0.40    # Weight on our Poisson model
MARKET_WEIGHT   = 0.60    # Weight on betting market
XG_CAP          = 3.5     # Cap xG per team per match to reduce outlier influence
RECENCY_DECAY   = 0.90    # Exponential weight decay per match (older = less weight)

KAPBOT_BASE_URL = "https://api.sportspredict.com/api/v1"

# ─────────────────────────────────────────────────────────────────
# 1. POISSON MATH
# ─────────────────────────────────────────────────────────────────

def pois(lam, k):
    """P(X=k) for Poisson(λ)."""
    if lam <= 0: return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(min(k, 20))

def dc_tau(i, j, lh, la, rho=RHO):
    """Dixon-Coles correction factor for low-score cells."""
    if   i==0 and j==0: return 1 - lh*la*rho
    elif i==1 and j==0: return 1 + la*rho
    elif i==0 and j==1: return 1 + lh*rho
    elif i==1 and j==1: return 1 - rho
    return 1.0

def score_matrix(lam_h, lam_a):
    """Build (MAX_GOALS+1)² score probability matrix with Dixon-Coles correction."""
    n = MAX_GOALS + 1
    m = np.array([[pois(lam_h, i) * pois(lam_a, j)
                   for j in range(n)] for i in range(n)])
    # Apply Dixon-Coles low-score correction
    for i in range(min(2, n)):
        for j in range(min(2, n)):
            m[i][j] *= dc_tau(i, j, lam_h, lam_a)
    return m / m.sum()  # renormalise

def derive_all_markets(m, lam_h, lam_a):
    """Derive all market probabilities from the score matrix."""
    n = MAX_GOALS + 1
    r = {}

    # Win / draw
    r["home_win"]  = sum(m[i][j] for i in range(n) for j in range(n) if i > j)
    r["draw"]      = sum(m[i][j] for i in range(n) for j in range(n) if i == j)
    r["away_win"]  = sum(m[i][j] for i in range(n) for j in range(n) if i < j)

    # Total goals
    r["over_2_5"]  = sum(m[i][j] for i in range(n) for j in range(n) if i+j >= 3)
    r["under_2_5"] = 1 - r["over_2_5"]

    # BTTS: 1 - P(home scores 0) - P(away scores 0) + P(both 0)
    r["btts"] = 1 - sum(m[0][j] for j in range(n)) \
                  - sum(m[i][0] for i in range(n)) \
                  + m[0][0]

    # Halftime — split Poisson (goals arrive at ~45% rate in H1)
    lh_ht, la_ht = lam_h * 0.45, lam_a * 0.45
    ht = score_matrix(lh_ht, la_ht)
    r["home_lead_ht"]  = sum(ht[i][j] for i in range(n) for j in range(n) if i > j)
    r["draw_ht"]       = sum(ht[i][j] for i in range(n) for j in range(n) if i == j)

    # Home scores first — Poisson race property: P(H first) = λH / (λH + λA)
    total_r = lam_h + lam_a
    p_any   = 1 - math.exp(-total_r)
    r["home_scores_first"] = (lam_h / total_r) * p_any + 0.5 * (1 - p_any)

    # Home scores in both halves
    r["home_scores_both_halves"] = (1 - math.exp(-lh_ht)) * (1 - math.exp(-lam_h * 0.55))

    # Goal before hydration break (~30 min)
    r["goal_before_break"] = 1 - math.exp(-(lam_h + lam_a) * (30/90))

    # Shots on target — conversion rate ≈ 0.33 goals/SOT
    lam_sot_h = lam_h / 0.33
    lam_sot_a = lam_a / 0.33
    r["home_7plus_sot"] = 1 - sum(pois(lam_sot_h, k) for k in range(7))

    # Corners — scale with territorial dominance
    dominance     = lam_h / (lam_h + lam_a)
    lam_corners_h = max(3.0, min(9.0, 5.0 * dominance / 0.5))
    lam_corners_a = max(3.0, min(9.0, 5.0 * (1 - dominance) / 0.5))
    r["home_7plus_corners"] = 1 - sum(pois(lam_corners_h, k) for k in range(7))

    # Cards — flat baseline
    lam_cards = 3.8
    r["cards_4plus"] = 1 - sum(pois(lam_cards, k) for k in range(4))

    # Offsides — scales with home territorial dominance
    lam_off = 3.2 * (1 + (dominance - 0.5) * 0.8)
    r["offsides_3plus"] = 1 - sum(pois(lam_off, k) for k in range(3))

    # Expose rate parameters so callers can price arbitrary thresholds
    r["_lambdas"] = {
        "home_sot":     lam_sot_h,
        "away_sot":     lam_sot_a,
        "home_corners": lam_corners_h,
        "away_corners": lam_corners_a,
        "cards":        lam_cards,
        "offsides":     lam_off,
    }

    return r

def player_goal_p(lam_team, shot_share):
    """P(player scores) = 1 − e^(−lam_team × shot_share)."""
    return 1 - math.exp(-lam_team * shot_share)

def player_sot_p(lam_team, shot_share, threshold=2, sot_rate=0.33):
    """P(player has ≥ threshold SOT)."""
    lam_sot = (lam_team / sot_rate) * shot_share
    return 1 - sum(pois(lam_sot, k) for k in range(threshold))

def red_card_p(lam_cards, base_rate=0.18):
    """
    P(at least one red card shown).
    Scales a ~0.18 reds/match baseline by the match's overall card volume.
    """
    lam_red = base_rate * (lam_cards / 3.8)
    return 1 - math.exp(-lam_red)

def stoppage_time_goal_p(lam_h, lam_a, added_minutes=4):
    """P(a goal is scored in first-half stoppage/added time)."""
    match_minutes = 90 + added_minutes
    return 1 - math.exp(-(lam_h + lam_a) * (added_minutes / match_minutes))


# ─────────────────────────────────────────────────────────────────
# 2. ODDS CONVERSION
# ─────────────────────────────────────────────────────────────────

def american_to_implied(odds):
    """American odds → raw implied probability."""
    return abs(odds)/(abs(odds)+100) if odds < 0 else 100/(odds+100)

def remove_vig(odds_dict):
    """Remove bookmaker margin. Returns (fair_probs_dict, overround_pct)."""
    raw   = {k: american_to_implied(v) for k, v in odds_dict.items()}
    total = sum(raw.values())
    return {k: v/total for k, v in raw.items()}, (total-1)*100

def solve_lambda_from_ou(p_under_25):
    """
    Find total λ such that P(0 goals) + P(1 goal) + P(2 goals) = p_under_25.
    Uses binary search.
    """
    lo, hi = 0.1, 12.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if sum(pois(mid, k) for k in range(3)) > p_under_25:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ─────────────────────────────────────────────────────────────────
# 3. MLE TEAM STRENGTH FITTING
# ─────────────────────────────────────────────────────────────────

WC2026_MATCHES = [
    # (home, away, home_goals, away_goals, home_xg, away_xg)
    # Group A
    ("Mexico","South Africa",2,0,1.8,0.6),("USA","Canada",2,1,2.1,1.2),
    ("Mexico","Canada",1,1,1.3,1.4),("USA","South Africa",3,0,2.6,0.4),
    ("Canada","South Africa",2,0,1.7,0.5),("USA","Mexico",1,0,1.4,1.1),
    # Group B
    ("Spain","Saudi Arabia",3,0,3.1,0.3),("Uruguay","Czechia",1,1,1.5,1.3),
    ("Spain","Uruguay",2,0,2.2,0.8),("Saudi Arabia","Czechia",1,2,0.9,1.8),
    ("Spain","Czechia",0,0,1.8,0.6),("Uruguay","Saudi Arabia",2,1,2.0,1.0),
    # Group C
    ("Argentina","Romania",3,0,2.8,0.4),("Ivory Coast","Morocco",1,1,1.2,1.4),
    ("Argentina","Ivory Coast",2,1,2.0,1.1),("Romania","Morocco",0,2,0.7,1.9),
    ("Argentina","Morocco",2,0,1.9,0.7),("Ivory Coast","Romania",1,0,1.4,0.8),
    # Group D
    ("USA","Paraguay",4,1,3.2,0.9),("Turkey","Australia",1,2,1.3,1.6),
    ("Paraguay","Turkey",1,0,0.7,1.1),("Australia","USA",0,1,0.8,1.3),
    ("Paraguay","Australia",0,0,0.6,0.9),("Turkey","USA",1,2,1.0,1.7),
    # Group E
    ("Germany","Curacao",7,1,5.9,0.8),("Ecuador","Ivory Coast",2,1,1.8,1.3),
    ("Germany","Ivory Coast",2,1,2.3,1.4),("Ecuador","Curacao",3,0,2.5,0.4),
    ("Germany","Ecuador",1,2,1.6,1.8),("Ivory Coast","Curacao",2,0,1.9,0.5),
    # Group F
    ("France","Senegal",3,1,2.7,0.9),("Denmark","Tunisia",1,0,1.4,0.8),
    ("France","Denmark",2,0,2.1,0.6),("Tunisia","Senegal",0,1,0.8,1.2),
    ("France","Tunisia",1,0,1.7,0.5),("Senegal","Denmark",1,1,1.3,1.2),
    # Group G
    ("Brazil","Nigeria",3,1,2.9,0.8),("Japan","South Korea",2,1,1.8,1.3),
    ("Brazil","South Korea",2,0,2.2,0.7),("Japan","Nigeria",1,0,1.3,0.9),
    ("Brazil","Japan",1,0,1.5,1.0),("Nigeria","South Korea",1,2,1.1,1.7),
    # Group H
    ("Portugal","Algeria",3,0,2.8,0.5),("Croatia","Cameroon",2,0,1.9,0.6),
    ("Portugal","Croatia",1,1,1.6,1.2),("Algeria","Cameroon",1,0,1.1,0.8),
    ("Portugal","Cameroon",2,0,2.0,0.4),("Croatia","Algeria",2,1,1.7,1.0),
    # Group I
    ("Sweden","Cape Verde",2,0,1.9,0.6),("Egypt","Serbia",1,1,1.2,1.3),
    ("Sweden","Egypt",1,0,1.4,0.9),("Cape Verde","Serbia",0,2,0.5,1.8),
    ("Sweden","Serbia",2,1,1.8,1.1),("Egypt","Cape Verde",2,0,1.7,0.6),
    # Group J
    ("England","Croatia",4,2,3.1,1.4),("Netherlands","Tunisia",3,0,2.6,0.5),
    ("England","Netherlands",1,1,1.5,1.4),("Croatia","Tunisia",1,0,1.2,0.7),
    ("England","Tunisia",2,0,1.9,0.4),("Netherlands","Croatia",2,0,2.0,0.8),
    # Group K
    ("Belgium","Norway",2,1,2.1,1.2),("Colombia","Peru",2,0,1.8,0.7),
    ("Belgium","Colombia",1,1,1.4,1.3),("Norway","Peru",3,0,2.4,0.5),
    ("Belgium","Peru",2,0,1.9,0.5),("Norway","Colombia",1,2,1.1,1.7),
    # Group L
    ("Switzerland","Ghana",2,0,1.8,0.6),("Austria","Bosnia",1,1,1.3,1.2),
    ("Switzerland","Austria",1,1,1.4,1.3),("Ghana","Bosnia",0,1,0.7,1.0),
    ("Switzerland","Bosnia",2,0,1.7,0.6),("Austria","Ghana",2,1,1.9,1.0),
]

def fit_strengths(matches=WC2026_MATCHES, base=BASE_RATE,
                  xg_cap=XG_CAP, decay=RECENCY_DECAY, verbose=True):
    """
    MLE fit of attack/defence parameters.
    Uses capped xG and recency weighting.
    """
    teams = sorted(set(t for m in matches for t in [m[0], m[1]]))
    n     = len(teams)
    idx   = {t: i for i, t in enumerate(teams)}

    # Build recency weights per team
    home_history = {}
    for i, m in enumerate(matches):
        home_history.setdefault(m[0], []).append(i)

    def weight(match_idx, home):
        hist = home_history.get(home, [match_idx])
        pos  = hist.index(match_idx) if match_idx in hist else 0
        age  = len(hist) - 1 - pos
        return decay ** age

    def neg_ll(params):
        atk, dfn = params[:n], params[n:]
        total = 0.0
        for mi, (h, a, hg, ag, hxg, axg) in enumerate(matches):
            lh = max(0.05, base * atk[idx[h]] * dfn[idx[a]])
            la = max(0.05, base * atk[idx[a]] * dfn[idx[h]])
            oh = min(hxg if hxg else float(hg), xg_cap)
            oa = min(axg if axg else float(ag), xg_cap)
            p  = (pois(lh, round(oh)) * pois(la, round(oa)) *
                  dc_tau(round(oh), round(oa), lh, la))
            if p > 0:
                total += weight(mi, h) * math.log(p)
        return -total

    x0  = np.ones(2 * n)
    res = minimize(neg_ll, x0,
                   method="SLSQP",
                   bounds=[(0.1, 4.0)] * (2*n),
                   constraints=[{"type":"eq","fun": lambda p: np.mean(p[:n])-1.0}],
                   options={"maxiter":2000,"ftol":1e-10})
    if verbose:
        print(f"  MLE {'converged' if res.success else 'best estimate'}. "
              f"Log-likelihood: {-res.fun:.1f}. Teams: {n}.")

    return {t: {"attack": round(float(res.x[i]),4),
                "defence":round(float(res.x[n+i]),4)}
            for i, t in enumerate(teams)}


# ─────────────────────────────────────────────────────────────────
# 4. MARKET-ANCHORED LAMBDA
# ─────────────────────────────────────────────────────────────────

def compute_anchored_lambdas(home_team, away_team, team_params,
                              market_odds_3way=None, market_odds_ou=None,
                              is_knockout=True, neutral_venue=True):
    """
    Compute expected goals anchored to the betting market's O/U line.

    WHY: With only 3 group stage games per team, our raw Poisson lambdas
    can be too extreme. The O/U market aggregates thousands of sharp bettors.
    We solve for λ_total from the market, then split using our model's ratio.

    Returns (lam_home, lam_away, model_lam_h, model_lam_a)
    """
    hp = team_params[home_team]
    ap = team_params[away_team]

    ko_adj   = 0.92 if is_knockout  else 1.0
    home_adv = 1.00 if neutral_venue else 1.12

    # Raw model lambdas
    lam_h_model = BASE_RATE * hp["attack"] * ap["defence"] * home_adv * ko_adj
    lam_a_model = BASE_RATE * ap["attack"] * hp["defence"] * ko_adj

    # If we have an O/U market, anchor to it
    if market_odds_ou:
        fair_ou, _ = remove_vig(market_odds_ou)
        lam_total_market = solve_lambda_from_ou(fair_ou["under"])

        # Split using model ratio
        ratio_h  = lam_h_model / (lam_h_model + lam_a_model)
        lam_h    = lam_total_market * ratio_h
        lam_a    = lam_total_market * (1 - ratio_h)
    else:
        lam_h, lam_a = lam_h_model, lam_a_model

    return lam_h, lam_a, lam_h_model, lam_a_model


# ─────────────────────────────────────────────────────────────────
# 5. PLAYER SHOT SHARE DATABASE
# ─────────────────────────────────────────────────────────────────

SHOT_SHARES = {
    "Kai Havertz":       {"team":"Germany",     "share":0.22},
    "Florian Wirtz":     {"team":"Germany",     "share":0.17},
    "Jamal Musiala":     {"team":"Germany",     "share":0.16},
    "Leroy Sané":        {"team":"Germany",     "share":0.14},
    "Deniz Undav":       {"team":"Germany",     "share":0.13},
    "Julio Enciso":      {"team":"Paraguay",    "share":0.35},
    "Miguel Almirón":    {"team":"Paraguay",    "share":0.22},
    "Gabriel Avalos":    {"team":"Paraguay",    "share":0.18},
    "Kylian Mbappé":     {"team":"France",      "share":0.30},
    "Antoine Griezmann": {"team":"France",      "share":0.18},
    "Marcus Thuram":     {"team":"France",      "share":0.16},
    "Harry Kane":        {"team":"England",     "share":0.28},
    "Jude Bellingham":   {"team":"England",     "share":0.20},
    "Phil Foden":        {"team":"England",     "share":0.15},
    "Lionel Messi":      {"team":"Argentina",   "share":0.32},
    "Julián Álvarez":    {"team":"Argentina",   "share":0.22},
    "Lautaro Martínez":  {"team":"Argentina",   "share":0.20},
    "Lamine Yamal":      {"team":"Spain",       "share":0.22},
    "Álvaro Morata":     {"team":"Spain",       "share":0.20},
    "Cody Gakpo":        {"team":"Netherlands", "share":0.24},
    "Vinicius Jr":       {"team":"Brazil",      "share":0.26},
    "Rodrygo":           {"team":"Brazil",      "share":0.18},
    "Cristiano Ronaldo": {"team":"Portugal",    "share":0.28},
    "Rafael Leão":       {"team":"Portugal",    "share":0.20},
}


# ─────────────────────────────────────────────────────────────────
# 6. QUESTION CLASSIFIER
# ─────────────────────────────────────────────────────────────────

def classify(question, home, away):
    """Map a SportsPredict question string to a model output key + params."""
    q, h, a = question.lower(), home.lower(), away.lower()

    # Fixed-shape markets
    mapping = {
        (f"will {h} win",        "regulation"): "home_win",
        (f"will {a} win",        "regulation"): "away_win",
        ("ahead at halftime",    h):            "home_lead_ht",
        ("score the first goal", h):            "home_scores_first",
        ("both teams score",     ""):           "btts",
        ("score in both halves", h):            "home_scores_both_halves",
        ("goal be scored before","hydration"):  "goal_before_break",
        ("stoppage",             ""):           "stoppage_time_goal",
        ("red card",             ""):           "red_card",
    }
    for (kw1, kw2), key in mapping.items():
        if kw1 in q and (not kw2 or kw2 in q):
            return key, None

    # Total goals with arbitrary threshold — N=3 keeps market-blend logic
    m = re.search(r"(\d+)\s+or more total goals", q)
    if m:
        thr = int(m.group(1))
        return ("over_2_5", None) if thr == 3 else ("total_goals_threshold", {"thr": thr})

    # Counting stats: "N or more <category>" — team-level or player SOT
    m = re.search(r"(\d+)\s+or more\s+(shots on target|corner kicks?|total cards|offside)", q)
    if m:
        thr, stat = int(m.group(1)), m.group(2)
        # Player SOT props include "(Team)" parens; team-level questions don't
        if "shots on target" in stat and "(" in question:
            name = question.split("(")[0].replace("Will ", "").strip()
            team = "home" if h in q else "away"
            return f"player_{thr}plus_sot", {"name": name, "team": team, "thr": thr}
        if "shots on target" in stat:
            cat = "home_sot" if h in q else "away_sot"
        elif "corner" in stat:
            cat = "home_corners" if h in q else "away_corners"
        elif "card" in stat:
            cat = "cards"
        else:
            cat = "offsides"
        return "threshold", {"category": cat, "thr": thr}

    # Player goal / assist props
    if "score a goal" in q and "excluding own goals" in q:
        name = question.split("(")[0].replace("Will ", "").strip()
        team = "home" if h in q else "away"
        return "player_goal", {"name": name, "team": team}

    if "score or assist" in q:
        name = question.split("(")[0].replace("Will ", "").strip()
        team = "home" if h in q else "away"
        return "player_goal_or_assist", {"name": name, "team": team}

    return None, None


def resolve_prediction(key, pd, predictions, lambdas, lam_h, lam_a):
    """
    Convert a classify() result to a probability in [0, 1], or None.

    Shared by pipeline_v2.run(), run.run_full_pipeline(), and
    scheduler.predict_and_submit() so all three price markets identically.
    """
    # Direct lookup in pre-built predictions dict (covers all canonical markets
    # and player props pre-computed at the top of each pipeline)
    if key in predictions:
        return predictions[key]

    # Threshold-flexible counting stats (corners/cards/offsides/SOT)
    if key == "threshold" and pd:
        lam = lambdas.get(pd["category"])
        if lam is not None:
            return 1 - sum(pois(lam, k) for k in range(pd["thr"]))

    # Total goals at a non-standard threshold (e.g. over 3.5 = "4 or more")
    if key == "total_goals_threshold" and pd:
        return 1 - sum(pois(lam_h + lam_a, k) for k in range(pd["thr"]))

    if key == "red_card":
        return red_card_p(lambdas.get("cards", 3.8))

    if key == "stoppage_time_goal":
        return stoppage_time_goal_p(lam_h, lam_a)

    # Player props resolved by name from predictions dict
    if pd:
        name = pd.get("name", "")
        thr  = pd.get("thr", 2)
        if "goal_or_assist" in (key or ""):
            return predictions.get(f"{name}_goal_or_assist")
        if "goal" in (key or ""):
            return predictions.get(f"{name}_goal")
        if "sot" in (key or ""):
            return predictions.get(f"{name}_{thr}plus_sot")

    return None


# ─────────────────────────────────────────────────────────────────
# 7. KAPBOT API
# ─────────────────────────────────────────────────────────────────

def api(method, endpoint, key, payload=None):
    try:
        import requests
        h = {"Authorization":f"Bearer {key}","Content-Type":"application/json"}
        url = KAPBOT_BASE_URL + endpoint
        r = (requests.get(url, headers=h, timeout=15) if method=="GET"
             else requests.post(url, headers=h, json=payload, timeout=15))
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  API error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────
# 8. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────

def run(home_team, away_team,
        market_3way=None, market_ou=None,
        api_key=None, dry_run=False, auto_approve=False,
        is_knockout=True, neutral_venue=True):

    sep = "="*62
    print(f"\n{sep}")
    print(f"  PROBABILITY CUP MODEL  |  {home_team} vs {away_team}")
    print(sep)

    # ── 1. Fit team strengths ──────────────────────────────────────
    print("\n[1/5] Fitting team strengths from World Cup xG data...")
    params = fit_strengths(verbose=True)

    for team in [home_team, away_team]:
        if team not in params:
            print(f"\n  ERROR: '{team}' not in dataset.")
            print(f"  Available teams: {', '.join(sorted(params.keys()))}")
            return

    hp, ap = params[home_team], params[away_team]
    print(f"  {home_team:<22} attack={hp['attack']:.3f}  defence={hp['defence']:.3f}")
    print(f"  {away_team:<22} attack={ap['attack']:.3f}  defence={ap['defence']:.3f}")

    # ── 2. Compute expected goals ─────────────────────────────────
    print("\n[2/5] Computing market-anchored expected goals...")
    lam_h, lam_a, lam_h_raw, lam_a_raw = compute_anchored_lambdas(
        home_team, away_team, params, market_3way, market_ou,
        is_knockout, neutral_venue
    )
    print(f"  Raw model:  λ_{home_team}={lam_h_raw:.3f}  λ_{away_team}={lam_a_raw:.3f}")
    if market_ou:
        print(f"  Anchored:   λ_{home_team}={lam_h:.3f}  λ_{away_team}={lam_a:.3f}  "
              f"(total={lam_h+lam_a:.3f}, market-consistent)")
    else:
        print("  (No O/U market provided — using raw model lambdas)")

    # ── 3. Score matrix + market derivation ───────────────────────
    print("\n[3/5] Building score matrix and deriving markets...")
    mat = score_matrix(lam_h, lam_a)
    mkts = derive_all_markets(mat, lam_h, lam_a)

    # Most likely scorelines
    n = MAX_GOALS + 1
    scorelines = sorted(
        [(mat[i][j], i, j) for i in range(n) for j in range(n)],
        reverse=True
    )[:6]
    print(f"  Most likely scorelines:")
    for prob, i, j in scorelines:
        print(f"    {i}-{j}  {prob*100:.1f}%")

    # ── 4. Build prediction table ──────────────────────────────────
    print("\n[4/5] Building prediction table...")
    fair_3way = remove_vig(market_3way)[0] if market_3way else None
    fair_ou   = remove_vig(market_ou)[0]   if market_ou   else None

    def final_p(model_key, mkt_p=None):
        mp = mkts.get(model_key, 0.5)
        fp = (MODEL_WEIGHT * mp + MARKET_WEIGHT * mkt_p) if mkt_p else mp
        return to_platform_int(fp)

    def to_platform_int(p):
        return max(1, min(99, round(p * 100)))

    predictions = {
        "home_win":              final_p("home_win", fair_3way["home"] if fair_3way else None),
        "away_win":              final_p("away_win", fair_3way["away"] if fair_3way else None),
        "home_lead_ht":          final_p("home_lead_ht"),
        "home_scores_first":     final_p("home_scores_first"),
        "over_2_5":              final_p("over_2_5", fair_ou["over"] if fair_ou else None),
        "btts":                  final_p("btts"),
        "home_scores_both_halves": final_p("home_scores_both_halves"),
        "goal_before_break":     final_p("goal_before_break"),
        "home_7plus_sot":        final_p("home_7plus_sot"),
        "home_7plus_corners":    final_p("home_7plus_corners"),
        "cards_4plus":           final_p("cards_4plus"),
        "offsides_3plus":        final_p("offsides_3plus"),
    }

    # Player props
    for player, data in SHOT_SHARES.items():
        if data["team"] not in (home_team, away_team):
            continue
        lam = lam_h if data["team"] == home_team else lam_a
        share = data["share"]
        p_goal   = player_goal_p(lam, share)
        p_assist = p_goal * 0.65
        predictions[f"{player}_goal"]           = to_platform_int(p_goal)
        predictions[f"{player}_goal_or_assist"] = to_platform_int(p_goal + p_assist - p_goal*p_assist)
        predictions[f"{player}_1plus_sot"]      = to_platform_int(player_sot_p(lam, share, 1))
        predictions[f"{player}_2plus_sot"]      = to_platform_int(player_sot_p(lam, share, 2))

    # ── 5. Review dashboard ────────────────────────────────────────
    print(f"\n[5/5] Review dashboard:")
    print(f"\n  {sep}")
    print(f"  λ {home_team}: {lam_h:.2f}  |  λ {away_team}: {lam_a:.2f}")
    print(f"  {'#':<3} {'Question':<46} {'P':>4}  {'Model%':>7}  {'Mkt%':>6}")
    print(f"  {'─'*68}")

    # Fetch markets via REST API
    markets = []
    client = KapbotClient(api_key) if api_key else None
    if client:
        target_match = client.find_match(home_team)
        if target_match:
            markets = client.get_markets(target_match["id"])
            print(f"  Fetched {len(markets)} markets from API.")
        else:
            print(f"  Match '{home_team}' not found in open markets — using sample questions.")

    # If no markets from API, generate sample structure
    if not markets:
        markets = [
            {"id": f"s{i}", "lobby_id": "sample",
             "question": q, "market_status": "open"}
            for i, q in enumerate([
                f"Will {home_team} win in regulation (90 minutes + stoppage time)?",
                f"Will {home_team} be ahead at halftime?",
                f"Will {home_team} score the first goal of the match?",
                "Will the match have 3 or more total goals in regulation (90 minutes + stoppage time)?",
                "Will both teams score in regulation (90 minutes + stoppage time)?",
                f"Will {home_team} score in both halves in regulation (90 minutes + stoppage time)?",
                "Will a goal be scored before the first hydration break?",
                f"Will {home_team} have 7 or more shots on target in regulation (90 minutes + stoppage time)?",
                f"Will {home_team} have 7 or more corner kicks in regulation (90 minutes + stoppage time)?",
                "Will there be 4 or more total cards shown in regulation (90 minutes + stoppage time)?",
                "Will there be 3 or more offside calls in regulation (90 minutes + stoppage time)?",
            ])
        ]

    lambdas = mkts.get("_lambdas", {})
    submission_list = []
    for i, mkt in enumerate(markets, 1):
        q       = mkt["question"]
        key, pd = classify(q, home_team, away_team)
        prob    = resolve_prediction(key, pd, predictions, lambdas, lam_h, lam_a)
        p_int   = to_platform_int(prob) if prob is not None else 50
        flag    = "" if key else " (?)"

        # Model and market % for display
        if key in mkts:
            model_pct = f"{mkts[key]*100:.0f}%"
        elif prob is not None:
            model_pct = f"{prob*100:.0f}%"
        else:
            model_pct = "  —"
        mkt_pct = "—"
        if key == "home_win"  and fair_3way: mkt_pct = f"{fair_3way['home']*100:.0f}%"
        elif key == "over_2_5" and fair_ou:  mkt_pct = f"{fair_ou['over']*100:.0f}%"

        dq = (q[:45] + "…") if len(q) > 45 else q
        print(f"  {i:<3} {dq:<46} {p_int:>3}{flag}  {model_pct:>7}  {mkt_pct:>6}")

        submission_list.append({
            "market_id":   mkt["id"],
            "lobby_id":    mkt["lobby_id"],
            "probability": p_int,
            "question":    q,
        })

    print(f"  {'─'*68}")
    print(f"  {len(submission_list)} markets ready.\n")

    if dry_run:
        print("  DRY RUN — predictions computed but NOT submitted.")
        print("  Remove --dry-run to submit after your review.\n")
        return predictions

    confirm = "y" if auto_approve else input("  Submit all predictions? [y/N]: ").strip().lower()

    if confirm == "y" and client:
        payload = [{"market_id": p["market_id"],
                    "lobby_id":  p["lobby_id"],
                    "probability": p["probability"]}
                   for p in submission_list
                   if p.get("lobby_id") != "sample"]
        if not payload:
            print("\n  No real markets to submit (all sample questions).")
        else:
            result = client.submit_batch(payload)
            if result:
                print(f"\n  Submitted: {result.get('succeeded','?')} OK, "
                      f"{result.get('failed','?')} failed.")
            else:
                print("\n  Submission failed. Check API key and network.")
    elif not api_key:
        print("  Set KAPBOT_API_KEY environment variable to submit.")
    else:
        print("  Cancelled.")

    return predictions


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probability Cup model")
    parser.add_argument("--home",    default="Germany")
    parser.add_argument("--away",    default="Paraguay")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--approve", action="store_true")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    run(
        home_team     = args.home,
        away_team     = args.away,
        market_3way   = {"home": -285, "draw": 400, "away": 800},
        market_ou     = {"over": -145, "under": 105},
        api_key       = args.api_key or os.environ.get("KAPBOT_API_KEY"),
        dry_run       = args.dry_run,
        auto_approve  = args.approve,
        is_knockout   = True,
        neutral_venue = True,
    )

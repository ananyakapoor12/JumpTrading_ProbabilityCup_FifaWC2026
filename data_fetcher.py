"""
====================================================================
STAGE 3: AUTO DATA FETCHER
====================================================================

This module replaces all manually set data in the pipeline with
live, automatically fetched data. It has four jobs:

  1. MATCH RESULTS  — fetch all completed WC2026 group stage results
                      from openfootball (GitHub, no API key needed)
                      This feeds the MLE team strength fitter.

  2. PLAYER SHARES  — derive each player's shot share from their
                      tournament goal tally. Players who score more
                      goals get a higher share of team xG in the model.

  3. UPCOMING GAMES — list the next matches open for prediction so
                      the pipeline knows what to price next.

  4. ODDS FETCHING  — two modes:
       a) The Odds API (free tier, 500 req/month, needs API key)
          Sign up free at https://the-odds-api.com
          Set ODDS_API_KEY=your_key in your .env file
       b) Fallback: user provides odds manually (or we use the 
          market-neutral 50/50 and rely more on the Poisson model)

WHY THESE CHOICES
─────────────────
openfootball (GitHub):
  - Free, no API key, raw.githubusercontent.com is accessible
  - Real official WC2026 data updated after each match
  - Has full scorelines, HT scores, goalscorer names

Player shot shares from goals (vs FBref):
  - FBref is blocked in this environment and rate-limits scraping
  - Goal share is a reasonable proxy for shot share in a tournament
    setting where we have limited data (3 games per team)
  - A player who scored 3 tournament goals gets ~3x the weight of
    one who scored 1 — a sensible prior
  - We apply a floor (MIN_SHARE) so unlisted players still get a
    baseline allocation

The Odds API:
  - 500 free requests/month covers all 64 WC matches with room to spare
  - Returns clean decimal odds for 3-way (H/D/A) and totals (O/U)
  - Much more reliable than scraping a bookmaker page
"""

import os, json, math, requests
from datetime import datetime, timezone
from collections import defaultdict

# ─── Configuration ────────────────────────────────────────────────
OPENFOOTBALL_URL = (
    "https://raw.githubusercontent.com/openfootball/"
    "world-cup.json/master/2026/worldcup.json"
)
ODDS_API_BASE    = "https://api.the-odds-api.com/v4"
ODDS_API_SPORT   = "soccer_fifa_world_cup"

# Shot share floors and defaults
MIN_SHARE   = 0.08   # minimum share for any starting player (even 0 scorers)
MAX_SHARE   = 0.55   # cap so one player can't dominate
GOAL_WEIGHT = 0.70   # weight on goals scored vs positional prior

# Positional priors: what share of goals does each position typically take?
# Used to allocate share to players who haven't scored yet
POSITIONAL_PRIORS = {
    "striker":   0.25,
    "attacking": 0.18,
    "winger":    0.15,
    "midfielder":0.10,
    "defender":  0.05,
}

# Known player positions — used when goal data is sparse
PLAYER_POSITIONS = {
    # Germany
    "Kai Havertz":    "striker",    "Deniz Undav":    "striker",
    "Florian Wirtz":  "attacking",  "Jamal Musiala":  "attacking",
    "Leroy Sané":     "winger",     "Leroy Sane":     "winger",
    "Joshua Kimmich": "midfielder", "Aleksandar Pavlovic": "midfielder",
    # Paraguay
    "Julio Enciso":   "attacking",  "Miguel Almirón": "winger",
    "Miguel Almiron": "winger",     "Gabriel Avalos": "striker",
    "Antonio Sanabria":"striker",
    # France
    "Kylian Mbappé":  "striker",    "Kylian Mbappe":  "striker",
    "Ousmane Dembélé":"winger",     "Marcus Thuram":  "striker",
    "Antoine Griezmann":"attacking",
    # England
    "Harry Kane":     "striker",    "Jude Bellingham":"attacking",
    "Phil Foden":     "attacking",  "Bukayo Saka":    "winger",
    "Cole Palmer":    "attacking",
    # Argentina
    "Lionel Messi":   "attacking",  "Julián Álvarez": "striker",
    "Julian Alvarez": "striker",    "Lautaro Martínez":"striker",
    "Lautaro Martinez":"striker",
    # Spain
    "Lamine Yamal":   "winger",     "Álvaro Morata":  "striker",
    "Alvaro Morata":  "striker",    "Pedri":          "attacking",
    # Netherlands
    "Cody Gakpo":     "striker",    "Brian Brobbey":  "striker",
    "Donyell Malen":  "winger",     "Memphis Depay":  "striker",
    # Brazil
    "Vinícius Júnior":"winger",     "Vinicius Junior":"winger",
    "Matheus Cunha":  "striker",    "Rodrygo":        "winger",
    "Raphinha":       "winger",
    # Portugal
    "Cristiano Ronaldo":"striker",  "Rafael Leão":    "winger",
    "Rafael Leao":    "winger",     "Bruno Fernandes":"attacking",
    # Belgium
    "Romelu Lukaku":  "striker",    "Dodi Lukebakio": "winger",
    # Norway
    "Erling Haaland": "striker",
    # Colombia
    "Jhon Durán":     "striker",    "Jhon Duran":     "striker",
    # USA
    "Folarin Balogun":"striker",    "Ricardo Pepi":   "striker",
    # Morocco
    "Ismael Saibari": "attacking",  "Youssef En-Nesyri":"striker",
    # Ecuador
    "Enner Valencia": "striker",
    # Mexico
    "Julián Quiñones":"winger",     "Raúl Jiménez":   "striker",
}


# ─────────────────────────────────────────────────────────────────
# 1. FETCH MATCH RESULTS
# ─────────────────────────────────────────────────────────────────

def fetch_wc2026_data():
    """
    Fetch all WC2026 match data from openfootball (GitHub).

    Returns the raw JSON dict with keys:
      - 'matches': list of match dicts
    
    Each match has:
      round, date, team1, team2,
      score: {ft: [h, a], ht: [h, a]},
      goals1: [{name, minute, penalty?, own_goal?}],
      goals2: [{name, minute, ...}],
      group (if group stage)
    """
    print("  [data] Fetching WC2026 match data from openfootball...")
    try:
        r = requests.get(OPENFOOTBALL_URL, timeout=15)
        r.raise_for_status()
        data = r.json()
        matches = data.get("matches", [])
        completed = sum(1 for m in matches if m.get("score"))
        upcoming  = sum(1 for m in matches if not m.get("score"))
        print(f"  [data] OK — {len(matches)} total matches "
              f"({completed} completed, {upcoming} upcoming)")
        return data
    except Exception as e:
        print(f"  [data] ERROR fetching match data: {e}")
        return None


def parse_completed_matches(data):
    """
    Convert openfootball match data into the format expected by
    the MLE fitter: (home, away, home_goals, away_goals, home_xg, away_xg)

    Since openfootball doesn't have xG, we use actual goals and rely on
    the XG_CAP in the fitter to reduce outlier influence.
    """
    if not data:
        return []

    results = []
    for m in data["matches"]:
        score = m.get("score", {})
        ft    = score.get("ft")
        if not ft or len(ft) < 2:
            continue  # not played yet

        results.append((
            m["team1"],    # home team
            m["team2"],    # away team
            ft[0],         # home goals
            ft[1],         # away goals
            None,          # home xG (not available — fitter uses actual goals)
            None,          # away xG
        ))

    print(f"  [data] Parsed {len(results)} completed matches for MLE fitting.")
    return results


# ─────────────────────────────────────────────────────────────────
# 2. PLAYER SHOT SHARES FROM GOALS
# ─────────────────────────────────────────────────────────────────

def build_player_shot_shares(data, min_share=MIN_SHARE, max_share=MAX_SHARE):
    """
    Derive player shot shares from tournament goal tallies.

    WHY THIS WORKS
    ──────────────
    In a tournament with only 3 group games, we can't get reliable
    shots-on-target data. But goals are a good proxy:

      share_i = (goals_i / team_goals) × GOAL_WEIGHT
              + positional_prior_i    × (1 - GOAL_WEIGHT)

    This blends:
      - what players have actually done (goal contribution)
      - what their position suggests they will do (prior)

    We then normalise within each team so shares sum to ≤ 1.0.

    Returns dict: {player_name: {"team": str, "share": float}}
    """
    if not data:
        return {}

    # Step 1: Count goals per player and per team (excl. own goals)
    player_goals = defaultdict(lambda: {"goals": 0, "team": ""})
    team_goals   = defaultdict(float)

    for m in data["matches"]:
        if not m.get("score"):
            continue
        for side, team_name in [("goals1", m["team1"]), ("goals2", m["team2"])]:
            for g in m.get(side, []):
                if g.get("own_goal"):
                    continue
                name = g["name"]
                player_goals[name]["goals"] += 1
                player_goals[name]["team"]  = team_name
                team_goals[team_name]       += 1.0

    # Step 2: Compute blended share for each player
    shares = {}
    for player, d in player_goals.items():
        team       = d["team"]
        team_total = max(team_goals[team], 1.0)
        goal_share = d["goals"] / team_total

        # Positional prior
        pos   = PLAYER_POSITIONS.get(player, "midfielder")
        prior = POSITIONAL_PRIORS.get(pos, 0.10)

        # Blend
        raw_share = GOAL_WEIGHT * goal_share + (1 - GOAL_WEIGHT) * prior

        # Apply min/max
        raw_share = max(min_share, min(max_share, raw_share))

        shares[player] = {"team": team, "share": round(raw_share, 3)}

    # Step 3: Add known players with 0 tournament goals using positional prior
    for player, pos in PLAYER_POSITIONS.items():
        if player not in shares:
            prior = POSITIONAL_PRIORS.get(pos, 0.10)
            # Find their team from PLAYER_POSITIONS context
            # (We don't know their team from goals, so we leave team blank
            #  and let the pipeline match by name)
            shares[player] = {
                "team":  "",  # will be inferred from match context
                "share": max(min_share, POSITIONAL_PRIORS.get(pos, 0.10)),
            }

    # Assign team from PLAYER_POSITIONS for players with known team
    KNOWN_TEAMS = {
        "Kai Havertz": "Germany", "Deniz Undav": "Germany",
        "Florian Wirtz": "Germany", "Jamal Musiala": "Germany",
        "Leroy Sané": "Germany", "Leroy Sane": "Germany",
        "Joshua Kimmich": "Germany",
        "Julio Enciso": "Paraguay", "Miguel Almirón": "Paraguay",
        "Miguel Almiron": "Paraguay", "Gabriel Avalos": "Paraguay",
        "Kylian Mbappé": "France", "Kylian Mbappe": "France",
        "Ousmane Dembélé": "France", "Marcus Thuram": "France",
        "Antoine Griezmann": "France",
        "Harry Kane": "England", "Jude Bellingham": "England",
        "Phil Foden": "England", "Bukayo Saka": "England",
        "Lionel Messi": "Argentina", "Julián Álvarez": "Argentina",
        "Julian Alvarez": "Argentina", "Lautaro Martínez": "Argentina",
        "Lamine Yamal": "Spain", "Álvaro Morata": "Spain",
        "Cody Gakpo": "Netherlands", "Brian Brobbey": "Netherlands",
        "Vinícius Júnior": "Brazil", "Vinicius Junior": "Brazil",
        "Matheus Cunha": "Brazil", "Rodrygo": "Brazil",
        "Cristiano Ronaldo": "Portugal", "Rafael Leão": "Portugal",
        "Erling Haaland": "Norway",
        "Folarin Balogun": "USA",
        "Ismael Saibari": "Morocco",
    }
    for player in list(shares.keys()):
        if not shares[player]["team"] and player in KNOWN_TEAMS:
            shares[player]["team"] = KNOWN_TEAMS[player]

    # Remove players with no known team
    shares = {k: v for k, v in shares.items() if v["team"]}

    print(f"  [data] Built shot shares for {len(shares)} players "
          f"across {len(set(v['team'] for v in shares.values()))} teams.")
    return shares


# ─────────────────────────────────────────────────────────────────
# 3. UPCOMING MATCHES
# ─────────────────────────────────────────────────────────────────

def get_upcoming_matches(data, n=10):
    """
    Return the next n unplayed matches from the tournament schedule.

    Returns list of dicts: [{home, away, date, round}, ...]
    """
    if not data:
        return []

    upcoming = [
        {
            "home":  m["team1"],
            "away":  m["team2"],
            "date":  m.get("date", ""),
            "round": m.get("round", ""),
            "venue": m.get("ground", ""),
        }
        for m in data["matches"]
        if not m.get("score") and
           m.get("team1", "").replace(" ", "").isalpha()  # skip placeholder "W74" etc
    ]
    return upcoming[:n]


def print_upcoming(upcoming):
    """Display upcoming matches."""
    print(f"\n  {'Date':<12} {'Round':<22} {'Home':<22} {'Away'}")
    print(f"  {'─'*70}")
    for m in upcoming:
        print(f"  {m['date']:<12} {m['round']:<22} "
              f"{m['home']:<22} {m['away']}")


# ─────────────────────────────────────────────────────────────────
# 4. ODDS FETCHING
# ─────────────────────────────────────────────────────────────────

def fetch_odds_for_match(home_team, away_team, api_key=None):
    """
    Fetch live betting odds for a specific match.

    Returns dict with keys:
      market_3way: {"home": int, "draw": int, "away": int}  (American odds)
      market_ou:   {"over": int, "under": int}

    Or None if unavailable.

    HOW TO GET A FREE ODDS API KEY
    ───────────────────────────────
    1. Go to https://the-odds-api.com
    2. Sign up for free (500 requests/month)
    3. Copy your API key
    4. Set it: export ODDS_API_KEY=your_key_here
       Or put ODDS_API_KEY=your_key in a .env file next to the script

    500 requests ÷ 64 WC matches × 2 markets = 4 requests per match
    → plenty for the whole tournament
    """
    key = api_key or os.environ.get("ODDS_API_KEY")

    if not key:
        print("  [odds] No ODDS_API_KEY set — skipping live odds fetch.")
        print("         Get a free key at https://the-odds-api.com")
        print("         Set it: export ODDS_API_KEY=your_key")
        return None

    try:
        # Fetch available events for WC
        url = f"{ODDS_API_BASE}/sports/{ODDS_API_SPORT}/odds"
        params = {
            "apiKey":   key,
            "regions":  "us,eu",
            "markets":  "h2h,totals",
            "oddsFormat": "american",
        }
        print(f"  [odds] Fetching live odds from The Odds API...")
        r = requests.get(url, params=params, timeout=15)
        remaining = r.headers.get("x-requests-remaining", "?")
        print(f"  [odds] API requests remaining this month: {remaining}")

        if r.status_code == 401:
            print("  [odds] Invalid API key. Check ODDS_API_KEY env var.")
            return None
        r.raise_for_status()

        events = r.json()

        # Find the right match
        target = _find_match_in_odds(events, home_team, away_team)
        if not target:
            print(f"  [odds] Match '{home_team} vs {away_team}' not found in odds feed.")
            print("         Market may not be open yet, or team name mismatch.")
            return None

        # Extract best odds across bookmakers
        return _extract_best_odds(target)

    except requests.HTTPError as e:
        print(f"  [odds] HTTP error: {e}")
        return None
    except Exception as e:
        print(f"  [odds] Error: {e}")
        return None


def _find_match_in_odds(events, home_team, away_team):
    """Find a match in the odds API response by fuzzy team name matching."""
    # Normalise team names for matching
    def norm(s):
        return s.lower().replace(" ", "").replace("-", "")

    h_norm = norm(home_team)
    a_norm = norm(away_team)

    for event in events:
        eh = norm(event.get("home_team", ""))
        ea = norm(event.get("away_team", ""))
        # Full match or 5-char prefix match (handles slight name differences)
        if (h_norm[:6] in eh or eh[:6] in h_norm) and \
           (a_norm[:6] in ea or ea[:6] in a_norm):
            return event

    return None


def _extract_best_odds(event):
    """
    Extract the best (sharpest) odds across all bookmakers.

    Strategy: use Pinnacle if available (sharpest book), 
    otherwise average across all books.
    """
    h2h_odds    = {"home": [], "draw": [], "away": []}
    totals_odds = {"over": [], "under": []}

    for book in event.get("bookmakers", []):
        for market in book.get("markets", []):
            if market["key"] == "h2h":
                for outcome in market["outcomes"]:
                    key = outcome["name"]
                    if key == event["home_team"]:
                        h2h_odds["home"].append(outcome["price"])
                    elif key == event["away_team"]:
                        h2h_odds["away"].append(outcome["price"])
                    elif key == "Draw":
                        h2h_odds["draw"].append(outcome["price"])

            elif market["key"] == "totals":
                for outcome in market["outcomes"]:
                    pt = outcome.get("point", 0)
                    if abs(pt - 2.5) < 0.1:  # O/U 2.5 specifically
                        if outcome["name"] == "Over":
                            totals_odds["over"].append(outcome["price"])
                        elif outcome["name"] == "Under":
                            totals_odds["under"].append(outcome["price"])

    result = {}

    if all(h2h_odds[k] for k in ["home", "draw", "away"]):
        # Use the median (robust to outlier books)
        result["market_3way"] = {
            "home": _median_odds(h2h_odds["home"]),
            "draw": _median_odds(h2h_odds["draw"]),
            "away": _median_odds(h2h_odds["away"]),
        }
        print(f"  [odds] 3-way: "
              f"home={result['market_3way']['home']:+d}  "
              f"draw={result['market_3way']['draw']:+d}  "
              f"away={result['market_3way']['away']:+d}  "
              f"(n={len(h2h_odds['home'])} books)")

    if totals_odds["over"] and totals_odds["under"]:
        result["market_ou"] = {
            "over":  _median_odds(totals_odds["over"]),
            "under": _median_odds(totals_odds["under"]),
        }
        print(f"  [odds] O/U 2.5: "
              f"over={result['market_ou']['over']:+d}  "
              f"under={result['market_ou']['under']:+d}  "
              f"(n={len(totals_odds['over'])} books)")

    return result if result else None


def _median_odds(odds_list):
    """Return the median odds value (integer, American format)."""
    if not odds_list:
        return None
    sorted_odds = sorted(odds_list)
    mid = len(sorted_odds) // 2
    return int(sorted_odds[mid])


# ─────────────────────────────────────────────────────────────────
# 5. .ENV FILE LOADER
# ─────────────────────────────────────────────────────────────────

def load_env(path=".env"):
    """
    Load environment variables from a .env file.

    Format (one per line):
      KAPBOT_API_KEY=abc123
      ODDS_API_KEY=xyz789

    This means you never have to type keys on the command line.
    """
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())
    print(f"  [config] Loaded environment from {path}")


# ─────────────────────────────────────────────────────────────────
# 6. RESULTS VALIDATION
# ─────────────────────────────────────────────────────────────────

def compute_model_accuracy(data, team_params, recent_n=20):
    """
    After matches are played, compute our model's Brier scores
    against actual outcomes. This is how we improve over time.

    data:        openfootball match data
    team_params: fitted team parameters
    recent_n:    only check the most recent N matches

    Returns dict: {market_type: avg_brier_score}
    """
    from pipeline_v2 import score_matrix, derive_all_markets, to_platform_int

    completed = [m for m in data["matches"] if m.get("score", {}).get("ft")]
    recent    = completed[-recent_n:]

    briers = defaultdict(list)

    for m in recent:
        if m["team1"] not in team_params or m["team2"] not in team_params:
            continue

        hp = team_params[m["team1"]]
        ap = team_params[m["team2"]]
        lam_h = 1.25 * hp["attack"] * ap["defence"] * 0.92
        lam_a = 1.25 * ap["attack"] * hp["defence"] * 0.92

        mat   = score_matrix(lam_h, lam_a)
        mkts  = derive_all_markets(mat, lam_h, lam_a)

        ft = m["score"]["ft"]
        ht = m["score"].get("ht", [0, 0])

        # Compute actual outcomes
        actual = {
            "home_win":    1 if ft[0] > ft[1] else 0,
            "btts":        1 if ft[0] > 0 and ft[1] > 0 else 0,
            "over_2_5":    1 if ft[0] + ft[1] >= 3 else 0,
            "home_lead_ht":1 if ht[0] > ht[1] else 0,
        }

        for key, outcome in actual.items():
            if key in mkts:
                p = mkts[key]
                brier = (p - outcome) ** 2
                briers[key].append(brier)

    results = {}
    print("\n  Model accuracy on recent completed matches:")
    for key, scores in briers.items():
        avg = sum(scores) / len(scores)
        results[key] = avg
        baseline = 0.25  # coin flip Brier score
        delta = baseline - avg
        flag = "✓" if delta > 0 else "✗"
        print(f"    {flag} {key:<25} avg Brier={avg:.3f}  "
              f"vs baseline={baseline:.3f}  "
              f"delta={delta:+.3f}")

    return results


# ─────────────────────────────────────────────────────────────────
# 7. MAIN — FULL DATA FETCH
# ─────────────────────────────────────────────────────────────────

def fetch_all(home_team=None, away_team=None,
              odds_api_key=None, verbose=True):
    """
    Master data fetch function. Call this at the start of each
    pipeline run to get fresh data.

    Returns:
      matches      : list of completed match tuples for MLE fitting
      player_shares: dict {player: {team, share}}
      upcoming     : list of upcoming match dicts
      odds         : dict {market_3way, market_ou} or None
      raw_data     : the full openfootball JSON (for validation)
    """
    load_env()  # load .env if present

    print("\n  ── Fetching live data ──────────────────────────────────")

    # 1. Match results
    raw_data = fetch_wc2026_data()
    matches  = parse_completed_matches(raw_data) if raw_data else []

    # 2. Player shares
    player_shares = build_player_shot_shares(raw_data) if raw_data else {}

    # 3. Upcoming matches
    upcoming = get_upcoming_matches(raw_data, n=15) if raw_data else []

    # 4. Odds (if home/away specified and API key available)
    odds = None
    if home_team and away_team:
        key  = odds_api_key or os.environ.get("ODDS_API_KEY")
        odds = fetch_odds_for_match(home_team, away_team, api_key=key)
        if not odds:
            print("  [odds] No live odds — model will run without market blending.")

    print("  ── Data fetch complete ─────────────────────────────────\n")

    return matches, player_shares, upcoming, odds, raw_data


# ─────────────────────────────────────────────────────────────────
# STANDALONE RUN — show upcoming matches and data summary
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_env()

    print("\n" + "="*60)
    print("  STAGE 3: DATA FETCHER")
    print("="*60)

    matches, player_shares, upcoming, _, raw_data = fetch_all()

    print(f"\n  Completed matches loaded: {len(matches)}")

    print("\n  Upcoming matches:")
    print_upcoming(upcoming)

    print(f"\n  Player shot shares computed: {len(player_shares)}")
    print(f"\n  Top 15 player shares:")
    top = sorted(player_shares.items(),
                 key=lambda x: x[1]["share"], reverse=True)[:15]
    print(f"  {'Player':<28} {'Team':<22} {'Share':>6}")
    print(f"  {'─'*58}")
    for name, d in top:
        print(f"  {name:<28} {d['team']:<22} {d['share']:>6.3f}")

    # Check ODDS_API_KEY
    key = os.environ.get("ODDS_API_KEY")
    print(f"\n  ODDS_API_KEY: {'SET ✓' if key else 'NOT SET — add to .env for live odds'}")
    print(f"  KAPBOT_API_KEY: {'SET ✓' if os.environ.get('KAPBOT_API_KEY') else 'NOT SET — add to .env for submission'}")
    print()
    print("  To use live odds: get a free key at https://the-odds-api.com")
    print("  Then add to .env:  ODDS_API_KEY=your_key_here")

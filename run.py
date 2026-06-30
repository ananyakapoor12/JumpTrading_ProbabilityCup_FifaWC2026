"""
====================================================================
PROBABILITY CUP — MASTER RUNNER
====================================================================

This is the single script you run for every match.

SETUP (one-time)
─────────────────
pip install requests beautifulsoup4 scipy numpy

Create a .env file in the same directory:
  KAPBOT_API_KEY=your_sportspredict_api_key
  ODDS_API_KEY=your_the_odds_api_key   (optional but recommended)

HOW TO RUN
──────────
# Show upcoming matches:
python run.py --list

# Predict a specific match (dry run — no submission):
python run.py --match "Germany vs Paraguay" --dry-run

# Predict and submit (asks for approval first):
python run.py --match "France vs Sweden"

# Auto-approve and submit (for scheduled/bot mode):
python run.py --match "France vs Sweden" --approve
"""

import os, sys, argparse
sys.path.insert(0, os.path.dirname(__file__))

from data_fetcher  import fetch_all, print_upcoming, compute_model_accuracy
from pipeline_v2   import (fit_strengths, compute_anchored_lambdas, score_matrix,
                            derive_all_markets, pois,
                            remove_vig, SHOT_SHARES, MODEL_WEIGHT, MARKET_WEIGHT,
                            classify, resolve_prediction)
from api_client    import KapbotClient
import math

# ── helpers ──────────────────────────────────────────────────────

def to_int(p):
    return max(1, min(99, round(p * 100)))

def player_goal_p(lam, share):
    return 1 - math.exp(-lam * share)

def player_sot_p(lam, share, thr=2, sot_rate=0.33):
    from pipeline_v2 import pois
    lam_sot = (lam / sot_rate) * share
    return 1 - sum(pois(lam_sot, k) for k in range(thr))


def run_full_pipeline(home_team, away_team,
                      dry_run=False, auto_approve=False,
                      odds_api_key=None, kapbot_key=None,
                      is_knockout=True, neutral_venue=True):

    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  PROBABILITY CUP  |  {home_team} vs {away_team}")
    print(sep)

    # ── 1. Fetch all live data ─────────────────────────────────────
    print("\n[1/6] Fetching live data...")
    matches, live_shares, upcoming, odds, raw_data = fetch_all(
        home_team=home_team, away_team=away_team,
        odds_api_key=odds_api_key
    )

    if not matches:
        print("  ERROR: Could not fetch match data. Check internet connection.")
        sys.exit(1)

    # ── 2. Fit team strengths ──────────────────────────────────────
    print("\n[2/6] Fitting team strengths (MLE on live xG data)...")
    params = fit_strengths(matches=matches, verbose=True)

    for team in [home_team, away_team]:
        if team not in params:
            print(f"\n  ERROR: '{team}' not in dataset.")
            avail = sorted(params.keys())
            print(f"  Available teams ({len(avail)}): {', '.join(avail[:20])}...")
            sys.exit(1)

    hp, ap = params[home_team], params[away_team]
    print(f"  {home_team:<22} attack={hp['attack']:.3f}  defence={hp['defence']:.3f}")
    print(f"  {away_team:<22} attack={ap['attack']:.3f}  defence={ap['defence']:.3f}")

    # ── 3. Compute expected goals ─────────────────────────────────
    print("\n[3/6] Computing expected goals...")
    market_3way = odds.get("market_3way") if odds else None
    market_ou   = odds.get("market_ou")   if odds else None

    lam_h, lam_a, lam_h_raw, lam_a_raw = compute_anchored_lambdas(
        home_team, away_team, params, market_3way, market_ou,
        is_knockout, neutral_venue
    )

    print(f"  Raw model:  λ_{home_team}={lam_h_raw:.3f}  λ_{away_team}={lam_a_raw:.3f}")
    if market_ou:
        print(f"  Anchored:   λ_{home_team}={lam_h:.3f}  λ_{away_team}={lam_a:.3f}  "
              f"(anchored to O/U market)")
    else:
        print(f"  (No live odds — using raw model lambdas)")

    # ── 4. Derive markets ─────────────────────────────────────────
    print("\n[4/6] Deriving market probabilities from score matrix...")
    mat  = score_matrix(lam_h, lam_a)
    mkts = derive_all_markets(mat, lam_h, lam_a)

    from pipeline_v2 import MAX_GOALS
    n = MAX_GOALS + 1
    top_scores = sorted(
        [(mat[i][j], i, j) for i in range(n) for j in range(n)],
        reverse=True
    )[:5]
    print("  Most likely scorelines: " +
          "  ".join(f"{i}-{j}({p*100:.0f}%)" for p, i, j in top_scores))

    # ── 5. Build predictions ───────────────────────────────────────
    print("\n[5/6] Building full prediction set...")

    fair_3way = remove_vig(market_3way)[0] if market_3way else None
    fair_ou   = remove_vig(market_ou)[0]   if market_ou   else None

    def fp(model_key, mkt_p=None):
        mp = mkts.get(model_key, 0.5)
        return MODEL_WEIGHT * mp + MARKET_WEIGHT * mkt_p if mkt_p else mp

    predictions = {
        "home_win":               fp("home_win",  fair_3way["home"] if fair_3way else None),
        "away_win":               fp("away_win",  fair_3way["away"] if fair_3way else None),
        "home_lead_ht":           fp("home_lead_ht"),
        "away_lead_ht":           fp("away_lead_ht"),
        "home_scores_first":      fp("home_scores_first"),
        "away_scores_first":      fp("away_scores_first"),
        "over_2_5":               fp("over_2_5",  fair_ou["over"] if fair_ou else None),
        "btts":                   fp("btts"),
        "home_scores_both_halves":fp("home_scores_both_halves"),
        "goal_before_break":      fp("goal_before_break"),
        "home_7plus_sot":         fp("home_7plus_sot"),
        "home_7plus_corners":     fp("home_7plus_corners"),
        "cards_4plus":            fp("cards_4plus"),
        "offsides_3plus":         fp("offsides_3plus"),
    }

    # Player props — merge live shares with hardcoded fallback
    all_shares = {**SHOT_SHARES, **live_shares}  # live overwrites static

    for player, d in all_shares.items():
        if d["team"] not in (home_team, away_team):
            continue
        lam   = lam_h if d["team"] == home_team else lam_a
        share = d["share"]
        p_g   = player_goal_p(lam, share)
        p_a   = p_g * 0.65
        predictions[f"{player}_goal"]           = p_g
        predictions[f"{player}_goal_or_assist"] = p_g + p_a - p_g * p_a
        predictions[f"{player}_1plus_sot"]      = player_sot_p(lam, share, 1)
        predictions[f"{player}_2plus_sot"]      = player_sot_p(lam, share, 2)

    # ── 6. Review dashboard ────────────────────────────────────────
    print(f"\n[6/6] Review dashboard:\n")
    print(f"  {sep}")
    print(f"  λ {home_team}: {lam_h:.2f}  |  λ {away_team}: {lam_a:.2f}  "
          f"| {'Live odds ✓' if odds else 'No live odds'}")
    print(f"\n  {'#':<4}{'Question':<48}{'P':>4}  {'Model':>6}  {'Market':>6}")
    print(f"  {'─'*70}")

    # Fetch kapbot markets via REST API
    markets = []
    client = KapbotClient(kapbot_key) if kapbot_key else None
    if client:
        target = client.find_match(home_team)
        if target:
            markets = client.get_markets(target["id"])

    if not markets:
        # Generate sample structure
        q_templates = [
            (f"Will {home_team} win in regulation (90 minutes + stoppage time)?",    "home_win"),
            (f"Will {home_team} be ahead at halftime?",                              "home_lead_ht"),
            (f"Will {home_team} score the first goal of the match?",                 "home_scores_first"),
            ("Will the match have 3 or more total goals in regulation (90 minutes + stoppage time)?","over_2_5"),
            ("Will both teams score in regulation (90 minutes + stoppage time)?",    "btts"),
            (f"Will {home_team} score in both halves in regulation (90 minutes + stoppage time)?","home_scores_both_halves"),
            ("Will a goal be scored before the first hydration break?",              "goal_before_break"),
            (f"Will {home_team} have 7 or more shots on target in regulation (90 minutes + stoppage time)?","home_7plus_sot"),
            (f"Will {home_team} have 7 or more corner kicks in regulation (90 minutes + stoppage time)?","home_7plus_corners"),
            ("Will there be 4 or more total cards shown in regulation (90 minutes + stoppage time)?","cards_4plus"),
            ("Will there be 3 or more offside calls in regulation (90 minutes + stoppage time)?","offsides_3plus"),
        ]
        markets = [
            {"id": f"s{i}", "lobby_id": "sample",
             "question": q, "_key": k, "market_status": "open"}
            for i, (q, k) in enumerate(q_templates)
        ]

    lambdas = mkts.get("_lambdas", {})
    submission_list = []
    for i, mkt in enumerate(markets, 1):
        q     = mkt["question"]
        key, pd = classify(q, home_team, away_team)

        # Use pre-tagged key if available (sample markets)
        if not key and "_key" in mkt:
            key = mkt["_key"]

        prob  = resolve_prediction(key, pd, predictions, lambdas, lam_h, lam_a)
        p_int = max(1, min(99, round(prob * 100))) if prob is not None else 50
        flag  = "" if key else " (?)"

        if key in mkts:
            model_pct = f"{mkts[key]*100:.0f}%"
        elif prob is not None:
            model_pct = f"{prob*100:.0f}%"
        else:
            model_pct = "  —"
        mkt_pct = "—"
        if key == "home_win" and fair_3way:  mkt_pct = f"{fair_3way['home']*100:.0f}%"
        if key == "over_2_5" and fair_ou:    mkt_pct = f"{fair_ou['over']*100:.0f}%"

        dq = (q[:47] + "…") if len(q) > 47 else q
        print(f"  {i:<4}{dq:<48}{p_int:>3}{flag}  {model_pct:>6}  {mkt_pct:>6}")

        submission_list.append({
            "market_id":   mkt["id"],
            "lobby_id":    mkt.get("lobby_id", ""),
            "probability": p_int,
            "question":    q,
        })

    print(f"  {'─'*70}")
    print(f"  {len(submission_list)} markets ready.\n")

    if dry_run:
        print("  DRY RUN — not submitting. Remove --dry-run to submit.\n")
        return predictions

    confirm = "y" if auto_approve else input("  Submit all predictions? [y/N]: ").strip().lower()

    if confirm == "y":
        if client:
            payload = [{"market_id": p["market_id"],
                        "lobby_id":  p["lobby_id"],
                        "probability": p["probability"]}
                       for p in submission_list
                       if p.get("lobby_id") != "sample"]
            if not payload:
                print("  No real markets to submit (all sample questions).")
            else:
                result = client.submit_batch(payload)
                if result:
                    print(f"  Submitted {result.get('succeeded','?')} OK, "
                          f"{result.get('failed','?')} failed.")
                else:
                    print("  Submission failed — check KAPBOT_API_KEY.")
        else:
            print("  No KAPBOT_API_KEY set — cannot submit.")
    else:
        print("  Cancelled.")

    return predictions


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_fetcher import load_env
    load_env()

    parser = argparse.ArgumentParser(description="Probability Cup model")
    parser.add_argument("--match",   default=None,
                        help='Match name e.g. "France vs Sweden"')
    parser.add_argument("--list",    action="store_true",
                        help="List upcoming matches")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--approve", action="store_true")
    parser.add_argument("--validate",action="store_true",
                        help="Check model accuracy on completed matches")
    args = parser.parse_args()

    kapbot_key = os.environ.get("KAPBOT_API_KEY")
    odds_key   = os.environ.get("ODDS_API_KEY")

    if args.list or not args.match:
        from data_fetcher import fetch_wc2026_data, get_upcoming_matches
        data = fetch_wc2026_data()
        print("\n  Upcoming matches:")
        print_upcoming(get_upcoming_matches(data, n=15))
        if not args.match:
            print("\n  Usage: python run.py --match \"France vs Sweden\" --dry-run\n")
        if not args.match:
            sys.exit(0)

    if args.validate:
        from data_fetcher import fetch_wc2026_data
        data = fetch_wc2026_data()
        params = fit_strengths(verbose=False)
        compute_model_accuracy(data, params)
        sys.exit(0)

    if args.match:
        if " vs " not in args.match:
            print("  ERROR: use format 'Team1 vs Team2'")
            sys.exit(1)
        home, away = [t.strip() for t in args.match.split(" vs ", 1)]
        run_full_pipeline(
            home_team    = home,
            away_team    = away,
            dry_run      = args.dry_run,
            auto_approve = args.approve,
            odds_api_key = odds_key,
            kapbot_key   = kapbot_key,
            is_knockout  = True,
            neutral_venue= True,
        )

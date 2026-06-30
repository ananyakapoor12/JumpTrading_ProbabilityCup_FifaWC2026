"""
====================================================================
STAGE 4C: THE SCHEDULER — MAIN BOT
====================================================================

This is the process that runs 24/7 during the tournament.

WHAT IT DOES
─────────────
Every 5 minutes, it:
  1. Fetches the latest fixture list from openfootball
  2. Finds the next unplayed match we haven't predicted yet
  3. If kickoff is within the FIRE_WINDOW (default: 35 min),
     it triggers the full prediction pipeline
  4. Applies lineup adjustments from known absences
  5. Runs the Poisson model with live odds (if available)
  6. Shows the prediction table and asks for approval
     (or auto-submits if --auto flag is set)
  7. Logs the submission to bot_state.json

HOW TO RUN
──────────
# Interactive mode (asks for approval before each submission):
python scheduler.py

# Fully automatic (no prompts — runs as a background process):
python scheduler.py --auto

# Run in background (Linux/Mac):
nohup python scheduler.py --auto > bot.log 2>&1 &

# Check status anytime:
python scheduler.py --status

# Set a manual override (e.g. you know Mbappé is injured):
python scheduler.py --override "France vs Sweden" home_win 68

# Stop the bot:
kill $(cat bot.pid)   OR just Ctrl+C

TIMING LOGIC
────────────
FIRE_WINDOW = 35 minutes before kickoff
  → We want to submit after lineups are confirmed (~60 min out)
    but well before markets close (kickoff)
  → 35 min gives us time for the model to run + you to review

CHECK_INTERVAL = 5 minutes
  → How often we look at the fixture list
  → Lightweight — just a JSON fetch

LINEUP_CHECK_ADVANCE = 70 minutes
  → How early we start watching for lineup news
  → Gives 35 min of lineup awareness before the fire window
"""

import os
import sys
import time
import signal
import argparse
import schedule
import threading
from datetime import datetime, timezone, timedelta

# Add model directory to path
sys.path.insert(0, os.path.dirname(__file__))

from data_fetcher   import fetch_wc2026_data, get_upcoming_matches, load_env
from lineup_impact  import (apply_lineup_adjustments, get_known_absences,
                             parse_lineup_from_text, PLAYER_IMPACTS)
from state_manager  import (load_state, save_state, already_submitted,
                             record_submission, record_skip, get_overrides,
                             set_override, print_status, match_key)
from pipeline_v2    import (fit_strengths, compute_anchored_lambdas, score_matrix,
                             derive_all_markets, remove_vig, SHOT_SHARES,
                             MODEL_WEIGHT, MARKET_WEIGHT, classify, resolve_prediction, pois)
from api_client     import KapbotClient

# ─── Configuration ────────────────────────────────────────────────
FIRE_WINDOW_MINUTES    = 35   # submit this many minutes before kickoff
CHECK_INTERVAL_MINUTES = 5    # check fixture list this often
LINEUP_WATCH_MINUTES   = 70   # start watching for lineup news this early

PID_FILE = os.path.join(os.path.dirname(__file__), "bot.pid")
LOG_FILE = os.path.join(os.path.dirname(__file__), "bot.log")


# ─── Logging ──────────────────────────────────────────────────────

def log(msg, level="INFO"):
    ts  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] [{level}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except IOError:
        pass


# ─── Time utilities ───────────────────────────────────────────────

def parse_match_kickoff(match):
    """
    Parse kickoff datetime from openfootball match dict.
    Returns UTC datetime or None.

    openfootball format: date="2026-06-30", time="21:00 UTC+1"
    We convert to UTC.
    """
    date_str = match.get("date", "")
    time_str = match.get("time", "")

    if not date_str:
        return None

    try:
        # Parse offset from time string e.g. "21:00 UTC-4" or "16:30 UTC-6"
        offset_hours = 0
        if "UTC" in time_str:
            parts = time_str.split("UTC")
            time_part = parts[0].strip()
            offset_str = parts[1].strip() if len(parts) > 1 else ""
            if offset_str:
                offset_hours = float(offset_str)
        elif time_str:
            time_part = time_str.strip()
        else:
            time_part = "21:00"

        dt_str = f"{date_str} {time_part}"
        naive  = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
        utc_dt = naive.replace(tzinfo=timezone.utc) - timedelta(hours=offset_hours)
        return utc_dt
    except (ValueError, AttributeError):
        # Fallback: use date only with default evening kickoff
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d")
            return d.replace(hour=20, tzinfo=timezone.utc)
        except ValueError:
            return None


def minutes_to_kickoff(match):
    """Return minutes until match kickoff. Negative = already started."""
    ko = parse_match_kickoff(match)
    if not ko:
        return None
    now = datetime.now(timezone.utc)
    return (ko - now).total_seconds() / 60


# ─── Helper: full prediction for one match ────────────────────────

def _to_int(p):
    return max(1, min(99, round(p * 100)))


def predict_and_submit(home_team, away_team, state,
                       auto_approve=False, kapbot_key=None,
                       odds_api_key=None, lineup_news=""):
    """
    Run the full prediction pipeline for one match and optionally submit.
    This is called by the scheduler when it's time to fire.
    """
    log(f"FIRING prediction pipeline for {home_team} vs {away_team}")

    # ── A. Fetch live match data + fit team strengths ──────────────
    wc_data = fetch_wc2026_data()
    if not wc_data:
        log("Cannot fetch match data — aborting.", "ERROR")
        return False

    completed_matches = []
    for m in wc_data["matches"]:
        ft = m.get("score", {}).get("ft")
        if ft:
            completed_matches.append(
                (m["team1"], m["team2"], ft[0], ft[1], None, None)
            )

    params = fit_strengths(matches=completed_matches, verbose=False)

    if home_team not in params or away_team not in params:
        log(f"Team not found in params: {home_team} or {away_team}", "ERROR")
        avail = sorted(params.keys())
        log(f"Available: {', '.join(avail[:20])}", "ERROR")
        return False

    # ── B. Fetch live odds ─────────────────────────────────────────
    market_3way = None
    market_ou   = None

    odds_key = odds_api_key or os.environ.get("ODDS_API_KEY")
    if odds_key:
        try:
            from data_fetcher import fetch_odds_for_match
            odds = fetch_odds_for_match(home_team, away_team, api_key=odds_key)
            if odds:
                market_3way = odds.get("market_3way")
                market_ou   = odds.get("market_ou")
        except Exception as e:
            log(f"Odds fetch failed: {e}", "WARN")

    # ── C. Compute base lambdas ────────────────────────────────────
    lam_h, lam_a, lam_h_raw, lam_a_raw = compute_anchored_lambdas(
        home_team, away_team, params, market_3way, market_ou,
        is_knockout=True, neutral_venue=True
    )

    # ── D. Apply lineup adjustments ───────────────────────────────
    # Known absences (hardcoded research)
    known = get_known_absences(home_team, away_team)

    # Parse any additional lineup news text
    extra = {}
    if lineup_news:
        extra = parse_lineup_from_text(lineup_news, home_team, away_team)

    # Merge: extra overrides known
    home_absent   = list(set(known["home_absent"]   + extra.get("home_absent", [])))
    away_absent   = list(set(known["away_absent"]   + extra.get("away_absent", [])))
    home_doubtful = list(set(known["home_doubtful"] + extra.get("home_doubtful", [])))
    away_doubtful = list(set(known["away_doubtful"] + extra.get("away_doubtful", [])))

    lam_h_adj, lam_a_adj, adj_log, corner_adj = apply_lineup_adjustments(
        lam_h, lam_a, home_team, away_team,
        home_absent, away_absent, home_doubtful, away_doubtful,
        verbose=True
    )

    lineup_summary = (
        f"absent: {home_absent+away_absent} | "
        f"doubtful: {home_doubtful+away_doubtful}"
    )

    # ── E. Build score matrix + derive markets ─────────────────────
    mat  = score_matrix(lam_h_adj, lam_a_adj)
    mkts = derive_all_markets(mat, lam_h_adj, lam_a_adj)

    # ── F. Build prediction dict ───────────────────────────────────
    fair_3way = remove_vig(market_3way)[0] if market_3way else None
    fair_ou   = remove_vig(market_ou)[0]   if market_ou   else None

    def fp(model_key, mkt_p=None):
        mp = mkts.get(model_key, 0.5)
        return MODEL_WEIGHT * mp + MARKET_WEIGHT * mkt_p if mkt_p else mp

    def player_goal_p(lam, share):
        import math
        return 1 - math.exp(-lam * share)

    def player_sot_p(lam, share, thr, sot_rate=0.33):
        lam_sot = (lam / sot_rate) * share
        return 1 - sum(pois(lam_sot, k) for k in range(thr))

    predictions = {
        "home_win":               fp("home_win",  fair_3way["home"] if fair_3way else None),
        "away_win":               fp("away_win",  fair_3way["away"] if fair_3way else None),
        "home_lead_ht":           fp("home_lead_ht"),
        "home_scores_first":      fp("home_scores_first"),
        "over_2_5":               fp("over_2_5",  fair_ou["over"] if fair_ou else None),
        "btts":                   fp("btts"),
        "home_scores_both_halves":fp("home_scores_both_halves"),
        "goal_before_break":      fp("goal_before_break"),
        "home_7plus_sot":         fp("home_7plus_sot"),
        "home_7plus_corners":     fp("home_7plus_corners", None),
        "cards_4plus":            fp("cards_4plus"),
        "offsides_3plus":         fp("offsides_3plus"),
    }

    # Player props
    for player, d in SHOT_SHARES.items():
        if d["team"] not in (home_team, away_team):
            continue
        lam   = lam_h_adj if d["team"] == home_team else lam_a_adj
        share = d["share"]
        p_g   = player_goal_p(lam, share)
        p_a   = p_g * 0.65
        predictions[f"{player}_goal"]           = p_g
        predictions[f"{player}_goal_or_assist"] = p_g + p_a - p_g * p_a
        predictions[f"{player}_1plus_sot"]      = player_sot_p(lam, share, 1)
        predictions[f"{player}_2plus_sot"]      = player_sot_p(lam, share, 2)

    # Apply any manual overrides from state
    overrides = get_overrides(state, home_team, away_team)
    if overrides:
        log(f"Applying {len(overrides)} manual overrides: {overrides}")
        predictions.update(overrides)

    # ── G. Print review dashboard ──────────────────────────────────
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  BOT PREDICTION  |  {home_team} vs {away_team}")
    print(f"  λ {home_team}: {lam_h_adj:.2f}  |  λ {away_team}: {lam_a_adj:.2f}")
    print(f"  {'Live odds ✓' if market_3way else 'No live odds'}  "
          f"| {'Lineup adj ✓' if adj_log else 'No lineup adj'}")
    print(sep)

    # Fetch kapbot markets via REST API
    markets = []
    client = KapbotClient(kapbot_key) if kapbot_key else None
    if client:
        target = client.find_match(home_team)
        if target:
            markets = client.get_markets(target["id"])
            log(f"Fetched {len(markets)} markets from REST API")
        else:
            log(f"Match '{home_team}' not found in open markets", "WARN")

    # Build sample structure if no kapbot markets
    if not markets:
        q_templates = [
            (f"Will {home_team} win in regulation (90 minutes + stoppage time)?",       "home_win"),
            (f"Will {home_team} be ahead at halftime?",                                 "home_lead_ht"),
            (f"Will {home_team} score the first goal of the match?",                    "home_scores_first"),
            ("Will the match have 3 or more total goals in regulation (90 minutes + stoppage time)?", "over_2_5"),
            ("Will both teams score in regulation (90 minutes + stoppage time)?",        "btts"),
            (f"Will {home_team} score in both halves in regulation (90 minutes + stoppage time)?", "home_scores_both_halves"),
            ("Will a goal be scored before the first hydration break?",                  "goal_before_break"),
            (f"Will {home_team} have 7 or more shots on target in regulation (90 minutes + stoppage time)?", "home_7plus_sot"),
            (f"Will {home_team} have 7 or more corner kicks in regulation (90 minutes + stoppage time)?",    "home_7plus_corners"),
            ("Will there be 4 or more total cards shown in regulation (90 minutes + stoppage time)?",        "cards_4plus"),
            ("Will there be 3 or more offside calls in regulation (90 minutes + stoppage time)?",            "offsides_3plus"),
        ]
        markets = [
            {"id": f"s{i}", "lobby_id": "sample",
             "question": q, "_key": k, "market_status": "open"}
            for i, (q, k) in enumerate(q_templates)
        ]

    submission_list = []
    print(f"\n  {'#':<4}{'Question':<48}{'P':>4}")
    print(f"  {'─'*56}")

    lambdas = mkts.get("_lambdas", {})
    for i, mkt in enumerate(markets, 1):
        q       = mkt["question"]
        key, pd = classify(q, home_team, away_team)
        if not key and "_key" in mkt:
            key = mkt["_key"]

        prob  = resolve_prediction(key, pd, predictions, lambdas, lam_h, lam_a)
        p_int = max(1, min(99, round(prob * 100))) if prob is not None else 50
        dq    = (q[:47] + "…") if len(q) > 47 else q
        print(f"  {i:<4}{dq:<48}{p_int:>3}")

        submission_list.append({
            "market_id":   mkt["id"],
            "lobby_id":    mkt.get("lobby_id", ""),
            "probability": p_int,
            "question":    q,
        })

    print(f"  {'─'*56}")
    print(f"  {len(submission_list)} markets ready.\n")

    # ── H. Get approval ────────────────────────────────────────────
    if auto_approve:
        confirm = "y"
        log("Auto-approve enabled — submitting without prompt.")
    else:
        confirm = input("  Submit these predictions? [y/N/skip]: ").strip().lower()

    if confirm == "skip":
        record_skip(state, home_team, away_team, "user skipped at review")
        log(f"User skipped {home_team} vs {away_team}")
        return False

    if confirm != "y":
        log("User cancelled submission.")
        return False

    # ── I. Submit ─────────────────────────────────────────────────
    if not kapbot_key:
        log("No KAPBOT_API_KEY — cannot submit. Set in .env file.", "ERROR")
        return False

    payload = [
        {"market_id": p["market_id"],
         "lobby_id":  p["lobby_id"],
         "probability": p["probability"]}
        for p in submission_list
        if p["lobby_id"] != "sample"
    ]

    if not payload:
        log("No real kapbot markets to submit (all sample).", "WARN")
        # Still record as "submitted" for state tracking
        record_submission(state, home_team, away_team,
                          len(submission_list), len(submission_list), 0,
                          lam_h_adj, lam_a_adj, lineup_summary)
        return True

    result = client.submit_batch(payload)

    if result:
        succeeded = result.get("succeeded", 0)
        failed    = result.get("failed",    0)
        log(f"Submitted {succeeded}/{len(payload)} predictions for "
            f"{home_team} vs {away_team} ({failed} failed)")
        record_submission(state, home_team, away_team,
                          len(payload), succeeded, failed,
                          lam_h_adj, lam_a_adj, lineup_summary)
        return succeeded > 0
    else:
        log("Submission failed — check KAPBOT_API_KEY and connection.", "ERROR")
        return False


# ─────────────────────────────────────────────────────────────────
# SCHEDULER CORE
# ─────────────────────────────────────────────────────────────────

def check_and_fire(auto_approve, kapbot_key, odds_api_key):
    """
    Main tick function — called every CHECK_INTERVAL_MINUTES.

    Checks the fixture list, finds the next unplayed match,
    and fires if within FIRE_WINDOW_MINUTES of kickoff.
    """
    log("Checking fixture list...")

    state    = load_state()
    wc_data  = fetch_wc2026_data()
    upcoming = get_upcoming_matches(wc_data, n=20) if wc_data else []

    if not upcoming:
        log("No upcoming matches found.")
        return

    # Find next match we haven't predicted yet
    next_match = None
    for m in upcoming:
        home, away = m["home"], m["away"]
        if not home.replace(" ", "").isalpha():
            continue  # skip placeholder teams like "W74"
        if already_submitted(state, home, away):
            log(f"  Already submitted: {home} vs {away} — skipping")
            continue
        mins = minutes_to_kickoff(m)
        if mins is None:
            continue
        if mins < -15:
            log(f"  Too late for {home} vs {away} (kicked off {-mins:.0f} min ago)")
            record_skip(state, home, away, f"already started ({-mins:.0f} min ago)")
            continue
        next_match = m
        next_match["minutes_to_ko"] = mins
        break

    if not next_match:
        log("No upcoming matches to predict.")
        return

    home    = next_match["home"]
    away    = next_match["away"]
    mins    = next_match["minutes_to_ko"]
    ko_time = parse_match_kickoff(next_match)

    log(f"Next match: {home} vs {away} | "
        f"Kickoff: {ko_time.strftime('%H:%M UTC') if ko_time else '?'} | "
        f"{mins:.0f} min away")

    # Update state with next match info
    state["next_match"] = match_key(home, away)
    state["next_fire"]  = (
        (ko_time - timedelta(minutes=FIRE_WINDOW_MINUTES)).isoformat()
        if ko_time else "unknown"
    )
    save_state(state)

    # LINEUP WATCH: if within watch window, log that we're monitoring
    if mins <= LINEUP_WATCH_MINUTES:
        log(f"Within lineup watch window ({mins:.0f} min). Monitoring for news.")

    # FIRE: if within fire window, run the prediction pipeline
    if mins <= FIRE_WINDOW_MINUTES:
        log(f"WITHIN FIRE WINDOW ({mins:.0f} min to kickoff). Triggering pipeline.")
        predict_and_submit(
            home_team    = home,
            away_team    = away,
            state        = state,
            auto_approve = auto_approve,
            kapbot_key   = kapbot_key,
            odds_api_key = odds_api_key,
            lineup_news  = "",   # In auto mode, we rely on get_known_absences
        )
    else:
        log(f"Waiting — {mins:.0f} min to kickoff "
            f"(fires at {FIRE_WINDOW_MINUTES} min mark).")


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────

def main():
    load_env()

    parser = argparse.ArgumentParser(description="Probability Cup scheduler bot")
    parser.add_argument("--auto",     action="store_true",
                        help="Auto-approve all submissions (no prompts)")
    parser.add_argument("--once",     action="store_true",
                        help="Run once and exit (no loop)")
    parser.add_argument("--status",   action="store_true",
                        help="Print bot status and exit")
    parser.add_argument("--override", nargs=3, metavar=("MATCH", "MARKET", "PROB"),
                        help='Set manual override e.g. --override "France vs Sweden" home_win 70')
    parser.add_argument("--clear",    action="store_true",
                        help="Clear bot state (reset all submissions)")
    args = parser.parse_args()

    kapbot_key = os.environ.get("KAPBOT_API_KEY")
    odds_key   = os.environ.get("ODDS_API_KEY")

    # -- Status --
    if args.status:
        print_status(load_state())
        return

    # -- Clear state --
    if args.clear:
        confirm = input("Clear all bot state? This resets all submissions. [y/N]: ")
        if confirm.lower() == "y":
            if os.path.exists(
                os.path.join(os.path.dirname(__file__), "bot_state.json")
            ):
                os.remove(os.path.join(os.path.dirname(__file__), "bot_state.json"))
            log("State cleared.")
        return

    # -- Override --
    if args.override:
        raw_match, market, prob_str = args.override
        try:
            prob = int(prob_str)
        except ValueError:
            print(f"ERROR: probability must be integer 1-99, got '{prob_str}'")
            return
        if " vs " not in raw_match:
            print("ERROR: match format must be 'Team1 vs Team2'")
            return
        home, away = raw_match.split(" vs ", 1)
        state = load_state()
        set_override(state, home.strip(), away.strip(), market, prob)
        return

    # -- Check API key --
    if not kapbot_key:
        log("WARNING: KAPBOT_API_KEY not set. Predictions won't be submitted.", "WARN")
        log("         Add KAPBOT_API_KEY=your_key to your .env file", "WARN")
    if not odds_key:
        log("NOTE: ODDS_API_KEY not set. Running without live odds.", "INFO")

    # -- Write PID file --
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    log(f"Bot started (PID {os.getpid()}). "
        f"Auto={args.auto}. Check interval={CHECK_INTERVAL_MINUTES}min. "
        f"Fire window={FIRE_WINDOW_MINUTES}min.")

    # Handle Ctrl+C gracefully
    def handle_exit(sig, frame):
        log("Bot shutting down.")
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
        sys.exit(0)

    signal.signal(signal.SIGINT,  handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    # -- Once mode --
    if args.once:
        check_and_fire(args.auto, kapbot_key, odds_key)
        return

    # -- Continuous scheduler loop --
    schedule.every(CHECK_INTERVAL_MINUTES).minutes.do(
        check_and_fire, args.auto, kapbot_key, odds_key
    )

    # Run immediately on start
    check_and_fire(args.auto, kapbot_key, odds_key)

    log(f"Scheduler running. Checking every {CHECK_INTERVAL_MINUTES} minutes.")
    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()

"""
====================================================================
STAGE 4B: STATE MANAGER
====================================================================

The scheduler runs continuously. It needs to know:
  - Which matches have already been predicted (don't re-submit)
  - Which matches are coming up and when to fire
  - The result of each submission (for logging/debugging)
  - Any manual overrides the user has set

We store state in a simple JSON file (bot_state.json).
This survives restarts — if the bot crashes and restarts,
it picks up exactly where it left off.

STATE SCHEMA
────────────
{
  "submitted": {
    "Germany vs Paraguay": {
      "submitted_at": "2026-06-29T19:46:22Z",
      "markets": 15,
      "succeeded": 15,
      "failed": 0,
      "lineup_news": "Schlotterbeck out, Gomez suspended",
      "lam_home": 2.07,
      "lam_away": 0.97,
    }
  },
  "skipped": {
    "Brazil vs Japan": {
      "reason": "market already closed",
      "at": "2026-06-29T21:00:00Z"
    }
  },
  "overrides": {
    "France vs Sweden": {
      "home_win": 75,    // override model output for this market
    }
  },
  "last_run": "2026-06-29T21:35:00Z",
  "next_match": "Netherlands vs Morocco",
  "next_fire": "2026-06-29T22:30:00Z"
}
"""

import json
import os
from datetime import datetime, timezone

STATE_FILE = os.path.join(os.path.dirname(__file__), "bot_state.json")


def _now_str():
    return datetime.now(timezone.utc).isoformat()


def load_state():
    """Load state from disk. Returns empty state if file doesn't exist."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {
        "submitted": {},
        "skipped":   {},
        "overrides": {},
        "last_run":  None,
        "next_match": None,
        "next_fire":  None,
    }


def save_state(state):
    """Persist state to disk."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def match_key(home, away):
    """Canonical match key."""
    return f"{home} vs {away}"


def already_submitted(state, home, away):
    """Return True if we've already submitted predictions for this match."""
    return match_key(home, away) in state["submitted"]


def record_submission(state, home, away, markets, succeeded, failed,
                      lam_home, lam_away, lineup_notes=""):
    """Record a successful (or partial) submission."""
    state["submitted"][match_key(home, away)] = {
        "submitted_at": _now_str(),
        "markets":      markets,
        "succeeded":    succeeded,
        "failed":       failed,
        "lineup_news":  lineup_notes,
        "lam_home":     round(lam_home, 3),
        "lam_away":     round(lam_away, 3),
    }
    state["last_run"] = _now_str()
    save_state(state)


def record_skip(state, home, away, reason):
    """Record that we skipped a match and why."""
    state["skipped"][match_key(home, away)] = {
        "reason": reason,
        "at":     _now_str(),
    }
    save_state(state)


def get_overrides(state, home, away):
    """Return any manual prediction overrides for this match."""
    return state["overrides"].get(match_key(home, away), {})


def set_override(state, home, away, market, probability):
    """Set a manual override for a specific market in a specific match."""
    key = match_key(home, away)
    state["overrides"].setdefault(key, {})[market] = probability
    save_state(state)
    print(f"  Override set: {key} → {market} = {probability}")


def print_status(state):
    """Print a summary of bot state."""
    print("\n  ── Bot State ───────────────────────────────────────────")
    print(f"  Last run: {state.get('last_run', 'never')}")
    print(f"  Next fire: {state.get('next_fire', 'not scheduled')}")

    submitted = state.get("submitted", {})
    if submitted:
        print(f"\n  Submitted predictions ({len(submitted)} matches):")
        for match, data in submitted.items():
            status = f"✓ {data['succeeded']}/{data['markets']}"
            lam = f"λ {data.get('lam_home','?')}/{data.get('lam_away','?')}"
            print(f"    {match:<35} {status}  {lam}")
            if data.get("lineup_news"):
                print(f"      News: {data['lineup_news']}")

    skipped = state.get("skipped", {})
    if skipped:
        print(f"\n  Skipped ({len(skipped)} matches):")
        for match, data in skipped.items():
            print(f"    {match:<35} {data['reason']}")

    overrides = state.get("overrides", {})
    if overrides:
        print(f"\n  Active overrides:")
        for match, markets in overrides.items():
            print(f"    {match}: {markets}")

    print("  ──────────────────────────────────────────────────────\n")

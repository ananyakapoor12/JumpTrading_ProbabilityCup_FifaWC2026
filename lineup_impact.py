"""
====================================================================
STAGE 4A: LINEUP IMPACT SYSTEM
====================================================================

WHY LINEUPS MATTER
──────────────────
A missing striker can reduce a team's expected goals by 15-25%.
A missing goalkeeper can increase expected conceded by 5-10%.
This directly affects every market we predict.

The lineup market closes at kickoff. Confirmed lineups drop
~60 minutes before. This is the single biggest last-minute edge —
the market doesn't always fully adjust before close.

HOW WE MODEL LINEUP IMPACT
────────────────────────────
Each player has an "impact factor" — how much their presence
multiplies their team's expected goals (attack impact) and
how much their absence changes goals conceded (defence impact).

  lam_home_adjusted = lam_home × Π(attack_impacts) × Π(defence_impacts_opp)

Attack impact > 1 means the player boosts scoring when present.
A value of 1.20 means the team scores 20% more goals with them.
When a player is ABSENT, we multiply by 1/impact (reduces scoring).

Defence impact: a leaky defender or missing key CB inflates lam_away.

This is conservative and transparent — you can see exactly what
each player's absence or presence does to the probability.

DATA SOURCE FOR LINEUPS
────────────────────────
From our environment, only GitHub (raw.githubusercontent.com) is
reachable at runtime. Confirmed lineups ~60min before kickoff are
not on GitHub.

Solution: the scheduler calls a lineup_fetcher function that:
  1. Tries to fetch from GitHub (openfootball updates post-kickoff)
  2. Falls back to web_search (when running inside Claude)
  3. Falls back to the hardcoded known squad as a neutral prior

The scheduler logs which method was used so you can verify.
"""

import math
from typing import Dict, Optional

# ─────────────────────────────────────────────────────────────────
# PLAYER IMPACT DATABASE
# ─────────────────────────────────────────────────────────────────
#
# Format: {player_name: {attack: float, defence: float, position: str}}
#
# attack:  multiplier on team's λ_home/away when this player STARTS
#          1.0 = average, 1.20 = 20% boost, 0.85 = 15% drag
# defence: multiplier on OPPONENT's λ when this player STARTS
#          < 1 means tighter defence (fewer conceded)
#          > 1 means leaky (more conceded)
#
# Impact values derived from:
#   - Tournament xG with/without player (where available)
#   - Historical club-level with/without analysis
#   - Position importance weighting (striker > defender for attack)
#
# These are intentionally conservative — we don't want to swing
# predictions wildly on a single absence.

PLAYER_IMPACTS: Dict[str, Dict] = {

    # ── GERMANY ──────────────────────────────────────────────────
    "Kai Havertz": {
        "attack": 1.12, "defence": 0.97, "position": "striker",
        "team": "Germany",
        "note": "First-choice striker, focal point. Without him Undav leads line."
    },
    "Florian Wirtz": {
        "attack": 1.18, "defence": 0.98, "position": "attacking_mid",
        "team": "Germany",
        "note": "Germany's most creative player. Irreplaceable at the 10."
    },
    "Jamal Musiala": {
        "attack": 1.14, "defence": 0.98, "position": "attacking_mid",
        "team": "Germany",
        "note": "Key dribbler and chance creator from half-space."
    },
    "Joshua Kimmich": {
        "attack": 1.05, "defence": 0.92, "position": "right_back",
        "team": "Germany",
        "note": "Primary set-piece taker. Without him, corners/freekicks suffer."
    },
    "Nico Schlotterbeck": {
        "attack": 1.00, "defence": 0.90, "position": "centre_back",
        "team": "Germany",
        "note": "INJURED — already ruled out of tournament. Do not apply."
    },
    "Manuel Neuer": {
        "attack": 1.00, "defence": 0.95, "position": "goalkeeper",
        "team": "Germany",
        "note": "Experienced shot-stopper. Backup is Nübel (slightly less reliable)."
    },
    "Leroy Sané": {
        "attack": 1.08, "defence": 0.99, "position": "winger",
        "team": "Germany",
        "note": "Pace on right wing. Replaceable with Leweling."
    },
    "Deniz Undav": {
        "attack": 1.10, "defence": 0.99, "position": "striker",
        "team": "Germany",
        "note": "Top scorer in tournament (3G). Danger from bench/if Havertz out."
    },

    # ── PARAGUAY ─────────────────────────────────────────────────
    "Julio Enciso": {
        "attack": 1.20, "defence": 1.00, "position": "attacking_mid",
        "team": "Paraguay",
        "note": "Paraguay's main creative threat. Highest xG contribution."
    },
    "Miguel Almirón": {
        "attack": 1.15, "defence": 0.98, "position": "winger",
        "team": "Paraguay",
        "note": "Returns from suspension. Pace on counter-attack."
    },
    "Gabriel Avalos": {
        "attack": 1.05, "defence": 1.00, "position": "striker",
        "team": "Paraguay",
        "note": "Hold-up play striker, limited xG but occupies CBs."
    },
    "Diego Gomez": {
        "attack": 1.10, "defence": 0.96, "position": "midfielder",
        "team": "Paraguay",
        "note": "SUSPENDED vs Germany. Key set-piece taker absence."
    },
    "Omar Alderete": {
        "attack": 1.00, "defence": 0.91, "position": "centre_back",
        "team": "Paraguay",
        "note": "QUESTIONABLE (knee). Starting CB, marshals defensive shape."
    },
    "Gustavo Gómez": {
        "attack": 1.00, "defence": 0.92, "position": "centre_back",
        "team": "Paraguay",
        "note": "Captain and organiser. Critical to defensive solidity."
    },

    # ── FRANCE ────────────────────────────────────────────────────
    "Kylian Mbappé": {
        "attack": 1.30, "defence": 0.99, "position": "striker",
        "team": "France",
        "note": "World-class. Absence massively reduces France attack."
    },
    "Ousmane Dembélé": {
        "attack": 1.15, "defence": 0.99, "position": "winger",
        "team": "France",
        "note": "4 goals in tournament. Co-leading scorer."
    },
    "Antoine Griezmann": {
        "attack": 1.12, "defence": 0.95, "position": "attacking_mid",
        "team": "France",
        "note": "Pressing and link-up. Both attack and defensive structure."
    },
    "Marcus Thuram": {
        "attack": 1.10, "defence": 0.99, "position": "striker",
        "team": "France",
        "note": "Physical striker, good in transitions."
    },
    "N'Golo Kanté": {
        "attack": 1.02, "defence": 0.88, "position": "midfielder",
        "team": "France",
        "note": "Elite defensive midfielder. Absence opens France to counters."
    },

    # ── ENGLAND ──────────────────────────────────────────────────
    "Harry Kane": {
        "attack": 1.22, "defence": 0.99, "position": "striker",
        "team": "England",
        "note": "3 tournament goals. England attack built around him."
    },
    "Jude Bellingham": {
        "attack": 1.18, "defence": 0.96, "position": "attacking_mid",
        "team": "England",
        "note": "Box-to-box. Both creative and defensive contributions."
    },
    "Phil Foden": {
        "attack": 1.12, "defence": 0.99, "position": "attacking_mid",
        "team": "England",
        "note": "Key in tight spaces. Irreplaceable at the 10."
    },
    "Bukayo Saka": {
        "attack": 1.10, "defence": 0.98, "position": "winger",
        "team": "England",
        "note": "Right wing threat. Defensive contribution too."
    },

    # ── ARGENTINA ────────────────────────────────────────────────
    "Lionel Messi": {
        "attack": 1.35, "defence": 1.00, "position": "attacking_mid",
        "team": "Argentina",
        "note": "6 tournament goals. Irreplaceable. Any injury is catastrophic."
    },
    "Julián Álvarez": {
        "attack": 1.15, "defence": 0.99, "position": "striker",
        "team": "Argentina",
        "note": "High-energy striker. Presses defenders. Important second goal threat."
    },
    "Rodrigo De Paul": {
        "attack": 1.05, "defence": 0.93, "position": "midfielder",
        "team": "Argentina",
        "note": "Engine of midfield. Loses his engine without De Paul."
    },

    # ── SPAIN ─────────────────────────────────────────────────────
    "Lamine Yamal": {
        "attack": 1.20, "defence": 0.99, "position": "winger",
        "team": "Spain",
        "note": "Youngest star. Creates overloads on right."
    },
    "Pedri": {
        "attack": 1.10, "defence": 0.93, "position": "midfielder",
        "team": "Spain",
        "note": "Controls tempo. Spain's best midfielder."
    },
    "Álvaro Morata": {
        "attack": 1.08, "defence": 0.99, "position": "striker",
        "team": "Spain",
        "note": "Movement and hold-up. Not a heavy scorer but important."
    },

    # ── NETHERLANDS ──────────────────────────────────────────────
    "Cody Gakpo": {
        "attack": 1.15, "defence": 0.99, "position": "striker",
        "team": "Netherlands",
        "note": "First-choice striker. Good in air and with feet."
    },
    "Brian Brobbey": {
        "attack": 1.12, "defence": 0.99, "position": "striker",
        "team": "Netherlands",
        "note": "3 tournament goals. Physical threat."
    },
    "Virgil van Dijk": {
        "attack": 1.02, "defence": 0.85, "position": "centre_back",
        "team": "Netherlands",
        "note": "Elite CB. His absence significantly increases goals conceded."
    },

    # ── BRAZIL ───────────────────────────────────────────────────
    "Vinícius Júnior": {
        "attack": 1.22, "defence": 1.00, "position": "winger",
        "team": "Brazil",
        "note": "4 goals, pace and dribbling. Most dangerous player."
    },
    "Matheus Cunha": {
        "attack": 1.14, "defence": 0.99, "position": "striker",
        "team": "Brazil",
        "note": "3 goals. Underrated. Key movement off the ball."
    },
    "Marquinhos": {
        "attack": 1.00, "defence": 0.88, "position": "centre_back",
        "team": "Brazil",
        "note": "Captain and organiser of defence."
    },

    # ── PORTUGAL ─────────────────────────────────────────────────
    "Cristiano Ronaldo": {
        "attack": 1.20, "defence": 1.00, "position": "striker",
        "team": "Portugal",
        "note": "Leader. Penalties and set pieces too."
    },
    "Bruno Fernandes": {
        "attack": 1.15, "defence": 0.95, "position": "attacking_mid",
        "team": "Portugal",
        "note": "Creative hub. Key-pass creator."
    },
    "Rafael Leão": {
        "attack": 1.12, "defence": 0.99, "position": "winger",
        "team": "Portugal",
        "note": "Pace on left. Difficult to replace."
    },
}


# ─────────────────────────────────────────────────────────────────
# SET-PIECE ADJUSTMENT
# ─────────────────────────────────────────────────────────────────
#
# Some markets are specifically affected by who takes set pieces.
# Corner count, for example, correlates with who takes them —
# an accurate corner-taker creates more dangerous situations,
# causing defenders to clear more carefully (→ fewer corners sometimes)
# but good takers get their team more corners via rebounds.
#
# More importantly: penalties and direct free kicks depend heavily
# on who is available.

SET_PIECE_TAKERS = {
    "Germany":     {"corners": ["Joshua Kimmich", "Florian Wirtz"],
                    "penalties": ["Kai Havertz", "İlkay Gündoğan"],
                    "freekicks": ["Joshua Kimmich", "Florian Wirtz"]},
    "Paraguay":    {"corners": ["Miguel Almirón", "Julio Enciso"],
                    "penalties": ["Julio Enciso", "Miguel Almirón"],
                    "freekicks": ["Diego Gomez", "Julio Enciso"]},
    "France":      {"corners": ["Antoine Griezmann", "Kylian Mbappé"],
                    "penalties": ["Kylian Mbappé", "Antoine Griezmann"],
                    "freekicks": ["Kylian Mbappé", "Antoine Griezmann"]},
    "England":     {"corners": ["Trent Alexander-Arnold", "Bukayo Saka"],
                    "penalties": ["Harry Kane"],
                    "freekicks": ["Harry Kane", "Trent Alexander-Arnold"]},
    "Argentina":   {"corners": ["Lionel Messi", "Leandro Paredes"],
                    "penalties": ["Lionel Messi"],
                    "freekicks": ["Lionel Messi"]},
    "Spain":       {"corners": ["Pedri", "Lamine Yamal"],
                    "penalties": ["Álvaro Morata"],
                    "freekicks": ["Pedri", "Lamine Yamal"]},
    "Netherlands": {"corners": ["Xavi Simons", "Cody Gakpo"],
                    "penalties": ["Cody Gakpo", "Memphis Depay"],
                    "freekicks": ["Xavi Simons"]},
    "Brazil":      {"corners": ["Rodrygo", "Raphinha"],
                    "penalties": ["Neymar Jr", "Vinícius Júnior"],
                    "freekicks": ["Rodrygo", "Raphinha"]},
    "Portugal":    {"corners": ["Bruno Fernandes", "Rafael Leão"],
                    "penalties": ["Cristiano Ronaldo"],
                    "freekicks": ["Cristiano Ronaldo", "Bruno Fernandes"]},
}


# ─────────────────────────────────────────────────────────────────
# LINEUP IMPACT CALCULATOR
# ─────────────────────────────────────────────────────────────────

def apply_lineup_adjustments(
    lam_home: float,
    lam_away: float,
    home_team: str,
    away_team: str,
    home_absent: list,    # confirmed absent/suspended players (home)
    away_absent: list,    # confirmed absent/suspended players (away)
    home_doubtful: list,  # doubtful — we apply 50% of the impact
    away_doubtful: list,
    verbose: bool = True,
):
    """
    Adjust expected goals based on known lineup news.

    Returns (lam_home_adj, lam_away_adj, adjustment_log)

    HOW THE MATH WORKS
    ──────────────────
    For each absent player:
      If they boost attack by 1.12x normally,
      their absence → team scores at 1/1.12 = 0.893x rate.
      Reduction = lam × (1 - 1/1.12) = lam × 0.107

    For doubtful players we apply 50% of the adjustment:
      lam × (1 - 0.5 × (1 - 1/impact))

    Defence adjustments affect the OPPONENT's lambda.
    """
    adj_log = []
    lam_h = lam_home
    lam_a = lam_away

    def apply_absence(lam_team, lam_opp, player_name, weight=1.0):
        """Apply absence impact to lambdas. Weight=0.5 for doubtful."""
        p = PLAYER_IMPACTS.get(player_name)
        if not p:
            return lam_team, lam_opp, None

        atk_impact = p["attack"]
        def_impact = p["defence"]

        # Attack adjustment: team scores less without this player
        if atk_impact != 1.0:
            # Absent: divide by atk_impact (undo the boost)
            reduction = 1 - (1 / atk_impact)
            lam_team = lam_team * (1 - weight * reduction)

        # Defence adjustment: opponent scores more without this player
        if def_impact != 1.0:
            increase = 1 - def_impact  # how much less they concede with player
            lam_opp = lam_opp * (1 + weight * increase)

        note = p.get("note", "")[:60]
        status = "absent" if weight == 1.0 else "doubtful(50%)"
        log = {
            "player":  player_name,
            "team":    p["team"],
            "status":  status,
            "atk_adj": f"×{1 - weight*(1-1/atk_impact):.3f}" if atk_impact != 1.0 else "—",
            "def_adj": f"×{1 + weight*abs(1-def_impact):.3f}" if def_impact != 1.0 else "—",
            "note":    note,
        }
        return lam_team, lam_opp, log

    # Home absent
    for player in home_absent:
        lam_h, lam_a, log = apply_absence(lam_h, lam_a, player, weight=1.0)
        if log:
            adj_log.append(log)

    # Home doubtful
    for player in home_doubtful:
        lam_h, lam_a, log = apply_absence(lam_h, lam_a, player, weight=0.5)
        if log:
            adj_log.append(log)

    # Away absent
    for player in away_absent:
        lam_a, lam_h, log = apply_absence(lam_a, lam_h, player, weight=1.0)
        if log:
            adj_log.append(log)

    # Away doubtful
    for player in away_doubtful:
        lam_a, lam_h, log = apply_absence(lam_a, lam_h, player, weight=0.5)
        if log:
            adj_log.append(log)

    # Set-piece adjustment: if primary corner/freekick taker is absent,
    # reduce corners market slightly
    corner_adj = 1.0
    for team, absent_list in [(home_team, home_absent), (away_team, away_absent)]:
        takers = SET_PIECE_TAKERS.get(team, {}).get("corners", [])
        for player in absent_list:
            if player in takers[:1]:  # primary taker absent
                corner_adj *= 0.92
                adj_log.append({
                    "player": player, "team": team,
                    "status": "absent(set-piece)",
                    "atk_adj": "—", "def_adj": "—",
                    "note": "Primary corner/freekick taker absent → corners -8%"
                })

    if verbose:
        if adj_log:
            print(f"\n  Lineup adjustments applied:")
            print(f"  {'Player':<25} {'Status':<18} {'Atk':>6} {'Def':>6}")
            print(f"  {'─'*58}")
            for log in adj_log:
                print(f"  {log['player']:<25} {log['status']:<18} "
                      f"{log['atk_adj']:>6} {log['def_adj']:>6}")
            print(f"\n  λ before: home={lam_home:.3f}  away={lam_away:.3f}")
            print(f"  λ after:  home={lam_h:.3f}  away={lam_a:.3f}")
            delta_h = (lam_h - lam_home) / lam_home * 100
            delta_a = (lam_a - lam_away) / lam_away * 100
            print(f"  Change:   home={delta_h:+.1f}%  away={delta_a:+.1f}%")
        else:
            print("  No lineup adjustments (no absences flagged).")

    return lam_h, lam_a, adj_log, corner_adj


# ─────────────────────────────────────────────────────────────────
# LINEUP NEWS FETCHER
# ─────────────────────────────────────────────────────────────────

def parse_lineup_from_text(text: str, home_team: str, away_team: str):
    """
    Parse lineup/injury text to extract absent and doubtful players.

    This is used when lineup news is provided as a text string
    (e.g. from a web search result or manual input).

    Returns: {
        home_absent: [...],
        away_absent: [...],
        home_doubtful: [...],
        away_doubtful: [...],
    }

    It scans for known player names near keywords like:
    "out", "absent", "suspended", "injured", "ruled out",
    "doubtful", "questionable", "fitness doubt"
    """
    result = {
        "home_absent":   [],
        "away_absent":   [],
        "home_doubtful": [],
        "away_doubtful": [],
    }

    absent_kws   = ["out", "absent", "suspended", "ruled out", "injured",
                    "miss", "will not play", "won't play", "unavailable",
                    "tournament-ending", "red card ban"]
    doubtful_kws = ["doubtful", "questionable", "fitness doubt", "50/50",
                    "late fitness test", "not fully fit", "carrying"]

    text_lower = text.lower()
    all_teams  = {home_team: "home", away_team: "away"}

    for player_name, p_data in PLAYER_IMPACTS.items():
        p_team  = p_data.get("team", "")
        if p_team not in (home_team, away_team):
            continue

        side = "home" if p_team == home_team else "away"
        name_lower = player_name.lower()

        if name_lower not in text_lower:
            continue

        # Find position of name in text
        pos = text_lower.find(name_lower)
        # Check ±150 chars around the name for keywords
        context = text_lower[max(0, pos-150):pos+150]

        is_absent   = any(kw in context for kw in absent_kws)
        is_doubtful = any(kw in context for kw in doubtful_kws)

        if is_absent:
            result[f"{side}_absent"].append(player_name)
        elif is_doubtful:
            result[f"{side}_doubtful"].append(player_name)

    return result


def get_known_absences(home_team: str, away_team: str):
    """
    Return confirmed absences we already know about from research.
    These are pre-loaded and don't require a web search.

    Updated manually before the tournament and after each match.
    """
    known = {
        # Germany vs Paraguay (Round of 32, June 29)
        ("Germany", "Paraguay"): {
            "home_absent":   ["Nico Schlotterbeck"],    # tournament-ending injury
            "away_absent":   ["Diego Gomez"],            # suspended (2nd yellow)
            "home_doubtful": ["Nathaniel Brown"],        # groin issue
            "away_doubtful": ["Omar Alderete"],          # knee problem
        },
        # France vs Sweden (Round of 32, June 30)
        ("France", "Sweden"): {
            "home_absent":   [],
            "away_absent":   [],
            "home_doubtful": [],
            "away_doubtful": [],
        },
        # England vs DR Congo (Round of 32, July 1)
        ("England", "DR Congo"): {
            "home_absent":   [],
            "away_absent":   [],
            "home_doubtful": [],
            "away_doubtful": [],
        },
        # Netherlands vs Morocco (Round of 32, June 29)
        ("Netherlands", "Morocco"): {
            "home_absent":   [],
            "away_absent":   [],
            "home_doubtful": [],
            "away_doubtful": [],
        },
    }
    return known.get((home_team, away_team), {
        "home_absent":   [],
        "away_absent":   [],
        "home_doubtful": [],
        "away_doubtful": [],
    })


# ─────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  LINEUP IMPACT TEST — Germany vs Paraguay")
    print("="*60)

    lam_h_base = 1.908   # from our Stage 2 anchored model
    lam_a_base = 0.967

    absences = get_known_absences("Germany", "Paraguay")
    print(f"\n  Known absences:")
    print(f"    Germany absent:    {absences['home_absent']}")
    print(f"    Paraguay absent:   {absences['away_absent']}")
    print(f"    Germany doubtful:  {absences['home_doubtful']}")
    print(f"    Paraguay doubtful: {absences['away_doubtful']}")

    lam_h_adj, lam_a_adj, log, corner_adj = apply_lineup_adjustments(
        lam_h_base, lam_a_base,
        "Germany", "Paraguay",
        home_absent   = absences["home_absent"],
        away_absent   = absences["away_absent"],
        home_doubtful = absences["home_doubtful"],
        away_doubtful = absences["away_doubtful"],
        verbose=True
    )

    print(f"\n  Corner adjustment factor: {corner_adj:.3f}")

    # Show impact on Germany win probability
    import sys
    sys.path.insert(0, ".")
    from pipeline_v2 import score_matrix, derive_all_markets
    def to_platform_int(p): return max(1, min(99, round(p*100)))

    mat_before = score_matrix(lam_h_base, lam_a_base)
    mat_after  = score_matrix(lam_h_adj,  lam_a_adj)
    m_before   = derive_all_markets(mat_before, lam_h_base, lam_a_base)
    m_after    = derive_all_markets(mat_after,  lam_h_adj,  lam_a_adj)

    print(f"\n  Market impact:")
    print(f"  {'Market':<30} {'Before':>8} {'After':>8} {'Delta':>8}")
    print(f"  {'─'*56}")
    for key in ["home_win", "btts", "over_2_5", "home_lead_ht", "home_scores_first"]:
        b = m_before[key] * 100
        a = m_after[key]  * 100
        print(f"  {key:<30} {b:>7.1f}% {a:>7.1f}% {a-b:>+7.1f}%")

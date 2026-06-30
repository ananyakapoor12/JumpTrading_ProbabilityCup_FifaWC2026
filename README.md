# Probability Cup — FIFA World Cup 2026 Quant Model

Automated match prediction system for the SportsPredict Probability Cup. Built around a Poisson football model with Dixon-Coles correction, MLE team strength fitting, live odds anchoring, and a REST API submission pipeline.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Mathematical Model](#2-mathematical-model)
   - [Poisson Score Matrix](#21-poisson-score-matrix)
   - [Dixon-Coles Correction](#22-dixon-coles-correction)
   - [MLE Team Strength Fitting](#23-mle-team-strength-fitting)
   - [Expected Goals Calculation](#24-expected-goals-calculation)
   - [Market Anchoring](#25-market-anchoring)
   - [Model-Market Blending](#26-model-market-blending)
3. [Derived Market Formulas](#3-derived-market-formulas)
   - [Match Result Markets](#31-match-result-markets)
   - [Goals Markets](#32-goals-markets)
   - [Halftime Markets](#33-halftime-markets)
   - [Time-Window Markets](#34-time-window-markets)
   - [Counting Stats](#35-counting-stats)
   - [Player Props](#36-player-props)
4. [Question Classifier](#4-question-classifier)
5. [Data Pipeline](#5-data-pipeline)
6. [Lineup Impact Engine](#6-lineup-impact-engine)
7. [How to Run](#7-how-to-run)
8. [Current Results](#8-current-results)
9. [Roadmap](#9-roadmap)

---

## 1. System Architecture

```
openfootball (GitHub) ──► parse_completed_matches ──► fit_strengths (MLE)
                                                              │
The Odds API ────────────────────────────────────────────────┤
                                                              ▼
                                              compute_anchored_lambdas
                                                              │
lineup_impact ───────────────────────────────────────────────┤
                                                              ▼
                                              score_matrix + derive_all_markets
                                                              │
                                              classify(question) + resolve_prediction
                                                              │
                                              KapbotClient.submit_batch()
                                                              │
                                              kap.noodles (SportsPredict profile)
```

**Files:**

| File | Role |
|---|---|
| `pipeline_v2.py` | Core engine — MLE, anchoring, score matrix, market derivation, classifier |
| `data_fetcher.py` | Live data — openfootball results, player shares, odds |
| `poisson_model.py` | Standalone reference Poisson implementation |
| `lineup_impact.py` | Pre-match lambda adjustments for absences/injuries |
| `api_client.py` | SportsPredict REST client (event → lobby → markets → submit) |
| `run.py` | Single-match CLI runner |
| `scheduler.py` | 24/7 tournament bot (fires 35 min before each kickoff) |
| `state_manager.py` | Persistence — tracks submitted matches, overrides, skip log |

---

## 2. Mathematical Model

### 2.1 Poisson Score Matrix

The fundamental assumption: goals scored by each team in a match are **independent Poisson random variables**.

$$P(\text{home scores } i) = \frac{e^{-\lambda_H} \cdot \lambda_H^i}{i!}, \quad P(\text{away scores } j) = \frac{e^{-\lambda_A} \cdot \lambda_A^j}{j!}$$

The full score matrix is an $(N+1) \times (N+1)$ grid (N = 10):

$$M_{ij} = P(\text{home}=i,\, \text{away}=j) = \text{Pois}(\lambda_H, i) \cdot \text{Pois}(\lambda_A, j)$$

All market probabilities are sums over subsets of this matrix.

---

### 2.2 Dixon-Coles Correction

The independent Poisson model overestimates the probability of 0-0 draws and underestimates 1-0, 0-1 scorelines. The [Dixon-Coles (1997)](https://www.tandfonline.com/doi/abs/10.2307/2986290) correction applies a multiplicative factor $\tau$ to low-score cells:

$$\tau(\lambda_H, \lambda_A, i, j, \rho) = \begin{cases}
1 - \lambda_H \lambda_A \rho & \text{if } i=0, j=0 \\
1 + \lambda_H \rho & \text{if } i=0, j=1 \\
1 + \lambda_A \rho & \text{if } i=1, j=0 \\
1 - \rho & \text{if } i=1, j=1 \\
1 & \text{otherwise}
\end{cases}$$

**Current value:** $\rho = -0.13$

After applying $\tau$, the full matrix is renormalized so probabilities sum to 1.

---

### 2.3 MLE Team Strength Fitting

Each team $k$ has two latent parameters: **attack** $\alpha_k$ and **defence** $\delta_k$.

**Expected goals for a single match** (home team $h$, away team $a$):

$$\lambda_H^{(model)} = \text{BASE\_RATE} \cdot \alpha_h \cdot \delta_a \cdot \text{HOME\_ADV} \cdot \text{KO\_ADJ}$$

$$\lambda_A^{(model)} = \text{BASE\_RATE} \cdot \alpha_a \cdot \delta_h$$

**Constants:** `BASE_RATE = 1.25`, `HOME_ADV = 1.10` (suppressed at neutral venues), `KO_ADJ = 0.92` (knockout games are tighter).

Parameters $\{\alpha_k, \delta_k\}$ are estimated by **Maximum Likelihood Estimation** over all completed WC2026 matches. The log-likelihood is:

$$\ell = \sum_{\text{matches}} \left[ \log P(\text{home goals} \mid \lambda_H) + \log P(\text{away goals} \mid \lambda_A) \right]$$

$$= \sum_{\text{matches}} \left[ g_H \log \lambda_H - \lambda_H + g_A \log \lambda_A - \lambda_A \right] + C$$

where $g_H, g_A$ are observed goals (capped at `XG_CAP = 3.5`).

**Recency weighting:** Older matches are down-weighted by $w = \gamma^{\text{age}}$ with $\gamma = 0.90$, so recent form matters more than results from earlier in the tournament.

**Identifiability constraint:** Mean attack across all teams is fixed to 1.0 (one degree of freedom removed).

**Optimizer:** SLSQP (Sequential Least Squares Programming via `scipy.optimize.minimize`).

---

### 2.4 Expected Goals Calculation

After fitting $\{\alpha_k, \delta_k\}$, the raw model lambdas for a specific match are:

$$\lambda_H^{(raw)} = 1.25 \cdot \alpha_H \cdot \delta_A \cdot 1.10 \cdot 0.92$$

$$\lambda_A^{(raw)} = 1.25 \cdot \alpha_A \cdot \delta_H \cdot 0.92$$

These are then anchored to the live betting market (see §2.5).

---

### 2.5 Market Anchoring

If a live Over/Under 2.5 market exists, we anchor the **total expected goals** to the market's implied value rather than trusting the model alone.

**Step 1 — Remove vig from O/U market:**

Given American odds $o^+$ (over) and $o^-$ (under), implied probabilities are:

$$p_{over}^{raw} = \frac{|o|}{|o|+100} \quad (o < 0), \qquad p_{over}^{raw} = \frac{100}{o+100} \quad (o > 0)$$

Fair (vig-free) probabilities:

$$p_{under}^{fair} = \frac{p_{under}^{raw}}{p_{over}^{raw} + p_{under}^{raw}}$$

**Step 2 — Solve for $\lambda_{total}$ via binary search:**

We want $\lambda^{*}$ such that:

$$P(\text{total goals} \leq 2 \mid \lambda^{*}) = p_{under}^{fair}$$

$$\sum_{k=0}^{2} \frac{e^{-\lambda^{*}} (\lambda^{*})^k}{k!} = p_{under}^{fair}$$

This is solved numerically (binary search over $\lambda^{*} \in [0.5, 8.0]$).

**Step 3 — Split $\lambda^{*}$ by model ratio:**

$$\lambda_H = \lambda^{*} \cdot \frac{\lambda_H^{(raw)}}{\lambda_H^{(raw)} + \lambda_A^{(raw)}}, \qquad \lambda_A = \lambda^{*} - \lambda_H$$

This preserves the **relative team strength** from the model while anchoring the **total scoring rate** to what the market says.

---

### 2.6 Model-Market Blending

For markets where a direct betting line exists (3-way win odds, O/U), the final probability blends model and market:

$$p_{final} = 0.40 \cdot p_{model} + 0.60 \cdot p_{market}^{fair}$$

**Rationale:** 60% weight to the market acknowledges that books aggregate huge amounts of information (team news, public money, sharp bets) that our model cannot observe. 40% to model preserves our edge when the model disagrees with the market. This ratio will be tuned based on Brier score analysis.

---

## 3. Derived Market Formulas

### 3.1 Match Result Markets

All computed by summing over the score matrix $M$:

$$P(\text{home win}) = \sum_{i > j} M_{ij}$$

$$P(\text{draw}) = \sum_{i = j} M_{ij}$$

$$P(\text{away win}) = \sum_{i < j} M_{ij}$$

---

### 3.2 Goals Markets

**Over/Under N.5 goals:**

$$P(\text{over } N\text{.5}) = 1 - \sum_{k=0}^{N} \text{Pois}(\lambda_H + \lambda_A,\, k)$$

This uses the total lambda directly, valid since sum of independent Poissons is Poisson.

**Both Teams to Score (BTTS):**

$$P(\text{BTTS}) = P(g_H \geq 1) \cdot P(g_A \geq 1) = (1 - e^{-\lambda_H})(1 - e^{-\lambda_A})$$

---

### 3.3 Halftime Markets

First half carries approximately **45% of total expected goals** (empirically, scoring slightly increases in the second half):

$$\lambda_H^{HT} = 0.45 \cdot \lambda_H, \qquad \lambda_A^{HT} = 0.45 \cdot \lambda_A$$

Half-time score matrix $M^{HT}$ is built identically to the full-time matrix. Then:

$$P(\text{home leads HT}) = \sum_{i > j} M^{HT}_{ij}, \qquad P(\text{away leads HT}) = 1 - P(\text{home leads HT}) - P(\text{draw HT})$$

**Home scores in both halves:**

$$P = P(g_H^{1H} \geq 1) \cdot P(g_H^{2H} \geq 1) = (1 - e^{-\lambda_H^{HT}})(1 - e^{-\lambda_H \cdot 0.55})$$

**Second half more goals than first half:**

$$P(G_{2H} > G_{1H}), \quad G_{1H} \sim \text{Pois}(0.45\lambda_{tot}),\quad G_{2H} \sim \text{Pois}(0.55\lambda_{tot})$$

$$= \sum_{g_2=1}^{N} \sum_{g_1=0}^{g_2-1} \text{Pois}(0.55\lambda_{tot}, g_2) \cdot \text{Pois}(0.45\lambda_{tot}, g_1)$$

---

### 3.4 Time-Window Markets

**Poisson process in a time window** — if goals arrive at rate $\lambda_{tot}$ per 90 minutes, then in a window of $t$ minutes:

$$P(\text{≥1 goal in } t \text{ min}) = 1 - e^{-\lambda_{tot} \cdot t/90}$$

Applied to specific markets:

| Market | Window | Formula |
|---|---|---|
| Goal before first hydration break (≈30') | $t=30$ | $1-e^{-\lambda_{tot}/3}$ |
| Goal after second hydration break (≈76'–90') | $t=15$ | $1-e^{-\lambda_{tot}/6}$ |
| Goal in first-half stoppage time | $t=4$ | $1-e^{-\lambda_{tot}\cdot 4/94}$ |

**Penalty awarded:**

Based on WC knockout tournament base rate of ≈0.28 penalties/match (constant, no per-team adjustment yet):

$$P(\text{penalty}) = 0.28$$

**Card after second hydration break** (last 15–20 min + any extra time):

$$P(\text{≥1 card in window}) = 1 - e^{-\lambda_{cards} \cdot f \cdot b}$$

where $f = 20/90$ (fraction of match in window), $b = 1.3$ (late-game card rate increases approximately 30% above uniform), and $\lambda_{cards}$ is the match card rate (default 3.5 cards/match).

---

### 3.5 Counting Stats

**Team-level counting stats** (shots on target, corners, cards, offsides) use pre-computed lambda values $\lambda_{stat}$ stored in the `_lambdas` dict from `derive_all_markets`:

$$P(\text{team has} \geq N \text{ stat}) = 1 - \sum_{k=0}^{N-1} \text{Pois}(\lambda_{stat}, k)$$

**Shots on target lambda:** Derived from goal rate via conversion rate $c = 0.33$:

$$\lambda_{SOT} = \lambda_{goals} / c = 3.03 \cdot \lambda_{goals}$$

**Corner lambda:** Scaled by team's attacking dominance share:

$$\lambda_{corners}^H = C_{mean} \cdot \frac{\lambda_H}{\lambda_H + \lambda_A}$$

where $C_{mean} = 10.5$ corners/match. For corner dominance markets:

$$P(\text{home more corners}) = \sum_{c_H=1}^{N} \sum_{c_A=0}^{c_H-1} \text{Pois}(\lambda_{corners}^H, c_H) \cdot \text{Pois}(\lambda_{corners}^A, c_A)$$

**Both teams carded** (assuming independence between teams):

$$P = \left(1 - e^{-\lambda_{cards}/2}\right)^2$$

---

### 3.6 Player Props

Each player has a **shot share** $s \in [0, 1]$ — their fraction of their team's shots. These are estimated from WC2026 goal tallies (live) with positional priors as fallback.

**Player goal probability:**

$$P(\text{player scores}) = 1 - e^{-\lambda_{team} \cdot s}$$

**Player shot on target probability** (threshold $\geq n$):

The player's expected SOT rate uses team SOT rate $c_{sot} = 0.33$ (goals/SOT):

$$\lambda_{player}^{SOT} = \frac{\lambda_{team}}{c_{sot}} \cdot s$$

$$P(\text{player} \geq n \text{ SOT}) = 1 - \sum_{k=0}^{n-1} \text{Pois}(\lambda_{player}^{SOT}, k)$$

**Player goal or assist:**

Let $p_g = P(\text{goal})$ and $p_a \approx 0.65 \cdot p_g$ (assist rate ≈ 65% of goal rate):

$$P(\text{goal or assist}) = p_g + p_a - p_g \cdot p_a = 1 - (1 - p_g)(1 - p_a)$$

---

## 4. Question Classifier

SportsPredict questions are free-text. The `classify(question, home, away)` function maps them to model keys using a **two-stage parsing strategy:**

**Stage 1 — Regex patterns (most specific first):**
- Extract numeric thresholds: `"N or more total goals"`, `"N or more shots on target"`, etc.
- Distinguish player SOT from team SOT via parenthetical team-name check: `"(TeamName)"` in the question
- Extract player names: `re.match(r'Will\s+(.+?)\s+score a goal', ...)` → strip trailing `(TeamName)`

**Stage 2 — Fixed-shape keyword mapping (after all regex):**
- Keyword pairs `(kw1, kw2)` both must appear in the question
- Examples: `("ahead at halftime", home)` → `home_lead_ht`

**Why order matters:** All SportsPredict questions include the suffix `"(90 minutes + stoppage time)"`, which means `"stoppage"` appears in every question. Fixed-shape patterns that match on `"stoppage"` alone would poison unrelated markets — the regex patterns run first to prevent this.

**All classified market types:**

| Key | Description |
|---|---|
| `home_win`, `away_win` | Regulation result |
| `home_lead_ht`, `away_lead_ht` | Half-time lead |
| `home_scores_first`, `away_scores_first` | First goalscorer team |
| `over_2_5`, `total_goals_threshold` | Goals over N.5 |
| `btts` | Both teams score |
| `home_scores_both_halves` | Team scores in each half |
| `goal_before_break` | Goal before first HB (≈30') |
| `goal_after_2nd_hydration` | Goal after second HB (≈76') |
| `stoppage_time_goal` | First-half stoppage goal |
| `first_half_over` | First half N+ goals |
| `more_goals_2h_vs_1h` | Second half outscores first |
| `penalty` | Penalty awarded |
| `red_card` | Red card shown |
| `threshold` | N+ corners / cards / offsides / SOT (team) |
| `corner_dominance` | Team has more corners |
| `both_teams_carded` | Both teams receive ≥1 card |
| `card_after_2nd_hydration` | Card shown after second HB |
| `player_goal` | Named player scores |
| `player_goal_or_assist` | Named player goal or assist |
| `player_Nplus_sot` | Named player N+ shots on target |

---

## 5. Data Pipeline

### Sources

| Source | What | Refresh |
|---|---|---|
| [openfootball/world-cup.json](https://github.com/openfootball/world-cup.json) | All match results, scorers, xG | Per run |
| [The Odds API](https://the-odds-api.com) | Live 3-way + O/U 2.5 betting lines | Per run |
| `SHOT_SHARES` (hardcoded) | Player shot allocation fallback | Static |

### Player Shot Share Estimation

If a player has $g_p$ goals and their team has $G$ total goals in WC2026:

$$s_p = 0.70 \cdot \frac{g_p}{G} + 0.30 \cdot \text{positional\_prior}(p)$$

Positional priors: striker = 0.25, attacker = 0.18, attacking mid = 0.14, midfielder = 0.10, winger = 0.12, defender = 0.05.

Shares are clamped to $[0.08, 0.55]$.

---

## 6. Lineup Impact Engine

Lineups are confirmed ≈60 minutes before kickoff — after the market has priced in public information but before our submission window (35 min pre-kickoff). This is an exploitable edge.

**Attack adjustment** when player $p$ is absent:

$$\lambda_H^{adj} = \lambda_H \cdot \left(1 - w \cdot \left(1 - \frac{1}{\alpha_p}\right)\right)$$

where $\alpha_p$ is the player's attack impact factor (e.g., Haaland = 1.35) and $w = 1.0$ (absent) or $w = 0.5$ (doubtful).

**Defence adjustment** — opponent's expected goals increase when a key defender is out:

$$\lambda_A^{adj} = \lambda_A \cdot \left(1 + w \cdot \left(1 - \frac{1}{\delta_p}\right)\right)$$

where $\delta_p$ is the player's defensive impact factor (e.g., van Dijk = 0.85).

**Corner taker absence** applies a further -8% multiplier to that team's corner rate.

---

## 7. How to Run

### Setup

```bash
pip install requests beautifulsoup4 scipy numpy

# Create .env in project root:
KAPBOT_API_KEY=sp_live_your_key_here
ODDS_API_KEY=your_odds_api_key        # optional but recommended
```

### Single Match

```bash
# List upcoming fixtures
python run.py --list

# Dry run (no submission)
python run.py --match "France vs Sweden" --dry-run

# Predict and submit (prompts for approval)
python run.py --match "France vs Sweden"

# Submit without prompt (bot mode)
python run.py --match "France vs Sweden" --approve
```

### 24/7 Bot

```bash
python scheduler.py           # interactive — asks before each submission
python scheduler.py --auto    # fully automatic
python scheduler.py --status  # show submission history and exit
python scheduler.py --override "France vs Sweden" home_win 72  # manual override
```

### Validate Model Accuracy

```bash
python run.py --validate      # Brier scores on completed matches
```

### Git Tags

Each match submission is tagged:

```bash
git tag <matchcode>-v1        # e.g. civ-v-nor-v1
git diff ned-v-mar-v1 civ-v-nor-v1  # what changed between matches
```

---

## 8. Current Results

### Submitted Matches

| Match | Submitted | Markets |
|---|---|---|
| Germany vs Paraguay | 2026-06-29 | 15 |
| Netherlands vs Morocco | 2026-06-30 | 15 |
| Ivory Coast vs Norway | 2026-06-30 | 15 |

### Settled Brier Scores (NED vs MAR)

Brier score: $B = (p - o)^2$ where $p \in [0,1]$ is predicted probability and $o \in \{0,1\}$ is outcome. Lower is better. Baseline (always predict 50%) = **0.25**.

| Market | Prediction | Brier |
|---|---|---|
| Morocco leads HT | 27% | **0.073** ✓ (Morocco did NOT lead — correct direction) |
| Netherlands scores first | 49% | 0.260 (very close to 50/50 — honest) |
| Goal before first hydration break | 52% | 0.270 |
| Goal in first-half stoppage | 50% (default) | 0.250 |
| Goal after second hydration break | 50% (default) | 0.250 |
| Second half more goals than first | 50% (default) | 0.250 |

The Morocco HT lead prediction (0.073) represents a strong signal — the model correctly assigned low probability to Morocco leading at half-time.

---

## 9. Roadmap

### Immediate (after Brier review each match)

- **Tune blend weights** — after 5+ settled matches, use Brier score breakdown per market type to estimate optimal `MODEL_WEIGHT` / `MARKET_WEIGHT` (currently fixed at 40/60)
- **Penalty rate refinement** — currently a flat 0.28 constant. Better: adjust by team foul rate and referee style (trackable from WC2026 data)
- **Expand `SHOT_SHARES`** — add Ødegaard, Sørloth, and other players not yet in live data so player props stop defaulting to 50%

### Medium-term

- **Live xG data source** — openfootball provides actual goals, not xG. Switch to a live xG feed (e.g. FBref or StatsBomb) so the MLE fits on expected goals rather than raw goals, reducing outlier noise
- **Dixon-Coles rho calibration** — $\rho = -0.13$ is a standard value from the original 1997 paper fitted on English football. Re-estimate on WC knockout data where draws and 0-0 are potentially rarer
- **Lineup automation** — `lineup_impact.py` has the engine; wire it to a live news source so absences are auto-detected 60 min pre-kickoff without manual input
- **Correlated markets** — some markets are not independent (e.g., `over_2_5` and `btts` are correlated). Pricing them jointly from the score matrix (already done) is correct, but the submission could exploit cross-market consistency as a signal

### Longer-term

- **Negative Binomial instead of Poisson** — Poisson assumes variance = mean. In practice, variance in match goal-scoring is slightly higher (overdispersion). NB with a small dispersion parameter could improve tail probabilities
- **Match-specific correlation** — RHO is currently fixed at -0.13 for all matches. High-intensity derbies or matches with tactical pressure may show different low-score clustering
- **Halftime lambda split calibration** — currently hardcoded 45%/55% first/second half. Fit this ratio from WC2026 observed HT/FT scores
- **Kelly-style confidence sizing** — when our model disagrees strongly with the market, flag the market for human review (potential edge), rather than blending toward the market

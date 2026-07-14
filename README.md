# Probability Cup - FIFA World Cup 2026 Quant Model

Automated football match prediction system for the SportsPredict Probability Cup.

The project combines:
- a Poisson scoreline model,
- Dixon-Coles low-score correction,
- MLE team strength fitting from tournament results,
- market anchoring from betting odds,
- lineup/news adjustments,
- and an API submission/scheduler workflow.

## Table of Contents

- [1. What This Project Does](#1-what-this-project-does)
- [2. Tech Stack](#2-tech-stack)
- [3. Project Structure](#3-project-structure)
- [4. Data Sources](#4-data-sources)
- [5. Mathematical Model](#5-mathematical-model)
- [6. End-to-End Pipeline](#6-end-to-end-pipeline)
- [7. Running the Project](#7-running-the-project)
- [8. Scheduler and State Management](#8-scheduler-and-state-management)
- [9. Market Mapping and Question Classification](#9-market-mapping-and-question-classification)
- [10. Calibration, Validation, and Risk Controls](#10-calibration-validation-and-risk-controls)
- [11. Assumptions and Limitations](#11-assumptions-and-limitations)
- [12. Operational Notes](#12-operational-notes)

## 1. What This Project Does

For each match, the system:

1. Fits team attack/defence strengths with maximum likelihood using tournament match data.
2. Computes expected goals $(\lambda_{home}, \lambda_{away})$.
3. Anchors total goals to live market totals (O/U 2.5) when available.
4. Builds a full scoreline probability matrix (0..10 goals each side).
5. Applies Dixon-Coles correction for low-score outcomes.
6. Derives market probabilities (1X2, BTTS, totals, halftime, props).
7. Blends model probabilities with fair market probabilities where available.
8. Optionally adjusts lambdas for lineup absences/doubtful players.
9. Submits predictions via SportsPredict batch API.

The project supports both:
- manual per-match execution, and
- continuous automated scheduling around kickoff windows.

## 2. Tech Stack

### Language and Runtime

- Python 3.10+ (recommended)

### Core Libraries

- `numpy`: matrix and numerical operations
- `scipy.optimize`: constrained MLE optimization (`SLSQP`)
- `requests`: HTTP integration (data + APIs)
- `schedule`: periodic polling loop in the bot scheduler
- `beautifulsoup4`: listed in install instructions (kept for extensible scraping flows)

### External Services

- OpenFootball JSON (historical/fixture data)
- The Odds API (optional market odds)
- SportsPredict/Kapbot API (event/lobby/market discovery and submissions)

## 3. Project Structure

- `pipeline_v2.py`
	- Core model math and end-to-end pricing/submission flow
	- Poisson matrix, Dixon-Coles, MLE fitting, market derivation, classifier
- `data_fetcher.py`
	- Tournament data ingestion, odds ingestion, player share generation, env loading
- `lineup_impact.py`
	- Player impact database + lineup-based lambda adjustments
- `api_client.py`
	- REST client for SportsPredict API endpoints
- `run.py`
	- Main CLI runner for per-match execution
- `scheduler.py`
	- Always-on bot with kickoff timing logic
- `state_manager.py`
	- Persistent bot state (`bot_state.json`) for submissions/skips/overrides
- `poisson_model.py`
	- Earlier explanatory/stage model implementation and derivation notes

## 4. Data Sources

### Match Data

- OpenFootball 2026 World Cup JSON feed is used to fetch completed and upcoming matches.
- Completed fixtures are transformed to tuples used by the MLE fitter:
	- `(home, away, home_goals, away_goals, home_xg, away_xg)`

### Odds Data (Optional)

- The Odds API provides:
	- 3-way prices (`home/draw/away`)
	- totals prices (`over/under 2.5`)
- Median odds across books are used for robustness.

### Lineup/Player Context

- Hardcoded known absences and player impact multipliers.
- Optional text parsing for injury/suspension signals.

## 5. Mathematical Model

### 5.1 Poisson Goal Process

Per team, goals are modeled as Poisson random variables:

$$
P(X = k) = e^{-\lambda} \frac{\lambda^k}{k!}
$$

with team-specific expected goals $\lambda$.

### 5.2 Scoreline Matrix

Assuming independent home and away goal processes:

$$
P(G_h=i, G_a=j) = P(G_h=i) \cdot P(G_a=j)
$$

The project computes this for $i,j \in [0,10]$, then normalizes after corrections.

### 5.3 Dixon-Coles Low-Score Correction

Low-score outcomes are adjusted using $\rho=-0.13$:

$$
τ(0,0)=1-\lambda_h\lambda_a\rho,
\quad
τ(1,0)=1+\lambda_a\rho,
\quad
τ(0,1)=1+\lambda_h\rho,
\quad
τ(1,1)=1-\rho
$$

Only these cells are adjusted, then matrix is renormalized.

### 5.4 Team Strength Model (Attack/Defence)

Base rate model:

$$
\lambda_h = b \cdot a_h \cdot d_a \cdot \text{homeAdv} \cdot \text{KOAdj}
$$
$$
\lambda_a = b \cdot a_a \cdot d_h \cdot \text{KOAdj}
$$

Where:
- $b$ is base scoring rate (`BASE_RATE=1.25`),
- $a_*$ are attack strengths,
- $d_*$ are defensive concession multipliers,
- knockout and venue adjustments are applied.

### 5.5 MLE Fitting Objective

Attack/defence parameters are estimated by maximizing weighted log-likelihood across matches.

- Uses capped observed goals/xG proxy (`XG_CAP=3.5`) to reduce outlier leverage.
- Applies recency weighting (`RECENCY_DECAY=0.90`) per team home-match history.
- Enforces identifiability with mean-attack constraint:
	- $\frac{1}{N}\sum_i a_i = 1$.
- Optimizer: SLSQP with box bounds `[0.1, 4.0]`.

### 5.6 Market Anchoring (Totals)

Given fair under-2.5 probability from de-vigged market, solve for total lambda:

$$
P(T \le 2 \mid \lambda_T)=\sum_{k=0}^{2} e^{-\lambda_T}\frac{\lambda_T^k}{k!}
$$

Binary search solves $\lambda_T$, then split by model ratio:

$$
\lambda_h = \lambda_T \cdot r_h,
\quad
\lambda_a = \lambda_T \cdot (1-r_h),
\quad
r_h = \frac{\lambda_h^{model}}{\lambda_h^{model}+\lambda_a^{model}}
$$

### 5.7 Probability Blending (Model + Market)

For markets with external prices:

$$
p_{final} = w_m p_{model} + w_k p_{market}
$$

with defaults:
- `MODEL_WEIGHT = 0.40`
- `MARKET_WEIGHT = 0.60`

No-market props are model-only.

### 5.8 Derived Markets and Props

The model prices many outcomes analytically from matrix/lambdas, including:

- 1X2 (`home_win`, `draw`, `away_win`)
- BTTS, over/under thresholds
- halftime leader (using first-half lambda split)
- first goalscorer team via Poisson race
- cards/corners/offsides count thresholds
- player goal / player SOT props from team lambda and shot-share
- event-window props (goal/card in specified minute windows)

## 6. End-to-End Pipeline

1. `fetch_all(...)` gets match data, shot shares, upcoming fixtures, and optional odds.
2. `fit_strengths(...)` estimates team parameters.
3. `compute_anchored_lambdas(...)` computes and optionally anchors lambdas.
4. `apply_lineup_adjustments(...)` modifies lambdas when lineup news exists.
5. `score_matrix(...)` + `derive_all_markets(...)` produce market probabilities.
6. `classify(...)` maps SportsPredict market question text to model keys.
7. `resolve_prediction(...)` returns probability for each market question.
8. Probabilities are converted to platform integer scale `[1..99]`.
9. Submission happens via `KapbotClient.submit_batch(...)` or update flow.

## 7. Running the Project

## 7.1 Install

```bash
pip install requests beautifulsoup4 scipy numpy schedule
```

## 7.2 Environment Variables

Create `.env` in project root:

```env
KAPBOT_API_KEY=sp_live_your_key_here
ODDS_API_KEY=your_the_odds_api_key_here
```

- `KAPBOT_API_KEY`: required for real submissions
- `ODDS_API_KEY`: optional, enables market anchoring/blending from live odds

## 7.3 Main CLI

List upcoming matches:

```bash
python run.py --list
```

Dry run for one match:

```bash
python run.py --match "Germany vs Paraguay" --dry-run
```

Approve and submit:

```bash
python run.py --match "France vs Sweden"
```

Auto-approve mode:

```bash
python run.py --match "France vs Sweden" --approve
```

## 7.4 Direct Pipeline Runner

```bash
python pipeline_v2.py --home Germany --away Paraguay --dry-run
```

## 8. Scheduler and State Management

`scheduler.py` is a continuous bot loop for tournament operation.

Key timing parameters:
- `CHECK_INTERVAL_MINUTES = 5`
- `FIRE_WINDOW_MINUTES = 35`
- `LINEUP_WATCH_MINUTES = 70`

Behavior:
- polls fixtures,
- detects next unsubmitted match,
- fires pipeline near kickoff,
- applies overrides,
- submits and records results.

Run scheduler once:

```bash
python scheduler.py --once
```

Run continuously (auto mode):

```bash
python scheduler.py --auto
```

Check status:

```bash
python scheduler.py --status
```

Set override:

```bash
python scheduler.py --override "France vs Sweden" home_win 70
```

State persistence is handled by `state_manager.py` in `bot_state.json`:
- submitted matches and outcomes,
- skipped matches and reasons,
- manual overrides,
- last run and next fire metadata.

## 9. Market Mapping and Question Classification

SportsPredict questions are natural language strings. The classifier in `pipeline_v2.py` maps these to internal keys using regex and keyword patterns.

Examples:
- "Will Team X win in regulation" -> `home_win` / `away_win`
- "3 or more total goals" -> threshold totals market
- player text markets -> player goal/SOT pricing keys

This enables consistent pricing of both canonical and non-canonical market wording.

## 10. Calibration, Validation, and Risk Controls

Validation helpers in `data_fetcher.py` compute Brier scores on recent completed matches for selected markets.

Risk controls built into the model:
- bounded parameter optimization,
- xG/goal cap for outlier suppression,
- recency weighting,
- market anchoring and blending,
- probability clipping to platform limits `[1..99]`.

## 11. Assumptions and Limitations

- Independence assumption between home/away goals is an approximation.
- Some prop markets (cards/corners/offsides) use stylized Poisson approximations.
- Data quality and freshness depend on external sources and API availability.
- Player props rely on shot-share heuristics when richer event data is unavailable.
- Hardcoded lineup priors require periodic maintenance.
- Fair-probability conversion uses proportional de-vig normalization.

## 12. Operational Notes

- Use `--dry-run` before live submissions.
- Keep `.env` secrets private and do not commit API keys.
- Re-run close to kickoff to incorporate latest odds and lineup changes.
- If markets already exist for you, use update-capable submit paths in the API client.

---

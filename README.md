# ⚽ FPL Agent

An end-to-end AI-powered Fantasy Premier League prediction system. Collects live data nightly, predicts points for all 838 players using XGBoost, selects the optimal £100m squad using linear programming, and generates a weekly briefing powered by Claude AI.

> **Season 2025-26 live results:** 64.1 avg pts/GW · 2,434 total points · top 10k pace · beat average manager by 534 points over 38 GWs

---

## Table of Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Quick start](#quick-start)
- [Weekly workflow](#weekly-workflow)
- [Pipeline phases](#pipeline-phases)
- [Season results](#season-results)
- [Testing](#testing)
- [Dashboard](#dashboard)
- [Configuration](#configuration)
- [Contributing / Learning](#contributing--learning)
- [Phase 8 roadmap](#phase-8-roadmap)

---

## What it does

Every week before the FPL deadline, the system:

1. **Collects** live data from the FPL API, Understat (xG), and sports news RSS feeds
2. **Engineers** 25+ features per player — rolling form, minutes trend, xGI, custom fixture difficulty, injury risk, ownership
3. **Predicts** points for all 838 active players using XGBoost trained on 4 seasons of data
4. **Selects** the optimal 15-player squad within FPL constraints (£100m budget, 3-per-club, valid formation) using linear programming
5. **Ranks captains** using a dedicated XGBoost classifier trained on historical top-scorer data
6. **Generates alternatives** — SAFE, DIFFERENTIAL, and VALUE scenario teams alongside the optimal pick
7. **Briefs** you with a Claude AI weekly summary — captain pick with reasoning, transfer suggestions, key risks before deadline
8. **Evaluates** output quality using DeepEval with Claude as judge (faithfulness, hallucination, relevancy)

```bash
python scheduler.py run      # Refresh all data (~15 mins)
python scheduler.py predict  # Generate predictions + captain scores (~2 mins)
python -m prediction.scenarios  # Generate 3 alternative teams
streamlit run dashboard.py   # View team in browser
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                          │
│  FPL API · Understat xG · BBC/Sky/Guardian RSS · Playwright │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                     DATA PIPELINE                            │
│  fpl_collector → understat_scraper → ETL → news_monitor     │
│  → injury_tracker → historical_loader                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                  FEATURE ENGINEERING                         │
│  feature_engineering → fixture_difficulty                    │
│  → lineup_probability → feature_store                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    PREDICTION MODELS                         │
│  XGBoost points predictor (MAE 0.934)                        │
│  Captain classifier (XGBoost binary, AUC-ROC 0.847)         │
│  LP team selector → optimal team + 3 scenario teams         │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                     LLM AGENT LAYER                          │
│  Claude AI → captain pick · transfers · risks · summary      │
│  DeepEval → faithfulness · hallucination · relevancy         │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                       INTERFACES                             │
│  Streamlit dashboard · Terminal report · GitHub Actions CI   │
│  Allure HTML report → GitHub Pages                           │
└─────────────────────────────────────────────────────────────┘
```

**Database:** PostgreSQL 15 (local) with 12 tables, 110,000+ rows  
**Scheduler:** launchd (Mac) — data pipeline at 02:00, predictions at 03:00  
**Stack:** Python 3.11, XGBoost, PuLP, SQLAlchemy, Anthropic Claude API, Streamlit, DeepEval

---

## Project Structure

```
fpl-agent/
├── agent/
│   ├── __init__.py
│   └── fpl_agent.py            # Claude API briefing generator
│
├── collectors/
│   ├── fpl_collector.py        # FPL API — players, fixtures, history
│   ├── understat_scraper.py    # xG/xA via Playwright response interception
│   ├── news_monitor.py         # RSS injury signals + Claude parsing
│   ├── injury_tracker.py       # Injury profiles from absence history
│   └── historical_loader.py   # vaastav archive loader (2022-25)
│
├── pipeline/
│   ├── __init__.py
│   ├── etl_pipeline.py         # Validate, transform, upsert all tables
│   ├── feature_engineering.py  # Rolling features, xG enrichment, positions
│   ├── feature_store.py        # Join all features → model_features table
│   ├── fixture_difficulty.py   # Custom xGA-based FDR
│   └── lineup_probability.py   # 4-signal availability model
│
├── prediction/
│   ├── __init__.py
│   ├── predict_points.py       # XGBoost training + prediction
│   ├── captain_model.py        # Captain classifier v2 (real ceiling features)
│   ├── team_selector.py        # PuLP LP optimisation
│   ├── scenarios.py            # SAFE / DIFFERENTIAL / VALUE team scenarios
│   └── backtest.py             # Walk-forward validation
│
├── tests/
│   ├── unit/
│   │   ├── test_signals.py     # 34 tests — lineup probability signals
│   │   ├── test_selector.py    # 21 tests — LP solver FPL constraints
│   │   ├── test_captain_model.py  # 17 tests — captain classifier features
│   │   └── test_scenarios.py   # 23 tests — alternative team scenarios
│   ├── integration/
│   │   └── test_pipeline.py    # 20 tests — DB integrity, features, predictions
│   └── evals/
│       ├── test_agent_evals.py    # 24 tests — Claude output quality
│       ├── test_deepeval_agent.py # 13 tests — DeepEval semantic metrics
│       └── test_dataset.py        # 52 tests — factual/reasoning/edge/adversarial
│
├── .github/
│   └── workflows/
│       └── eval.yml            # GitHub Actions CI — rule-based + DeepEval + Allure
│
├── data/raw/
├── db/
├── logs/
├── models/                     # Saved XGBoost models + agent cache + pending submissions
├── notebooks/
│
├── dashboard.py                # Streamlit web dashboard
├── weekly_predict.py           # Full pipeline entry point
├── scheduler.py                # APScheduler pipeline orchestrator
├── requirements.txt            # Production dependencies
├── requirements-ci.txt         # CI-only lean dependencies
└── pytest.ini
```

---

## Quick Start

### Prerequisites

- Python 3.11
- PostgreSQL 15 (`brew install postgresql@15` on Mac)
- Anthropic API key
- Playwright (for Understat scraping)

### Installation

```bash
# Clone
git clone <your-repo-url>
cd fpl-agent

# Virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Dependencies
pip install -r requirements.txt
playwright install chromium

# PostgreSQL setup
createdb fpl_agent
psql fpl_agent -c "CREATE USER fpl_user WITH PASSWORD 'fpl_password';"
psql fpl_agent -c "GRANT ALL PRIVILEGES ON DATABASE fpl_agent TO fpl_user;"

# Environment variables
export FPL_DB_URL="postgresql://fpl_user:fpl_password@localhost:5432/fpl_agent"
export ANTHROPIC_API_KEY="your-api-key-here"

# First run — load historical archive + collect current season (~20 mins)
python collectors/historical_loader.py   # Load 2022-25 archive
python scheduler.py run                  # Collect current season data

# Generate predictions
python scheduler.py predict

# View dashboard
streamlit run dashboard.py
```

---

## Weekly Workflow

```bash
# After each gameweek completes (Thursday/Friday)

# Step 1 — refresh all data (~15 mins)
python scheduler.py run

# Step 2 — generate predictions + captain scores (~2 mins)
python scheduler.py predict

# Step 3 — generate alternative team scenarios (~1 min)
python -m prediction.scenarios

# Step 4 — view team and briefing
streamlit run dashboard.py

# Step 5 — verify pipeline health
pytest -v -m "not deepeval"
```

**Nightly scheduler (launchd — Mac):**

The scheduler runs automatically in the background after setup:

```bash
# Start (survives reboots)
launchctl load ~/Library/LaunchAgents/com.fpl-agent.scheduler.plist

# Stop
launchctl unload ~/Library/LaunchAgents/com.fpl-agent.scheduler.plist

# Check status
launchctl list com.fpl-agent.scheduler
```

Schedule:
- `02:00` — Full data pipeline (FPL → Understat → ETL → News → Injuries)
- `03:00` — Prediction pipeline (Features → XGBoost → Captain → Team → Claude)
- `06:00, 12:00, 18:00, 22:00` — News monitor refresh

---

## Pipeline Phases

### Phase 1 — Data Collection

| Script | Source | Output | Volume |
|---|---|---|---|
| `fpl_collector.py` | FPL API | players, fixtures, player_history | ~830 players/GW |
| `understat_scraper.py` | Understat (Playwright) | understat_players | 2,200+ records |
| `news_monitor.py` | RSS + Claude | lineup_signals | 300+/run |
| `injury_tracker.py` | player_history | injury_profiles | 729 profiles |
| `historical_loader.py` | vaastav GitHub archive | player_history_archive | 103,194 rows |

### Phase 2 — Data Storage

PostgreSQL with 12 tables. ETL validates nulls, outliers, duplicates and upserts on conflict.

Key fix: `player_history_archive` now includes `position` and `team` columns from the vaastav CSVs, eliminating the UNK position issue that affected all historical training data.

### Phase 3 — Feature Engineering

25+ features per player per gameweek:

- **Form:** `avg_points_5gw`, `avg_points_10gw`, `avg_points_season`, `points_trend`
- **Minutes:** `avg_minutes_5gw`, `started_rate_5gw`, `full_game_rate_5gw`
- **Attacking:** `avg_goals_5gw`, `avg_assists_5gw`, `avg_ict_5gw`, `xgi_season`, `xgi_per90_season`
- **Defensive:** `clean_sheet_rate_5gw`, `avg_saves_5gw`, `avg_goals_conceded_5gw`
- **Fixture:** custom xGA-based FDR, `opponent_xga_5`, `was_home`
- **Availability:** `lineup_probability` (4-signal weighted), `fpl_status`
- **Ownership:** `selected_by_percent`, `transfers_in_event`
- **Injury:** `injury_prone_score`, `games_since_return`

**Lineup probability** combines 4 signals:

```
lineup_prob = 0.50 × FPL status signal
            + 0.25 × news signal (RSS/Claude parsed)
            + 0.15 × minutes trend signal
            + 0.10 × injury history signal
```

### Phase 4 — Points Prediction Model

**XGBoost Regressor:**
- Train: 2022-25 seasons (80,930 rows, deduplicated)
- Validate: 2025-26 GW1-30 (29,338 rows)
- MAE: **0.934 pts/player** | RMSE: 1.913
- Top-10 accuracy: 60%
- Top features: `avg_points_5gw` (29%), `avg_minutes_5gw` (16%), `avg_points_10gw` (15%)

**LP Team Selector (PuLP):**
- Maximises adjusted_points (predicted × lineup_probability)
- Enforces all FPL constraints: 2GK/5DEF/5MID/3FWD, ≤£100m, max 3/club
- Captain overridden by captain classifier score
- Solves in ~2 seconds

### Phase 8a — Captain Classifier

**XGBoost Binary Classifier:**
- Target: "was this player the top scorer in their GW?"
- Train: 2022-25 (80,930 rows, 140 positives)
- Validate: 2025-26 (29,338 rows, 49 positives)
- AUC-ROC: **0.847**
- In-sample top-1 accuracy: **11.5%** (vs 6.7% random baseline)

Key features:
- `max_points_5gw` — actual ceiling in last 5 GWs (real, not proxy)
- `last_3gw_top10_rate` — was player in global top 10 scorers recently?
- `is_high/medium/low_ownership` — ownership tier (not compound ratio)
- `selected_by_percent` — raw FPL ownership

Note: 2025-26 out-of-sample accuracy was lower than expected due to unusually random top scorers that season (Ballard, J.Timber, Zubimendi). Model expected to perform at ~11-15% in a typical season.

### Phase 8c — Alternative Team Scenarios

Three scenarios generated alongside the optimal team:

| Scenario | Strategy | GW38 result |
|---|---|---|
| **SAFE** | Only starters with linp ≥ 0.85 | 51 pts |
| **DIFFERENTIAL** | Excludes top-3 template players by price×form | 54 pts |
| **VALUE** | Forces minimum £98m spend | 54 pts |

Consensus picks (in all scenarios) = must-own players regardless of strategy.

### Phase 5 — LLM Agent Layer

Claude reads the full prediction context and produces four XML-tagged sections:

- **`<captain>`** — recommendation with reasoning and vice-captain
- **`<transfers>`** — up to 3 in/out pairs with price delta and rationale
- **`<risks>`** — 2-3 concerns to monitor before deadline
- **`<summary>`** — one paragraph overview with contrarian insight

Response cached locally (~£0.01/call, ignored on re-runs unless `--refresh`).

---

## Season Results

**2025-26 live season (38 GWs):**

```
Total points      : 2,434
Avg pts/GW        : 64.1
Best GW           : GW17 (94 pts)
Worst GW          : GW34 (39 pts)
Above avg (≥50)   : 33/38 GWs
Big weeks (≥70)   : 14/38 GWs
```

**Benchmark comparison:**

| Benchmark | Pts/GW | 38 GW total |
|---|---|---|
| Average manager | 50.0 | 1,900 |
| Top 100k | 58.0 | 2,204 |
| **Our model** | **64.1** | **2,434** |
| Top 10k | 65.0 | 2,470 |
| Top 1k | 72.0 | 2,736 |

**36 points off top 10k** — less than 1 point per gameweek. The gap is entirely captaincy: GW37 wrong captain (Thiago vs Watkins) cost 13 pts alone.

---

## Testing

**204 tests total.**

```bash
pytest -v                          # All tests including DeepEval
pytest -v -m "not deepeval"        # 191 tests — no API cost
pytest tests/unit/ -v              # Fast unit tests only (~2s)
pytest tests/integration/ -v       # Requires DB connection
pytest tests/evals/ -v             # Requires agent cache + API
```

**Test layers:**

| File | Tests | What it covers |
|---|---|---|
| `test_signals.py` | 34 | Lineup probability signal boundary values |
| `test_selector.py` | 21 | LP solver FPL constraint validation |
| `test_captain_model.py` | 17 | Captain classifier feature engineering |
| `test_scenarios.py` | 23 | Alternative scenario constraints and output |
| `test_pipeline.py` | 20 | DB integrity, feature engineering, predictions |
| `test_agent_evals.py` | 24 | Claude output — structure, content, factuality, quality |
| `test_deepeval_agent.py` | 13 | Semantic — faithfulness, hallucination, relevancy |
| `test_dataset.py` | 52 | 52-prompt dataset: factual recall, reasoning, edge cases, adversarial |

**CI pipeline (GitHub Actions):**
- Rule-based evals: runs on every push (free)
- DeepEval semantic metrics: runs every Saturday (~£0.02/run, Claude judge)
- Allure HTML report: auto-published to GitHub Pages after each run

**Hallucination gate:** if `HallucinationMetric` score exceeds 0.20 on any section, the CI job fails and blocks the merge.

---

## Dashboard

```bash
streamlit run dashboard.py
# Opens at http://localhost:8501
```

Five tabs:

| Tab | Content |
|---|---|
| **Team** | SVG pitch view, captain/VC badges, probability bars, FDR chips |
| **Predictions** | Top 30 players, position filter, all metrics including captain_score |
| **Fixtures** | Upcoming fixtures — custom xGA-based FDR vs FPL built-in |
| **Watchlist** | Injury/doubt players with news, probability, 5GW form |
| **Briefing** | Claude captain/transfers/risks/summary + backtest bar chart |

---

## Configuration

**Required environment variables:**

```bash
export FPL_DB_URL="postgresql://fpl_user:fpl_password@localhost:5432/fpl_agent"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

**Optional (for FPL auto-submission — Phase 8e, 2026-27):**

```bash
export FPL_EMAIL="your@fpl-email.com"
export FPL_PASSWORD="yourpassword"
export FPL_TEAM_ID="123456"   # from FPL URL: fantasy.premierleague.com/entry/[ID]/history
```

**SQLite fallback:** if `FPL_DB_URL` is not set, all scripts fall back to `db/fpl.db`. Good for development without PostgreSQL.

**Key constants:**

| Constant | Default | File |
|---|---|---|
| `BUDGET` | £100m | `prediction/team_selector.py` |
| `MAX_PER_CLUB` | 3 | `prediction/team_selector.py` |
| `SAFE_LINP_THRESHOLD` | 0.85 | `prediction/scenarios.py` |
| `VALUE_MIN_SPEND` | £98m | `prediction/scenarios.py` |
| `DIFF_EXCLUDE_TOP_N` | 3 | `prediction/scenarios.py` |
| `SAFE_LINP_THRESHOLD` | 0.85 | `pipeline/lineup_probability.py` |

---

## Contributing / Learning

This project covers the full data science stack end to end — ideal for learning by doing.

**Suggested learning path:**

| Level | Focus | Task |
|---|---|---|
| Beginner | Data engineering | Add a feature to `pipeline/feature_engineering.py` + write a unit test |
| Beginner | SQL | Explore `player_history` in PostgreSQL, build a points chart |
| Intermediate | Feature engineering | Add set piece data (corners/free kicks) as a new signal |
| Intermediate | Modelling | Try LightGBM instead of XGBoost — compare MAE |
| Intermediate | LLM testing | Add a new test to `tests/evals/test_dataset.py` |
| Advanced | Captain classifier | Improve `prediction/captain_model.py` with better ceiling features |
| Advanced | DGW multiplier | Detect DGWs and multiply predicted points by 1.8 in `feature_store.py` |
| Advanced | Transfer planner | 3-GW fixture lookahead for optimal transfer sequencing |

**What makes this project good for learning:**
- Real messy data (non-ASCII names, null prices, wrong API flags, duplicate rows)
- Real constraints (budget, club limits, formation rules, PostgreSQL vs SQLite)
- Real evaluation (backtested against actual FPL scores, not toy datasets)
- Ground truth feedback — wrong predictions show up immediately in your GW score
- Full stack — from API scraping through ML to LLM evaluation to CI/CD

---

## Phase 8 Roadmap (2026-27 Season)

**Completed this off-season:**

| Phase | Feature | Status |
|---|---|---|
| 8a | Captain classifier v2 (real ceiling features, ownership tiers) | ✅ Done |
| 8b | 52-prompt evaluation dataset (factual/reasoning/edge/adversarial) | ✅ Done |
| 8c | Alternative team scenarios (SAFE/DIFFERENTIAL/VALUE) | ✅ Done |
| 8d | Data quality fixes (position UNK, duplicate rows, archive reload) | ✅ Done |

**Planned for 2026-27:**

| Phase | Feature | Est. pts gain |
|---|---|---|
| 8e | FPL auto-submission with approval gate | — |
| 8f | Transfer window monitor (new player detection) | — |
| 8g | DGW multiplier (×1.8 for double gameweek teams) | +3 pts/GW |
| 8h | Transfer planner (3-GW fixture lookahead) | +3 pts/GW |
| 8i | Real ceiling features from player_history archive | +2 pts/GW |
| 8j | Set piece data (corners/free kicks ownership) | +1 pts/GW |

**Target for 2026-27: ~70 pts/GW (top 1k territory)**

---

## Acknowledgements

- [FPL API](https://fantasy.premierleague.com/api/) — live player and fixture data
- [Understat](https://understat.com/) — xG/xA statistics
- [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) — historical FPL data archive
- [Anthropic Claude](https://www.anthropic.com/) — AI briefing and evaluation judge
- [PuLP](https://coin-or.github.io/pulp/) — LP optimisation
- [DeepEval](https://github.com/confident-ai/deepeval) — LLM evaluation framework

---

*Built during the 2025-26 FPL season. Final result: 2,434 pts (64.1 avg pts/GW) — top 10k pace, beat average manager by 534 points over 38 GWs. Phase 8 improvements targeting top 1k for 2026-27.*

# ⚽ FPL Agent

An end-to-end AI-powered Fantasy Premier League prediction system. Collects live data nightly, predicts points for all 838 players using XGBoost, selects the optimal £100m squad using linear programming, and generates a weekly briefing powered by Claude AI.

> **Season 2025-26 backtest:** 51.5 avg pts/GW — above the average FPL manager (50 pts/GW)

---

## Table of Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Quick start](#quick-start)
- [Weekly workflow](#weekly-workflow)
- [Pipeline phases](#pipeline-phases)
- [Model performance](#model-performance)
- [Testing](#testing)
- [Dashboard](#dashboard)
- [Configuration](#configuration)
- [Contributing / Learning](#contributing--learning)
- [Phase 8 roadmap](#phase-8-roadmap)

---

## What it does

Every week before the FPL deadline, the system:

1. **Collects** live data from the FPL API, Understat (xG), and sports news RSS feeds
2. **Engineers** 25 features per player — rolling form, minutes trend, xGI, custom fixture difficulty, injury risk
3. **Predicts** points for all 838 active players using an XGBoost model trained on 4 seasons of historical data
4. **Selects** the optimal 15-player squad within FPL constraints (£100m budget, 3-per-club, valid formation) using linear programming
5. **Briefs** you with a Claude AI weekly summary — captain pick with reasoning, transfer suggestions, key risks before deadline

```
python scheduler.py run      # Refresh all data (~15 mins)
python scheduler.py predict  # Generate predictions (~2 mins)
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
│  → injury_tracker                                            │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                  FEATURE ENGINEERING                         │
│  feature_engineering → fixture_difficulty                    │
│  → lineup_probability → feature_store                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    PREDICTION MODEL                          │
│  XGBoost (MAE 0.934) → LP team selector → predictions table │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                     LLM AGENT LAYER                          │
│  Claude AI → captain pick · transfers · risks · summary      │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                       INTERFACES                             │
│  Streamlit dashboard · Terminal report · Cached briefing     │
└─────────────────────────────────────────────────────────────┘
```

**Database:** PostgreSQL 15 (local) with 11 tables, 166,000+ rows  
**Scheduler:** APScheduler — data pipeline at 02:00, predictions at 03:00  
**Stack:** Python 3.11, XGBoost, PuLP, SQLAlchemy, Anthropic Claude API, Streamlit

---

## Project Structure

```
fpl-agent/
├── collectors/
│   ├── fpl_collector.py        # FPL API — players, fixtures, history
│   ├── understat_scraper.py    # xG/xA via Playwright response interception
│   ├── news_monitor.py         # RSS injury signals + Claude parsing
│   └── injury_tracker.py       # Injury profiles from absence history
│
├── data/raw/                   # Dated JSON snapshots
├── db/                         # SQLite fallback (dev)
├── logs/                       # Pipeline logs (daily rotation)
├── models/                     # Saved model + backtest results + agent cache
├── notebooks/                  # Exploration
│
├── tests/
│   ├── unit/
│   │   ├── test_signals.py     # 34 tests — lineup probability signal functions
│   │   └── test_selector.py    # 21 tests — LP solver FPL constraint validation
│   ├── integration/
│   │   └── test_pipeline.py    # 20 tests — DB integrity, features, predictions
│   ├── evals/
│   │   └── test_agent_evals.py # 24 tests — Claude output quality (5 layers)
│   └── conftest.py
│
├── etl_pipeline.py             # Validate, transform, upsert all tables
├── feature_engineering.py      # Rolling features, xG enrichment, injury profiles
├── fixture_difficulty.py       # Custom FDR from rolling xGA data
├── lineup_probability.py       # 4-signal availability model
├── feature_store.py            # Join all features into model_features table
├── predict_points.py           # XGBoost training + prediction
├── team_selector.py            # PuLP LP optimisation
├── backtest.py                 # Walk-forward validation GW5-30
├── fpl_agent.py                # Claude API briefing generator
├── weekly_predict.py           # Single entry point for prediction pipeline
├── scheduler.py                # APScheduler pipeline orchestrator
├── dashboard.py                # Streamlit web dashboard
├── pytest.ini
└── requirements.txt
```

---

## Quick Start

### Prerequisites

- Python 3.11
- PostgreSQL 15 (via Homebrew on Mac: `brew install postgresql@15`)
- Anthropic API key (for Claude briefing)
- Playwright (for Understat scraping)

### Installation

```bash
# Clone the repo
git clone <your-repo-url>
cd fpl-agent

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
playwright install chromium

# Set up PostgreSQL
createdb fpl_agent
psql fpl_agent -c "CREATE USER fpl_user WITH PASSWORD 'fpl_password';"
psql fpl_agent -c "GRANT ALL PRIVILEGES ON DATABASE fpl_agent TO fpl_user;"

# Environment variables
export FPL_DB_URL="postgresql://fpl_user:fpl_password@localhost:5432/fpl_agent"
export ANTHROPIC_API_KEY="your-api-key-here"

# First run — collect all data (takes ~15 mins)
python scheduler.py run

# Generate predictions
python scheduler.py predict

# Launch dashboard
streamlit run dashboard.py
```

---

## Weekly Workflow

```bash
# Thursday/Friday after gameweek completes

# Step 1 — refresh all data (~15 mins)
python scheduler.py run

# Step 2 — generate GW predictions (~2 mins)
python scheduler.py predict

# Step 3 — view team and briefing
streamlit run dashboard.py

# Step 4 — run tests to verify nothing broke
pytest -v
```

**Or for hands-free operation (leave running overnight):**

```bash
python scheduler.py schedule
```

Schedule:
- `02:00` — Full data pipeline (FPL → Understat → ETL → News → Injuries)
- `03:00` — Prediction pipeline (Features → XGBoost → Team → Claude briefing)
- `06:00, 12:00, 18:00, 22:00` — News monitor refresh

**Quick modes:**

```bash
python scheduler.py run       # Full data refresh
python scheduler.py predict   # Predictions only (use existing data)
python scheduler.py news      # News monitor only (fast)
python scheduler.py schedule  # Background scheduler (runs forever)

python weekly_predict.py                   # Full pipeline
python weekly_predict.py --no-collect      # Skip data refresh
python weekly_predict.py --report-only     # Report from existing predictions

python fpl_agent.py            # Generate Claude briefing
python fpl_agent.py --refresh  # Force new API call (ignore cache)
```

---

## Pipeline Phases

### Phase 1 — Data Collection

| Script | Source | Output | Rows |
|---|---|---|---|
| `fpl_collector.py` | FPL API | players, fixtures, player_history | 27,657/GW |
| `understat_scraper.py` | Understat (Playwright) | understat_players | 2,218 |
| `news_monitor.py` | RSS + Claude | lineup_signals | 300+/run |
| `injury_tracker.py` | player_history | injury_profiles | 727 profiles |

### Phase 2 — Data Storage

PostgreSQL with 11 tables. ETL validates nulls, outliers, duplicates and upserts on conflict. Historical archive: 103,194 rows from 2022-26.

### Phase 3 — Feature Engineering

25 features per player per gameweek:

- **Form:** `avg_points_5gw`, `avg_points_10gw`, `points_trend`
- **Minutes:** `avg_minutes_5gw`, `started_rate_5gw`, `full_game_rate_5gw`
- **Attacking:** `avg_goals_5gw`, `avg_assists_5gw`, `avg_ict_5gw`, `xgi_season`
- **Defensive:** `clean_sheet_rate_5gw`, `avg_saves_5gw`
- **Fixture:** custom xGA-based FDR (not FPL's 1-5 scale), `opponent_xga_5`
- **Availability:** `lineup_probability` (4-signal weighted model), `fpl_status`
- **Injury:** `injury_prone_score`, `games_since_return`

**Lineup probability** combines 4 signals:

```
lineup_prob = 0.50 × FPL status signal
            + 0.25 × news signal (RSS/Claude parsed)
            + 0.15 × minutes trend signal
            + 0.10 × injury history signal
```

**Custom FDR** — built from rolling 5-game xGA per team. Leeds had the strongest defence (xGA 0.917), Newcastle the weakest (xGA 2.337). Significantly differs from FPL's built-in ratings.

### Phase 4 — Prediction Model

**XGBoost Regressor:**
- Train: 2022-25 seasons (80,930 rows)
- Validate: 2025-26 GW1-30 (45,350 rows)
- MAE: **0.934 pts/player** | RMSE: 1.913
- Top-10 accuracy: 60%
- Top features: `avg_points_5gw` (30%), `avg_points_10gw` (16%), `avg_minutes_5gw` (15%)

**LP Team Selector (PuLP):**
- Maximises adjusted_points (predicted × lineup_probability)
- Enforces all FPL constraints: 2GK/5DEF/5MID/3FWD, ≤£100m, max 3/club, valid formation
- Captain = highest adjusted_points starter
- Solves in ~2 seconds

### Phase 5 — LLM Agent Layer

Claude reads the full prediction context (top 25 players, selected team, injury watchlist, fixture difficulty, defence rankings) and produces four sections:

- **Captain** — one recommendation with reasoning and vice-captain
- **Transfers** — up to 3 in/out pairs with price delta and rationale
- **Risks** — 2-3 concerns to monitor before deadline
- **Summary** — one paragraph weekly overview with contrarian insight

Response is cached locally — re-runs don't call the API unless `--refresh` is passed (~£0.01/call).

### Phase 6 — Backtesting

Walk-forward validation across 2025-26 GW5-30:

| Metric | Value |
|---|---|
| Gameweeks tested | 26 |
| Avg pts/GW | **51.5** |
| Best GW | GW6 — 80 pts |
| Worst GW | GW7 — 31 pts |
| Captain accuracy | 19% |
| Prediction MAE | 12.72 pts/GW |

Benchmarks: average manager 50 pts/GW · top 10k 65 pts/GW · top 1k 72 pts/GW

---

## Testing

**99 tests, all passing.**

```bash
pytest -v                    # All 99 tests
pytest tests/unit/ -v        # Fast — no DB needed (~7s)
pytest tests/integration/ -v # Requires DB connection (~2s)
pytest tests/evals/ -v       # Requires agent cache (~1s)
pytest -m unit               # By marker
pytest -m "not slow"         # Skip slow tests
```

**Test layers:**

| Layer | Tests | What it covers |
|---|---|---|
| Unit | 55 | Signal boundary values, LP solver FPL constraints, edge cases |
| Integration | 20 | DB integrity, feature engineering, prediction pipeline |
| Evals | 24 | Claude output quality — structural, content, factual, quality, regression |

**The eval layer** tests non-deterministic LLM output against criteria rather than exact strings:

```python
# Traditional testing (wrong for LLMs)
assert response == "Captain Thiago"

# Eval testing (correct)
assert mentions_player_name(response)
assert mentions_fixture(response)
assert gives_reasoning(response)
assert prices_in_valid_range(response)
```

**Two bugs found by tests before they reached production:**
- Status `n` (not in squad) was returning 0.70 lineup probability instead of ≤0.02
- NaN prices crashed the LP solver — fixed with safe float conversion

---

## Dashboard

```bash
streamlit run dashboard.py
# Opens at http://localhost:8501
```

Five tabs:

| Tab | Content |
|---|---|
| **Team** | SVG pitch view with formation, captain/VC badges, probability bars, FDR chips |
| **Predictions** | Top 30 players with position filter, all metrics |
| **Fixtures** | Upcoming fixtures — custom FDR vs FPL FDR comparison |
| **Watchlist** | Injury/doubt players with news, probability, form |
| **Briefing** | Claude captain/transfers/risks/summary + backtest bar chart |

---

## Configuration

**Environment variables:**

```bash
export FPL_DB_URL="postgresql://fpl_user:fpl_password@localhost:5432/fpl_agent"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

**SQLite fallback (no PostgreSQL needed for dev):**

If `FPL_DB_URL` is not set, all scripts fall back to `db/fpl.db` (SQLite). Good for local development and testing without a running PostgreSQL instance.

**Key constants (editable per script):**

| Constant | Default | Location |
|---|---|---|
| `BUDGET` | £100m | `team_selector.py` |
| `MAX_PER_CLUB` | 3 | `team_selector.py` |
| `BACKTEST_GW_START` | 5 | `backtest.py` |
| `BACKTEST_GW_END` | 30 | `backtest.py` |
| `WEIGHT_FPL_STATUS` | 0.50 | `lineup_probability.py` |
| `WEIGHT_NEWS` | 0.25 | `lineup_probability.py` |
| `WEIGHT_MINUTES` | 0.15 | `lineup_probability.py` |
| `WEIGHT_INJURY` | 0.10 | `lineup_probability.py` |

---

## Contributing / Learning

This project covers the full data science stack end to end — ideal for learning by doing.

**Suggested learning path:**

| Level | Focus | Task |
|---|---|---|
| Beginner | Data engineering | Add a new feature to `feature_engineering.py` and write a unit test |
| Beginner | SQL | Explore `player_history` in PostgreSQL, build a form chart |
| Intermediate | Feature engineering | Add set piece data (corners/free kicks) as a new signal |
| Intermediate | Modelling | Try LightGBM instead of XGBoost — compare MAE |
| Intermediate | LLM testing | Add a new eval test to `test_agent_evals.py` |
| Advanced | Captaincy | Build a dedicated captain classifier (`captain_model.py`) |
| Advanced | DGW | Implement a double gameweek points multiplier (×1.8) |
| Advanced | Transfer planner | 3-GW fixture lookahead for optimal transfer sequencing |

**What makes this project good for learning:**
- Real messy data (non-ASCII names, null prices, wrong API flags)
- Real constraints (budget, club limits, formation rules)
- Real evaluation (backtested against actual FPL scores, not toy datasets)
- Full stack — from API scraping to LP optimisation to LLM prompting to web UI
- Instant honest feedback — wrong predictions show up in your GW score

---

## Phase 8 Roadmap (2026-27 Season)

Priority improvements for next season:

| Improvement | Estimated gain | Status |
|---|---|---|
| Captain classifier (separate XGBoost for captaincy) | +5 pts/GW | Planned |
| DGW multiplier (×1.8 for double gameweek teams) | +3 pts/GW | Planned |
| Transfer planner (3-GW lookahead) | +3 pts/GW | Planned |
| Budget floor constraint (≥£98m in LP) | +2 pts/GW | Planned |
| Chip timing strategy (TC/BB/WC) | +3 pts/GW | Planned |
| Folder structure refactor | — | Planned |
| Set piece data integration | +1 pts/GW | Stretch |
| Differential picker (low ownership + high ceiling) | +1 pts/GW | Stretch |

Target with Phase 8 improvements: **~67 pts/GW (top 10k territory)**

---

## Acknowledgements

- [FPL API](https://fantasy.premierleague.com/api/) — live player and fixture data
- [Understat](https://understat.com/) — xG/xA statistics
- [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) — historical FPL data archive
- [Anthropic Claude](https://www.anthropic.com/) — AI briefing layer
- [PuLP](https://coin-or.github.io/pulp/) — LP optimisation

---

*Built during the 2025-26 FPL season. Currently averaging above the average FPL manager. Phase 8 (captain classifier, DGW handling, transfer planner) planned for the off-season.*

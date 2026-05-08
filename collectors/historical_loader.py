"""
historical_loader.py  —  Phase 1, Component 1b: Historical Data Loader
=======================================================================

CONCEPTS TO UNDERSTAND BEFORE READING THIS SCRIPT
--------------------------------------------------

1. Why a separate historical archive table?
   We keep historical data in player_history_archive separate from
   player_history (current season) for a clean reason: the current season
   table gets refreshed every week by the FPL collector. The archive is
   static — it never changes once loaded. Mixing them would mean either
   overwriting history on every run or complex deduplication logic.
   Separate tables = simple, predictable behaviour.

2. Where the data comes from — vaastav's GitHub repo
   A community contributor has scraped and archived every FPL gameweek
   going back to 2016/17 in clean CSV format. It is the standard dataset
   used by almost every FPL data science project. Each CSV file represents
   one gameweek and contains per-player stats identical in shape to what
   our FPL collector produces.
   Repo: https://github.com/vaastav/Fantasy-Premier-League

3. How we fetch CSV files directly from GitHub
   GitHub exposes raw file content at raw.githubusercontent.com URLs.
   We construct the URL for each gameweek CSV programmatically and fetch
   it with requests — same fetch_url pattern from the FPL collector.
   pandas can read a CSV directly from a URL with pd.read_csv(url),
   which means no intermediate file saving needed.

4. Why we use if_exists='append' here
   Unlike the FPL collector which replaces tables on each run, the archive
   loader appends to the table. We only need to run this script once per
   season (or once ever for past seasons). Appending means we can safely
   add a new season's data without touching what's already there.

5. Season column — why it matters
   When we train the model in Phase 4, we need to know which season each
   row came from. This lets us weight recent seasons more heavily than
   older ones (a 2024/25 performance is more predictive of 2026/27 than
   a 2022/23 one). We add a 'season' column to every row we load.
"""

import requests
import pandas as pd
import time
from io import StringIO
from datetime import datetime
from sqlalchemy import create_engine, text


# =============================================================================
# CONFIGURATION
# =============================================================================

# Base URL pattern for vaastav's raw CSV files on GitHub
# We format this with season string and gameweek number per request
VAASTAV_BASE = (
    "https://raw.githubusercontent.com/vaastav/"
    "Fantasy-Premier-League/master/data/{season}/gws/gw{gw}.csv"
)

# Four seasons to load — format matches vaastav's folder naming convention
SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]

# Maximum gameweeks to attempt per season.
# We try all 38 but handle missing ones gracefully — some seasons have
# fewer completed gameweeks (e.g. current season mid-run, or COVID years).
MAX_GW = 38

DB_PATH = "db/fpl.db"

# Columns we want from each gameweek CSV.
# vaastav's CSVs have slightly different column names to the FPL API
# so we map them to our standard names below.
COLUMNS_WANTED = [
    "name",             # Player name (vaastav uses full name, not web_name)
    "element",          # FPL player ID — use this to join with players table
    "fixture",
    "round",            # Gameweek number
    "opponent_team",
    "was_home",
    "kickoff_time",
    "total_points",     # THE TARGET VARIABLE
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "value",            # Price at time of this gameweek (in tenths of £m)
    "transfers_balance",
    "selected",
    "team_h_score",
    "team_a_score",
]


# =============================================================================
# DATABASE
# =============================================================================

def get_database_engine():
    """Returns SQLAlchemy engine connected to the local SQLite database."""
    return create_engine(f"sqlite:///{DB_PATH}")


# =============================================================================
# CSV FETCHER
# =============================================================================

def fetch_gameweek_csv(season, gw):
    """
    Fetches a single gameweek CSV from vaastav's GitHub repo.
    Returns a pandas DataFrame on success, None if the file doesn't exist
    or the request fails. Missing gameweeks (e.g. blank GWs or future GWs
    in the current season) return a 404 which we handle silently.
    """
    url = VAASTAV_BASE.format(season=season, gw=gw)
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 404:
            return None  # Gameweek doesn't exist yet — not an error
        response.raise_for_status()
        # pd.read_csv can read from a string buffer — no need to save to disk
        df = pd.read_csv(StringIO(response.text))
        return df
    except requests.exceptions.RequestException:
        return None


# =============================================================================
# COLUMN NORMALISER
# =============================================================================

def normalise_columns(df, season, gw):
    """
    Standardises column names and adds metadata columns.

    vaastav's CSV column names are mostly consistent with the FPL API but
    have minor variations across seasons. We keep only the columns we want,
    handle missing ones gracefully, and add season/gw tags to every row
    so we can filter and weight by recency during model training.
    """
    # Keep only columns that exist in this CSV and that we want
    keep = [c for c in COLUMNS_WANTED if c in df.columns]
    df = df[keep].copy()

    # Add metadata so we know exactly where each row came from
    df["season"] = season      # e.g. "2023-24"
    df["gameweek"] = gw        # redundant with 'round' but explicit is better

    # Coerce numeric columns — some CSVs have mixed types due to missing values
    numeric_cols = [
        "total_points", "minutes", "goals_scored", "assists", "clean_sheets",
        "goals_conceded", "bonus", "bps", "value", "selected",
        "influence", "creativity", "threat", "ict_index",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# =============================================================================
# SEASON LOADER
# =============================================================================

def load_season(season, engine):
    """
    Loads all available gameweeks for one season into player_history_archive.
    Iterates GW1–GW38, fetches each CSV, normalises it, and appends to DB.
    Stops early if two consecutive gameweeks return 404 — this handles
    current seasons where later gameweeks haven't been played yet.
    """
    print(f"\n  Loading {season}...")
    total_rows = 0
    consecutive_missing = 0

    for gw in range(1, MAX_GW + 1):
        df = fetch_gameweek_csv(season, gw)

        if df is None or df.empty:
            consecutive_missing += 1
            if consecutive_missing >= 2:
                # Two missing GWs in a row = we've passed the end of this season
                print(f"    Stopped at GW{gw-1} (no more data)")
                break
            continue

        consecutive_missing = 0
        df = normalise_columns(df, season, gw)
        df.to_sql("player_history_archive", engine, if_exists="append", index=False)
        total_rows += len(df)

        print(f"    GW{gw:02d} — {len(df)} rows", end="\r")
        time.sleep(0.3)  # Polite rate limiting for GitHub's servers

    print(f"    ✓ {season} complete — {total_rows} rows loaded        ")
    return total_rows


# =============================================================================
# DEDUPLICATION
# =============================================================================

def deduplicate_archive(engine):
    """
    Removes duplicate rows from player_history_archive.

    If the loader is run more than once (e.g. to add a new season),
    we want to ensure there are no duplicate gameweek entries per player.
    We identify duplicates by the combination of element + round + season
    and keep only the first occurrence.
    """
    print("\n  Deduplicating archive...")
    with engine.connect() as conn:
        # SQLite-compatible deduplication using rowid
        conn.execute(text("""
            DELETE FROM player_history_archive
            WHERE rowid NOT IN (
                SELECT MIN(rowid)
                FROM player_history_archive
                GROUP BY element, round, season
            )
        """))
        conn.commit()

        result = conn.execute(text("SELECT COUNT(*) FROM player_history_archive"))
        count = result.fetchone()[0]
    print(f"  ✓ Archive contains {count:,} rows after deduplication")
    return count


# =============================================================================
# VERIFICATION QUERY
# =============================================================================

def verify_archive(engine):
    """
    Prints a summary of what's in the archive broken down by season.
    Lets us confirm all four seasons loaded correctly before proceeding.
    """
    print("\n  Archive summary:")
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT season,
                   COUNT(DISTINCT element) as players,
                   COUNT(DISTINCT round)   as gameweeks,
                   COUNT(*)                as total_rows,
                   ROUND(AVG(total_points), 2) as avg_points
            FROM player_history_archive
            GROUP BY season
            ORDER BY season
        """))
        rows = result.fetchall()

    print(f"  {'Season':<12} {'Players':>8} {'GWs':>6} {'Rows':>8} {'Avg Pts':>8}")
    print(f"  {'-'*46}")
    total = 0
    for row in rows:
        print(f"  {row[0]:<12} {row[1]:>8} {row[2]:>6} {row[3]:>8,} {row[4]:>8}")
        total += row[3]
    print(f"  {'-'*46}")
    print(f"  {'TOTAL':<12} {'':>8} {'':>6} {total:>8,}")


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_historical_load(seasons=None):
    """
    Loads historical FPL data for all specified seasons into the archive table.
    Safe to run multiple times — deduplication handles any overlap.
    Pass a subset of seasons to add just one new season e.g. ["2025-26"].
    """
    if seasons is None:
        seasons = SEASONS

    print("=" * 60)
    print(f"FPL Historical Loader  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Seasons: {', '.join(seasons)}")
    print("=" * 60)

    engine = get_database_engine()
    grand_total = 0

    for season in seasons:
        rows = load_season(season, engine)
        grand_total += rows

    deduplicate_archive(engine)
    verify_archive(engine)

    print(f"\n{'='*60}")
    print(f"  ✓ Total rows loaded: {grand_total:,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_historical_load()

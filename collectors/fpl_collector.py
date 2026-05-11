"""
fpl_collector.py  —  Phase 1, Component 1: FPL API Data Collector
==================================================================

CONCEPTS TO UNDERSTAND BEFORE READING THIS SCRIPT
--------------------------------------------------

1. REST API
   The FPL API is a REST API — a set of URLs that return JSON data when you
   call them with an HTTP GET request. No login required. Think of each URL
   as a specific question you can ask FPL's servers:
     "Give me all players"          →  /bootstrap-static/
     "Give me all fixtures"         →  /fixtures/
     "Give me player 233's history" →  /element-summary/233/

2. JSON → DataFrame → SQL  (the ETL pattern)
   Every data pipeline follows Extract, Transform, Load (ETL):
     Extract   = fetch raw JSON from the API
     Transform = shape it into a clean table using pandas DataFrame
     Load      = write that table into a database
   We follow this pattern for every endpoint. Understand it once, apply
   it to every data source in the project.

3. Why SQLite (not PostgreSQL yet)?
   SQLite is a database that lives in a single file on your machine.
   No server, no Docker, no config. Perfect for Phase 1 when you are
   the only one reading and writing. We migrate to PostgreSQL in Phase 2
   when multiple collectors need to write simultaneously.

4. Why save raw JSON AND a database?
   The database is optimised for fast queries. The JSON is your audit trail.
   When a prediction looks wrong in GW20, you open the JSON from that day
   and see exactly what the API returned. Think of it as keeping test logs
   alongside your test results.

5. Rate limiting  (being a polite API client)
   The element-summary endpoint requires one call per player (~500 players).
   Hammering 500 requests in 2 seconds will get your IP temporarily blocked.
   A 0.5s sleep between calls gives ~120 requests/minute — fast enough to
   finish in ~4 minutes, slow enough not to trigger a block.

6. Checkpointing
   If the script crashes at player 490 out of 500, we lose nothing because
   we flush to the database every 50 players. Same logic you would use in
   test automation to persist partial results before a suite times out.
"""

import requests
import json
import os
import time
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_URL      = "https://fantasy.premierleague.com/api"
BOOTSTRAP_URL = f"{BASE_URL}/bootstrap-static/"
FIXTURES_URL  = f"{BASE_URL}/fixtures/"
ELEMENT_URL   = f"{BASE_URL}/element-summary/{{player_id}}/"

DATA_DIR = "data/raw"
DB_PATH  = "db/fpl.db"


# =============================================================================
# SETUP
# =============================================================================

def setup_directories():
    """Creates local folders for raw JSON snapshots and the database file."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs("db", exist_ok=True)
    print("✓ Directories ready")


def get_database_engine():
    import os
    pg_url = os.environ.get("FPL_DB_URL")
    if pg_url:
        return create_engine(pg_url)
    return create_engine(f"sqlite:///{DB_PATH}")


# =============================================================================
# CORE HTTP FETCHER
# =============================================================================

def fetch_url(url, label=""):
    """
    Single reusable HTTP GET wrapper used for every API call in this project.
    Centralising HTTP logic here means if the FPL API ever needs a new header,
    cookie, or auth token, we change it in one place only.
    Returns parsed JSON as a dict on success, None on any failure.
    """
    headers = {
        # Some APIs reject requests with no User-Agent. We mimic a browser
        # to avoid 403 Forbidden responses.
        "User-Agent": "Mozilla/5.0 (compatible; FPL-Agent/1.0)"
    }
    try:
        print(f"  Fetching {label or url}...")
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()  # Throws on 4xx / 5xx status codes
        return response.json()
    except requests.exceptions.Timeout:
        print(f"  ✗ Timeout: {label}")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"  ✗ HTTP error: {label} — {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Request failed: {label} — {e}")
        return None


def save_raw_json(data, filename):
    """
    Writes a dated JSON snapshot to disk  (e.g. bootstrap_20260901.json).
    Dating the filename lets us replay any historical gameweek during
    backtesting without making a live API call.
    """
    date_str = datetime.now().strftime("%Y%m%d")
    filepath = os.path.join(DATA_DIR, f"{filename}_{date_str}.json")
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  ✓ Saved raw JSON → {filepath}")


# =============================================================================
# BOOTSTRAP-STATIC  (master endpoint — players, teams, gameweeks)
# =============================================================================

def collect_bootstrap(engine):
    """
    Fetches /bootstrap-static/ — the single richest endpoint in the FPL API.
    One call returns all players, all teams, and the full gameweek schedule.
    We split that response into three separate database tables.

    Column selection is intentional: the raw response has 60+ fields per player.
    We keep only what the prediction model will actually use. Adding columns
    later is trivial; cleaning up irrelevant ones mid-project is not.
    """
    print("\n[1/3] bootstrap-static...")
    data = fetch_url(BOOTSTRAP_URL, "bootstrap-static")
    if not data:
        return False
    save_raw_json(data, "bootstrap")

    # --- PLAYERS  (data["elements"] is the FPL name for the players list) ---
    players_df = pd.DataFrame(data["elements"])
    player_cols = [
        "id", "first_name", "second_name", "web_name",
        "team",                           # Team ID — join with teams table for name
        "element_type",                   # 1=GK  2=DEF  3=MID  4=FWD
        "now_cost",                       # Price in tenths of £m  (130 = £13.0m)
        "total_points", "points_per_game", "form",
        "selected_by_percent",            # Ownership %
        "minutes", "goals_scored", "assists", "clean_sheets",
        "goals_conceded", "yellow_cards", "red_cards",
        "bonus", "bps",                   # Bonus points and raw BPS score
        "influence", "creativity", "threat", "ict_index",
        "chance_of_playing_this_round",   # 0–100 from FPL team news
        "chance_of_playing_next_round",
        "news",                           # Free-text injury note
        "news_added",
        "status",                         # a=available  d=doubtful  i=injured  u=unavailable
        "cost_change_start",              # Price movement since season start
        "transfers_in_event",             # GW transfer volume — popularity signal
        "transfers_out_event",
    ]
    keep = [c for c in player_cols if c in players_df.columns]
    players_df[keep].to_sql("players", engine, if_exists="replace", index=False)
    print(f"  ✓ {len(players_df)} players saved")

    # --- TEAMS ---
    teams_df = pd.DataFrame(data["teams"])
    team_cols = [
        "id", "name", "short_name", "strength",
        "strength_overall_home", "strength_overall_away",
        "strength_attack_home",  "strength_attack_away",
        "strength_defence_home", "strength_defence_away",
    ]
    keep = [c for c in team_cols if c in teams_df.columns]
    teams_df[keep].to_sql("teams", engine, if_exists="replace", index=False)
    print(f"  ✓ {len(teams_df)} teams saved")

    # --- GAMEWEEKS  (FPL's internal name for gameweeks is 'events') ---
    gw_df = pd.DataFrame(data["events"])
    gw_cols = [
        "id", "name", "deadline_time",
        "average_entry_score", "highest_score",
        "is_current", "is_next", "is_previous",
        "finished", "data_checked",
    ]
    keep = [c for c in gw_cols if c in gw_df.columns]
    gw_df[keep].to_sql("gameweeks", engine, if_exists="replace", index=False)
    print(f"  ✓ {len(gw_df)} gameweeks saved")

    return True


# =============================================================================
# FIXTURES  (full season schedule + fixture difficulty ratings)
# =============================================================================

def collect_fixtures(engine):
    """
    Fetches /fixtures/ and saves every match in the season.

    We use this later to calculate upcoming fixture difficulty per player,
    flag blank gameweeks (no match that week), and double gameweeks
    (two matches in one week — a major FPL advantage if you own the right players).
    team_h_difficulty / team_a_difficulty are FPL's own FDR scores (1=easy, 5=hard).
    Phase 3 replaces these with our own xGA-based difficulty model.
    """
    print("\n[2/3] fixtures...")
    data = fetch_url(FIXTURES_URL, "fixtures")
    if not data:
        return False
    save_raw_json(data, "fixtures")

    fixtures_df = pd.DataFrame(data)
    fixture_cols = [
        "id",
        "event",              # Gameweek number (None if not yet scheduled to a GW)
        "team_h", "team_a",   # Home / away team IDs
        "team_h_score", "team_a_score",
        "kickoff_time", "finished",
        "team_h_difficulty",  # FPL FDR for home team
        "team_a_difficulty",  # FPL FDR for away team
    ]
    keep = [c for c in fixture_cols if c in fixtures_df.columns]
    fixtures_df[keep].to_sql("fixtures", engine, if_exists="replace", index=False)
    print(f"  ✓ {len(fixtures_df)} fixtures saved")
    return True


# =============================================================================
# ELEMENT-SUMMARY  (per-player gameweek-by-gameweek history)
# =============================================================================

def collect_player_histories(engine, max_players=None):
    # Truncate existing history — we rebuild fresh each run
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE player_history RESTART IDENTITY CASCADE"))
        conn.commit()
    print("  Cleared existing player history")
    
    print("\n[3/3] player histories...")
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT id, web_name FROM players")).fetchall()

    players = rows[:max_players] if max_players else rows
    print(f"  Processing {len(players)} players...")

    batch = []

    for i, (player_id, name) in enumerate(players):
        data = fetch_url(
            ELEMENT_URL.format(player_id=player_id),
            f"{name} ({i+1}/{len(players)})"
        )
        if data and "history" in data:
            for record in data["history"]:
                record["player_id"] = player_id  # Tag every row so we can JOIN later
                batch.append(record)

        time.sleep(0.5)  # Rate limiting — see concept note at the top of this file

        # Flush to DB every 50 players (checkpoint against crashes)
        if batch and (i + 1) % 50 == 0:
            _flush_history_batch(batch, engine)
            batch = []
            print(f"  ✓ Checkpoint: {i+1}/{len(players)} players")

    if batch:
        _flush_history_batch(batch, engine)

    print("  ✓ All histories saved")
    return True


def _flush_history_batch(records, engine):
    """Writes a batch of gameweek history records to the player_history table."""
    df = pd.DataFrame(records)
    history_cols = [
        "player_id", "element", "fixture", "round",
        "opponent_team", "was_home", "kickoff_time",
        "total_points",
        "minutes", "goals_scored", "assists", "clean_sheets",
        "goals_conceded", "own_goals", "penalties_saved", "penalties_missed",
        "yellow_cards", "red_cards", "saves", "bonus", "bps",
        "influence", "creativity", "threat", "ict_index",
        "value", "transfers_balance", "selected",
    ]
    keep = [c for c in history_cols if c in df.columns]
    df = df[keep]

    # Deduplicate — FPL API occasionally returns duplicate GW records
    df = df.drop_duplicates(subset=["player_id", "round"], keep="last")

    df.to_sql("player_history", engine, if_exists="append", index=False)


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_collection(max_players=None):
    """
    Orchestrates the full pipeline: setup → bootstrap → fixtures → histories.
    Call with max_players=20 during development, None for a full production run.
    """
    print("=" * 60)
    print(f"FPL Data Collection  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    setup_directories()
    engine = get_database_engine()

    results = {
        "bootstrap" : collect_bootstrap(engine),
        "fixtures"  : collect_fixtures(engine),
        "histories" : collect_player_histories(engine, max_players=max_players),
    }

    print("\n" + "=" * 60)
    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗'}  {name}")
    print("=" * 60)


if __name__ == "__main__":
    run_collection(max_players=None)  # Change to any integer for a quick test run (e.g. 20)

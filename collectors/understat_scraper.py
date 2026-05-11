"""
understat_scraper.py  —  Phase 1, Component 2: Understat xG/xA Scraper
=======================================================================

CONCEPTS TO UNDERSTAND BEFORE READING THIS SCRIPT
--------------------------------------------------

1. What is xG and why does it matter for FPL?
   Expected Goals (xG) measures shot quality — a tap-in from 5 yards might
   be xG=0.80, a long-range effort xG=0.03. It tells you what SHOULD have
   happened, not what did. This matters for FPL prediction because:
   - A striker with 0 goals but 3.2xG over 4 weeks is unlucky — buy him
   - A striker with 4 goals but 0.8xG is overperforming — he will regress
   - The model uses xG to predict future points, not past luck

2. What is xA?
   Expected Assists measures the quality of chances created. If you play a
   through ball that creates a 0.5xG chance, you get 0.5xA regardless of
   whether your teammate scores. Identifies creative players before their
   assist numbers show up.

3. xGChain and xGBuildup — the bonus metrics we discovered
   xGChain: xG value of every move a player was involved in that led to a
   shot. Rewards players who contribute to attacks even without scoring/assisting.
   xGBuildup: same but excludes the final pass and shot — pure team play contribution.
   These are more predictive of future FPL involvement than raw xG/xA alone.

4. How we collect the data — Playwright response interception
   Understat loads its data via an internal API call (getLeagueData) after
   the page renders. We use Playwright to load the page in a headless browser,
   intercept that API response as it arrives, and extract the JSON directly.
   This is cleaner than scraping HTML — we get structured data with zero parsing.

5. Why this approach is fast
   The old approach needed one HTTP request per player (~500 requests, 15+ mins).
   This approach needs ONE Playwright page load per season (4 total).
   All 530 players for a season arrive in a single 570kb response.
   Total runtime: ~2 minutes instead of 15+.

6. Two bonus fields from the player data
   xGChain and xGBuildup were not in the original plan but Understat provides
   them for free. We collect them now — they may become valuable features
   in the prediction model during Phase 3.
"""

import json
import time
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text
from playwright.sync_api import sync_playwright


# =============================================================================
# CONFIGURATION
# =============================================================================

# One URL per season — Understat uses the start year only
LEAGUE_URLS = {
    "2022-23": "https://understat.com/league/EPL/2022",
    "2023-24": "https://understat.com/league/EPL/2023",
    "2024-25": "https://understat.com/league/EPL/2024",
    "2025-26": "https://understat.com/league/EPL/2025",
}

DB_PATH = "db/fpl.db"


# =============================================================================
# DATABASE
# =============================================================================

def get_database_engine():
    import os
    pg_url = os.environ.get("FPL_DB_URL")
    if pg_url:
        return create_engine(pg_url)
    return create_engine(f"sqlite:///{DB_PATH}")


# =============================================================================
# CORE INTERCEPTOR — one function, one page load, all player data
# =============================================================================

def fetch_league_season(season_label, url, browser):
    """
    Loads one Understat league page in a headless browser and intercepts
    the getLeagueData API response as it arrives.

    Returns a dict with keys: 'players', 'teams', 'dates'
    or None if the interception failed.

    We pass the browser object in rather than creating a new one per call —
    reusing the same browser instance is faster and uses less memory.
    """
    captured = {}

    def handle_response(response):
        # We only care about Understat's internal data endpoint
        if "getLeagueData" in response.url:
            try:
                captured.update(response.json())
            except Exception:
                pass

    page = browser.new_page()
    page.on("response", handle_response)

    try:
        print(f"  Loading {season_label}...")
        page.goto(url, timeout=30000)
        page.wait_for_load_state("networkidle", timeout=30000)
    except Exception as e:
        print(f"  ✗ Failed to load {season_label}: {e}")
        page.close()
        return None

    page.close()

    if not captured:
        print(f"  ✗ No data intercepted for {season_label}")
        return None

    return captured


# =============================================================================
# DATA PARSER — converts raw API response into clean DataFrames
# =============================================================================

def parse_players(raw_players, season_label):
    """
    Converts the raw players list from getLeagueData into a clean DataFrame.

    The raw data has all numeric fields as strings (e.g. "26.34") — we cast
    them to proper numeric types so they work correctly in the model later.

    We also compute xGI (xG + xA) here as a convenience column since it's
    the single most useful attacking metric for FPL prediction.
    """
    rows = []
    for p in raw_players:
        rows.append({
            "understat_id"  : p.get("id"),
            "player_name"   : p.get("player_name"),
            "season"        : season_label,
            "team"          : p.get("team_title"),
            "position"      : p.get("position"),
            "games"         : int(p.get("games", 0)),
            "minutes"       : int(p.get("time", 0)),
            "goals"         : int(p.get("goals", 0)),
            "xG"            : float(p.get("xG", 0)),
            "assists"       : int(p.get("assists", 0)),
            "xA"            : float(p.get("xA", 0)),
            "xGI"           : float(p.get("xG", 0)) + float(p.get("xA", 0)),
            "shots"         : int(p.get("shots", 0)),
            "key_passes"    : int(p.get("key_passes", 0)),
            "yellow_cards"  : int(p.get("yellow_cards", 0)),
            "red_cards"     : int(p.get("red_cards", 0)),
            "npg"           : int(p.get("npg", 0)),        # Non-penalty goals
            "npxG"          : float(p.get("npxG", 0)),     # Non-penalty xG
            "xGChain"       : float(p.get("xGChain", 0)),  # Full attack chain involvement
            "xGBuildup"     : float(p.get("xGBuildup", 0)),# Build-up play contribution
        })

    return pd.DataFrame(rows)


def parse_teams(raw_teams, season_label):
    """
    Converts the raw teams dict from getLeagueData into a clean DataFrame.
    Team-level xG/xGA data is used in Phase 3 to build our custom fixture
    difficulty model — better than FPL's built-in FDR.

    raw_teams is a dict keyed by team_id, each value containing home/away splits.
    """
    rows = []
    for team_id, team_data in raw_teams.items():
        # Each team has 'title', 'id', and nested 'history' list
        title = team_data.get("title", "")
        history = team_data.get("history", [])

        for match in history:
            rows.append({
                "understat_team_id" : team_id,
                "team_name"         : title,
                "season"            : season_label,
                "date"              : match.get("date"),
                "xG"                : float(match.get("xG", 0)),    # xG scored
                "xGA"               : float(match.get("xGA", 0)),   # xG conceded
                "goals"             : int(match.get("scored", 0)),
                "goals_against"     : int(match.get("missed", 0)),
                "was_home"          : match.get("h_a") == "h",
                "result"            : match.get("result"),          # w/d/l
                "pts"               : int(match.get("pts", 0)),
            })

    return pd.DataFrame(rows)


# =============================================================================
# MAIN COLLECTION LOOP
# =============================================================================

def collect_all_seasons(engine):
    """
    Iterates over all four seasons, loads each page once via Playwright,
    intercepts the API response, parses it, and saves to the database.
    """
    # Truncate before re-collecting — rebuilt fresh each run
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE understat_players, understat_teams RESTART IDENTITY CASCADE"))
        conn.commit()
    print("  Cleared existing Understat data")

    print("\n[1/2] Collecting season-level player and team xG data...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        for season_label, url in LEAGUE_URLS.items():
            # Fresh dict per season — avoids cross-season contamination
            captured = {}

            def handle_response(response, cap=captured):
                if "getLeagueData" in response.url:
                    try:
                        cap.update(response.json())
                    except Exception:
                        pass

            page = browser.new_page()
            page.on("response", handle_response)

            try:
                print(f"  Loading {season_label}...")
                page.goto(url, timeout=30000)
                page.wait_for_load_state("networkidle", timeout=30000)
            except Exception as e:
                print(f"  ✗ Failed to load {season_label}: {e}")
                page.close()
                continue

            page.close()

            if not captured:
                print(f"  ✗ No data for {season_label}")
                continue

            if "players" in captured:
                players_df = parse_players(captured["players"], season_label)
                players_df.columns = [c.lower() for c in players_df.columns]
                players_df.to_sql("understat_players", engine, if_exists="append", index=False)
                print(f"  ✓ {season_label} — {len(players_df)} players saved")

            if "teams" in captured:
                teams_df = parse_teams(captured["teams"], season_label)
                if not teams_df.empty:
                    teams_df.columns = [c.lower() for c in teams_df.columns]
                    teams_df.to_sql("understat_teams", engine, if_exists="append", index=False)

            time.sleep(2)

        browser.close()


# =============================================================================
# DEDUPLICATION
# =============================================================================

def deduplicate(engine):
    """
    Removes any duplicate rows that could arise from running the script
    more than once. Identifies duplicates by understat_id + season for
    players, and team_name + date for team rows.
    PostgreSQL uses ctid (internal row ID) instead of SQLite's rowid.
    """
    print("\n  Deduplicating...")
    with engine.connect() as conn:
        conn.execute(text("""
            DELETE FROM understat_players
            WHERE ctid NOT IN (
                SELECT MIN(ctid) FROM understat_players
                GROUP BY understat_id, season
            )
        """))
        conn.execute(text("""
            DELETE FROM understat_teams
            WHERE ctid NOT IN (
                SELECT MIN(ctid) FROM understat_teams
                GROUP BY understat_team_id, date
            )
        """))
        conn.commit()
    print("  ✓ Deduplication complete")


# =============================================================================
# VERIFICATION
# =============================================================================

def verify(engine):
    """
    Prints a summary table of collected data broken down by season.
    Sanity check: Haaland should top xG for 2025-26, Salah should be
    near the top of xA. If the names look wrong, something went sideways.
    """
    print("\n[2/2] Verification")

    with engine.connect() as conn:
        # Season summary
        result = conn.execute(text("""
            SELECT season,
                   COUNT(*)              as players,
                   ROUND(SUM(xG), 1)    as total_xG,
                   ROUND(AVG(xGI), 3)   as avg_xGI
            FROM understat_players
            GROUP BY season
            ORDER BY season
        """))
        rows = result.fetchall()

    print(f"\n  {'Season':<12} {'Players':>8} {'Total xG':>10} {'Avg xGI':>9}")
    print(f"  {'-'*43}")
    for row in rows:
        print(f"  {row[0]:<12} {row[1]:>8} {row[2]:>10} {row[3]:>9}")

    # Top 10 by xGI for current season
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT player_name, team, position,
                   ROUND(xG, 2)      as xG,
                   ROUND(xA, 2)      as xA,
                   ROUND(xGI, 2)     as xGI,
                   ROUND(xGChain, 2) as xGChain
            FROM understat_players
            WHERE season = '2025-26'
            ORDER BY xGI DESC
            LIMIT 10
        """))
        top = result.fetchall()

    print(f"\n  Top 10 by xGI — 2025/26 season:")
    print(f"  {'Player':<24} {'Team':<22} {'Pos':<6} {'xG':>6} {'xA':>6} {'xGI':>6} {'xGChain':>8}")
    print(f"  {'-'*78}")
    for row in top:
        print(f"  {row[0]:<24} {row[1]:<22} {row[2]:<6} {row[3]:>6} {row[4]:>6} {row[5]:>6} {row[6]:>8}")


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_understat_collection():
    """
    Runs the full Understat collection pipeline.
    Four page loads, ~2 minutes total, all seasons done.
    """
    print("=" * 60)
    print(f"Understat Scraper  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Seasons: {', '.join(LEAGUE_URLS.keys())}")
    print("=" * 60)

    engine = get_database_engine()

    collect_all_seasons(engine)
    deduplicate(engine)
    verify(engine)

    print(f"\n{'='*60}")
    print("  Understat collection complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_understat_collection()

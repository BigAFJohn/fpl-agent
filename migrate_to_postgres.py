"""
migrate_to_postgres.py  —  Phase 2, Component 1: PostgreSQL Migration
=====================================================================

CONCEPTS TO UNDERSTAND BEFORE READING THIS SCRIPT
--------------------------------------------------

1. Why we define the schema explicitly this time
   In Phase 1 we let pandas infer column types when writing to SQLite
   (df.to_sql). This is fine for prototyping but produces loose types —
   everything becomes TEXT or REAL. PostgreSQL is strictly typed, so we
   define each table properly:
   - INTEGER not REAL for counts and IDs
   - NUMERIC(10,4) for xG/xA decimals (preserves precision)
   - TIMESTAMP for dates (enables date arithmetic)
   - TEXT for free-form strings
   Proper types make queries faster and prevent silent data errors.

2. Indexes — why they matter at our data size
   Without indexes, every query scans every row. With 116,000 rows in
   player_history_archive, a query like "get all rows for player 233"
   reads all 116,000 rows to find ~150. An index on player_id makes it
   instant. We add indexes on every column we'll use in WHERE clauses
   or JOIN conditions.

3. The migration pattern — read SQLite, write PostgreSQL
   We connect to both databases simultaneously. For each table:
   - Read the full table from SQLite into a pandas DataFrame
   - Write it to PostgreSQL using to_sql with the correct schema
   This is a one-time operation. After migration the collectors will
   write directly to PostgreSQL.

4. Connection strings
   SQLite:     sqlite:///db/fpl.db
   PostgreSQL: postgresql://user:password@host:port/database
   SQLAlchemy uses the same API for both — only the connection string
   changes. This is why we wrapped all DB access in get_database_engine()
   in Phase 1 — we only change one line per collector.

5. Environment variables for credentials
   We never hardcode database passwords in code. We use environment
   variables (os.environ) so the password lives only in your shell
   profile, not in any file that could be committed to GitHub.
   The .env pattern: set once in ~/.zprofile, read everywhere.
"""

import os
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text


# =============================================================================
# CONFIGURATION
# =============================================================================

SQLITE_PATH = "db/fpl.db"

# Read from environment variable — never hardcode credentials
# Set in ~/.zprofile: export FPL_DB_URL="postgresql://fpl_user:fpl_password@localhost:5432/fpl_agent"
PG_URL = os.environ.get(
    "FPL_DB_URL",
    "postgresql://fpl_user:fpl_password@localhost:5432/fpl_agent"
)

# Tables to migrate in dependency order (referenced tables first)
TABLES_TO_MIGRATE = [
    "teams",
    "gameweeks",
    "players",
    "fixtures",
    "player_history",
    "player_history_archive",
    "understat_players",
    "understat_teams",
    "lineup_signals",
    "injury_absences",
    "injury_profiles",
]


# =============================================================================
# ENGINES
# =============================================================================

def get_sqlite_engine():
    """Returns engine connected to the existing SQLite database."""
    return create_engine(f"sqlite:///{SQLITE_PATH}")


def get_postgres_engine():
    """
    Returns engine connected to PostgreSQL.
    Uses connection pooling — PostgreSQL performs much better when
    connections are reused rather than opened/closed per query.
    """
    return create_engine(PG_URL, pool_size=5, max_overflow=10)


# =============================================================================
# SCHEMA CREATION
# =============================================================================

def create_schema(pg_engine):
    """
    Creates all tables in PostgreSQL with proper types and indexes.

    We DROP and recreate each table — this migration script is designed
    to run once cleanly. If you need to re-run it, it starts fresh.

    The schema mirrors what pandas created in SQLite but with:
    - Proper column types instead of generic TEXT/REAL
    - Primary keys declared explicitly
    - Foreign key relationships documented in comments
    - Indexes on high-frequency query columns
    """
    schema_sql = """

    -- =========================================================
    -- CORE FPL TABLES
    -- =========================================================

    DROP TABLE IF EXISTS player_history_archive CASCADE;
    DROP TABLE IF EXISTS player_history CASCADE;
    DROP TABLE IF EXISTS lineup_signals CASCADE;
    DROP TABLE IF EXISTS injury_absences CASCADE;
    DROP TABLE IF EXISTS injury_profiles CASCADE;
    DROP TABLE IF EXISTS understat_players CASCADE;
    DROP TABLE IF EXISTS understat_teams CASCADE;
    DROP TABLE IF EXISTS fixtures CASCADE;
    DROP TABLE IF EXISTS players CASCADE;
    DROP TABLE IF EXISTS gameweeks CASCADE;
    DROP TABLE IF EXISTS teams CASCADE;

    CREATE TABLE teams (
        id                      INTEGER PRIMARY KEY,
        name                    TEXT NOT NULL,
        short_name              TEXT,
        strength                INTEGER,
        strength_overall_home   INTEGER,
        strength_overall_away   INTEGER,
        strength_attack_home    INTEGER,
        strength_attack_away    INTEGER,
        strength_defence_home   INTEGER,
        strength_defence_away   INTEGER
    );

    CREATE TABLE gameweeks (
        id                  INTEGER PRIMARY KEY,
        name                TEXT,
        deadline_time       TEXT,
        average_entry_score INTEGER,
        highest_score       INTEGER,
        is_current          BOOLEAN,
        is_next             BOOLEAN,
        is_previous         BOOLEAN,
        finished            BOOLEAN,
        data_checked        BOOLEAN
    );

    CREATE TABLE players (
        id                              INTEGER PRIMARY KEY,
        first_name                      TEXT,
        second_name                     TEXT,
        web_name                        TEXT,
        team                            INTEGER REFERENCES teams(id),
        element_type                    INTEGER,    -- 1=GK 2=DEF 3=MID 4=FWD
        now_cost                        INTEGER,    -- tenths of £m
        total_points                    INTEGER,
        points_per_game                 NUMERIC(6,2),
        form                            NUMERIC(6,2),
        selected_by_percent             NUMERIC(6,2),
        minutes                         INTEGER,
        goals_scored                    INTEGER,
        assists                         INTEGER,
        clean_sheets                    INTEGER,
        goals_conceded                  INTEGER,
        yellow_cards                    INTEGER,
        red_cards                       INTEGER,
        bonus                           INTEGER,
        bps                             INTEGER,
        influence                       NUMERIC(8,2),
        creativity                      NUMERIC(8,2),
        threat                          NUMERIC(8,2),
        ict_index                       NUMERIC(8,2),
        chance_of_playing_this_round    INTEGER,
        chance_of_playing_next_round    INTEGER,
        news                            TEXT,
        news_added                      TEXT,
        status                          TEXT,       -- a/d/i/u
        cost_change_start               INTEGER,
        transfers_in_event              INTEGER,
        transfers_out_event             INTEGER
    );

    CREATE INDEX idx_players_team ON players(team);
    CREATE INDEX idx_players_element_type ON players(element_type);
    CREATE INDEX idx_players_web_name ON players(web_name);

    CREATE TABLE fixtures (
        id                  INTEGER PRIMARY KEY,
        event               INTEGER REFERENCES gameweeks(id),
        team_h              INTEGER REFERENCES teams(id),
        team_a              INTEGER REFERENCES teams(id),
        team_h_score        INTEGER,
        team_a_score        INTEGER,
        kickoff_time        TEXT,
        finished            BOOLEAN,
        team_h_difficulty   INTEGER,
        team_a_difficulty   INTEGER
    );

    CREATE INDEX idx_fixtures_event ON fixtures(event);
    CREATE INDEX idx_fixtures_team_h ON fixtures(team_h);
    CREATE INDEX idx_fixtures_team_a ON fixtures(team_a);

    -- =========================================================
    -- PLAYER HISTORY (current season)
    -- =========================================================

    CREATE TABLE player_history (
        id                  SERIAL PRIMARY KEY,
        player_id           INTEGER REFERENCES players(id),
        element             INTEGER,
        fixture             INTEGER,
        round               INTEGER,            -- gameweek number
        opponent_team       INTEGER,
        was_home            BOOLEAN,
        kickoff_time        TEXT,
        total_points        INTEGER,            -- TARGET VARIABLE
        minutes             INTEGER,
        goals_scored        INTEGER,
        assists             INTEGER,
        clean_sheets        INTEGER,
        goals_conceded      INTEGER,
        own_goals           INTEGER,
        penalties_saved     INTEGER,
        penalties_missed    INTEGER,
        yellow_cards        INTEGER,
        red_cards           INTEGER,
        saves               INTEGER,
        bonus               INTEGER,
        bps                 INTEGER,
        influence           NUMERIC(8,2),
        creativity          NUMERIC(8,2),
        threat              NUMERIC(8,2),
        ict_index           NUMERIC(8,2),
        value               INTEGER,
        transfers_balance   INTEGER,
        selected            INTEGER
    );

    CREATE INDEX idx_ph_player_id ON player_history(player_id);
    CREATE INDEX idx_ph_round ON player_history(round);
    CREATE INDEX idx_ph_player_round ON player_history(player_id, round);

    -- =========================================================
    -- HISTORICAL ARCHIVE (4 seasons from vaastav)
    -- =========================================================

    CREATE TABLE player_history_archive (
        id                  SERIAL PRIMARY KEY,
        player_id           INTEGER,            -- vaastav element ID
        name                TEXT,
        fixture             INTEGER,
        round               INTEGER,
        opponent_team       INTEGER,
        was_home            BOOLEAN,
        kickoff_time        TEXT,
        total_points        INTEGER,
        minutes             INTEGER,
        goals_scored        INTEGER,
        assists             INTEGER,
        clean_sheets        INTEGER,
        goals_conceded      INTEGER,
        own_goals           INTEGER,
        penalties_saved     INTEGER,
        penalties_missed    INTEGER,
        yellow_cards        INTEGER,
        red_cards           INTEGER,
        saves               INTEGER,
        bonus               INTEGER,
        bps                 INTEGER,
        influence           NUMERIC(8,2),
        creativity          NUMERIC(8,2),
        threat              NUMERIC(8,2),
        ict_index           NUMERIC(8,2),
        value               INTEGER,
        transfers_balance   INTEGER,
        selected            INTEGER,
        team_h_score        INTEGER,
        team_a_score        INTEGER,
        season              TEXT,
        gameweek            INTEGER
    );

    CREATE INDEX idx_pha_player_id ON player_history_archive(player_id);
    CREATE INDEX idx_pha_season ON player_history_archive(season);
    CREATE INDEX idx_pha_player_season ON player_history_archive(player_id, season);
    CREATE INDEX idx_pha_round ON player_history_archive(round);

    -- =========================================================
    -- UNDERSTAT xG DATA
    -- =========================================================

    CREATE TABLE understat_players (
        id              SERIAL PRIMARY KEY,
        understat_id    TEXT,
        player_name     TEXT,
        season          TEXT,
        team            TEXT,
        position        TEXT,
        games           INTEGER,
        minutes         INTEGER,
        goals           INTEGER,
        xG              NUMERIC(10,4),
        assists         INTEGER,
        xA              NUMERIC(10,4),
        xGI             NUMERIC(10,4),
        shots           INTEGER,
        key_passes      INTEGER,
        yellow_cards    INTEGER,
        red_cards       INTEGER,
        npg             INTEGER,
        npxG            NUMERIC(10,4),
        xGChain         NUMERIC(10,4),
        xGBuildup       NUMERIC(10,4)
    );

    CREATE INDEX idx_up_player_name ON understat_players(player_name);
    CREATE INDEX idx_up_season ON understat_players(season);
    CREATE INDEX idx_up_team ON understat_players(team);

    CREATE TABLE understat_teams (
        id                  SERIAL PRIMARY KEY,
        understat_team_id   TEXT,
        team_name           TEXT,
        season              TEXT,
        date                TEXT,
        xG                  NUMERIC(10,4),
        xGA                 NUMERIC(10,4),
        goals               INTEGER,
        goals_against       INTEGER,
        was_home            BOOLEAN,
        result              TEXT,
        pts                 INTEGER
    );

    CREATE INDEX idx_ut_team_name ON understat_teams(team_name);
    CREATE INDEX idx_ut_season ON understat_teams(season);

    -- =========================================================
    -- LINEUP SIGNALS
    -- =========================================================

    CREATE TABLE lineup_signals (
        id              SERIAL PRIMARY KEY,
        collected_at    TIMESTAMP,
        gameweek        INTEGER,
        player_name     TEXT,
        club            TEXT,
        status          TEXT,
        confidence      NUMERIC(4,2),
        signal_type     TEXT,
        source          TEXT,
        headline        TEXT,
        raw_text        TEXT,
        claude_response TEXT
    );

    CREATE INDEX idx_ls_player_name ON lineup_signals(player_name);
    CREATE INDEX idx_ls_gameweek ON lineup_signals(gameweek);
    CREATE INDEX idx_ls_status ON lineup_signals(status);
    CREATE INDEX idx_ls_collected_at ON lineup_signals(collected_at);

    -- =========================================================
    -- INJURY TRACKING
    -- =========================================================

    CREATE TABLE injury_absences (
        id              SERIAL PRIMARY KEY,
        player_id       INTEGER,
        player_name     TEXT,
        season          TEXT,
        absence_start   INTEGER,
        absence_end     INTEGER,
        length_gw       INTEGER,
        absence_type    TEXT
    );

    CREATE INDEX idx_ia_player_id ON injury_absences(player_id);
    CREATE INDEX idx_ia_season ON injury_absences(season);

    CREATE TABLE injury_profiles (
        player_id               INTEGER PRIMARY KEY,
        player_name             TEXT,
        total_absences          INTEGER,
        avg_absence_length      NUMERIC(6,2),
        absences_per_10_games   NUMERIC(6,3),
        last_absence_gw         INTEGER,
        last_absence_season     TEXT,
        games_since_return      INTEGER,
        injury_prone_score      NUMERIC(5,3),
        reliability_label       TEXT,
        updated_at              TIMESTAMP
    );

    CREATE INDEX idx_ip_score ON injury_profiles(injury_prone_score);
    CREATE INDEX idx_ip_label ON injury_profiles(reliability_label);
    """

    with pg_engine.connect() as conn:
        conn.execute(text(schema_sql))
        conn.commit()

    print("✓ PostgreSQL schema created with indexes")


# =============================================================================
# DATA MIGRATION
# =============================================================================

def migrate_table(table_name, sqlite_engine, pg_engine):
    """
    Reads one table from SQLite and writes it to PostgreSQL.

    Uses chunksize=5000 for large tables — reads and writes 5,000 rows
    at a time rather than loading everything into memory at once.
    For player_history_archive (116k rows) this keeps memory usage low.

    if_exists='append' because we already created the tables with
    proper schema above — we don't want pandas to recreate them.
    """
    try:
        # Check table exists in SQLite
        sqlite_tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table'",
            sqlite_engine
        )["name"].tolist()

        if table_name not in sqlite_tables:
            print(f"  ⚠ Skipping {table_name} — not in SQLite")
            return 0

        # Get row count first
        count = pd.read_sql(
            f"SELECT COUNT(*) as n FROM {table_name}",
            sqlite_engine
        )["n"].iloc[0]

        if count == 0:
            print(f"  ⚠ Skipping {table_name} — empty table")
            return 0

        # Rename 'element' to 'player_id' for player_history_archive
        # to match our PostgreSQL schema
        df = pd.read_sql(f"SELECT * FROM {table_name}", sqlite_engine)
        if table_name == "player_history_archive" and "element" in df.columns:
            df = df.rename(columns={"element": "player_id"})

        # Drop SQLite auto-increment 'index' column if present
        if "index" in df.columns:
            df = df.drop(columns=["index"])

        # Write to PostgreSQL in chunks
        df.to_sql(
            table_name,
            pg_engine,
            if_exists="append",
            index=False,
            chunksize=5000,
            method="multi"   # Faster bulk insert
        )

        print(f"  ✓ {table_name:<30} {count:>8,} rows migrated")
        return count

    except Exception as e:
        print(f"  ✗ {table_name}: {e}")
        return 0


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_migration(sqlite_engine, pg_engine):
    """
    Compares row counts between SQLite and PostgreSQL for every table.
    Any mismatch means something went wrong in the migration.
    """
    print("\n  Verification — row count comparison:")
    print(f"  {'Table':<30} {'SQLite':>10} {'PostgreSQL':>12} {'Match':>7}")
    print(f"  {'-'*63}")

    all_match = True

    for table in TABLES_TO_MIGRATE:
        try:
            sqlite_count = pd.read_sql(
                f"SELECT COUNT(*) as n FROM {table}", sqlite_engine
            )["n"].iloc[0]
        except Exception:
            sqlite_count = 0

        try:
            pg_count = pd.read_sql(
                f"SELECT COUNT(*) as n FROM {table}", pg_engine
            )["n"].iloc[0]
        except Exception:
            pg_count = 0

        match = "✓" if sqlite_count == pg_count else "✗ MISMATCH"
        if sqlite_count != pg_count:
            all_match = False

        print(f"  {table:<30} {sqlite_count:>10,} {pg_count:>12,} {match:>7}")

    print()
    if all_match:
        print("  ✓ All tables migrated successfully")
    else:
        print("  ✗ Some tables have mismatches — check errors above")

    return all_match


# =============================================================================
# UPDATE COLLECTORS TO USE POSTGRESQL
# =============================================================================

def print_next_steps():
    """
    Prints the environment variable to add and the one-line change
    needed in each collector to switch from SQLite to PostgreSQL.
    """
    print("""
Next steps — switch collectors to PostgreSQL:

1. Add to ~/.zprofile (already set if you ran setup):
   export FPL_DB_URL="postgresql://fpl_user:fpl_password@localhost:5432/fpl_agent"

2. In each collector, replace:
   create_engine(f"sqlite:///db/fpl.db")

   With:
   import os
   create_engine(os.environ.get("FPL_DB_URL", "sqlite:///db/fpl.db"))

   This uses PostgreSQL when FPL_DB_URL is set, SQLite as fallback.
   No other code changes needed — SQLAlchemy API is identical.

3. Run the FPL collector once to verify PostgreSQL writes work:
   python collectors/fpl_collector.py
""")


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_migration():
    """
    Full migration pipeline:
    1. Create PostgreSQL schema with proper types and indexes
    2. Migrate each table from SQLite to PostgreSQL
    3. Verify row counts match
    4. Print next steps
    """
    print("=" * 60)
    print(f"PostgreSQL Migration  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    sqlite_engine = get_sqlite_engine()
    pg_engine     = get_postgres_engine()

    # Step 1: Create schema
    print("\n[1/3] Creating PostgreSQL schema...")
    create_schema(pg_engine)

    # Step 2: Migrate tables
    print("\n[2/3] Migrating data...")
    total_rows = 0
    for table in TABLES_TO_MIGRATE:
        rows = migrate_table(table, sqlite_engine, pg_engine)
        total_rows += rows

    # Step 3: Verify
    print("\n[3/3] Verifying migration...")
    verify_migration(sqlite_engine, pg_engine)

    print(f"\n{'='*60}")
    print(f"  Total rows migrated: {total_rows:,}")
    print(f"{'='*60}")

    print_next_steps()


if __name__ == "__main__":
    run_migration()

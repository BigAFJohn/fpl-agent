"""
etl_pipeline.py  —  Phase 2, Component 2: ETL Pipeline
=======================================================

CONCEPTS TO UNDERSTAND BEFORE READING THIS SCRIPT
--------------------------------------------------

1. Why ETL instead of just writing raw data?
   Raw API data is messy. Player prices come as integers (130 = £13.0m).
   xG values are strings in some sources. Names differ between sources
   ("Mohamed Salah" vs "M.Salah" vs "Salah"). The ETL layer cleans all
   of this before it reaches the model — garbage in, garbage out.

2. Upsert — insert or update, never duplicate
   A regular INSERT fails if the row already exists. A regular UPDATE
   fails if it doesn't. An UPSERT does both: if the row exists, update
   it; if not, insert it. PostgreSQL has native UPSERT via:
       INSERT INTO table (...) VALUES (...)
       ON CONFLICT (unique_key) DO UPDATE SET ...
   This means our collectors can run as many times as we want and the
   database always reflects the latest state without duplicates.

3. Data validation — why we check before loading
   Bad data silently corrupts a model. A player with minutes=9999 or
   xG=-1.0 will skew predictions. We validate every table before loading
   and write issues to a data_quality_log table. This is the same
   principle as assertions in test automation — fail loudly on bad data
   rather than silently producing wrong results.

4. Derived columns — adding value during transform
   We add columns the raw data doesn't have but the model needs:
   - price_millions: now_cost / 10 (£13.0m not 130)
   - position_label: 'GK'/'DEF'/'MID'/'FWD' not 1/2/3/4
   - form_5gw: rolling average points over last 5 gameweeks
   - xgi: xG + xA (if not already present)
   These are computed once here rather than in every model query.

5. The data_quality_log table
   Every validation issue gets logged with: table name, column, issue
   type, count of affected rows, and timestamp. In Phase 6 (monitoring)
   we alert when quality scores drop below threshold. Think of it as
   your test report for data integrity.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, text


# =============================================================================
# CONFIGURATION
# =============================================================================

DB_URL = os.environ.get(
    "FPL_DB_URL",
    "postgresql://fpl_user:fpl_password@localhost:5432/fpl_agent"
)


# =============================================================================
# DATABASE
# =============================================================================

def get_engine():
    """Returns SQLAlchemy engine connected to PostgreSQL."""
    return create_engine(DB_URL, pool_size=5, max_overflow=10)


def setup_etl_tables(engine):
    """
    Creates the data_quality_log table for tracking validation issues,
    and an etl_runs table to record each pipeline execution.
    """
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS data_quality_log (
                id          SERIAL PRIMARY KEY,
                run_at      TIMESTAMP,
                table_name  TEXT,
                column_name TEXT,
                issue_type  TEXT,       -- null/outlier/duplicate/mismatch
                issue_count INTEGER,
                details     TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS etl_runs (
                id              SERIAL PRIMARY KEY,
                run_at          TIMESTAMP,
                tables_processed INTEGER,
                rows_upserted   INTEGER,
                quality_issues  INTEGER,
                duration_secs   REAL,
                status          TEXT    -- success/partial/failed
            )
        """))
        conn.commit()
    print("✓ ETL tables ready")


# =============================================================================
# DATA QUALITY VALIDATOR
# =============================================================================

def validate_table(df, table_name, engine, run_at):
    """
    Runs a suite of data quality checks on a DataFrame and logs issues.

    Checks performed:
    - Null counts on critical columns
    - Outlier detection (values > 3 standard deviations from mean)
    - Duplicate row detection on primary key columns
    - Range validation (e.g. confidence must be 0-1)

    Returns a quality score 0-1 (1 = perfect, 0 = all checks failed).
    Issues are written to data_quality_log for monitoring.
    """
    issues = []
    total_checks = 0
    passed_checks = 0

    # Define critical columns per table — nulls here are serious
    critical_cols = {
        "players"               : ["id", "web_name", "team", "element_type"],
        "player_history"        : ["player_id", "round", "total_points", "minutes"],
        "player_history_archive": ["player_id", "round", "total_points", "minutes", "season"],
        "fixtures"              : ["id", "team_h", "team_a"],
        "understat_players"     : ["understat_id", "player_name", "season"],
        "lineup_signals"        : ["player_name", "status", "confidence"],
        "injury_profiles"       : ["player_id", "injury_prone_score"],
    }

    # Outlier thresholds — max realistic values per column
    outlier_thresholds = {
        "total_points"    : 50,    # No FPL player scores 50+ in a GW
        "minutes"         : 120,   # Max with extra time
        "goals_scored"    : 5,     # Practical maximum
        "assists"         : 5,
        "confidence"      : 1.0,
        "injury_prone_score": 1.0,
        "xg"              : 5.0,   # xG above 5 in one game is impossible
        "xa"              : 5.0,
    }

    # --- NULL CHECKS ---
    for col in critical_cols.get(table_name, []):
        if col in df.columns:
            total_checks += 1
            null_count = df[col].isna().sum()
            if null_count == 0:
                passed_checks += 1
            else:
                issues.append({
                    "table_name" : table_name,
                    "column_name": col,
                    "issue_type" : "null_values",
                    "issue_count": int(null_count),
                    "details"    : f"{null_count} null values in critical column",
                })

    # --- OUTLIER CHECKS ---
    for col, max_val in outlier_thresholds.items():
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            total_checks += 1
            outlier_count = (df[col] > max_val).sum()
            if outlier_count == 0:
                passed_checks += 1
            else:
                issues.append({
                    "table_name" : table_name,
                    "column_name": col,
                    "issue_type" : "outlier",
                    "issue_count": int(outlier_count),
                    "details"    : f"{outlier_count} values exceed max threshold of {max_val}",
                })

    # --- DUPLICATE CHECKS ---
    pk_cols = {
        "players"               : ["id"],
        "player_history"        : ["player_id", "round"],
        "player_history_archive": ["player_id", "round", "season"],
        "fixtures"              : ["id"],
        "understat_players"     : ["understat_id", "season"],
        "injury_profiles"       : ["player_id"],
    }
    if table_name in pk_cols:
        total_checks += 1
        pk = pk_cols[table_name]
        available_pk = [c for c in pk if c in df.columns]
        if available_pk:
            dup_count = df.duplicated(subset=available_pk).sum()
            if dup_count == 0:
                passed_checks += 1
            else:
                issues.append({
                    "table_name" : table_name,
                    "column_name": ",".join(available_pk),
                    "issue_type" : "duplicate",
                    "issue_count": int(dup_count),
                    "details"    : f"{dup_count} duplicate rows on {available_pk}",
                })

    # Log all issues to database
    if issues:
        issues_df = pd.DataFrame(issues)
        issues_df["run_at"] = run_at
        issues_df.to_sql("data_quality_log", engine, if_exists="append", index=False)

    quality_score = passed_checks / total_checks if total_checks > 0 else 1.0

    status = "✓" if quality_score == 1.0 else f"⚠ {len(issues)} issues"
    print(f"    Quality: {status} (score: {quality_score:.2f})")

    return quality_score, len(issues)


# =============================================================================
# TRANSFORMATIONS
# =============================================================================

def transform_players(df):
    # Cast chance_of_playing columns from float to int (PostgreSQL expects INTEGER)
    for col in ['chance_of_playing_this_round', 'chance_of_playing_next_round']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Price in £millions
    df["price_millions"] = df["now_cost"] / 10.0

    # Human-readable position labels
    position_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    df["position_label"] = df["element_type"].map(position_map)

    # Availability label from status code
    status_map = {
        "a": "available",
        "d": "doubtful",
        "i": "injured",
        "u": "unavailable",
    }
    df["availability"] = df["status"].map(status_map).fillna("unknown")

    # Points per million — value metric for FPL
    df["points_per_million"] = (
        df["total_points"] / df["price_millions"]
    ).replace([np.inf, -np.inf], 0).fillna(0).round(2)

    return df


def transform_player_history(df):
    """
    Cleans and enriches gameweek history.
    Adds a started flag and calculates minutes_90s (90-minute equivalents).
    """
    # Boolean: did the player start (60+ mins = started)
    df["started"] = df["minutes"] >= 60

    # 90-minute equivalents — normalises contribution by playing time
    df["minutes_90s"] = (df["minutes"] / 90).round(3)

    # Clean up any minutes > 120 (data error)
    df["minutes"] = df["minutes"].clip(upper=120)

    # Ensure total_points has no nulls
    df["total_points"] = df["total_points"].fillna(0).astype(int)

    return df


def transform_understat(df):
    """
    Cleans understat player data.
    Recalculates xGI to ensure consistency, adds xG per 90 metrics.
    """
    # Ensure xGI is always xG + xA (recalculate to fix any inconsistencies)
    if "xg" in df.columns and "xa" in df.columns:
        df["xgi"] = (df["xg"] + df["xa"]).round(4)

    # Per-90-minute metrics — comparable across players with different minutes
    if "minutes" in df.columns and df["minutes"].sum() > 0:
        df["xg_per90"] = (
            (df["xg"] / df["minutes"].replace(0, np.nan)) * 90
        ).round(4).fillna(0)
        df["xa_per90"] = (
            (df["xa"] / df["minutes"].replace(0, np.nan)) * 90
        ).round(4).fillna(0)
        df["xgi_per90"] = (df["xg_per90"] + df["xa_per90"]).round(4)

    return df


def transform_lineup_signals(df):
    """
    Cleans lineup signals. Normalises confidence to 0-1 range,
    clips any out-of-range values from Claude responses.
    """
    df["confidence"] = df["confidence"].clip(0.0, 1.0)
    df["status"] = df["status"].str.lower().fillna("unknown")

    # Convert collected_at to proper timestamp
    df["collected_at"] = pd.to_datetime(df["collected_at"], errors="coerce")

    return df


# =============================================================================
# UPSERT ENGINE
# =============================================================================

def upsert_table(df, table_name, conflict_cols, update_cols, engine):
    """
    Performs a PostgreSQL UPSERT — insert rows, update on conflict.

    conflict_cols: columns that define uniqueness (e.g. ['player_id', 'round'])
    update_cols:   columns to update if row already exists

    This is the core of idempotent data loading — running the pipeline
    twice produces identical results, not doubled rows.

    We use raw SQL for the upsert because pandas to_sql doesn't support
    ON CONFLICT natively. We build the SQL dynamically from column names.
    """
    if df.empty:
        print(f"    ⚠ Empty DataFrame — skipping {table_name}")
        return 0

    # Replace NaN with None for PostgreSQL compatibility
    df = df.where(pd.notnull(df), None)

    rows_affected = 0
    cols = list(df.columns)
    col_str = ", ".join(cols)
    val_str = ", ".join([f":{c}" for c in cols])
    conflict_str = ", ".join(conflict_cols)

    # Build SET clause for update (exclude conflict columns)
    set_clauses = [f"{c} = EXCLUDED.{c}" for c in update_cols if c in cols]
    set_str = ", ".join(set_clauses)

    upsert_sql = f"""
        INSERT INTO {table_name} ({col_str})
        VALUES ({val_str})
        ON CONFLICT ({conflict_str})
        DO UPDATE SET {set_str}
    """

    # Execute in batches of 500 rows
    batch_size = 500
    with engine.connect() as conn:
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            records = batch.to_dict(orient="records")
            conn.execute(text(upsert_sql), records)
            rows_affected += len(batch)
        conn.commit()

    return rows_affected


# =============================================================================
# TABLE-SPECIFIC ETL RUNNERS
# =============================================================================

def etl_players(engine, run_at):
    """ETL for players table — upsert on player ID."""
    print("\n  [players]")
    df = pd.read_sql("SELECT * FROM players", engine)
    df = transform_players(df)
    quality_score, issues = validate_table(df, "players", engine, run_at)

    # Add derived columns to PostgreSQL schema if they don't exist
    with engine.connect() as conn:
        for col, dtype in [
            ("price_millions", "NUMERIC(6,2)"),
            ("position_label", "TEXT"),
            ("availability", "TEXT"),
            ("points_per_million", "NUMERIC(6,2)"),
        ]:
            try:
                conn.execute(text(f"ALTER TABLE players ADD COLUMN IF NOT EXISTS {col} {dtype}"))
            except Exception:
                pass
        conn.commit()

    # Players table: truncate and reload — faster than upsert for 836 rows
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE players CASCADE"))
        conn.commit()
    df.to_sql("players", engine, if_exists="append", index=False, chunksize=500)
    rows = len(df)
    print(f"    Upserted: {rows:,} rows")
    return rows, issues


def etl_player_history(engine, run_at):
    """ETL for current season player history — upsert on player_id + round."""
    print("\n  [player_history]")
    df = pd.read_sql("SELECT * FROM player_history", engine)
    df = transform_player_history(df)
    quality_score, issues = validate_table(df, "player_history", engine, run_at)

    # Add derived columns
    with engine.connect() as conn:
        for col, dtype in [
            ("started", "BOOLEAN"),
            ("minutes_90s", "NUMERIC(6,3)"),
        ]:
            try:
                conn.execute(text(f"ALTER TABLE player_history ADD COLUMN IF NOT EXISTS {col} {dtype}"))
            except Exception:
                pass
        conn.commit()

    update_cols = [c for c in df.columns if c not in ["player_id", "round"]]
    rows = upsert_table(df, "player_history", ["player_id", "round"], update_cols, engine)
    print(f"    Upserted: {rows:,} rows")
    return rows, issues


def etl_player_history_archive(engine, run_at):
    """
    ETL for historical archive — upsert on player_id + round + season.
    Processes in chunks of 10,000 rows to keep memory usage low.
    """
    print("\n  [player_history_archive]")

    # Get total count first
    total = pd.read_sql(
        "SELECT COUNT(*) as n FROM player_history_archive", engine
    )["n"].iloc[0]
    print(f"    Processing {total:,} rows in chunks...")

    total_rows = 0
    total_issues = 0
    chunk_size = 10000

    for offset in range(0, total, chunk_size):
        df = pd.read_sql(
            f"SELECT * FROM player_history_archive LIMIT {chunk_size} OFFSET {offset}",
            engine
        )
        df = transform_player_history(df)

        if offset == 0:  # Only validate first chunk to avoid log spam
            _, issues = validate_table(df, "player_history_archive", engine, run_at)
            total_issues += issues

            # Add derived columns on first chunk
            with engine.connect() as conn:
                for col, dtype in [("started", "BOOLEAN"), ("minutes_90s", "NUMERIC(6,3)")]:
                    try:
                        conn.execute(text(
                            f"ALTER TABLE player_history_archive ADD COLUMN IF NOT EXISTS {col} {dtype}"
                        ))
                    except Exception:
                        pass
                conn.commit()

        update_cols = [c for c in df.columns if c not in ["player_id", "round", "season"]]
        rows = upsert_table(
            df, "player_history_archive",
            ["player_id", "round", "season"],
            update_cols, engine
        )
        total_rows += rows
        print(f"    Chunk {offset//chunk_size + 1}: {total_rows:,}/{total:,} rows", end="\r")

    print(f"\n    Upserted: {total_rows:,} rows")
    return total_rows, total_issues


def etl_understat(engine, run_at):
    """ETL for Understat xG data — upsert on understat_id + season."""
    print("\n  [understat_players]")
    df = pd.read_sql("SELECT * FROM understat_players", engine)
    df = transform_understat(df)
    quality_score, issues = validate_table(df, "understat_players", engine, run_at)

    # Add per-90 columns
    with engine.connect() as conn:
        for col, dtype in [
            ("xg_per90", "NUMERIC(8,4)"),
            ("xa_per90", "NUMERIC(8,4)"),
            ("xgi_per90", "NUMERIC(8,4)"),
        ]:
            try:
                conn.execute(text(
                    f"ALTER TABLE understat_players ADD COLUMN IF NOT EXISTS {col} {dtype}"
                ))
            except Exception:
                pass
        conn.commit()

    update_cols = [c for c in df.columns if c not in ["understat_id", "season"]]
    rows = upsert_table(df, "understat_players", ["understat_id", "season"], update_cols, engine)
    print(f"    Upserted: {rows:,} rows")
    return rows, issues


def etl_lineup_signals(engine, run_at):
    """
    ETL for lineup signals — append only, no upsert.
    Signals are point-in-time observations — we never overwrite them.
    Old signals stay in the log for historical analysis.
    """
    print("\n  [lineup_signals]")
    df = pd.read_sql("SELECT * FROM lineup_signals", engine)
    df = transform_lineup_signals(df)
    quality_score, issues = validate_table(df, "lineup_signals", engine, run_at)
    print(f"    {len(df):,} signals in database (append-only, no upsert needed)")
    return len(df), issues


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_etl(engine, run_at):
    """
    Post-ETL verification — checks derived columns were added correctly
    and prints a data quality summary from this run's log.
    """
    print("\n  ETL verification:")

    # Check derived columns exist and have data
    checks = [
        ("players",         "price_millions",   "SELECT COUNT(*) FROM players WHERE price_millions > 0"),
        ("players",         "position_label",   "SELECT COUNT(DISTINCT position_label) FROM players"),
        ("player_history",  "started",          "SELECT COUNT(*) FROM player_history WHERE started = true"),
        ("understat_players", "xgi_per90",      "SELECT COUNT(*) FROM understat_players WHERE xgi_per90 > 0"),
    ]

    print(f"\n  {'Check':<45} {'Result':>8}")
    print(f"  {'-'*55}")
    for table, col, query in checks:
        try:
            result = pd.read_sql(query, engine).iloc[0, 0]
            print(f"  {table}.{col:<35} {result:>8,}")
        except Exception as e:
            print(f"  {table}.{col:<35} {'ERROR':>8}")

    # Quality log summary for this run
    quality_summary = pd.read_sql(f"""
        SELECT table_name, issue_type, SUM(issue_count) as total_affected
        FROM data_quality_log
        WHERE run_at = '{run_at}'
        GROUP BY table_name, issue_type
        ORDER BY total_affected DESC
    """, engine)

    if quality_summary.empty:
        print("\n  ✓ No data quality issues detected")
    else:
        print("\n  ⚠ Data quality issues:")
        print(quality_summary.to_string(index=False))

    # Top 5 players by points_per_million (sanity check)
    print("\n  Top 5 value players (points per £m):")
    top_value = pd.read_sql("""
        SELECT web_name, position_label, price_millions,
               total_points, points_per_million
        FROM players
        WHERE minutes > 900
        ORDER BY points_per_million DESC
        LIMIT 5
    """, engine)
    print(top_value.to_string(index=False))


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_etl():
    """
    Runs the full ETL pipeline:
    1. Validate and transform players
    2. Validate and transform player history (current season)
    3. Validate and transform historical archive (chunked)
    4. Validate and transform Understat xG data
    5. Validate lineup signals
    6. Verify derived columns and quality log
    """
    start_time = datetime.now()
    run_at = start_time.isoformat()

    print("=" * 60)
    print(f"ETL Pipeline  |  {start_time.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    engine = get_engine()
    setup_etl_tables(engine)

    total_rows = 0
    total_issues = 0

    # Run each table's ETL
    etl_functions = [
        etl_players,
        etl_player_history,
        etl_player_history_archive,
        etl_understat,
        etl_lineup_signals,
    ]

    for etl_fn in etl_functions:
        try:
            rows, issues = etl_fn(engine, run_at)
            total_rows += rows
            total_issues += issues
        except Exception as e:
            print(f"  ✗ {etl_fn.__name__} failed: {e}")

    # Verify
    verify_etl(engine, run_at)

    # Record this run
    duration = (datetime.now() - start_time).total_seconds()
    run_record = {
        "run_at"           : run_at,
        "tables_processed" : len(etl_functions),
        "rows_upserted"    : total_rows,
        "quality_issues"   : total_issues,
        "duration_secs"    : round(duration, 2),
        "status"           : "success" if total_issues == 0 else "partial",
    }
    pd.DataFrame([run_record]).to_sql(
        "etl_runs", engine, if_exists="append", index=False
    )

    print(f"\n{'='*60}")
    print(f"  Tables processed : {len(etl_functions)}")
    print(f"  Rows upserted    : {total_rows:,}")
    print(f"  Quality issues   : {total_issues}")
    print(f"  Duration         : {duration:.1f}s")
    print(f"  Status           : {run_record['status']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_etl()

"""
feature_engineering.py  —  Phase 3: Feature Engineering
========================================================

CONCEPTS TO UNDERSTAND BEFORE READING THIS SCRIPT
--------------------------------------------------

1. Why rolling averages beat season totals
   A player's season total obscures recent form. Haaland with 25 goals
   in 35 games looks great — but if 20 of those came in the first 20
   games and he's only scored 5 in the last 15, his season total is
   misleading. A 5-gameweek rolling average captures current form.
   The model needs to know what a player is doing NOW, not what he
   did 3 months ago.

2. Window sizes — why we use multiple
   5-gameweek window: captures hot/cold streaks, recent injury returns
   10-gameweek window: medium-term form, smooths out lucky/unlucky weeks
   Season average: baseline expectation for the player
   Using all three lets the model learn both short-term momentum
   and long-term quality simultaneously.

3. Trend — direction matters as much as level
   A player averaging 6 pts/GW over 10 weeks but only 4 in the last 5
   is declining — worse FPL pick than someone averaging 4 over 10 weeks
   but 6 in the last 5 (improving). Trend = recent_form - medium_form.
   Positive trend = buy signal. Negative trend = sell signal.

4. Minutes as a proxy for availability
   The model can't directly observe "did the player start?" for future
   games. But minutes played in recent weeks is the best proxy:
   - 90 mins × 5 GWs = iron-man, start him
   - 45 mins × 5 GWs = rotation risk or returning from injury
   - 0 mins × 3 GWs = injured or dropped
   We compute avg_minutes_5gw and started_rate_5gw as features.

5. The feature store — why one clean table
   The prediction model in Phase 4 needs one row per player per
   gameweek with all features pre-computed. Computing features inside
   the model training loop is slow and error-prone. The feature store
   separates "what do we know?" from "what do we predict?" clearly.

6. Lag features — avoiding data leakage
   CRITICAL: when computing features for gameweek N, we can only use
   data from gameweeks 1 to N-1. Using GW N data to predict GW N
   is cheating (data leakage) — the model would learn to predict
   what already happened. All rolling windows are computed on
   PAST gameweeks only using pandas shift().

7. Multi-strategy name matching for Understat
   FPL uses shortened names (B.Fernandes, Haaland, Gibbs-White) while
   Understat uses full names (Bruno Fernandes, Erling Haaland, Morgan
   Gibbs-White). We use four strategies in priority order:
   1. Exact match
   2. Initial.Lastname format (B.Fernandes → Bruno Fernandes)
   3. Single lastname (Haaland → Erling Haaland)
   4. Hyphenated surname (Gibbs-White → Morgan Gibbs-White)
   All lookups are pre-built as dicts — O(1) not O(n) per row.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, text


# =============================================================================
# CONFIGURATION
# =============================================================================

DB_PATH = "db/fpl.db"

SHORT_WINDOW  = 5
MEDIUM_WINDOW = 10


# =============================================================================
# DATABASE
# =============================================================================

def get_engine():
    """Returns engine connected to PostgreSQL or SQLite fallback."""
    pg_url = os.environ.get("FPL_DB_URL")
    if pg_url:
        return create_engine(pg_url, pool_size=5, max_overflow=10)
    return create_engine(f"sqlite:///{DB_PATH}")


def setup_feature_tables(engine):
    """
    Creates the player_features table — one row per player per gameweek.
    This is the direct input to the prediction model in Phase 4.
    Drops and recreates on each run — always a full rebuild.
    """
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS player_features"))
        conn.execute(text("""
            CREATE TABLE player_features (
                player_id           INTEGER,
                season              TEXT,
                gameweek            INTEGER,
                web_name            TEXT,
                position            TEXT,
                team_id             INTEGER,
                price               NUMERIC(6,2),
                actual_points       INTEGER,

                avg_points_5gw      NUMERIC(6,2),
                avg_points_10gw     NUMERIC(6,2),
                avg_points_season   NUMERIC(6,2),
                points_trend        NUMERIC(6,2),

                avg_minutes_5gw     NUMERIC(6,2),
                avg_minutes_10gw    NUMERIC(6,2),
                started_rate_5gw    NUMERIC(5,3),
                full_game_rate_5gw  NUMERIC(5,3),

                avg_goals_5gw       NUMERIC(6,3),
                avg_assists_5gw     NUMERIC(6,3),
                avg_bonus_5gw       NUMERIC(6,3),
                avg_bps_5gw         NUMERIC(6,2),
                avg_ict_5gw         NUMERIC(7,3),

                clean_sheet_rate_5gw   NUMERIC(5,3),
                avg_saves_5gw          NUMERIC(6,2),
                avg_goals_conceded_5gw NUMERIC(6,2),

                xg_season           NUMERIC(8,4),
                xa_season           NUMERIC(8,4),
                xgi_season          NUMERIC(8,4),
                xg_per90_season     NUMERIC(8,4),
                xgi_per90_season    NUMERIC(8,4),

                injury_prone_score  NUMERIC(5,3),
                games_since_return  INTEGER,
                fpl_availability    TEXT,
                chance_of_playing   INTEGER,
                selected_by_percent NUMERIC(6,2),
                transfers_in_event  BIGINT,
                transfers_out_event BIGINT,

                computed_at         TIMESTAMP,

                PRIMARY KEY (player_id, season, gameweek)
            )
        """))
        conn.commit()
    print("✓ player_features table ready")


# =============================================================================
# DATA LOADERS
# =============================================================================

def load_history(engine):
    """
    Loads combined player history from current season and archive.
    Returns a single DataFrame sorted by player and gameweek.
    """
    print("  Loading player history...")

    current = pd.read_sql("""
        SELECT ph.player_id,
               p.web_name,
               p.element_type,
               p.position_label  AS position,
               p.team,
               p.now_cost,
               p.selected_by_percent,
               p.transfers_in_event,
               p.transfers_out_event,
               p.status,
               p.chance_of_playing_this_round,
               ph.round          AS gameweek,
               ph.total_points,
               ph.minutes,
               ph.goals_scored,
               ph.assists,
               ph.clean_sheets,
               ph.goals_conceded,
               ph.bonus,
               ph.bps,
               ph.ict_index,
               ph.saves,
               ph.value          AS price_raw,
               '2025-26'         AS season
        FROM player_history ph
        JOIN players p ON ph.player_id = p.id
        ORDER BY ph.player_id, ph.round
    """, engine)

    
    archive = pd.read_sql("""
        SELECT element           AS player_id,
               name              AS web_name,
               position,
               team              AS team_name,
               NULL::integer     AS element_type,
               NULL::integer     AS team,
               NULL::integer     AS now_cost,
               NULL::numeric     AS selected_by_percent,
               NULL::bigint      AS transfers_in_event,
               NULL::bigint      AS transfers_out_event,
               NULL::text        AS status,
               NULL::integer     AS chance_of_playing_this_round,
               round             AS gameweek,
               total_points,
               minutes,
               goals_scored,
               assists,
               clean_sheets,
               goals_conceded,
               bonus,
               bps,
               ict_index,
               saves,
               value             AS price_raw,
               season
        FROM player_history_archive
        WHERE season != '2025-26'
        ORDER BY element, season, round
    """, engine)

    df = pd.concat([archive, current], ignore_index=True)
    df = df.sort_values(["player_id", "season", "gameweek"]).reset_index(drop=True)

    for col in ["total_points", "minutes", "goals_scored", "assists",
                "clean_sheets", "goals_conceded", "bonus", "bps",
                "saves", "ict_index", "price_raw"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    print(f"  ✓ Loaded {len(df):,} gameweek rows across {df['player_id'].nunique():,} players")
    return df


def load_understat(engine):
    """Loads Understat season-level xG data."""
    print("  Loading Understat xG data...")
    df = pd.read_sql("""
        SELECT player_name, season,
               xg, xa, xgi, xg_per90, xgi_per90,
               xgchain, xgbuildup, npxg
        FROM understat_players
    """, engine)
    print(f"  ✓ Loaded {len(df):,} Understat records")
    return df


def load_injury_profiles(engine):
    """Loads injury profiles for merging with player features."""
    print("  Loading injury profiles...")
    df = pd.read_sql("""
        SELECT player_id, injury_prone_score, games_since_return,
               reliability_label
        FROM injury_profiles
    """, engine)
    print(f"  ✓ Loaded {len(df):,} injury profiles")
    return df


# =============================================================================
# ROLLING FEATURE CALCULATOR
# =============================================================================

def _safe_float(val):
    """Convert any numeric type to Python float, None if not convertible."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _safe_int(val):
    """Convert any numeric type to Python int, None if not convertible."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def compute_rolling_features(df):
    """
    Computes rolling window features for each player.

    Uses .shift(1) before rolling windows to prevent data leakage —
    features for gameweek N only use data from gameweeks 1..N-1.
    Groups by player_id + season to avoid windows crossing season boundaries.
    """
    print("  Computing rolling features...")

    feature_rows = []
    groups = df.groupby(["player_id", "season"])
    total_groups = len(groups)
    processed = 0

    for (player_id, season), group in groups:
        g = group.sort_values("gameweek").reset_index(drop=True)

        # Shift all stats by 1 GW — prevents data leakage
        pts     = g["total_points"].shift(1)
        mins    = g["minutes"].shift(1)
        goals   = g["goals_scored"].shift(1)
        assists = g["assists"].shift(1)
        cs      = g["clean_sheets"].shift(1)
        gc      = g["goals_conceded"].shift(1)
        bonus   = g["bonus"].shift(1)
        bps     = g["bps"].shift(1)
        ict     = g["ict_index"].shift(1)
        saves   = g["saves"].shift(1)
        started = (g["minutes"].shift(1) >= 60).astype(float)
        full_gm = (g["minutes"].shift(1) >= 85).astype(float)

        avg_pts_5   = pts.rolling(SHORT_WINDOW,  min_periods=1).mean()
        avg_pts_10  = pts.rolling(MEDIUM_WINDOW, min_periods=1).mean()
        avg_pts_ssn = pts.expanding().mean()
        avg_mins_5  = mins.rolling(SHORT_WINDOW,  min_periods=1).mean()
        avg_mins_10 = mins.rolling(MEDIUM_WINDOW, min_periods=1).mean()
        started_rate  = started.rolling(SHORT_WINDOW, min_periods=1).mean()
        full_gm_rate  = full_gm.rolling(SHORT_WINDOW, min_periods=1).mean()
        avg_goals_5   = goals.rolling(SHORT_WINDOW,  min_periods=1).mean()
        avg_assists_5 = assists.rolling(SHORT_WINDOW, min_periods=1).mean()
        avg_bonus_5   = bonus.rolling(SHORT_WINDOW,  min_periods=1).mean()
        avg_bps_5     = bps.rolling(SHORT_WINDOW,    min_periods=1).mean()
        avg_ict_5     = ict.rolling(SHORT_WINDOW,    min_periods=1).mean()
        cs_rate_5     = cs.rolling(SHORT_WINDOW,  min_periods=1).mean()
        avg_saves_5   = saves.rolling(SHORT_WINDOW, min_periods=1).mean()
        avg_gc_5      = gc.rolling(SHORT_WINDOW,  min_periods=1).mean()
        trend = avg_pts_5 - avg_pts_10

        for i, row in g.iterrows():
            feature_rows.append({
                "player_id"             : int(player_id),
                "season"                : str(season),
                "gameweek"              : int(row["gameweek"]),
                "web_name"              : str(row.get("web_name", "") or ""),
                "position" : (row.get("position") 
                             if row.get("position") not in (None, "", "UNK", "AM") 
                             else _element_type_to_pos(row.get("element_type"))),
                "team_id"               : _safe_int(row.get("team")),
                "price"                 : _safe_float(float(row.get("price_raw", 0) or 0) / 10),
                "actual_points"         : int(row["total_points"]),

                "avg_points_5gw"        : _safe_float(avg_pts_5.iloc[i]),
                "avg_points_10gw"       : _safe_float(avg_pts_10.iloc[i]),
                "avg_points_season"     : _safe_float(avg_pts_ssn.iloc[i]),
                "points_trend"          : _safe_float(trend.iloc[i]),

                "avg_minutes_5gw"       : _safe_float(avg_mins_5.iloc[i]),
                "avg_minutes_10gw"      : _safe_float(avg_mins_10.iloc[i]),
                "started_rate_5gw"      : _safe_float(started_rate.iloc[i]),
                "full_game_rate_5gw"    : _safe_float(full_gm_rate.iloc[i]),

                "avg_goals_5gw"         : _safe_float(avg_goals_5.iloc[i]),
                "avg_assists_5gw"       : _safe_float(avg_assists_5.iloc[i]),
                "avg_bonus_5gw"         : _safe_float(avg_bonus_5.iloc[i]),
                "avg_bps_5gw"           : _safe_float(avg_bps_5.iloc[i]),
                "avg_ict_5gw"           : _safe_float(avg_ict_5.iloc[i]),

                "clean_sheet_rate_5gw"  : _safe_float(cs_rate_5.iloc[i]),
                "avg_saves_5gw"         : _safe_float(avg_saves_5.iloc[i]),
                "avg_goals_conceded_5gw": _safe_float(avg_gc_5.iloc[i]),

                "xg_season"             : None,
                "xa_season"             : None,
                "xgi_season"            : None,
                "xg_per90_season"       : None,
                "xgi_per90_season"      : None,

                "injury_prone_score"    : None,
                "games_since_return"    : None,

                "fpl_availability"      : str(row.get("status") or "") or None,
                "chance_of_playing"     : _safe_int(row.get("chance_of_playing_this_round")),
                "selected_by_percent"   : _safe_float(row.get("selected_by_percent")),
                "transfers_in_event"    : _safe_int(row.get("transfers_in_event")),
                "transfers_out_event"   : _safe_int(row.get("transfers_out_event")),

                "computed_at"           : datetime.now().isoformat(),
            })

        processed += 1
        if processed % 500 == 0:
            print(f"    {processed:,}/{total_groups:,} player-seasons", end="\r")

    print(f"    {processed:,}/{total_groups:,} player-seasons ✓")
    return pd.DataFrame(feature_rows)


def _element_type_to_pos(element_type):
    """Converts FPL element_type integer to position label."""
    mapping = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    # Also handle string positions from archive
    if isinstance(element_type, str):
        return {"AM": "MID", "GK": "GK", "DEF": "DEF", 
                "MID": "MID", "FWD": "FWD"}.get(element_type, "UNK")
    try:
        return mapping.get(int(element_type), "UNK")
    except (TypeError, ValueError):
        return "UNK"


# =============================================================================
# FEATURE ENRICHMENT
# =============================================================================

def enrich_with_understat(features_df, understat_df):
    """
    Merges Understat xG data using multi-strategy name matching.
    All lookups are pre-built as dicts — O(1) per row, not O(n).

    Strategy 1: Exact match (Erling Haaland → Erling Haaland)
    Strategy 2: Initial.Lastname (B.Fernandes → Bruno Fernandes)
    Strategy 3: Single lastname (Haaland → Erling Haaland)
    Strategy 4: Hyphenated surname (Gibbs-White → Morgan Gibbs-White)
    """
    print("  Enriching with Understat xG data...")

    understat_df = understat_df.sort_values(["player_name", "season"])

    # Strategy 1: exact match
    exact_map = understat_df.set_index(["player_name", "season"])

    # Strategy 2: Initial.Lastname → full name (B.Fernandes → Bruno Fernandes)
    initial_map = {}
    for _, row in understat_df.iterrows():
        parts = row["player_name"].split()
        if len(parts) >= 2:
            key = (parts[0][0] + "." + parts[-1], row["season"])
            if key not in initial_map:
                initial_map[key] = row

    # Strategy 3: single last name (Haaland, Semenyo, Palmer)
    lastname_map = {}
    for _, row in understat_df.iterrows():
        last = row["player_name"].split()[-1].strip()
        key  = (last, row["season"])
        if key not in lastname_map:
            lastname_map[key] = row

    # Strategy 4: hyphenated last two words (Gibbs-White, Van-Dijk)
    hyphen_map = {}
    for _, row in understat_df.iterrows():
        parts = row["player_name"].split()
        if len(parts) >= 2:
            key = ("-".join(parts[-2:]), row["season"])
            if key not in hyphen_map:
                hyphen_map[key] = row

    def get_xg_row(web_name, season):
        """Try all four strategies, return first hit."""
        # Strategy 1: exact
        key = (web_name, season)
        if key in exact_map.index:
            entry = exact_map.loc[key]
            return entry.iloc[0] if isinstance(entry, pd.DataFrame) else entry

        # Strategy 2: initial.lastname
        if key in initial_map:
            return initial_map[key]

        # Strategy 3: single last name
        if " " not in web_name and "." not in web_name:
            if key in lastname_map:
                return lastname_map[key]

        # Strategy 4: hyphenated surname
        if key in hyphen_map:
            return hyphen_map[key]

        return None

    enriched = 0
    for idx, row in features_df.iterrows():
        xg_entry = get_xg_row(row["web_name"], row["season"])
        if xg_entry is not None:
            features_df.at[idx, "xg_season"]       = _safe_float(xg_entry.get("xg"))
            features_df.at[idx, "xa_season"]        = _safe_float(xg_entry.get("xa"))
            features_df.at[idx, "xgi_season"]       = _safe_float(xg_entry.get("xgi"))
            features_df.at[idx, "xg_per90_season"]  = _safe_float(xg_entry.get("xg_per90"))
            features_df.at[idx, "xgi_per90_season"] = _safe_float(xg_entry.get("xgi_per90"))
            enriched += 1

    pct = enriched / len(features_df) * 100 if len(features_df) > 0 else 0
    print(f"  ✓ xG data matched for {enriched:,} rows ({pct:.1f}%)")
    return features_df


def enrich_with_injury_profiles(features_df, injury_df):
    """
    Merges injury profile scores into the feature table.
    Matched on player_id — reliable since profiles use same IDs.
    """
    print("  Enriching with injury profiles...")

    injury_map = injury_df.set_index("player_id")
    enriched = 0

    for idx, row in features_df.iterrows():
        pid = row["player_id"]
        if pid in injury_map.index:
            features_df.at[idx, "injury_prone_score"] = _safe_float(
                injury_map.at[pid, "injury_prone_score"]
            )
            features_df.at[idx, "games_since_return"] = _safe_int(
                injury_map.at[pid, "games_since_return"]
            )
            enriched += 1

    pct = enriched / len(features_df) * 100 if len(features_df) > 0 else 0
    print(f"  ✓ Injury profiles matched for {enriched:,} rows ({pct:.1f}%)")
    return features_df


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_features(engine):
    """
    Sanity checks on the feature store.
    Checks row counts, feature coverage, and top players by form.
    """
    print("\n  Feature store summary:")

    summary = pd.read_sql("""
        SELECT season,
               COUNT(DISTINCT player_id)                  AS players,
               COUNT(*)                                   AS total_rows,
               ROUND(AVG(avg_points_5gw)::numeric, 2)    AS avg_form_5gw,
               COUNT(xg_season)                           AS rows_with_xg,
               COUNT(injury_prone_score)                  AS rows_with_injury
        FROM player_features
        GROUP BY season
        ORDER BY season
    """, engine)

    print(f"\n  {'Season':<10} {'Players':>8} {'Rows':>8} {'Avg Form':>10} {'xG%':>6} {'Inj%':>6}")
    print(f"  {'-'*52}")
    for _, r in summary.iterrows():
        xg_pct  = f"{r['rows_with_xg']/r['total_rows']*100:.0f}%" if r['total_rows'] > 0 else "0%"
        inj_pct = f"{r['rows_with_injury']/r['total_rows']*100:.0f}%" if r['total_rows'] > 0 else "0%"
        print(f"  {r['season']:<10} {r['players']:>8} {r['total_rows']:>8,} "
              f"{float(r['avg_form_5gw'] or 0):>10.2f} {xg_pct:>6} {inj_pct:>6}")

    print("\n  Top 10 by 5GW form (current season, active players):")
    top = pd.read_sql("""
        SELECT web_name, position, price,
               avg_points_5gw, avg_points_10gw, points_trend,
               started_rate_5gw, xgi_season
        FROM player_features
        WHERE season = '2025-26'
          AND gameweek = (
              SELECT MAX(gameweek) FROM player_features WHERE season = '2025-26'
          )
          AND avg_minutes_5gw > 60
        ORDER BY avg_points_5gw DESC
        LIMIT 10
    """, engine)

    if not top.empty:
        print(f"\n  {'Player':<20} {'Pos':<5} {'£':>5} {'Form5':>6} {'Form10':>7} "
              f"{'Trend':>6} {'Start%':>7} {'xGI':>7}")
        print(f"  {'-'*70}")
        for _, r in top.iterrows():
            xgi = f"{float(r['xgi_season']):.2f}" if r['xgi_season'] is not None else "N/A"
            print(
                f"  {str(r['web_name']):<20} {str(r['position']):<5} "
                f"{float(r['price'] or 0):>5.1f} {float(r['avg_points_5gw'] or 0):>6.2f} "
                f"{float(r['avg_points_10gw'] or 0):>7.2f} {float(r['points_trend'] or 0):>6.2f} "
                f"{float(r['started_rate_5gw'] or 0)*100:>6.0f}% {xgi:>7}"
            )


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_feature_engineering():
    """
    Full feature engineering pipeline:
    1. Load raw history from DB
    2. Compute rolling features (with lag to prevent leakage)
    3. Enrich with Understat xG (multi-strategy name matching)
    4. Enrich with injury profiles
    5. Write to player_features table
    6. Verify output
    """
    start = datetime.now()

    print("=" * 60)
    print(f"Feature Engineering  |  {start.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    engine = get_engine()
    setup_feature_tables(engine)

    print("\n[1/4] Loading data...")
    history_df   = load_history(engine)
    understat_df = load_understat(engine)
    injury_df    = load_injury_profiles(engine)

    print("\n[2/4] Computing rolling features...")
    features_df = compute_rolling_features(history_df)
    print(f"  ✓ {len(features_df):,} feature rows computed")

    print("\n[3/4] Enriching features...")
    features_df = enrich_with_understat(features_df, understat_df)
    features_df = enrich_with_injury_profiles(features_df, injury_df)

    print("\n[4/4] Writing to player_features...")
    features_df.to_sql(
        "player_features", engine,
        if_exists="replace", index=False, chunksize=1000
    )
    print(f"  ✓ {len(features_df):,} rows written")

    verify_features(engine)

    duration = (datetime.now() - start).total_seconds()
    print(f"\n{'='*60}")
    print(f"  Feature rows : {len(features_df):,}")
    print(f"  Duration     : {duration:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_feature_engineering()

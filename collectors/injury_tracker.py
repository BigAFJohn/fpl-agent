"""
injury_tracker.py  —  Phase 1, Component 4: Injury History Tracker
===================================================================

CONCEPTS TO UNDERSTAND BEFORE READING THIS SCRIPT
--------------------------------------------------

1. Why injury history predicts future availability
   Injuries are not random. Some players break down every 6-8 weeks
   regardless of manager rotation (injury-prone). Others play 38 games
   a season for years (reliable). The model needs to know which is which.
   A player FPL marks as "available" with a history of 4 injuries this
   season is a very different proposition to one with 0 injuries.

2. How we detect absences from existing data
   We already have gameweek-by-gameweek minutes played in player_history.
   An absence is a run of gameweeks where minutes=0 AFTER a period of
   playing. Pattern: playing → 0 mins → 0 mins → playing = absence detected.
   2+ consecutive zero-minute gameweeks = classified as absence.

3. The injury_prone_score — how it's calculated
   Combines three signals into a 0-1 score:
   - Frequency: absences per 10 games played (weight 0.5)
   - Duration:  average length of absences in gameweeks (weight 0.3)
   - Recency:   how recently was the last absence (weight 0.2)
   Score 0.7+ = high risk. Score 0.3- = reliable.

4. FPL player ID reassignment — why we use 2025/26 archive as the bridge
   FPL reassigns player element IDs every season when players move clubs.
   However the 2025/26 archive data (loaded from vaastav) uses the same
   element IDs as the current player_history table — confirmed 820/832
   matching IDs. So we join injury_profiles → archive 2025-26 → player_history
   using element ID. This gives us current season minutes for active players.

5. games_since_return — the re-injury risk window
   The first 3 games after returning from injury carry 2-3x the normal
   re-injury risk. We track this so Phase 3 can apply a temporary
   confidence penalty to recently-returned players.
"""

import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text


# =============================================================================
# CONFIGURATION
# =============================================================================

DB_PATH = "db/fpl.db"

MIN_ABSENCE_LENGTH = 2
MIN_GAMES_FOR_PROFILE = 10

SEASON_WEIGHTS = {
    "2022-23": 0.5,
    "2023-24": 0.7,
    "2024-25": 0.9,
    "2025-26": 1.0,
}


# =============================================================================
# DATABASE
# =============================================================================

def get_database_engine():
    """Returns SQLAlchemy engine connected to the local SQLite database."""
    return create_engine(f"sqlite:///{DB_PATH}")


def setup_injury_tables(engine):
    """Creates injury_absences and injury_profiles tables if they don't exist."""
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS injury_absences (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id       INTEGER,
                player_name     TEXT,
                season          TEXT,
                absence_start   INTEGER,
                absence_end     INTEGER,
                length_gw       INTEGER,
                absence_type    TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS injury_profiles (
                player_id               INTEGER PRIMARY KEY,
                player_name             TEXT,
                total_absences          INTEGER,
                avg_absence_length      REAL,
                absences_per_10_games   REAL,
                last_absence_gw         INTEGER,
                last_absence_season     TEXT,
                games_since_return      INTEGER,
                injury_prone_score      REAL,
                reliability_label       TEXT,
                updated_at              TEXT
            )
        """))
        conn.commit()
    print("✓ Injury tables ready")


# =============================================================================
# ABSENCE DETECTOR
# =============================================================================

def detect_absences_for_player(player_rows, player_id, player_name, season):
    """
    Detects absence periods from a single player's sorted gameweek rows.
    Returns list of absence dicts. Classifies by length:
    2 GWs = suspension/rotation, 3-5 = short injury, 6+ = long injury.
    """
    if len(player_rows) < MIN_GAMES_FOR_PROFILE:
        return []

    rows = player_rows.sort_values("round").reset_index(drop=True)
    absences = []
    i = 0

    while i < len(rows):
        if rows.iloc[i]["minutes"] == 0:
            absence_start_gw  = rows.iloc[i]["round"]
            absence_start_idx = i
            j = i

            while j < len(rows) and rows.iloc[j]["minutes"] == 0:
                j += 1

            absence_length = j - absence_start_idx

            if absence_length >= MIN_ABSENCE_LENGTH:
                absence_end_gw = (
                    rows.iloc[j]["round"] if j < len(rows)
                    else absence_start_gw + absence_length
                )
                if absence_length <= 2:
                    absence_type = "suspension_or_rotation"
                elif absence_length <= 5:
                    absence_type = "short_injury"
                else:
                    absence_type = "long_injury"

                absences.append({
                    "player_id"    : player_id,
                    "player_name"  : player_name,
                    "season"       : season,
                    "absence_start": int(absence_start_gw),
                    "absence_end"  : int(absence_end_gw),
                    "length_gw"    : absence_length,
                    "absence_type" : absence_type,
                })
            i = j
        else:
            i += 1

    return absences


# =============================================================================
# ABSENCE COLLECTION
# =============================================================================

def collect_all_absences(engine):
    """
    Runs absence detection across all players in player_history_archive
    plus current season from player_history.

    player_history_archive uses 'element' (vaastav naming).
    player_history uses 'player_id' (FPL API naming).
    Both are normalised to 'player_id' in the queries.
    """
    print("\n[1/3] Detecting absences from player history...")

    with engine.connect() as conn:
        conn.execute(text("DELETE FROM injury_absences"))
        conn.commit()

    archive_df = pd.read_sql("""
        SELECT element as player_id,
               name    as web_name,
               round, minutes, season
        FROM player_history_archive
        WHERE minutes IS NOT NULL
        ORDER BY element, season, round
    """, engine)

    current_df = pd.read_sql("""
        SELECT ph.player_id,
               p.web_name,
               ph.round, ph.minutes,
               '2025-26' as season
        FROM player_history ph
        LEFT JOIN players p ON ph.player_id = p.id
        WHERE ph.minutes IS NOT NULL
        ORDER BY ph.player_id, ph.round
    """, engine)

    df = pd.concat([archive_df, current_df], ignore_index=True)
    df = df.drop_duplicates(subset=["player_id", "round", "season"])

    total_absences = []
    players_processed = 0

    for (player_id, season), group in df.groupby(["player_id", "season"]):
        player_name = group["web_name"].iloc[0] or f"Player {player_id}"
        absences = detect_absences_for_player(group, player_id, player_name, season)
        total_absences.extend(absences)
        players_processed += 1

    if total_absences:
        pd.DataFrame(total_absences).to_sql(
            "injury_absences", engine, if_exists="append", index=False
        )

    print(f"  ✓ Processed {players_processed:,} player-seasons")
    print(f"  ✓ Detected {len(total_absences):,} absence periods")
    return len(total_absences)


# =============================================================================
# PROFILE BUILDER
# =============================================================================

def build_injury_profiles(engine):
    """
    Aggregates absence history into per-player injury profiles.
    Uses recency-weighted frequency and duration to compute injury_prone_score.
    """
    print("\n[2/3] Building injury profiles...")

    absences_df = pd.read_sql("SELECT * FROM injury_absences", engine)
    if absences_df.empty:
        print("  ✗ No absences found")
        return 0

    absences_df["weight"] = absences_df["season"].map(SEASON_WEIGHTS).fillna(0.5)

    games_df = pd.read_sql("""
        SELECT element as player_id, season, COUNT(*) as games_played
        FROM player_history_archive WHERE minutes > 0
        GROUP BY element, season
        UNION ALL
        SELECT player_id, '2025-26', COUNT(*)
        FROM player_history WHERE minutes > 0
        GROUP BY player_id
    """, engine)
    games_df["weight"] = games_df["season"].map(SEASON_WEIGHTS).fillna(0.5)
    games_df["weighted_games"] = games_df["games_played"] * games_df["weight"]
    total_weighted_games = games_df.groupby("player_id")["weighted_games"].sum()

    last_gw_map = dict(pd.read_sql("""
        SELECT player_id, MAX(round) as last_gw
        FROM player_history WHERE minutes > 0
        GROUP BY player_id
    """, engine).values)

    profiles = []
    now = datetime.now().isoformat()

    for player_id, player_absences in absences_df.groupby("player_id"):
        player_name    = player_absences["player_name"].iloc[0]
        weighted_games = total_weighted_games.get(player_id, 0)
        if weighted_games < MIN_GAMES_FOR_PROFILE:
            continue

        weighted_absence_count = player_absences["weight"].sum()
        avg_length  = (player_absences["length_gw"] * player_absences["weight"]).sum() / player_absences["weight"].sum()
        frequency   = (weighted_absence_count / weighted_games) * 10

        current_absences = player_absences[player_absences["season"] == "2025-26"]
        if not current_absences.empty:
            last_abs            = current_absences.sort_values("absence_end").iloc[-1]
            last_absence_gw     = int(last_abs["absence_end"])
            last_absence_season = "2025-26"
            current_gw          = last_gw_map.get(player_id, last_absence_gw)
            games_since_return  = max(0, int(current_gw) - last_absence_gw)
        else:
            last_abs            = player_absences.sort_values("absence_end").iloc[-1]
            last_absence_gw     = int(last_abs["absence_end"])
            last_absence_season = last_abs["season"]
            games_since_return  = 999  # No absence this season

        freq_score = min(frequency / 2.0, 1.0)
        dur_score  = min(avg_length / 6.0, 1.0)
        recency_score = (
            0.8 if games_since_return <= 3 else
            0.4 if games_since_return <= 10 else
            0.1 if games_since_return == 999 else 0.2
        )

        score = round((freq_score * 0.5) + (dur_score * 0.3) + (recency_score * 0.2), 3)
        label = (
            "very_risky" if score >= 0.7 else
            "risky"      if score >= 0.5 else
            "moderate"   if score >= 0.3 else
            "reliable"
        )

        profiles.append({
            "player_id"            : int(player_id),
            "player_name"          : player_name,
            "total_absences"       : len(player_absences),
            "avg_absence_length"   : round(float(avg_length), 2),
            "absences_per_10_games": round(float(frequency), 3),
            "last_absence_gw"      : last_absence_gw,
            "last_absence_season"  : last_absence_season,
            "games_since_return"   : games_since_return if games_since_return != 999 else -1,
            "injury_prone_score"   : score,
            "reliability_label"    : label,
            "updated_at"           : now,
        })

    if profiles:
        pd.DataFrame(profiles).to_sql(
            "injury_profiles", engine, if_exists="replace", index=False
        )
        print(f"  ✓ Built profiles for {len(profiles):,} players")

    return len(profiles)


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_profiles(engine):
    """
    Profile distribution + high-risk and reliable player lists.

    Joins injury_profiles → player_history_archive (2025-26) → player_history
    using element ID — confirmed 820/832 match. This handles FPL's ID
    reassignment across seasons without needing fragile name matching.
    """
    print("\n[3/3] Profile summary:")

    # Distribution table
    rows = pd.read_sql("""
        SELECT reliability_label,
               COUNT(*)                          as players,
               ROUND(AVG(injury_prone_score), 3) as avg_score,
               ROUND(AVG(avg_absence_length), 2) as avg_length,
               ROUND(AVG(absences_per_10_games), 3) as avg_freq
        FROM injury_profiles
        GROUP BY reliability_label
        ORDER BY avg_score DESC
    """, engine)

    print(f"\n  {'Label':<14} {'Players':>8} {'Avg Score':>10} {'Avg Len':>8} {'Freq/10':>8}")
    print(f"  {'-'*52}")
    for _, r in rows.iterrows():
        print(f"  {r['reliability_label']:<14} {r['players']:>8} {r['avg_score']:>10} {r['avg_length']:>8} {r['avg_freq']:>8}")

    # High risk — use element ID bridge through 2025-26 archive
    print(f"\n  🚨 High risk — recently returned (active players only):")
    risky = pd.read_sql("""
        SELECT ip.player_name,
               ip.injury_prone_score     as score,
               ip.games_since_return     as gs_ret,
               ip.total_absences         as absences,
               ip.avg_absence_length     as avg_len,
               SUM(ph.minutes)           as mins_this_season
        FROM injury_profiles ip
        JOIN player_history_archive arc
            ON ip.player_id = arc.element AND arc.season = '2025-26'
        JOIN player_history ph
            ON arc.element = ph.player_id
        WHERE ip.games_since_return BETWEEN 0 AND 5
          AND ip.injury_prone_score >= 0.5
        GROUP BY ip.player_id
        HAVING mins_this_season > 90
        ORDER BY ip.injury_prone_score DESC
        LIMIT 10
    """, engine)

    if not risky.empty:
        print(f"  {'Player':<24} {'Score':>6} {'GS Ret':>7} {'Abs':>5} {'AvgLen':>7} {'Mins':>6}")
        print(f"  {'-'*59}")
        for _, r in risky.iterrows():
            print(f"  {r['player_name']:<24} {r['score']:>6} {r['gs_ret']:>7} {r['absences']:>5} {r['avg_len']:>7} {int(r['mins_this_season']):>6}")
    else:
        print("  None currently in re-injury risk window")

    # Reliable players
    print(f"\n  ✅ Most reliable active players:")
    reliable = pd.read_sql("""
        SELECT ip.player_name,
               ip.injury_prone_score  as score,
               ip.total_absences      as absences,
               SUM(ph.minutes)        as mins_this_season,
               p.total_points
        FROM injury_profiles ip
        JOIN player_history_archive arc
            ON ip.player_id = arc.element AND arc.season = '2025-26'
        JOIN player_history ph
            ON arc.element = ph.player_id
        JOIN players p
            ON ph.player_id = p.id
        WHERE ip.reliability_label = 'reliable'
        GROUP BY ip.player_id
        HAVING mins_this_season > 500
           AND p.total_points > 80
        ORDER BY ip.injury_prone_score ASC, p.total_points DESC
        LIMIT 10
    """, engine)

    if not reliable.empty:
        print(f"  {'Player':<24} {'Score':>6} {'Absences':>9} {'Mins':>6} {'FPL Pts':>8}")
        print(f"  {'-'*57}")
        for _, r in reliable.iterrows():
            print(f"  {r['player_name']:<24} {r['score']:>6} {r['absences']:>9} {int(r['mins_this_season']):>6} {int(r['total_points']):>8}")
    else:
        print("  No reliable players found with current filters")


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_injury_tracker():
    """Full pipeline: detect absences → build profiles → verify."""
    print("=" * 60)
    print(f"Injury Tracker  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    engine = get_database_engine()
    setup_injury_tables(engine)

    absence_count = collect_all_absences(engine)
    profile_count = build_injury_profiles(engine)
    verify_profiles(engine)

    print(f"\n{'='*60}")
    print(f"  Absences detected : {absence_count:,}")
    print(f"  Profiles built    : {profile_count:,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_injury_tracker()

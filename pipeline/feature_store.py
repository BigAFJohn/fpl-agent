"""
feature_store.py  —  Phase 3, Component 4: Feature Store Consolidation
=======================================================================

CONCEPTS TO UNDERSTAND BEFORE READING THIS SCRIPT
--------------------------------------------------

1. Why we need a feature store
   We now have three separate tables with complementary information:
   - player_features: rolling form, minutes, xG (one row per player/GW)
   - fixture_difficulty: custom FDR per fixture (one row per fixture)
   - lineup_probability: availability signals (one row per player)

   The Phase 4 model needs ALL of this in one flat table. Joining
   at training time is slow and error-prone. The feature store
   pre-joins everything into model_features — one row per player
   per gameweek with every feature the model needs, ready to go.

2. What model_features contains
   For each player-gameweek:
   - Identity: player_id, season, gameweek, name, position, team, price
   - Target: actual_points (null for future gameweeks)
   - Form: rolling averages (5gw, 10gw), trend
   - Minutes: avg_minutes, started_rate, full_game_rate
   - Attacking: goals, assists, bonus, ICT, xG, xA, xGI
   - Defensive: clean_sheet_rate, saves, goals_conceded
   - Fixture: custom FDR (home/away), opponent xGA, FPL FDR
   - Availability: lineup_probability, fpl_status, chance_of_playing
   - Injury: injury_prone_score, games_since_return

3. The join strategy
   player_features x fixture_difficulty: join on team_id + gameweek
   player_features x lineup_probability: join on player_id + gameweek
   For future gameweeks, actual_points is NULL -- this is expected.
   The model trains on rows where actual_points IS NOT NULL, and
   predicts on rows where it IS NULL (upcoming fixtures).

4. Home vs away fixture difficulty
   A player's fixture difficulty depends on whether their team is
   playing at home or away. We store both home_attack_fdr and
   away_attack_fdr in the fixture table, then select the right
   one based on whether the player's team is home or away in
   each fixture. This is the was_home flag in player_history.

5. Current GW rows -- the prediction target
   For the most recent completed gameweek, actual_points is known.
   For the NEXT gameweek (upcoming), actual_points is NULL.
   We include both in model_features. The upcoming rows are what
   Phase 4 uses to generate weekly team recommendations.

6. Fixture lookup fix (Phase 7b)
   For prediction rows (latest GW), we must use the NEXT UNPLAYED
   fixture, not the completed GW fixture. GW36 form data should
   reference GW37 fixtures. Without this fix, the model predicts
   using already-played opponents which is meaningless.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, text


# =============================================================================
# DATABASE
# =============================================================================

DB_PATH = "db/fpl.db"


def get_engine():
    pg_url = os.environ.get("FPL_DB_URL")
    if pg_url:
        return create_engine(pg_url, pool_size=5, max_overflow=10)
    return create_engine(f"sqlite:///{DB_PATH}")


def setup_model_features_table(engine):
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS model_features"))
        conn.execute(text("""
            CREATE TABLE model_features (
                player_id               INTEGER,
                season                  TEXT,
                gameweek                INTEGER,
                web_name                TEXT,
                position                TEXT,
                team_id                 INTEGER,
                team_name               TEXT,
                price                   NUMERIC(6,2),
                was_home                BOOLEAN,
                actual_points           INTEGER,
                avg_points_5gw          NUMERIC(6,2),
                avg_points_10gw         NUMERIC(6,2),
                avg_points_season       NUMERIC(6,2),
                points_trend            NUMERIC(6,2),
                avg_minutes_5gw         NUMERIC(6,2),
                avg_minutes_10gw        NUMERIC(6,2),
                started_rate_5gw        NUMERIC(5,3),
                full_game_rate_5gw      NUMERIC(5,3),
                avg_goals_5gw           NUMERIC(6,3),
                avg_assists_5gw         NUMERIC(6,3),
                avg_bonus_5gw           NUMERIC(6,3),
                avg_bps_5gw             NUMERIC(6,2),
                avg_ict_5gw             NUMERIC(7,3),
                clean_sheet_rate_5gw    NUMERIC(5,3),
                avg_saves_5gw           NUMERIC(6,2),
                avg_goals_conceded_5gw  NUMERIC(6,2),
                xg_season               NUMERIC(8,4),
                xa_season               NUMERIC(8,4),
                xgi_season              NUMERIC(8,4),
                xg_per90_season         NUMERIC(8,4),
                xgi_per90_season        NUMERIC(8,4),
                fixture_fdr             NUMERIC(4,2),
                opponent_xga_5          NUMERIC(8,4),
                fpl_fdr                 INTEGER,
                opponent_name           TEXT,
                lineup_probability      NUMERIC(5,3),
                fpl_status              TEXT,
                fpl_chance              INTEGER,
                confidence_level        TEXT,
                injury_prone_score      NUMERIC(5,3),
                games_since_return      INTEGER,
                selected_by_percent     NUMERIC(6,2),
                transfers_in_event      BIGINT,
                transfers_out_event     BIGINT,
                computed_at             TIMESTAMP,
                PRIMARY KEY (player_id, season, gameweek)
            )
        """))
        conn.commit()
    print("✓ model_features table ready")


# =============================================================================
# BUILD FEATURE STORE
# =============================================================================

def build_model_features(engine):
    print("\n[1/3] Loading source tables...")

    pf = pd.read_sql("""
        SELECT pf.*,
               ph.was_home,
               t.name AS team_name
        FROM player_features pf
        LEFT JOIN player_history ph
            ON pf.player_id = ph.player_id
           AND pf.gameweek  = ph.round
        LEFT JOIN teams t
            ON pf.team_id = t.id
        ORDER BY pf.player_id, pf.season, pf.gameweek
    """, engine)
    print(f"  ✓ player_features: {len(pf):,} rows")

    fd = pd.read_sql("""
        SELECT * FROM fixture_difficulty
        ORDER BY gameweek
    """, engine)
    print(f"  ✓ fixture_difficulty: {len(fd):,} rows")

    lp = pd.read_sql("""
        SELECT player_id, gameweek,
               lineup_probability,
               fpl_status, fpl_chance,
               confidence_level,
               injury_prone_score
        FROM lineup_probability
    """, engine)
    print(f"  ✓ lineup_probability: {len(lp):,} rows")

    print("\n[2/3] Joining features...")

    # Build fixture lookup -- split into home and away rows
    home_fix = fd[["gameweek", "home_team", "away_team",
                   "home_attack_fdr", "home_opp_xga_5",
                   "fpl_fdr_home", "finished"]].copy()
    home_fix.columns = ["gameweek", "team_name", "opponent_name",
                        "fixture_fdr", "opponent_xga_5", "fpl_fdr", "finished"]
    home_fix["was_home_fix"] = True

    away_fix = fd[["gameweek", "away_team", "home_team",
                   "away_attack_fdr", "away_opp_xga_5",
                   "fpl_fdr_away", "finished"]].copy()
    away_fix.columns = ["gameweek", "team_name", "opponent_name",
                        "fixture_fdr", "opponent_xga_5", "fpl_fdr", "finished"]
    away_fix["was_home_fix"] = False

    all_fix = pd.concat([home_fix, away_fix], ignore_index=True)
    fixture_map = all_fix.set_index(["team_name", "gameweek"])

    # Build per-team lookup of next unplayed fixture gameweek
    # Used to redirect prediction rows to upcoming fixtures
    unplayed = all_fix[all_fix["finished"] == False].copy()
    next_fix_gw = (
        unplayed.groupby("team_name")["gameweek"]
        .min()
        .to_dict()
    )

    # Latest GW in current season -- these are the prediction rows
    latest_gw = int(
        pf[pf["season"] == "2025-26"]["gameweek"].max()
    ) if not pf[pf["season"] == "2025-26"].empty else 0

    # Build lineup probability lookup
    lp_map = lp.set_index(["player_id", "gameweek"])

    def _f(val):
        if val is None:
            return None
        try:
            f = float(val)
            return None if np.isnan(f) else f
        except (TypeError, ValueError):
            return None

    def _i(val):
        f = _f(val)
        return int(f) if f is not None else None

    def get_fix(team_name, gameweek, col):
        key = (team_name, gameweek)
        if key not in fixture_map.index:
            return None
        val = fixture_map.loc[key]
        if isinstance(val, pd.DataFrame):
            val = val.iloc[0]
        v = val.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return v

    def is_finished(team_name, gameweek):
        """Returns True if this team's fixture for this GW is finished."""
        key = (team_name, gameweek)
        if key not in fixture_map.index:
            return True
        val = fixture_map.loc[key]
        if isinstance(val, pd.DataFrame):
            return bool(val["finished"].iloc[0])
        finished = val.get("finished")
        return bool(finished) if finished is not None else True

    rows = []
    total = len(pf)
    processed = 0

    for _, row in pf.iterrows():
        team_name = str(row.get("team_name") or "")
        gameweek  = int(row["gameweek"])
        player_id = int(row["player_id"])
        season    = str(row["season"])

        fix_fdr  = None
        opp_xga  = None
        fpl_fdr  = None
        opp_name = None

        if season == "2025-26" and team_name:
            # KEY FIX: For prediction rows (latest completed GW), look up
            # the next UNPLAYED fixture rather than the completed one.
            # This ensures Thiago shows Crystal Palace (GW37) not Man City
            # (GW36 already played), Rogers shows Liverpool not Burnley, etc.
            fix_gw = gameweek
            if gameweek == latest_gw and is_finished(team_name, gameweek):
                fix_gw = next_fix_gw.get(team_name, gameweek)

            fix_fdr  = get_fix(team_name, fix_gw, "fixture_fdr")
            opp_xga  = get_fix(team_name, fix_gw, "opponent_xga_5")
            fpl_fdr  = get_fix(team_name, fix_gw, "fpl_fdr")
            opp_name = get_fix(team_name, fix_gw, "opponent_name")

        lp_prob   = None
        lp_status = None
        lp_chance = None
        lp_conf   = None
        inj_score = _f(row.get("injury_prone_score"))
        gsr       = _i(row.get("games_since_return"))

        lp_key = (player_id, gameweek)
        if lp_key in lp_map.index:
            lp_row = lp_map.loc[lp_key]
            if isinstance(lp_row, pd.DataFrame):
                lp_row = lp_row.iloc[0]
            lp_prob   = lp_row.get("lineup_probability")
            lp_status = lp_row.get("fpl_status")
            lp_chance = lp_row.get("fpl_chance")
            lp_conf   = lp_row.get("confidence_level")
            lp_inj    = lp_row.get("injury_prone_score")
            if lp_inj is not None:
                inj_score = _f(lp_inj)

        rows.append({
            "player_id"             : player_id,
            "season"                : season,
            "gameweek"              : gameweek,
            "web_name"              : str(row.get("web_name") or ""),
            "position"              : str(row.get("position") or "UNK"),
            "team_id"               : _i(row.get("team_id")),
            "team_name"             : team_name,
            "price"                 : _f(row.get("price")),
            "was_home"              : bool(row["was_home"]) if row.get("was_home") is not None else None,
            "actual_points"         : _i(row.get("actual_points")),
            "avg_points_5gw"        : _f(row.get("avg_points_5gw")),
            "avg_points_10gw"       : _f(row.get("avg_points_10gw")),
            "avg_points_season"     : _f(row.get("avg_points_season")),
            "points_trend"          : _f(row.get("points_trend")),
            "avg_minutes_5gw"       : _f(row.get("avg_minutes_5gw")),
            "avg_minutes_10gw"      : _f(row.get("avg_minutes_10gw")),
            "started_rate_5gw"      : _f(row.get("started_rate_5gw")),
            "full_game_rate_5gw"    : _f(row.get("full_game_rate_5gw")),
            "avg_goals_5gw"         : _f(row.get("avg_goals_5gw")),
            "avg_assists_5gw"       : _f(row.get("avg_assists_5gw")),
            "avg_bonus_5gw"         : _f(row.get("avg_bonus_5gw")),
            "avg_bps_5gw"           : _f(row.get("avg_bps_5gw")),
            "avg_ict_5gw"           : _f(row.get("avg_ict_5gw")),
            "clean_sheet_rate_5gw"  : _f(row.get("clean_sheet_rate_5gw")),
            "avg_saves_5gw"         : _f(row.get("avg_saves_5gw")),
            "avg_goals_conceded_5gw": _f(row.get("avg_goals_conceded_5gw")),
            "xg_season"             : _f(row.get("xg_season")),
            "xa_season"             : _f(row.get("xa_season")),
            "xgi_season"            : _f(row.get("xgi_season")),
            "xg_per90_season"       : _f(row.get("xg_per90_season")),
            "xgi_per90_season"      : _f(row.get("xgi_per90_season")),
            "fixture_fdr"           : _f(fix_fdr),
            "opponent_xga_5"        : _f(opp_xga),
            "fpl_fdr"               : _i(fpl_fdr),
            "opponent_name"         : str(opp_name) if opp_name else None,
            "lineup_probability"    : _f(lp_prob),
            "fpl_status"            : str(lp_status) if lp_status else None,
            "fpl_chance"            : _i(lp_chance),
            "confidence_level"      : str(lp_conf) if lp_conf else None,
            "injury_prone_score"    : inj_score,
            "games_since_return"    : gsr,
            "selected_by_percent"   : _f(row.get("selected_by_percent")),
            "transfers_in_event"    : _i(row.get("transfers_in_event")),
            "transfers_out_event"   : _i(row.get("transfers_out_event")),
            "computed_at"           : datetime.now().isoformat(),
        })

        processed += 1
        if processed % 10000 == 0:
            print(f"    {processed:,}/{total:,} rows", end="\r")

    print(f"    {processed:,}/{total:,} rows ✓")

    mf = pd.DataFrame(rows)
    mf.to_sql("model_features", engine, if_exists="replace",
              index=False, chunksize=1000)
    print(f"  ✓ {len(mf):,} rows written to model_features")
    return mf


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_model_features(engine):
    print("\n[3/3] Feature store verification")

    coverage = pd.read_sql(text("""
        SELECT season,
               COUNT(*)                              AS rows,
               COUNT(DISTINCT player_id)             AS players,
               COUNT(fixture_fdr)                    AS with_fixture,
               COUNT(lineup_probability)             AS with_lineup,
               COUNT(xgi_season)                     AS with_xg,
               COUNT(injury_prone_score)             AS with_injury,
               ROUND(AVG(actual_points)::numeric, 2) AS avg_pts
        FROM model_features
        GROUP BY season
        ORDER BY season
    """), engine)

    print(f"\n  {'Season':<10} {'Rows':>7} {'Players':>8} {'Fix%':>5} "
          f"{'Lin%':>5} {'xG%':>5} {'Inj%':>5} {'AvgPts':>7}")
    print(f"  {'-'*58}")
    for _, r in coverage.iterrows():
        rows = int(r["rows"])
        fix_pct = f"{int(r['with_fixture'])/rows*100:.0f}%" if rows else "0%"
        lin_pct = f"{int(r['with_lineup'])/rows*100:.0f}%"  if rows else "0%"
        xg_pct  = f"{int(r['with_xg'])/rows*100:.0f}%"      if rows else "0%"
        inj_pct = f"{int(r['with_injury'])/rows*100:.0f}%"   if rows else "0%"
        avg_pts = f"{float(r['avg_pts']):.2f}" if r["avg_pts"] is not None else "N/A"
        print(f"  {r['season']:<10} {rows:>7,} {int(r['players']):>8} "
              f"{fix_pct:>5} {lin_pct:>5} {xg_pct:>5} {inj_pct:>5} {avg_pts:>7}")

    print("\n  Top GW picks (form x lineup_prob x fixture ease):")
    top = pd.read_sql(text("""
        SELECT web_name, position, team_name, price,
               avg_points_5gw,
               lineup_probability,
               fixture_fdr,
               opponent_name,
               xgi_season,
               ROUND(
                   (avg_points_5gw * lineup_probability
                    * (6.0 - COALESCE(fixture_fdr, 3.0)) / 5.0)::numeric
               , 2) AS composite_score
        FROM model_features
        WHERE season = '2025-26'
          AND gameweek = (
              SELECT MAX(gameweek) FROM model_features WHERE season = '2025-26'
          )
          AND avg_minutes_5gw > 45
          AND lineup_probability > 0.5
          AND fixture_fdr IS NOT NULL
        ORDER BY composite_score DESC
        LIMIT 15
    """), engine)

    if not top.empty:
        print(f"\n  {'Player':<20} {'Pos':<4} {'£':>5} {'Form5':>6} "
              f"{'LinProb':>8} {'FDR':>4} {'Opponent':<22} {'Score':>6}")
        print(f"  {'-'*82}")
        for _, r in top.iterrows():
            print(
                f"  {str(r['web_name']):<20} {str(r['position']):<4} "
                f"{float(r['price'] or 0):>5.1f} "
                f"{float(r['avg_points_5gw'] or 0):>6.2f} "
                f"{float(r['lineup_probability'] or 0):>8.2f} "
                f"{float(r['fixture_fdr'] or 0):>4.1f} "
                f"{str(r['opponent_name'] or ''):<22} "
                f"{float(r['composite_score'] or 0):>6.2f}"
            )


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_feature_store():
    start = datetime.now()

    print("=" * 60)
    print(f"Feature Store  |  {start.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    engine = get_engine()
    setup_model_features_table(engine)
    mf = build_model_features(engine)
    verify_model_features(engine)

    duration = (datetime.now() - start).total_seconds()
    print(f"\n{'='*60}")
    print(f"  model_features rows : {len(mf):,}")
    print(f"  Duration            : {duration:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_feature_store()

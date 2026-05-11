"""
fixture_difficulty.py  —  Phase 3, Component 2: Fixture Difficulty Model
=========================================================================

CONCEPTS TO UNDERSTAND BEFORE READING THIS SCRIPT
--------------------------------------------------

1. Why FPL's built-in FDR is insufficient
   FPL rates fixture difficulty 1-5 based on team strength ratings that
   update slowly — maybe once every few weeks. This means:
   - A team that has conceded 12 goals in 5 games still shows FDR=2
   - A newly promoted team in great form shows FDR=1 (easy) all season
   - Home/away is blended into one number, losing resolution
   Our model rebuilds FDR from xGA (expected goals against) per match,
   which updates every gameweek and reflects actual defensive performance.

2. xGA as the core signal
   xGA (expected goals against) measures the quality of chances a team
   conceded, not just actual goals. A team that concedes 0 goals but
   faced 3.5xGA got lucky — their defence is actually weak. A team that
   conceded 2 goals from 0.4xGA was unlucky — their defence is strong.
   Using xGA instead of actual goals gives a cleaner difficulty signal.

3. How we calculate custom FDR
   For each upcoming fixture we ask: "How easy is it to score against
   this opponent?" We answer with:
   - Opponent's rolling 5-game xGA allowed (higher = easier to score)
   - Normalised to a 1-5 scale matching FPL's convention
   - Split by home/away (teams concede ~15% more goals at home when
     they're the away team — actually away teams face harder defence
     at home grounds)
   - Adjusted for recent trend (improving or declining defence)

4. Rolling window choice — 5 games
   We use 5 games rather than season average because defensive form
   changes quickly. A team that sacked their manager 3 weeks ago has
   a completely different defensive setup. 5 games captures current
   reality; season average is stale by GW20.

5. The output — fixture_difficulty table
   One row per fixture with our custom FDR for both home and away team.
   This joins with player_features in Phase 4 so the model knows how
   hard each player's upcoming fixture is.

6. Upcoming fixtures — GW+1 through GW+5
   We compute difficulty for the next 5 gameweeks per team, not just
   the next one. This lets the model reason about fixture runs:
   - 5 easy fixtures = strong buy signal for that team's players
   - 5 hard fixtures = avoid even good players
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, text


# =============================================================================
# CONFIGURATION
# =============================================================================

DB_PATH     = "db/fpl.db"
ROLL_WINDOW = 5       # Games to use for rolling xGA
MIN_GAMES   = 3       # Minimum games needed for a reliable estimate
FDR_MIN     = 1.0     # Minimum FDR score (easiest)
FDR_MAX     = 5.0     # Maximum FDR score (hardest)


# =============================================================================
# DATABASE
# =============================================================================

def get_engine():
    """Returns engine connected to PostgreSQL or SQLite fallback."""
    pg_url = os.environ.get("FPL_DB_URL")
    if pg_url:
        return create_engine(pg_url, pool_size=5, max_overflow=10)
    return create_engine(f"sqlite:///{DB_PATH}")


def setup_difficulty_tables(engine):
    """
    Creates two tables:
    - team_defence_ratings: rolling xGA per team per gameweek
    - fixture_difficulty: custom FDR per fixture for both teams
    """
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS team_defence_ratings"))
        conn.execute(text("DROP TABLE IF EXISTS fixture_difficulty"))

        conn.execute(text("""
            CREATE TABLE team_defence_ratings (
                team_name           TEXT,
                season              TEXT,
                gameweek            INTEGER,
                xga_rolling_5       NUMERIC(8,4),  -- rolling 5-game xGA conceded
                xga_season_avg      NUMERIC(8,4),  -- season average xGA conceded
                xga_trend           NUMERIC(8,4),  -- recent vs medium term trend
                defence_strength    NUMERIC(5,3),  -- 0-1, lower = stronger defence
                PRIMARY KEY (team_name, season, gameweek)
            )
        """))

        conn.execute(text("""
            CREATE TABLE fixture_difficulty (
                fixture_id          INTEGER,
                season              TEXT,
                gameweek            INTEGER,
                kickoff_time        TEXT,
                home_team           TEXT,
                away_team           TEXT,

                -- Custom FDR scores (1=easy, 5=hard)
                home_attack_fdr     NUMERIC(4,2),  -- how hard is it for HOME team to score?
                away_attack_fdr     NUMERIC(4,2),  -- how hard is it for AWAY team to score?

                -- Raw xGA values for reference
                home_opp_xga_5      NUMERIC(8,4),  -- away team's recent xGA (home team attacks vs this)
                away_opp_xga_5      NUMERIC(8,4),  -- home team's recent xGA (away team attacks vs this)

                -- FPL's built-in FDR for comparison
                fpl_fdr_home        INTEGER,
                fpl_fdr_away        INTEGER,

                -- Is this fixture completed?
                finished            BOOLEAN,

                PRIMARY KEY (fixture_id, season)
            )
        """))
        conn.commit()
    print("✓ Fixture difficulty tables ready")


# =============================================================================
# TEAM DEFENCE RATINGS
# =============================================================================

def build_team_defence_ratings(engine):
    """
    Builds rolling xGA ratings for every team across all seasons.

    Uses Understat team match data which has per-match xGA for each team.
    For each team+season we compute:
    - 5-game rolling xGA conceded (using shift to avoid leakage)
    - Season average xGA
    - Trend (rolling5 vs rolling10)
    - Defence strength score normalised 0-1
    """
    print("\n[1/3] Building team defence ratings...")

    # Load Understat team match data
    teams_df = pd.read_sql("""
        SELECT team_name, season, date,
               xga    AS xga_conceded,
               was_home,
               result, pts
        FROM understat_teams
        ORDER BY team_name, season, date
    """, engine)

    if teams_df.empty:
        print("  ✗ No Understat team data found")
        return 0

    print(f"  Loaded {len(teams_df):,} team match records")

    # Add a gameweek proxy — rank matches within season by date
    teams_df["date"] = pd.to_datetime(teams_df["date"])
    teams_df["gameweek_proxy"] = teams_df.groupby(
        ["team_name", "season"]
    )["date"].rank(method="first").astype(int)

    rating_rows = []

    for (team_name, season), group in teams_df.groupby(["team_name", "season"]):
        g = group.sort_values("gameweek_proxy").reset_index(drop=True)

        # Shift xGA by 1 to prevent leakage (same principle as rolling features)
        xga = g["xga_conceded"].shift(1)

        rolling_5  = xga.rolling(ROLL_WINDOW,  min_periods=MIN_GAMES).mean()
        rolling_10 = xga.rolling(10,            min_periods=MIN_GAMES).mean()
        season_avg = xga.expanding().mean()
        trend      = rolling_5 - rolling_10

        for i, row in g.iterrows():
            rating_rows.append({
                "team_name"       : str(team_name),
                "season"          : str(season),
                "gameweek"        : int(row["gameweek_proxy"]),
                "xga_rolling_5"   : float(rolling_5.iloc[i])  if not pd.isna(rolling_5.iloc[i])  else None,
                "xga_season_avg"  : float(season_avg.iloc[i]) if not pd.isna(season_avg.iloc[i]) else None,
                "xga_trend"       : float(trend.iloc[i])      if not pd.isna(trend.iloc[i])      else None,
                "defence_strength": None,  # Normalised below
            })

    ratings_df = pd.DataFrame(rating_rows)

    # Normalise xga_rolling_5 to 0-1 defence_strength score
    # Low xGA conceded = strong defence = low score (hard to score against)
    # High xGA conceded = weak defence = high score (easy to score against)
    xga_vals = ratings_df["xga_rolling_5"].dropna()
    if len(xga_vals) > 0:
        xga_min = xga_vals.quantile(0.05)
        xga_max = xga_vals.quantile(0.95)
        ratings_df["defence_strength"] = (
            (ratings_df["xga_rolling_5"] - xga_min) / (xga_max - xga_min)
        ).clip(0, 1).round(3)

    ratings_df.to_sql(
        "team_defence_ratings", engine,
        if_exists="replace", index=False, chunksize=1000
    )
    print(f"  ✓ {len(ratings_df):,} team-gameweek ratings built")
    return len(ratings_df)


# =============================================================================
# FIXTURE DIFFICULTY SCORES
# =============================================================================

def build_fixture_difficulty(engine):
    """
    Joins fixtures with team defence ratings to produce custom FDR scores.

    For each fixture we look up the OPPONENT's recent defensive strength:
    - Home team's FDR = away team's defensive strength (how hard is away defence?)
    - Away team's FDR = home team's defensive strength (how hard is home defence?)

    Higher opponent xGA conceded = lower FDR (easier fixture).
    Lower opponent xGA conceded = higher FDR (harder fixture).

    We then normalise to 1-5 scale matching FPL convention.
    """
    print("\n[2/3] Building fixture difficulty scores...")

    # Load fixtures with team names
    fixtures_df = pd.read_sql("""
        SELECT f.id         AS fixture_id,
               f.event      AS gameweek,
               f.kickoff_time,
               ht.name      AS home_team,
               at.name      AS away_team,
               f.team_h_difficulty AS fpl_fdr_home,
               f.team_a_difficulty AS fpl_fdr_away,
               f.finished,
               '2025-26'    AS season
        FROM fixtures f
        JOIN teams ht ON f.team_h = ht.id
        JOIN teams at ON f.team_a = at.id
        WHERE f.event IS NOT NULL
        ORDER BY f.event, f.id
    """, engine)

    print(f"  Loaded {len(fixtures_df):,} fixtures")

    # Get latest defence rating per team (most recent gameweek)
    latest_ratings = pd.read_sql("""
        SELECT DISTINCT ON (team_name)
               team_name,
               xga_rolling_5,
               xga_season_avg,
               defence_strength
        FROM team_defence_ratings
        WHERE season = '2025-26'
          AND xga_rolling_5 IS NOT NULL
        ORDER BY team_name, gameweek DESC
    """, engine)

    rating_map = latest_ratings.set_index("team_name")
    print(f"  Team ratings available for {len(rating_map)} teams")

    def get_defence_strength(team_name):
        """Returns defence_strength for a team, or 0.5 (neutral) if not found."""
        if team_name in rating_map.index:
            return float(rating_map.at[team_name, "defence_strength"] or 0.5)
        return 0.5

    def get_xga_5(team_name):
        """Returns xga_rolling_5 for a team, or None if not found."""
        if team_name in rating_map.index:
            val = rating_map.at[team_name, "xga_rolling_5"]
            return float(val) if val is not None else None
        return None

    def normalise_to_fdr(defence_strength):
        """
        Converts defence_strength (0-1) to FDR (1-5).
        defence_strength=0 means strongest defence → FDR=5 (hardest to score against)
        defence_strength=1 means weakest defence  → FDR=1 (easiest to score against)
        We invert because high xGA conceded = weak defence = easy fixture.
        """
        if defence_strength is None:
            return 3.0  # Neutral default
        # Invert: weak defence (high xga) = easy = low FDR
        inverted = 1.0 - defence_strength
        return round(FDR_MIN + inverted * (FDR_MAX - FDR_MIN), 2)

    difficulty_rows = []

    for _, fixture in fixtures_df.iterrows():
        home_team = fixture["home_team"]
        away_team = fixture["away_team"]

        # Home team attacks against away team's defence
        away_defence = get_defence_strength(away_team)
        # Away team attacks against home team's defence
        home_defence = get_defence_strength(home_team)

        # Home advantage adjustment — home teams concede less
        # Empirically ~10-15% xGA reduction at home, so away defence is slightly harder
        home_advantage = 0.05
        away_defence_adj = min(1.0, away_defence + home_advantage)

        difficulty_rows.append({
            "fixture_id"      : int(fixture["fixture_id"]),
            "season"          : str(fixture["season"]),
            "gameweek"        : int(fixture["gameweek"]) if fixture["gameweek"] else None,
            "kickoff_time"    : str(fixture["kickoff_time"] or ""),
            "home_team"       : home_team,
            "away_team"       : away_team,
            "home_attack_fdr" : normalise_to_fdr(away_defence_adj),
            "away_attack_fdr" : normalise_to_fdr(home_defence),
            "home_opp_xga_5"  : get_xga_5(away_team),
            "away_opp_xga_5"  : get_xga_5(home_team),
            "fpl_fdr_home"    : int(fixture["fpl_fdr_home"]) if fixture["fpl_fdr_home"] else None,
            "fpl_fdr_away"    : int(fixture["fpl_fdr_away"]) if fixture["fpl_fdr_away"] else None,
            "finished"        : bool(fixture["finished"]),
        })

    difficulty_df = pd.DataFrame(difficulty_rows)
    difficulty_df.to_sql(
        "fixture_difficulty", engine,
        if_exists="replace", index=False, chunksize=1000
    )
    print(f"  ✓ {len(difficulty_df):,} fixture difficulty scores built")
    return len(difficulty_df)


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_difficulty(engine):
    """
    Sanity checks:
    1. Compare our custom FDR vs FPL's FDR for upcoming fixtures
    2. Show easiest and hardest upcoming fixtures
    3. Show team defensive rankings
    """
    print("\n[3/3] Verification")

    # Upcoming fixtures with both FDR scores
    print("\n  Upcoming fixtures — Custom FDR vs FPL FDR:")
    upcoming = pd.read_sql("""
        SELECT home_team, away_team, gameweek,
               home_attack_fdr AS home_custom,
               away_attack_fdr AS away_custom,
               fpl_fdr_home,
               fpl_fdr_away,
               home_opp_xga_5,
               away_opp_xga_5
        FROM fixture_difficulty
        WHERE finished = FALSE
          AND gameweek IS NOT NULL
        ORDER BY gameweek, home_attack_fdr
        LIMIT 15
    """, engine)

    if not upcoming.empty:
        print(f"\n  {'Home':<22} {'Away':<22} {'GW':>4} "
              f"{'H-Cus':>6} {'H-FPL':>6} {'A-Cus':>6} {'A-FPL':>6}")
        print(f"  {'-'*72}")
        for _, r in upcoming.iterrows():
            print(
                f"  {str(r['home_team']):<22} {str(r['away_team']):<22} "
                f"{int(r['gameweek'] or 0):>4} "
                f"{float(r['home_custom'] or 0):>6.2f} {int(r['fpl_fdr_home'] or 0):>6} "
                f"{float(r['away_custom'] or 0):>6.2f} {int(r['fpl_fdr_away'] or 0):>6}"
            )

    # Team defensive rankings
    print("\n  Team defensive rankings (current season, lower = stronger):")
    teams = pd.read_sql("""
        SELECT DISTINCT ON (team_name)
               team_name,
               ROUND(xga_rolling_5::numeric, 3)  AS xga_5gw,
               ROUND(xga_season_avg::numeric, 3) AS xga_season,
               ROUND(defence_strength::numeric, 3) AS strength
        FROM team_defence_ratings
        WHERE season = '2025-26'
          AND xga_rolling_5 IS NOT NULL
        ORDER BY team_name, gameweek DESC
    """, engine)

    if not teams.empty:
        teams = teams.sort_values("xga_5gw")
        print(f"\n  {'Team':<25} {'xGA/5gw':>8} {'xGA/ssn':>8} {'Strength':>9}")
        print(f"  {'-'*54}")
        for _, r in teams.iterrows():
            print(
                f"  {str(r['team_name']):<25} "
                f"{float(r['xga_5gw'] or 0):>8.3f} "
                f"{float(r['xga_season'] or 0):>8.3f} "
                f"{float(r['strength'] or 0):>9.3f}"
            )


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_fixture_difficulty():
    """
    Full fixture difficulty pipeline:
    1. Build rolling xGA defence ratings per team per gameweek
    2. Join with fixtures to produce custom FDR scores
    3. Verify output against FPL's built-in FDR
    """
    start = datetime.now()

    print("=" * 60)
    print(f"Fixture Difficulty  |  {start.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    engine = get_engine()
    setup_difficulty_tables(engine)

    ratings_count    = build_team_defence_ratings(engine)
    difficulty_count = build_fixture_difficulty(engine)
    verify_difficulty(engine)

    duration = (datetime.now() - start).total_seconds()
    print(f"\n{'='*60}")
    print(f"  Team ratings  : {ratings_count:,}")
    print(f"  Fixtures      : {difficulty_count:,}")
    print(f"  Duration      : {duration:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_fixture_difficulty()

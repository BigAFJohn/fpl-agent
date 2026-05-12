"""
lineup_probability.py  —  Phase 3, Component 3: Lineup Probability Score
=========================================================================

CONCEPTS TO UNDERSTAND BEFORE READING THIS SCRIPT
--------------------------------------------------

1. Why lineup probability is the most important single feature
   A player with 0.9 xGI per 90 minutes is worthless if he plays 0
   minutes. Lineup probability gates every other prediction — it answers
   "will this player even be on the pitch?" before we ask "how many
   points will he score?". Get this wrong and the entire model fails.

2. The four signal sources we combine
   Each source has different reliability and update frequency:

   a) FPL official status (highest reliability, updates 2-3x/week)
      - status='a' + chance=100 → near-certain starter
      - status='d' + chance=75  → likely starter, monitor
      - status='i'              → almost certain non-starter
      Source confidence weight: 0.40

   b) News signals from Claude parsing (medium reliability, real-time)
      - Aggregated confidence from lineup_signals table
      - Multiple signals per player weighted by recency
      Source confidence weight: 0.30

   c) Recent minutes trend (lagging but reliable)
      - avg_minutes_5gw > 75 → regular starter
      - started_rate_5gw > 0.8 → reliable starter
      - Doesn't tell us about THIS week but reflects pattern
      Source confidence weight: 0.20

   d) Injury history (background risk)
      - injury_prone_score adjusts probability down for risky players
      - games_since_return < 3 triggers re-injury risk penalty
      Source confidence weight: 0.10

3. Signal combination — weighted average with overrides
   We compute a weighted average of all signals but apply hard overrides:
   - FPL status='i' (injured) → cap at 0.05 regardless of other signals
   - FPL status='u' (unavailable) → cap at 0.02
   - FPL chance_of_playing=0 → cap at 0.05
   - FPL chance_of_playing=100 + no news → floor at 0.75
   These overrides prevent the model from starting an injured player
   just because his recent minutes trend is high.

4. Recency weighting for news signals
   News from yesterday matters more than news from 3 weeks ago.
   We weight news signals by age: signals from the last 24hrs get
   full weight, signals from 7 days ago get 0.3 weight. This ensures
   the lineup probability responds quickly to breaking injury news.

5. Output table — lineup_probability
   One row per player per gameweek with:
   - lineup_probability (0.0-1.0)
   - signal breakdown (which sources contributed what)
   - confidence_level (high/medium/low based on signal agreement)
   This feeds directly into player_features and the Phase 4 model.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from sqlalchemy import create_engine, text


# =============================================================================
# CONFIGURATION
# =============================================================================

DB_PATH = "db/fpl.db"

# Signal source weights — must sum to 1.0
WEIGHT_FPL_STATUS  = 0.40
WEIGHT_NEWS        = 0.30
WEIGHT_MINUTES     = 0.20
WEIGHT_INJURY      = 0.10

# Hard override thresholds
MAX_PROB_INJURED     = 0.05
MAX_PROB_UNAVAILABLE = 0.02
MAX_PROB_CHANCE_ZERO = 0.05
MIN_PROB_AVAILABLE   = 0.75   # Floor when FPL marks available + no negative news


# =============================================================================
# DATABASE
# =============================================================================

def get_engine():
    """Returns engine connected to PostgreSQL or SQLite fallback."""
    pg_url = os.environ.get("FPL_DB_URL")
    if pg_url:
        return create_engine(pg_url, pool_size=5, max_overflow=10)
    return create_engine(f"sqlite:///{DB_PATH}")


def setup_lineup_table(engine):
    """
    Creates the lineup_probability table.
    One row per player per gameweek with probability and signal breakdown.
    """
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS lineup_probability"))
        conn.execute(text("""
            CREATE TABLE lineup_probability (
                player_id           INTEGER,
                gameweek            INTEGER,
                web_name            TEXT,
                team_name           TEXT,

                -- The key output
                lineup_probability  NUMERIC(5,3),   -- 0.0 to 1.0

                -- Signal components (for debugging and model explainability)
                fpl_signal          NUMERIC(5,3),   -- from FPL status/chance
                news_signal         NUMERIC(5,3),   -- from Claude-parsed news
                minutes_signal      NUMERIC(5,3),   -- from recent minutes trend
                injury_signal       NUMERIC(5,3),   -- from injury profile

                -- Signal metadata
                fpl_status          TEXT,
                fpl_chance          INTEGER,
                news_count          INTEGER,        -- number of recent signals
                avg_news_confidence NUMERIC(5,3),
                injury_prone_score  NUMERIC(5,3),
                avg_minutes_5gw     NUMERIC(6,2),
                started_rate_5gw    NUMERIC(5,3),

                -- Confidence in our estimate
                confidence_level    TEXT,           -- high/medium/low
                signal_agreement    NUMERIC(5,3),   -- 0-1, how much signals agree

                computed_at         TIMESTAMP,

                PRIMARY KEY (player_id, gameweek)
            )
        """))
        conn.commit()
    print("✓ lineup_probability table ready")


# =============================================================================
# SIGNAL BUILDERS
# =============================================================================

def build_fpl_signal(status, chance):
    """
    Converts FPL status and chance_of_playing into a 0-1 probability.

    FPL's chance_of_playing is the most direct signal we have — it's
    set by FPL staff based on official club communications. We trust it
    more than any other source.
    """
    if chance is not None:
        return float(chance) / 100.0

    # Fall back to status mapping if chance not set
    status_map = {
        "a": 0.85,   # Available — not guaranteed starter but probably plays
        "d": 0.50,   # Doubtful — 50/50
        "i": 0.05,   # Injured — almost certainly out
        "s": 0.02,   # Suspended — almost certainly out
        "u": 0.02,   # Unavailable — not in squad
    }
    return status_map.get(status, 0.70)


def build_news_signal(player_name, current_gw, news_df):
    """
    Aggregates recent news signals for a player into a single probability.

    Applies recency weighting — recent signals matter more:
    - Last 24hrs: weight 1.0
    - Last 3 days: weight 0.7
    - Last 7 days: weight 0.4
    - Older: weight 0.2

    Returns (signal_value, signal_count, avg_confidence) tuple.
    """
    if news_df is None or news_df.empty:
        return 0.70, 0, None  # Neutral if no news

    # Filter to this player's signals
    player_news = news_df[
        news_df["player_name"].str.lower() == player_name.lower()
    ].copy()

    if player_news.empty:
        return 0.70, 0, None

    now = datetime.now(timezone.utc)

    # Apply recency weighting
    weighted_signals = []
    for _, signal in player_news.iterrows():
        try:
            collected = pd.to_datetime(signal["collected_at"], utc=True)
            age_hours = (now - collected).total_seconds() / 3600
        except Exception:
            age_hours = 168  # Default to 7 days if parse fails

        if age_hours <= 24:
            recency_weight = 1.0
        elif age_hours <= 72:
            recency_weight = 0.7
        elif age_hours <= 168:
            recency_weight = 0.4
        else:
            recency_weight = 0.2

        # Convert status to probability
        status_prob = {
            "available"  : 0.85,
            "doubt"      : 0.45,
            "injured"    : 0.05,
            "unavailable": 0.02,
        }.get(str(signal.get("status", "")).lower(), 0.70)

        # Blend status probability with signal confidence
        raw_conf = float(signal.get("confidence", 0.5) or 0.5)
        blended  = (status_prob * raw_conf) + (0.70 * (1 - raw_conf))

        weighted_signals.append(blended * recency_weight)

    if not weighted_signals:
        return 0.70, 0, None

    avg_signal = sum(weighted_signals) / len(weighted_signals)
    avg_conf   = float(player_news["confidence"].mean())

    return round(avg_signal, 3), len(player_news), round(avg_conf, 3)


def build_minutes_signal(avg_minutes_5gw, started_rate_5gw):
    """
    Converts recent minutes trend into a 0-1 starting probability.

    This is a lagging signal — it tells us the PATTERN of recent weeks,
    not specifically this week. A player who has started 5/5 games with
    85+ mins each is a reliable starter unless something changed.
    """
    if avg_minutes_5gw is None or started_rate_5gw is None:
        return 0.70  # Neutral default

    avg_mins   = float(avg_minutes_5gw)
    start_rate = float(started_rate_5gw)

    # Base probability from start rate
    base = start_rate * 0.85  # 100% start rate → 85% probability

    # Boost for full 90s
    if avg_mins >= 85:
        base = min(1.0, base + 0.10)
    elif avg_mins >= 70:
        base = min(1.0, base + 0.05)
    elif avg_mins < 30:
        base = base * 0.6  # Mostly coming off bench

    return round(float(base), 3)


def build_injury_signal(injury_prone_score, games_since_return):
    """
    Adjusts probability based on injury history.
    This is a background risk signal — it nudges probabilities down
    for players who historically break down often.

    Returns a multiplier (0.5-1.0) rather than an absolute probability.
    Applied as a modifier to the combined signal rather than a standalone source.
    """
    if injury_prone_score is None:
        return 1.0  # No adjustment if no profile

    score  = float(injury_prone_score)
    gsr    = int(games_since_return) if games_since_return is not None else 999

    # Re-injury risk window — first 3 games back
    if 0 <= gsr <= 3:
        risk_multiplier = 0.75
    elif 4 <= gsr <= 6:
        risk_multiplier = 0.90
    else:
        risk_multiplier = 1.0

    # Injury-prone penalty
    if score >= 0.7:
        prone_multiplier = 0.88
    elif score >= 0.5:
        prone_multiplier = 0.94
    else:
        prone_multiplier = 1.0

    return round(risk_multiplier * prone_multiplier, 3)


# =============================================================================
# MAIN PROBABILITY CALCULATOR
# =============================================================================

def compute_lineup_probabilities(engine):
    """
    Computes lineup probability for every active player for the current
    and next gameweek.

    Loads all signal sources, combines them with weights, applies hard
    overrides for injured/unavailable players, and writes results.
    """
    print("\n[1/3] Loading signal sources...")

    # Current players with FPL status
    players_df = pd.read_sql("""
        SELECT p.id AS player_id,
               p.web_name,
               t.name AS team_name,
               p.status,
               p.chance_of_playing_this_round AS chance,
               p.element_type,
               p.news
        FROM players p
        JOIN teams t ON p.team = t.id
        ORDER BY p.web_name
    """, engine)
    print(f"  ✓ {len(players_df):,} active players loaded")

    # Recent news signals (last 7 days)
    news_df = pd.read_sql("""
        SELECT player_name, status, confidence, collected_at, signal_type
        FROM lineup_signals
        WHERE collected_at >= NOW() - INTERVAL '7 days'
        ORDER BY collected_at DESC
    """, engine)
    print(f"  ✓ {len(news_df):,} recent news signals loaded")

    # Player features for minutes trend
    features_df = pd.read_sql("""
        SELECT player_id,
               avg_minutes_5gw,
               started_rate_5gw,
               injury_prone_score,
               games_since_return,
               gameweek
        FROM player_features
        WHERE season = '2025-26'
          AND gameweek = (
              SELECT MAX(gameweek) FROM player_features WHERE season = '2025-26'
          )
    """, engine)
    features_map = features_df.set_index("player_id")
    print(f"  ✓ {len(features_df):,} player feature rows loaded")

    # Current gameweek
    gw_result = pd.read_sql("""
        SELECT id FROM gameweeks
        WHERE is_current = TRUE
        LIMIT 1
    """, engine)
    current_gw = int(gw_result["id"].iloc[0]) if not gw_result.empty else 36

    print(f"\n[2/3] Computing probabilities for GW{current_gw}...")

    probability_rows = []

    for _, player in players_df.iterrows():
        player_id  = int(player["player_id"])
        web_name   = str(player["web_name"])
        status     = str(player["status"] or "a")
        chance     = int(player["chance"]) if player["chance"] is not None else None

        # Get features for this player
        feat = features_map.loc[player_id] if player_id in features_map.index else None

        def _feat_float(key):
            if feat is None:
                return None
            val = feat[key]
            if val is None:
                return None
            try:
                f = float(val)
                return None if np.isnan(f) else f
            except (TypeError, ValueError):
                return None

        def _feat_int(key):
            f = _feat_float(key)
            return int(f) if f is not None else None

        avg_mins   = _feat_float("avg_minutes_5gw")
        start_rate = _feat_float("started_rate_5gw")
        inj_score  = _feat_float("injury_prone_score")
        gsr        = _feat_int("games_since_return")

        # Build individual signals
        fpl_sig     = build_fpl_signal(status, chance)
        news_sig, news_count, avg_news_conf = build_news_signal(web_name, current_gw, news_df)
        mins_sig    = build_minutes_signal(avg_mins, start_rate)
        injury_mult = build_injury_signal(inj_score, gsr)

        # Weighted combination
        combined = (
            fpl_sig   * WEIGHT_FPL_STATUS +
            news_sig  * WEIGHT_NEWS +
            mins_sig  * WEIGHT_MINUTES +
            0.70      * WEIGHT_INJURY    # Injury contributes via multiplier below
        )

        # Apply injury multiplier
        combined = combined * injury_mult

        # Hard overrides — FPL official status takes precedence
        if status == "i" or (chance is not None and chance == 0):
            combined = min(combined, MAX_PROB_INJURED)
        elif status == "u":
            combined = min(combined, MAX_PROB_UNAVAILABLE)
        elif status == "a" and chance is None and news_count == 0:
            combined = max(combined, MIN_PROB_AVAILABLE)
        elif status == "s":
            combined = min(combined, 0.02)

        combined = round(float(np.clip(combined, 0.0, 1.0)), 3)

        # Signal agreement — how consistent are the signals?
        signals      = [fpl_sig, news_sig, mins_sig]
        signal_std   = float(np.std(signals))
        agreement    = round(max(0.0, 1.0 - signal_std), 3)

        # Confidence level
        if agreement >= 0.8 and news_count > 0:
            confidence_level = "high"
        elif agreement >= 0.6:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        probability_rows.append({
            "player_id"          : player_id,
            "gameweek"           : current_gw,
            "web_name"           : web_name,
            "team_name"          : str(player["team_name"]),
            "lineup_probability" : combined,
            "fpl_signal"         : round(fpl_sig, 3),
            "news_signal"        : round(news_sig, 3),
            "minutes_signal"     : round(mins_sig, 3),
            "injury_signal"      : round(injury_mult, 3),
            "fpl_status"         : status,
            "fpl_chance"         : chance,
            "news_count"         : news_count,
            "avg_news_confidence": avg_news_conf,
            "injury_prone_score" : inj_score,
            "avg_minutes_5gw"    : avg_mins,
            "started_rate_5gw"   : start_rate,
            "confidence_level"   : confidence_level,
            "signal_agreement"   : agreement,
            "computed_at"        : datetime.now(timezone.utc).isoformat(),
        })

    prob_df = pd.DataFrame(probability_rows)
    prob_df.to_sql("lineup_probability", engine, if_exists="replace", index=False)
    print(f"  ✓ {len(prob_df):,} lineup probabilities computed")
    return prob_df


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_lineup_probabilities(engine):
    """
    Sanity checks:
    1. Distribution of probabilities
    2. Notable doubts and injuries
    3. High-value players with low probability (watch list)
    """
    print("\n[3/3] Verification")

    dist = pd.read_sql(text("""
        SELECT
            CASE
                WHEN lineup_probability >= 0.85 THEN 'near-certain (85%+)'
                WHEN lineup_probability >= 0.65 THEN 'likely (65-85%)'
                WHEN lineup_probability >= 0.40 THEN 'doubt (40-65%)'
                WHEN lineup_probability >= 0.10 THEN 'unlikely (10-40%)'
                ELSE 'out (<10%)'
            END AS bucket,
            COUNT(*) AS players,
            ROUND(AVG(lineup_probability)::numeric, 3) AS avg_prob
        FROM lineup_probability
        GROUP BY bucket
        ORDER BY avg_prob DESC
    """), engine)

    print(f"\n  Probability distribution:")
    print(f"  {'Bucket':<25} {'Players':>8} {'Avg Prob':>9}")
    print(f"  {'-'*44}")
    for _, r in dist.iterrows():
        print(f"  {r['bucket']:<25} {r['players']:>8} {float(r['avg_prob'] or 0):>9.3f}")

    doubts = pd.read_sql(text("""
        SELECT lp.web_name, lp.team_name,
               lp.lineup_probability AS prob,
               lp.fpl_status, lp.fpl_chance,
               lp.news_count,
               pf.avg_points_5gw
        FROM lineup_probability lp
        LEFT JOIN player_features pf
            ON lp.player_id = pf.player_id
           AND pf.season = '2025-26'
           AND pf.gameweek = (SELECT MAX(gameweek) FROM player_features WHERE season = '2025-26')
        WHERE lp.lineup_probability BETWEEN 0.20 AND 0.75
          AND pf.avg_points_5gw > 4
        ORDER BY pf.avg_points_5gw DESC
        LIMIT 10
    """), engine)

    print(f"\n  ⚠ Notable doubts (20-75% probability, high form):")
    if not doubts.empty:
        print(f"  {'Player':<22} {'Team':<20} {'Prob':>5} {'Status':>7} {'News':>5} {'Form5':>6}")
        print(f"  {'-'*65}")
        for _, r in doubts.iterrows():
            print(
                f"  {str(r['web_name']):<22} {str(r['team_name']):<20} "
                f"{float(r['prob'] or 0):>5.2f} {str(r['fpl_status'] or ''):>7} "
                f"{int(r['news_count'] or 0):>5} {float(r['avg_points_5gw'] or 0):>6.2f}"
            )

    reliable = pd.read_sql(text("""
        SELECT lp.web_name, lp.team_name,
               lp.lineup_probability AS prob,
               lp.confidence_level,
               pf.avg_points_5gw,
               pf.xgi_season
        FROM lineup_probability lp
        LEFT JOIN player_features pf
            ON lp.player_id = pf.player_id
           AND pf.season = '2025-26'
           AND pf.gameweek = (SELECT MAX(gameweek) FROM player_features WHERE season = '2025-26')
        WHERE lp.lineup_probability >= 0.85
          AND pf.avg_points_5gw > 5
        ORDER BY pf.avg_points_5gw DESC
        LIMIT 10
    """), engine)

    print(f"\n  Top reliable starters (prob > 0.85, active):")
    if not reliable.empty:
        print(f"  {'Player':<22} {'Team':<20} {'Prob':>5} {'Conf':>6} {'Form5':>6} {'xGI':>6}")
        print(f"  {'-'*67}")
        for _, r in reliable.iterrows():
            xgi = f"{float(r['xgi_season']):.2f}" if r["xgi_season"] is not None else "N/A"
            print(
                f"  {str(r['web_name']):<22} {str(r['team_name']):<20} "
                f"{float(r['prob'] or 0):>5.2f} {str(r['confidence_level'] or ''):>6} "
                f"{float(r['avg_points_5gw'] or 0):>6.2f} {xgi:>6}"
            )


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_lineup_probability():
    """
    Full lineup probability pipeline:
    1. Load all signal sources
    2. Compute weighted probability per player
    3. Apply hard overrides for injured/unavailable
    4. Verify output
    """
    start = datetime.now()

    print("=" * 60)
    print(f"Lineup Probability  |  {start.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    engine = get_engine()
    setup_lineup_table(engine)

    prob_df = compute_lineup_probabilities(engine)
    verify_lineup_probabilities(engine)

    duration = (datetime.now() - start).total_seconds()
    print(f"\n{'='*60}")
    print(f"  Players scored : {len(prob_df):,}")
    print(f"  Duration       : {duration:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_lineup_probability()

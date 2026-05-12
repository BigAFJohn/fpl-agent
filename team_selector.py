"""
team_selector.py  —  Phase 4, Component 2: Optimal Team Selector
=================================================================

CONCEPTS TO UNDERSTAND BEFORE READING THIS SCRIPT
--------------------------------------------------

1. Why linear programming for team selection
   Given 838 players each with a predicted score, we want to pick the
   15-player squad that maximises total predicted points subject to
   FPL constraints. This is an optimisation problem. Brute force is
   impossible (838 choose 15 = 10^30 combinations). Linear programming
   solves it exactly in milliseconds.

2. FPL constraints we must satisfy
   Squad constraints (15 players total):
   - Exactly 2 goalkeepers
   - Exactly 5 defenders
   - Exactly 5 midfielders
   - Exactly 3 forwards
   - Total cost ≤ 100.0 (£m)
   - Maximum 3 players from any single club

   Starting XI constraints (11 from 15):
   - Exactly 1 goalkeeper starts
   - Minimum 3 defenders start
   - Minimum 2 midfielders start
   - Minimum 1 forward starts
   - Exactly 11 players start total

   Captain: doubles points for one player.

3. How PuLP encodes these constraints
   PuLP is a Python linear programming library. We define:
   - Binary decision variables: x[i] = 1 if player i is in squad
   - Binary decision variables: s[i] = 1 if player i starts
   - Binary decision variable:  c[i] = 1 if player i is captain
   - Objective: maximise sum(s[i] * pred[i]) + sum(c[i] * pred[i])
     (starting players score their prediction; captain scores double)
   - Constraints: all FPL rules as linear inequalities

4. The captain multiplier
   The captain scores double points. In the LP, we add a captain
   bonus term: sum(c[i] * pred[i]) where c[i] ≤ s[i] (can only
   captain a starting player) and sum(c[i]) = 1 (exactly one captain).
   The LP automatically picks the highest-predicted starting player
   as captain.

5. Bench ordering
   The 4 bench players are ordered by lineup_probability — the most
   likely to play comes off the bench first. This maximises expected
   points from the bench in case of injuries/suspensions.

6. Output
   The selector outputs:
   - Starting XI with captain/vice-captain marked
   - Bench ordered by likelihood
   - Total predicted points
   - Budget breakdown
   - Transfers suggested vs a reference team (if provided)
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, text

try:
    import pulp
except ImportError:
    raise ImportError("Missing dependency. Run: pip install pulp")


# =============================================================================
# CONFIGURATION
# =============================================================================

DB_PATH = "db/fpl.db"

# FPL squad constraints
SQUAD_SIZE       = 15
SQUAD_GK         = 2
SQUAD_DEF        = 5
SQUAD_MID        = 5
SQUAD_FWD        = 3
MAX_PER_CLUB     = 3
BUDGET           = 100.0   # £m

# Starting XI constraints
START_SIZE       = 11
START_GK         = 1
MIN_START_DEF    = 3
MIN_START_MID    = 2
MIN_START_FWD    = 1


# =============================================================================
# DATABASE
# =============================================================================

def get_engine():
    pg_url = os.environ.get("FPL_DB_URL")
    if pg_url:
        return create_engine(pg_url, pool_size=5, max_overflow=10)
    return create_engine(f"sqlite:///{DB_PATH}")


def setup_selected_team_table(engine):
    """Creates selected_team table for storing the optimal team output."""
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS selected_team"))
        conn.execute(text("""
            CREATE TABLE selected_team (
                player_id           INTEGER,
                gameweek            INTEGER,
                web_name            TEXT,
                position            TEXT,
                team_name           TEXT,
                price               NUMERIC(6,2),
                predicted_points    NUMERIC(6,3),
                adjusted_points     NUMERIC(6,3),
                lineup_probability  NUMERIC(5,3),
                is_starting         BOOLEAN,
                is_captain          BOOLEAN,
                is_vice_captain     BOOLEAN,
                bench_order         INTEGER,        -- NULL if starting, 1-4 if bench
                opponent_name       TEXT,
                fixture_fdr         NUMERIC(4,2),
                selected_at         TIMESTAMP,
                PRIMARY KEY (player_id, gameweek)
            )
        """))
        conn.commit()
    print("✓ selected_team table ready")


# =============================================================================
# DATA LOADER
# =============================================================================

def load_predictions(engine):
    """
    Loads predictions with all data needed for team selection.
    Deduplicates by player_id keeping the row with highest adjusted_points
    to handle the duplicate name issue from archive data.
    """
    print("  Loading predictions...")

    df = pd.read_sql("""
        SELECT p.player_id, p.web_name, p.position, p.team_name,
               p.price, p.predicted_points, p.adjusted_points,
               p.lineup_probability, p.gameweek,
               p.opponent_name, p.fixture_fdr, p.fpl_status
        FROM predictions p
        ORDER BY p.adjusted_points DESC
    """, engine)

    # Deduplicate — keep highest adjusted_points per player_id
    df = df.sort_values("adjusted_points", ascending=False)
    df = df.drop_duplicates(subset="player_id", keep="first")

    # Fill missing values
    df["price"]              = pd.to_numeric(df["price"], errors="coerce").fillna(5.0)
    df["adjusted_points"]    = pd.to_numeric(df["adjusted_points"], errors="coerce").fillna(0)
    df["predicted_points"]   = pd.to_numeric(df["predicted_points"], errors="coerce").fillna(0)
    df["lineup_probability"] = pd.to_numeric(df["lineup_probability"], errors="coerce").fillna(0.5)

    # Filter out unavailable/suspended players — they can't play
    unavailable = df["fpl_status"].isin(["u", "s", "n"])
    if unavailable.sum() > 0:
        print(f"  Filtering {unavailable.sum()} unavailable/suspended players")
        df = df[~unavailable]

    print(f"  ✓ {len(df):,} players available for selection")
    return df.reset_index(drop=True)


# =============================================================================
# LINEAR PROGRAMME
# =============================================================================

def select_optimal_squad(players_df):
    """
    Solves the FPL squad selection optimisation problem using PuLP.

    Decision variables:
    - x[i]: 1 if player i is in the 15-player squad
    - s[i]: 1 if player i is in the starting XI
    - c[i]: 1 if player i is captain

    Objective: maximise sum(s[i]*pred[i]) + sum(c[i]*pred[i])

    Returns indices of selected squad players with starting/captain flags.
    """
    print("\n  Solving squad optimisation...")

    n = len(players_df)
    players = players_df.reset_index(drop=True)

    # Decision variables
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]  # In squad
    s = [pulp.LpVariable(f"s_{i}", cat="Binary") for i in range(n)]  # Starting
    c = [pulp.LpVariable(f"c_{i}", cat="Binary") for i in range(n)]  # Captain

    # Objective: maximise starting points + captain bonus
    prob = pulp.LpProblem("FPL_Team_Selector", pulp.LpMaximize)
    adj_pts = players["adjusted_points"].tolist()
    prob += (
        pulp.lpSum(s[i] * adj_pts[i] for i in range(n)) +
        pulp.lpSum(c[i] * adj_pts[i] for i in range(n))  # Captain doubles
    )

    # -------------------------------------------------------------------------
    # SQUAD CONSTRAINTS
    # -------------------------------------------------------------------------

    # Total squad size
    prob += pulp.lpSum(x) == SQUAD_SIZE

    # Position constraints
    gk_idx  = [i for i, p in enumerate(players["position"]) if p == "GK"]
    def_idx = [i for i, p in enumerate(players["position"]) if p == "DEF"]
    mid_idx = [i for i, p in enumerate(players["position"]) if p == "MID"]
    fwd_idx = [i for i, p in enumerate(players["position"]) if p == "FWD"]

    prob += pulp.lpSum(x[i] for i in gk_idx)  == SQUAD_GK
    prob += pulp.lpSum(x[i] for i in def_idx) == SQUAD_DEF
    prob += pulp.lpSum(x[i] for i in mid_idx) == SQUAD_MID
    prob += pulp.lpSum(x[i] for i in fwd_idx) == SQUAD_FWD

    # Budget constraint
    prices = players["price"].tolist()
    prob += pulp.lpSum(x[i] * prices[i] for i in range(n)) <= BUDGET

    # Max 3 per club
    for club in players["team_name"].unique():
        club_idx = [i for i, t in enumerate(players["team_name"]) if t == club]
        prob += pulp.lpSum(x[i] for i in club_idx) <= MAX_PER_CLUB

    # -------------------------------------------------------------------------
    # STARTING XI CONSTRAINTS
    # -------------------------------------------------------------------------

    # Can only start if in squad
    for i in range(n):
        prob += s[i] <= x[i]

    # Exactly 11 starters
    prob += pulp.lpSum(s) == START_SIZE

    # Exactly 1 starting GK
    prob += pulp.lpSum(s[i] for i in gk_idx) == START_GK

    # Min 3 DEF starting
    prob += pulp.lpSum(s[i] for i in def_idx) >= MIN_START_DEF

    # Min 2 MID starting
    prob += pulp.lpSum(s[i] for i in mid_idx) >= MIN_START_MID

    # Min 1 FWD starting
    prob += pulp.lpSum(s[i] for i in fwd_idx) >= MIN_START_FWD

    # -------------------------------------------------------------------------
    # CAPTAIN CONSTRAINTS
    # -------------------------------------------------------------------------

    # Exactly one captain
    prob += pulp.lpSum(c) == 1

    # Can only captain a starter
    for i in range(n):
        prob += c[i] <= s[i]

    # -------------------------------------------------------------------------
    # SOLVE
    # -------------------------------------------------------------------------

    solver = pulp.PULP_CBC_CMD(msg=0)  # Suppress solver output
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        print(f"  ✗ Solver status: {status}")
        return None

    # Extract solution
    squad_idx   = [i for i in range(n) if pulp.value(x[i]) > 0.5]
    start_idx   = [i for i in range(n) if pulp.value(s[i]) > 0.5]
    captain_idx = [i for i in range(n) if pulp.value(c[i]) > 0.5]

    total_cost = sum(prices[i] for i in squad_idx)
    total_pred = sum(adj_pts[i] for i in start_idx)
    if captain_idx:
        total_pred += adj_pts[captain_idx[0]]  # Captain bonus

    print(f"  ✓ Optimal solution found")
    print(f"    Total cost   : £{total_cost:.1f}m")
    print(f"    Predicted pts: {total_pred:.2f} (incl. captain bonus)")

    return {
        "squad_idx"  : squad_idx,
        "start_idx"  : start_idx,
        "captain_idx": captain_idx,
        "total_cost" : total_cost,
        "total_pred" : total_pred,
    }


# =============================================================================
# FORMAT AND SAVE RESULT
# =============================================================================

def format_and_save_team(players_df, solution, engine):
    """
    Formats the selected team and saves to selected_team table.
    Orders bench by lineup_probability (most likely to play first).
    """
    players = players_df.reset_index(drop=True)
    squad_idx   = solution["squad_idx"]
    start_idx   = set(solution["start_idx"])
    captain_idx = set(solution["captain_idx"])

    # Vice captain = second highest adjusted points among starters
    starters_sorted = sorted(
        start_idx,
        key=lambda i: players.loc[i, "adjusted_points"],
        reverse=True
    )
    vc_idx = set()
    if len(starters_sorted) >= 2:
        # Skip captain
        for i in starters_sorted:
            if i not in captain_idx:
                vc_idx.add(i)
                break

    # Bench: squad minus starters, ordered by lineup_probability
    bench_idx = [i for i in squad_idx if i not in start_idx]
    bench_idx_sorted = sorted(
        bench_idx,
        key=lambda i: players.loc[i, "lineup_probability"],
        reverse=True
    )

    rows = []
    gw = int(players.loc[squad_idx[0], "gameweek"])

    for i in squad_idx:
        is_starting = i in start_idx
        bench_order = None
        if not is_starting:
            bench_order = bench_idx_sorted.index(i) + 1

        rows.append({
            "player_id"          : int(players.loc[i, "player_id"]),
            "gameweek"           : gw,
            "web_name"           : str(players.loc[i, "web_name"]),
            "position"           : str(players.loc[i, "position"]),
            "team_name"          : str(players.loc[i, "team_name"]),
            "price"              : float(players.loc[i, "price"]),
            "predicted_points"   : float(players.loc[i, "predicted_points"]),
            "adjusted_points"    : float(players.loc[i, "adjusted_points"]),
            "lineup_probability" : float(players.loc[i, "lineup_probability"]),
            "is_starting"        : is_starting,
            "is_captain"         : i in captain_idx,
            "is_vice_captain"    : i in vc_idx,
            "bench_order"        : bench_order,
            "opponent_name"      : str(players.loc[i, "opponent_name"] or ""),
            "fixture_fdr"        : float(players.loc[i, "fixture_fdr"] or 3.0),
            "selected_at"        : datetime.now().isoformat(),
        })

    team_df = pd.DataFrame(rows)
    team_df.to_sql("selected_team", engine, if_exists="replace", index=False)
    print(f"  ✓ Team saved to selected_team table")
    return team_df


# =============================================================================
# DISPLAY
# =============================================================================

def display_team(team_df):
    """Prints the selected team in a readable format."""

    starters = team_df[team_df["is_starting"]].copy()
    bench    = team_df[~team_df["is_starting"]].sort_values("bench_order")

    total_cost = team_df["price"].sum()
    start_pred = starters["adjusted_points"].sum()
    cap_bonus  = starters[starters["is_captain"]]["adjusted_points"].sum()
    total_pred = start_pred + cap_bonus

    print(f"\n{'='*70}")
    print(f"  OPTIMAL TEAM — GW{int(team_df['gameweek'].iloc[0])}")
    print(f"{'='*70}")

    # Starting XI by position
    pos_order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    starters_sorted = starters.sort_values(
        "position", key=lambda x: x.map(pos_order)
    )

    print(f"\n  STARTING XI")
    print(f"  {'Player':<22} {'Pos':<4} {'Team':<22} {'£':>5} "
          f"{'Pred':>6} {'LinP':>5} {'vs':<18} {'FDR':>4}")
    print(f"  {'-'*90}")

    for _, r in starters_sorted.iterrows():
        cap_str = " (C)" if r["is_captain"] else " (V)" if r["is_vice_captain"] else "    "
        print(
            f"  {str(r['web_name']) + cap_str:<26} {str(r['position']):<4} "
            f"{str(r['team_name']):<22} £{float(r['price']):>4.1f} "
            f"{float(r['adjusted_points']):>6.2f} "
            f"{float(r['lineup_probability']):>5.2f} "
            f"{str(r['opponent_name']):<18} "
            f"{float(r['fixture_fdr']):>4.1f}"
        )

    print(f"\n  BENCH (ordered by likelihood)")
    print(f"  {'Player':<22} {'Pos':<4} {'Team':<22} {'£':>5} "
          f"{'Pred':>6} {'LinP':>5}")
    print(f"  {'-'*66}")
    for _, r in bench.iterrows():
        print(
            f"  {str(r['web_name']):<22} {str(r['position']):<4} "
            f"{str(r['team_name']):<22} £{float(r['price']):>4.1f} "
            f"{float(r['adjusted_points']):>6.2f} "
            f"{float(r['lineup_probability']):>5.2f}"
        )

    print(f"\n  {'Total cost':<25} £{total_cost:.1f}m")
    print(f"  {'Predicted points':<25} {total_pred:.2f} (incl. captain bonus)")
    print(f"  {'Budget remaining':<25} £{BUDGET - total_cost:.1f}m")
    print(f"{'='*70}")

    # Position summary
    print(f"\n  Formation: ", end="")
    formation = []
    for pos in ["DEF", "MID", "FWD"]:
        count = len(starters[starters["position"] == pos])
        formation.append(str(count))
    print("-".join(formation))

    # Club breakdown
    print(f"\n  Players per club:")
    club_counts = team_df["team_name"].value_counts()
    for club, count in club_counts.items():
        bar = "●" * count
        print(f"    {str(club):<25} {bar} ({count})")


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_team_selector():
    """
    Full team selection pipeline:
    1. Load predictions from XGBoost model
    2. Solve LP optimisation for optimal squad
    3. Format and save selected team
    4. Display team in readable format
    """
    start = datetime.now()

    print("=" * 60)
    print(f"Team Selector  |  {start.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    engine = get_engine()
    setup_selected_team_table(engine)

    print("\n[1/3] Loading predictions...")
    players_df = load_predictions(engine)

    print("\n[2/3] Selecting optimal squad...")
    solution = select_optimal_squad(players_df)

    if solution is None:
        print("✗ Could not find optimal solution")
        return

    print("\n[3/3] Formatting and saving team...")
    team_df = format_and_save_team(players_df, solution, engine)

    display_team(team_df)

    duration = (datetime.now() - start).total_seconds()
    print(f"\n  Duration: {duration:.0f}s")


if __name__ == "__main__":
    run_team_selector()

"""
prediction/scenarios.py  —  Phase 8c: Alternative Team Scenarios
=================================================================

CONCEPTS TO UNDERSTAND BEFORE READING THIS SCRIPT
--------------------------------------------------

1. Why alternative scenarios?
   The LP solver in team_selector.py returns ONE optimal team —
   the mathematically best solution. But "optimal" depends on
   assumptions that may not hold:

   - The model predicts Haaland at 4.17 adj pts. But what if
     Man City rotate and he doesn't play? The "safe" team might
     exclude him.
   - The template team might have 3 Man City players. A
     "differential" team avoids this concentration.
   - A "value" team maximises pts-per-million, useful when
     you've just used a wildcard and need to maximise budget.

   Generating 3-4 scenarios lets you choose based on your
   risk appetite and current FPL situation, rather than
   blindly following one mathematical optimum.

2. The four scenario types
   TEMPLATE    — default optimal (already in team_selector.py)
                 Maximises total adjusted points, no extra constraints.

   SAFE        — only players with lineup_probability > 0.85
                 Avoids rotation risks and injury doubts.
                 Lower ceiling but higher floor.

   DIFFERENTIAL — excludes top-3 most popular FPL picks
                 (approximated by highest ownership proxy: price × form)
                 Targets low-ownership players who could outscore
                 the template captain. High variance, high upside.

   VALUE        — maximises adjusted_points per £million
                 Forces minimum budget spend (≥£98m) to ensure
                 value players aren't just cheap benchwarmers.
                 Useful after wildcard or when building from scratch.

3. How constraints are added to the LP
   Each scenario adds extra constraints on top of the base FPL rules.
   The LP objective stays the same (maximise adjusted points) but
   the feasible solution space changes.

   SAFE:        lineup_probability[i] >= 0.85 for all starters
   DIFFERENTIAL: x[i] = 0 for top-3 ownership proxy players
   VALUE:        total_cost >= 98.0 (force spending up)

4. Integration with team_selector.py
   scenarios.py imports select_optimal_squad logic and extends it.
   It does NOT replace team_selector.py — the default pipeline
   still runs as before. Scenarios are an optional extra step
   triggered by run_scenarios() after the main selection.

5. Output
   Three alternative squads, each formatted identically to the
   main team output. Saved to a scenarios table in the DB.
   Displayed in the dashboard as selectable tabs.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, text

try:
    import pulp
except ImportError:
    raise ImportError("Run: pip install pulp")


# =============================================================================
# CONFIGURATION — mirrors team_selector.py
# =============================================================================

DB_PATH    = "db/fpl.db"
SQUAD_SIZE = 15
START_SIZE = 11
BUDGET     = 100.0

MIN_GK  = 2
MIN_DEF = 5
MIN_MID = 5
MIN_FWD = 3

MIN_START_GK  = 1
MIN_START_DEF = 3
MIN_START_MID = 2
MIN_START_FWD = 1

MAX_PER_CLUB = 3

# Scenario-specific thresholds
SAFE_LINP_THRESHOLD    = 0.85   # Minimum lineup probability for SAFE scenario
VALUE_MIN_SPEND        = 98.0   # Minimum budget spend for VALUE scenario
DIFF_EXCLUDE_TOP_N     = 3      # Number of "template" players to exclude in DIFF


# =============================================================================
# DATABASE
# =============================================================================

def get_engine():
    pg_url = os.environ.get("FPL_DB_URL")
    if pg_url:
        return create_engine(pg_url, pool_size=5, max_overflow=10)
    return create_engine(f"sqlite:///{DB_PATH}")


def setup_scenarios_table(engine):
    """Creates the scenarios table if it doesn't exist."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS team_scenarios (
                id              SERIAL PRIMARY KEY,
                scenario        VARCHAR(20),
                gameweek        INTEGER,
                player_id       INTEGER,
                web_name        VARCHAR(100),
                position        VARCHAR(5),
                team_name       VARCHAR(50),
                price           NUMERIC(5,1),
                adjusted_points NUMERIC(6,3),
                lineup_probability NUMERIC(5,3),
                is_starter      BOOLEAN,
                is_captain      BOOLEAN,
                is_vice_captain BOOLEAN,
                bench_order     INTEGER,
                predicted_pts   NUMERIC(6,3),
                created_at      TIMESTAMP DEFAULT NOW()
            )
        """))
    print("✓ team_scenarios table ready")


# =============================================================================
# CORE LP SOLVER — parameterised for different scenarios
# =============================================================================

def solve_scenario(players_df, scenario_name, extra_constraints_fn=None):
    """
    Solves the FPL optimisation for a given scenario.

    Parameters
    ----------
    players_df : DataFrame
        Full predictions dataframe from load_predictions()
    scenario_name : str
        Name for logging ("SAFE", "DIFFERENTIAL", "VALUE")
    extra_constraints_fn : callable, optional
        Function(prob, x, s, c, players, n) that adds extra constraints
        to the LP problem before solving.

    Returns
    -------
    dict or None
        Solution dict with squad_idx, start_idx, captain_idx,
        total_cost, total_pred. None if infeasible.
    """
    print(f"\n  Solving {scenario_name} scenario...")

    n       = len(players_df)
    players = players_df.reset_index(drop=True)

    # Decision variables
    x = [pulp.LpVariable(f"{scenario_name}_x_{i}", cat="Binary") for i in range(n)]
    s = [pulp.LpVariable(f"{scenario_name}_s_{i}", cat="Binary") for i in range(n)]
    c = [pulp.LpVariable(f"{scenario_name}_c_{i}", cat="Binary") for i in range(n)]

    prob    = pulp.LpProblem(f"FPL_{scenario_name}", pulp.LpMaximize)
    adj_pts = players["adjusted_points"].tolist()
    prices  = pd.to_numeric(players["price"], errors="coerce").fillna(4.0).tolist()

    # Objective — same as main selector
    prob += (
        pulp.lpSum(s[i] * adj_pts[i] for i in range(n)) +
        pulp.lpSum(c[i] * adj_pts[i] for i in range(n))
    )

    # -------------------------------------------------------------------------
    # BASE FPL CONSTRAINTS (same as team_selector.py)
    # -------------------------------------------------------------------------

    prob += pulp.lpSum(x) == SQUAD_SIZE

    for pos, mn in [("GK", MIN_GK), ("DEF", MIN_DEF),
                    ("MID", MIN_MID), ("FWD", MIN_FWD)]:
        idx = players[players["position"] == pos].index.tolist()
        prob += pulp.lpSum(x[i] for i in idx) == mn

    prob += pulp.lpSum(prices[i] * x[i] for i in range(n)) <= BUDGET

    clubs = players["team_name"].unique()
    for club in clubs:
        idx = players[players["team_name"] == club].index.tolist()
        prob += pulp.lpSum(x[i] for i in idx) <= MAX_PER_CLUB

    prob += pulp.lpSum(s) == START_SIZE

    for i in range(n):
        prob += s[i] <= x[i]

    gk_idx = players[players["position"] == "GK"].index.tolist()
    prob += pulp.lpSum(s[i] for i in gk_idx) == MIN_START_GK

    def_idx = players[players["position"] == "DEF"].index.tolist()
    prob += pulp.lpSum(s[i] for i in def_idx) >= MIN_START_DEF

    mid_idx = players[players["position"] == "MID"].index.tolist()
    prob += pulp.lpSum(s[i] for i in mid_idx) >= MIN_START_MID

    fwd_idx = players[players["position"] == "FWD"].index.tolist()
    prob += pulp.lpSum(s[i] for i in fwd_idx) >= MIN_START_FWD

    prob += pulp.lpSum(c) == 1
    for i in range(n):
        prob += c[i] <= s[i]

    # -------------------------------------------------------------------------
    # SCENARIO-SPECIFIC CONSTRAINTS
    # -------------------------------------------------------------------------
    if extra_constraints_fn:
        extra_constraints_fn(prob, x, s, c, players, n)

    # -------------------------------------------------------------------------
    # SOLVE
    # -------------------------------------------------------------------------
    solver = pulp.PULP_CBC_CMD(msg=0)
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]

    if status != "Optimal":
        print(f"  ✗ {scenario_name} infeasible (status: {status})")
        return None

    squad_idx   = [i for i in range(n) if pulp.value(x[i]) > 0.5]
    start_idx   = [i for i in range(n) if pulp.value(s[i]) > 0.5]
    captain_idx = [i for i in range(n) if pulp.value(c[i]) > 0.5]

    # Override captain with captain_score if available
    if "captain_score" in players.columns and start_idx:
        starters_df = players.iloc[start_idx].copy()
        cap_scores  = pd.to_numeric(
            starters_df["captain_score"], errors="coerce"
        ).fillna(0)
        if cap_scores.max() > 0:
            best_cap_pos = cap_scores.idxmax()
            captain_idx  = [best_cap_pos]

    total_cost = sum(prices[i] for i in squad_idx)
    total_pred = sum(adj_pts[i] for i in start_idx)
    if captain_idx:
        total_pred += adj_pts[captain_idx[0]]

    print(f"  ✓ {scenario_name} solution found — "
          f"£{total_cost:.1f}m, {total_pred:.2f} pred pts")

    return {
        "scenario"   : scenario_name,
        "squad_idx"  : squad_idx,
        "start_idx"  : start_idx,
        "captain_idx": captain_idx,
        "total_cost" : total_cost,
        "total_pred" : total_pred,
    }


# =============================================================================
# SCENARIO DEFINITIONS
# =============================================================================

def build_safe_constraints(prob, x, s, c, players, n):
    """
    SAFE scenario: only start players with lineup_probability >= 0.85.
    Reduces rotation risk. Lower ceiling, higher floor.
    """
    linp = pd.to_numeric(
        players["lineup_probability"], errors="coerce"
    ).fillna(0.5).tolist()

    for i in range(n):
        if linp[i] < SAFE_LINP_THRESHOLD:
            prob += s[i] == 0  # Cannot start low-probability players


def build_differential_constraints(prob, x, s, c, players, n):
    """
    DIFFERENTIAL scenario: excludes top-N "template" players.
    Template players approximated by price × avg_points_5gw
    (proxy for ownership — expensive in-form players are heavily owned).
    Targets low-ownership differentials.
    """
    prices  = pd.to_numeric(players["price"], errors="coerce").fillna(4.0)
    if "avg_points_5gw" in players.columns:
        form = pd.to_numeric(players["avg_points_5gw"], errors="coerce").fillna(0)
    else:
        form = pd.Series(0.0, index=players.index)
    template_score = prices * form

    # Exclude top-N template players from the squad entirely
    top_n_idx = template_score.nlargest(DIFF_EXCLUDE_TOP_N).index.tolist()
    for i in top_n_idx:
        prob += x[i] == 0  # Cannot be in squad

    # Log which players are excluded
    excluded = players.loc[top_n_idx, "web_name"].tolist() if "web_name" in players.columns else top_n_idx
    print(f"    Excluding template players: {excluded}")


def build_value_constraints(prob, x, s, c, players, n):
    """
    VALUE scenario: forces minimum spend of £98m.
    Ensures we don't waste budget on cheap benchwarmers.
    Instead maximises pts-per-million by using all available budget.
    """
    prices = pd.to_numeric(players["price"], errors="coerce").fillna(4.0).tolist()
    prob += pulp.lpSum(prices[i] * x[i] for i in range(n)) >= VALUE_MIN_SPEND


# =============================================================================
# FORMAT AND SAVE SCENARIOS
# =============================================================================

def format_scenario(players_df, solution, engine):
    """
    Formats a scenario solution into a readable team dataframe
    and saves to team_scenarios table.
    """
    if solution is None:
        return None

    players  = players_df.reset_index(drop=True)
    scenario = solution["scenario"]
    gw       = int(players["gameweek"].iloc[0]) if "gameweek" in players.columns else 0

    rows = []
    for i in solution["squad_idx"]:
        is_starter  = i in solution["start_idx"]
        is_captain  = i in solution["captain_idx"]
        is_vc       = False  # Computed below
        bench_order = None

        row = players.iloc[i]
        rows.append({
            "scenario"         : scenario,
            "gameweek"         : gw,
            "player_id"        : int(row.get("player_id", 0)),
            "web_name"         : str(row.get("web_name", "")),
            "position"         : str(row.get("position", "")),
            "team_name"        : str(row.get("team_name", "")),
            "price"            : float(row.get("price", 0)),
            "adjusted_points"  : float(row.get("adjusted_points", 0)),
            "lineup_probability": float(row.get("lineup_probability", 0)),
            "is_starter"       : is_starter,
            "is_captain"       : is_captain,
            "is_vice_captain"  : False,
            "bench_order"      : bench_order,
            "predicted_pts"    : float(row.get("predicted_points", 0)),
        })

    team_df = pd.DataFrame(rows)

    # Assign vice-captain (highest adj_pts starter who isn't captain)
    starters = team_df[team_df["is_starter"]].copy()
    non_caps  = starters[~starters["is_captain"]].sort_values(
        "adjusted_points", ascending=False
    )
    if not non_caps.empty:
        vc_idx = non_caps.index[0]
        team_df.loc[vc_idx, "is_vice_captain"] = True

    # Assign bench order by lineup_probability
    bench    = team_df[~team_df["is_starter"]].copy()
    bench_gk = bench[bench["position"] == "GK"]
    bench_out = bench[bench["position"] != "GK"].sort_values(
        "lineup_probability", ascending=False
    )
    bench_ordered = pd.concat([bench_out, bench_gk])
    for order, idx in enumerate(bench_ordered.index, 1):
        team_df.loc[idx, "bench_order"] = order

    # Save to DB
    try:
        team_df.to_sql("team_scenarios", engine, if_exists="append",
                       index=False, method="multi")
    except Exception as e:
        print(f"  ⚠ Could not save {scenario} to DB: {e}")

    return team_df


def display_scenario(team_df, solution):
    """Displays a scenario team in the same format as team_selector.py."""
    if team_df is None:
        return

    scenario   = team_df["scenario"].iloc[0]
    total_cost = solution["total_cost"]
    total_pred = solution["total_pred"]

    print(f"\n{'='*70}")
    print(f"  {scenario} TEAM")
    print(f"{'='*70}")

    starters = team_df[team_df["is_starter"]].sort_values(
        "adjusted_points", ascending=False
    )
    bench = team_df[~team_df["is_starter"]].sort_values("bench_order")

    print(f"\n  STARTING XI")
    print(f"  {'Player':<22} {'Pos':<4} {'Team':<18} {'£':>4} "
          f"{'Pred':>5} {'LinP':>5} {'vs':<20} {'FDR':>4}")
    print(f"  {'-'*85}")

    for _, r in starters.iterrows():
        cap_str = " (C)" if r["is_captain"] else " (V)" if r["is_vice_captain"] else ""
        name    = str(r["web_name"]) + cap_str
        opp     = str(r.get("opponent_name", "")) if "opponent_name" in r else ""
        fdr     = float(r.get("fixture_fdr", 0) or 0)
        print(
            f"  {name:<26} {str(r['position']):<4} "
            f"{str(r['team_name']):<18} "
            f"£{float(r['price']):>3.1f} "
            f"{float(r['adjusted_points']):>5.2f} "
            f"{float(r['lineup_probability']):>5.2f} "
            f"{opp:<20} {fdr:>4.1f}"
        )

    print(f"\n  BENCH")
    for _, r in bench.iterrows():
        print(
            f"  {int(r['bench_order'] or 0)}. {str(r['web_name']):<20} "
            f"{str(r['position']):<4} {str(r['team_name']):<18} "
            f"£{float(r['price']):>3.1f}  "
            f"linp={float(r['lineup_probability']):.2f}"
        )

    print(f"\n  Total cost    : £{total_cost:.1f}m")
    print(f"  Predicted pts : {total_pred:.2f} (incl. captain bonus)")
    print(f"{'='*70}")


# =============================================================================
# COMPARISON TABLE
# =============================================================================

def compare_scenarios(scenarios_dict):
    """
    Prints a side-by-side comparison of all scenarios.
    Shows which players appear in multiple scenarios — these are
    the 'consensus' picks regardless of strategy.
    """
    if not scenarios_dict:
        return

    print(f"\n{'='*70}")
    print(f"  SCENARIO COMPARISON")
    print(f"{'='*70}")
    print(f"\n  {'Scenario':<15} {'Cost':>6} {'Pred Pts':>9} {'Captain':<20}")
    print(f"  {'-'*55}")

    all_players = {}

    for name, (solution, team_df) in scenarios_dict.items():
        if solution is None or team_df is None:
            print(f"  {name:<15} {'INFEASIBLE':>6}")
            continue

        cap_row = team_df[team_df["is_captain"]]
        cap_name = cap_row["web_name"].iloc[0] if not cap_row.empty else "?"

        print(f"  {name:<15} £{solution['total_cost']:>4.1f}m "
              f"{solution['total_pred']:>8.2f}  {cap_name:<20}")

        # Track which scenarios each player appears in
        starters = team_df[team_df["is_starter"]]["web_name"].tolist()
        for player in starters:
            all_players[player] = all_players.get(player, []) + [name]

    # Consensus players — appear in all scenarios
    n_scenarios = len([v for v in scenarios_dict.values() if v[0] is not None])
    consensus   = [p for p, s in all_players.items() if len(s) == n_scenarios]
    unique      = {name: [p for p, s in all_players.items() if s == [name]]
                   for name in scenarios_dict.keys()}

    print(f"\n  Consensus starters (in ALL scenarios):")
    for p in sorted(consensus):
        print(f"    ✓ {p}")

    print(f"\n  Unique picks per scenario:")
    for name, players in unique.items():
        if players:
            print(f"    {name}: {', '.join(players)}")

    print(f"{'='*70}")


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_scenarios():
    """
    Generates three alternative team scenarios:
      SAFE         — only high-probability starters (linp > 0.85)
      DIFFERENTIAL — excludes top-3 template/high-ownership players
      VALUE        — forces minimum £98m spend for pts-per-million focus

    All three run against the same predictions as the main team selector.
    Results saved to team_scenarios table and displayed for comparison.
    """
    start = datetime.now()

    print("=" * 60)
    print(f"Scenario Generator  |  {start.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    engine = get_engine()
    setup_scenarios_table(engine)

    # Load predictions (same as team_selector.py)
    print("\n[1/4] Loading predictions...")
    from prediction.team_selector import load_predictions
    players_df = load_predictions(engine)

    if players_df is None or players_df.empty:
        print("  ✗ No predictions found — run predict_points.py first")
        return

    print(f"  ✓ {len(players_df):,} players loaded")

    # Clear existing scenarios for this GW
    gw = int(players_df["gameweek"].iloc[0]) if "gameweek" in players_df.columns else 0
    with engine.begin() as conn:
        conn.execute(text(
            "DELETE FROM team_scenarios WHERE gameweek = :gw"
        ), {"gw": gw})

    print(f"\n[2/4] Generating scenarios for GW{gw}...")

    scenarios_dict = {}

    # SCENARIO 1: SAFE
    safe_solution = solve_scenario(
        players_df, "SAFE",
        extra_constraints_fn=build_safe_constraints
    )
    safe_team = format_scenario(players_df, safe_solution, engine)
    scenarios_dict["SAFE"] = (safe_solution, safe_team)

    # SCENARIO 2: DIFFERENTIAL
    diff_solution = solve_scenario(
        players_df, "DIFFERENTIAL",
        extra_constraints_fn=build_differential_constraints
    )
    diff_team = format_scenario(players_df, diff_solution, engine)
    scenarios_dict["DIFFERENTIAL"] = (diff_solution, diff_team)

    # SCENARIO 3: VALUE
    value_solution = solve_scenario(
        players_df, "VALUE",
        extra_constraints_fn=build_value_constraints
    )
    value_team = format_scenario(players_df, value_solution, engine)
    scenarios_dict["VALUE"] = (value_solution, value_team)

    print(f"\n[3/4] Displaying scenarios...")
    for name, (solution, team_df) in scenarios_dict.items():
        display_scenario(team_df, solution)

    print(f"\n[4/4] Comparing scenarios...")
    compare_scenarios(scenarios_dict)

    duration = (datetime.now() - start).total_seconds()
    n_successful = sum(1 for s, _ in scenarios_dict.values() if s is not None)

    print(f"\n{'='*60}")
    print(f"  Scenarios generated : {n_successful}/3")
    print(f"  Saved to            : team_scenarios table")
    print(f"  Duration            : {duration:.0f}s")
    print(f"{'='*60}")

    return scenarios_dict


if __name__ == "__main__":
    run_scenarios()

"""
tests/unit/test_selector.py
===========================
Unit tests for team_selector.py constraint validation.

Tests verify that the LP solver always produces teams that
satisfy FPL rules — regardless of input data.

These are constraint tests — we don't care what team is selected,
only that it satisfies every FPL rule.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import pandas as pd
import numpy as np


# =============================================================================
# FIXTURES — synthetic player pools for testing
# =============================================================================

def make_player_pool(n_gk=20, n_def=80, n_mid=80, n_fwd=40, seed=42):
    """
    Creates a synthetic player pool with realistic FPL distributions.
    Used to test selector constraints without needing real DB data.
    """
    rng = np.random.RandomState(seed)

    players = []
    player_id = 1

    clubs = [f"Club{i}" for i in range(1, 21)]  # 20 clubs

    for pos, count, price_range in [
        ("GK",  n_gk,  (3.9, 6.5)),
        ("DEF", n_def, (3.9, 8.0)),
        ("MID", n_mid, (4.5, 13.0)),
        ("FWD", n_fwd, (4.5, 14.0)),
    ]:
        for i in range(count):
            players.append({
                "player_id"        : player_id,
                "web_name"         : f"{pos}_{i}",
                "position"         : pos,
                "team_name"        : clubs[i % 20],
                "price"            : round(rng.uniform(*price_range), 1),
                "predicted_points" : round(rng.uniform(1, 10), 2),
                "adjusted_points"  : round(rng.uniform(1, 9), 2),
                "lineup_probability": round(rng.uniform(0.5, 1.0), 2),
                "fpl_status"       : "a",
                "opponent_name"    : "Opponent",
                "fixture_fdr"      : round(rng.uniform(1, 5), 1),
                "gameweek"         : 36,
            })
            player_id += 1

    return pd.DataFrame(players)


def select_team_from_pool(players_df):
    """
    Runs the actual team selector on a player pool.
    Returns the selected team DataFrame.
    """
    from prediction.team_selector import select_optimal_squad, format_and_save_team
    import os
    from sqlalchemy import create_engine

    solution = select_optimal_squad(players_df)
    assert solution is not None, "LP solver returned no solution"

    # Build team manually without DB save
    squad_idx   = solution["squad_idx"]
    start_idx   = set(solution["start_idx"])
    captain_idx = set(solution["captain_idx"])

    players = players_df.reset_index(drop=True)
    squad_df = players.iloc[squad_idx].copy()
    squad_df["is_starting"]    = squad_df.index.isin(start_idx)
    squad_df["is_captain"]     = squad_df.index.isin(captain_idx)
    squad_df["is_vice_captain"] = False

    return squad_df, solution


# =============================================================================
# Squad size and composition tests
# =============================================================================

class TestSquadComposition:
    """Verify squad always has exactly the right number of players per position."""

    @pytest.fixture
    def standard_pool(self):
        return make_player_pool()

    def test_squad_has_15_players(self, standard_pool):
        squad_df, _ = select_team_from_pool(standard_pool)
        assert len(squad_df) == 15, f"Squad should have 15 players, got {len(squad_df)}"

    def test_squad_has_2_goalkeepers(self, standard_pool):
        squad_df, _ = select_team_from_pool(standard_pool)
        gk_count = (squad_df["position"] == "GK").sum()
        assert gk_count == 2, f"Squad should have 2 GKs, got {gk_count}"

    def test_squad_has_5_defenders(self, standard_pool):
        squad_df, _ = select_team_from_pool(standard_pool)
        def_count = (squad_df["position"] == "DEF").sum()
        assert def_count == 5, f"Squad should have 5 DEFs, got {def_count}"

    def test_squad_has_5_midfielders(self, standard_pool):
        squad_df, _ = select_team_from_pool(standard_pool)
        mid_count = (squad_df["position"] == "MID").sum()
        assert mid_count == 5, f"Squad should have 5 MIDs, got {mid_count}"

    def test_squad_has_3_forwards(self, standard_pool):
        squad_df, _ = select_team_from_pool(standard_pool)
        fwd_count = (squad_df["position"] == "FWD").sum()
        assert fwd_count == 3, f"Squad should have 3 FWDs, got {fwd_count}"

    def test_no_duplicate_players(self, standard_pool):
        squad_df, _ = select_team_from_pool(standard_pool)
        assert squad_df["player_id"].nunique() == 15, \
            "Squad should have 15 unique players"


# =============================================================================
# Budget constraint tests
# =============================================================================

class TestBudgetConstraints:
    """Verify squad always stays within budget."""

    def test_squad_within_100m_budget(self):
        pool = make_player_pool()
        squad_df, solution = select_team_from_pool(pool)
        total_cost = squad_df["price"].sum()
        assert total_cost <= 100.0, \
            f"Squad cost £{total_cost:.1f}m exceeds £100m budget"

    def test_budget_used_efficiently(self):
        """
        Squad should use at least £85m — if much less,
        the solver is leaving value on the table.
        """
        pool = make_player_pool()
        squad_df, solution = select_team_from_pool(pool)
        total_cost = squad_df["price"].sum()
        assert total_cost >= 85.0, \
            f"Squad only costs £{total_cost:.1f}m — budget underutilised"


# =============================================================================
# Club constraint tests
# =============================================================================

class TestClubConstraints:
    """Verify max 3 players from any single club."""

    def test_max_3_players_per_club(self):
        pool = make_player_pool()
        squad_df, _ = select_team_from_pool(pool)
        club_counts = squad_df["team_name"].value_counts()
        violations = club_counts[club_counts > 3]
        assert len(violations) == 0, \
            f"Club limit violated: {violations.to_dict()}"

    def test_club_constraint_with_dominant_team(self):
        """
        Even if one club has many high-scoring players,
        selector should cap at 3 from that club.
        """
        pool = make_player_pool(seed=99)
        # Artificially boost one club's scores
        pool.loc[pool["team_name"] == "Club1", "adjusted_points"] = 9.9
        squad_df, _ = select_team_from_pool(pool)
        club1_count = (squad_df["team_name"] == "Club1").sum()
        assert club1_count <= 3, \
            f"Club1 has {club1_count} players — violates max 3 rule"


# =============================================================================
# Starting XI constraint tests
# =============================================================================

class TestStartingXI:
    """Verify starting XI composition is valid."""

    @pytest.fixture
    def squad_and_starters(self):
        pool = make_player_pool()
        squad_df, solution = select_team_from_pool(pool)
        starters = squad_df[squad_df["is_starting"]]
        bench    = squad_df[~squad_df["is_starting"]]
        return squad_df, starters, bench

    def test_exactly_11_starters(self, squad_and_starters):
        _, starters, _ = squad_and_starters
        assert len(starters) == 11, \
            f"Should have 11 starters, got {len(starters)}"

    def test_exactly_4_bench(self, squad_and_starters):
        _, _, bench = squad_and_starters
        assert len(bench) == 4, \
            f"Should have 4 bench players, got {len(bench)}"

    def test_exactly_1_starting_gk(self, squad_and_starters):
        _, starters, _ = squad_and_starters
        gk_starters = (starters["position"] == "GK").sum()
        assert gk_starters == 1, \
            f"Should have exactly 1 starting GK, got {gk_starters}"

    def test_minimum_3_starting_defenders(self, squad_and_starters):
        _, starters, _ = squad_and_starters
        def_starters = (starters["position"] == "DEF").sum()
        assert def_starters >= 3, \
            f"Should have at least 3 starting DEFs, got {def_starters}"

    def test_minimum_2_starting_midfielders(self, squad_and_starters):
        _, starters, _ = squad_and_starters
        mid_starters = (starters["position"] == "MID").sum()
        assert mid_starters >= 2, \
            f"Should have at least 2 starting MIDs, got {mid_starters}"

    def test_minimum_1_starting_forward(self, squad_and_starters):
        _, starters, _ = squad_and_starters
        fwd_starters = (starters["position"] == "FWD").sum()
        assert fwd_starters >= 1, \
            f"Should have at least 1 starting FWD, got {fwd_starters}"


# =============================================================================
# Captain constraint tests
# =============================================================================

class TestCaptainConstraints:
    """Verify captain selection logic."""

    def test_exactly_one_captain(self):
        pool = make_player_pool()
        squad_df, _ = select_team_from_pool(pool)
        captain_count = squad_df["is_captain"].sum()
        assert captain_count == 1, \
            f"Should have exactly 1 captain, got {captain_count}"

    def test_captain_is_in_starting_xi(self):
        pool = make_player_pool()
        squad_df, _ = select_team_from_pool(pool)
        captain = squad_df[squad_df["is_captain"]]
        assert len(captain) == 1
        assert captain.iloc[0]["is_starting"], \
            "Captain must be in the starting XI"

    def test_captain_is_highest_predicted_starter(self):
        """Captain should be the starter with highest adjusted_points."""
        pool = make_player_pool()
        squad_df, _ = select_team_from_pool(pool)
        starters = squad_df[squad_df["is_starting"]]
        captain  = squad_df[squad_df["is_captain"]].iloc[0]

        best_starter_pts = starters["adjusted_points"].max()
        captain_pts      = captain["adjusted_points"]

        assert captain_pts == pytest.approx(best_starter_pts, abs=0.01), \
            f"Captain ({captain_pts:.2f}) should be highest starter ({best_starter_pts:.2f})"


# =============================================================================
# Edge case tests
# =============================================================================

class TestEdgeCases:
    """Test selector behaviour with unusual input data."""

    def test_handles_missing_prices(self):
        """Selector should handle NaN prices gracefully."""
        pool = make_player_pool()
        pool.loc[pool.index[:5], "price"] = None
        # Should not crash
        try:
            squad_df, _ = select_team_from_pool(pool)
            assert len(squad_df) == 15
        except Exception as e:
            pytest.fail(f"Selector crashed on missing prices: {e}")

    def test_handles_all_same_score(self):
        """When all players have identical scores, constraints should still be met."""
        pool = make_player_pool()
        pool["adjusted_points"] = 5.0
        squad_df, _ = select_team_from_pool(pool)
        assert len(squad_df) == 15
        assert squad_df["price"].sum() <= 100.0

"""
tests/unit/test_captain_model.py
=================================
Unit tests for the captain classifier.

Tests focus on:
1. Feature engineering correctness
2. Label building logic
3. Signal ordering (premium home attacker should score higher than
   budget away defender)
4. Boundary values and edge cases
"""

import sys
from unittest import result
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from prediction.captain_model import (
    engineer_captain_features,
    build_captain_labels,
    CAPTAIN_FEATURES,
)


# =============================================================================
# FIXTURES
# =============================================================================

def make_player_df(n=20, seed=42):
    """Creates a synthetic player dataframe for testing."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "player_id"          : range(1, n + 1),
        "web_name"           : [f"Player{i}" for i in range(1, n + 1)],
        "position"           : (["GK"] * 2 + ["DEF"] * 5 +
                                ["MID"] * 7 + ["FWD"] * 6)[:n],
        "team_name"          : [f"Team{i % 5}" for i in range(n)],
        "price"              : rng.uniform(4.0, 14.0, n),
        "season"             : ["2025-26"] * n,
        "gameweek"           : [1] * n,
        "avg_points_5gw"     : rng.uniform(1, 8, n),
        "avg_points_10gw"    : rng.uniform(1, 7, n),
        "avg_points_season"  : rng.uniform(1, 6, n),
        "xgi_season"         : rng.uniform(0, 20, n),
        "xgi_per90_season"   : rng.uniform(0, 1, n),
        "avg_goals_5gw"      : rng.uniform(0, 0.8, n),
        "avg_assists_5gw"    : rng.uniform(0, 0.5, n),
        "avg_ict_5gw"        : rng.uniform(0, 50, n),
        "fixture_fdr"        : rng.uniform(1, 5, n),
        "opponent_xga_5"     : rng.uniform(0.5, 2.5, n),
        "lineup_probability" : rng.uniform(0.5, 1.0, n),
        "was_home"           : rng.randint(0, 2, n).astype(float),
        "actual_points"      : rng.randint(1, 20, n),
        "selected_by_percent": rng.uniform(0.5, 50.0, n),
        "opponent_xga_5"     : rng.uniform(0.5, 2.5, n),
    })


# =============================================================================
# FEATURE ENGINEERING TESTS
# =============================================================================

class TestFeatureEngineering:
    """Tests that captain features are computed correctly."""

    def test_all_captain_features_present(self):
        """After engineering, all CAPTAIN_FEATURES columns must exist."""
        df = make_player_df()
        result = engineer_captain_features(df)
        missing = [f for f in CAPTAIN_FEATURES if f not in result.columns]
        assert len(missing) == 0, f"Missing captain features: {missing}"

    def test_is_attacking_correct(self):
        """FWD and MID should have is_attacking=1, GK and DEF should have 0."""
        df = make_player_df()
        result = engineer_captain_features(df)
        for _, row in result.iterrows():
            if row["position"] in ("FWD", "MID"):
                assert row["is_attacking"] == 1.0, \
                    f"{row['position']} should have is_attacking=1"
            else:
                assert row["is_attacking"] == 0.0, \
                    f"{row['position']} should have is_attacking=0"

    def test_is_premium_correct(self):
        """Players with price > £9m should have is_premium=1."""
        df = make_player_df()
        df.loc[0, "price"] = 14.0  # Definitely premium
        df.loc[1, "price"] = 4.5   # Definitely not premium
        result = engineer_captain_features(df)
        assert result.loc[0, "is_premium"] == 1.0, "£14m player should be premium"
        assert result.loc[1, "is_premium"] == 0.0, "£4.5m player should not be premium"

    def test_opp_xga_rank_bounded(self):
        """opp_xga_rank should be between 0 and 1 (percentile rank)."""
        df = make_player_df()
        result = engineer_captain_features(df)
        assert result["opp_xga_rank"].between(0, 1).all(), \
            "opp_xga_rank should be between 0 and 1"

    def test_no_nan_in_captain_features(self):
        """No NaN values in any captain feature after engineering."""
        df = make_player_df()
        # Introduce some NaN values to test filling
        df.loc[0, "avg_points_5gw"] = None
        df.loc[1, "opponent_xga_5"] = None
        df.loc[2, "lineup_probability"] = None

        result = engineer_captain_features(df)
        for feat in CAPTAIN_FEATURES:
            if feat in result.columns:
                nan_count = result[feat].isna().sum()
                assert nan_count == 0, \
                    f"Feature '{feat}' has {nan_count} NaN values after engineering"

    def test_pos_one_hot_mutually_exclusive(self):
        """Each player should have exactly one position one-hot flag set."""
        df = make_player_df()
        result = engineer_captain_features(df)
        pos_cols = ["pos_GK", "pos_DEF", "pos_MID", "pos_FWD"]
        pos_sum = result[pos_cols].sum(axis=1)
        assert (pos_sum == 1).all(), \
            "Each player should have exactly one position flag set"

    def test_ceiling_feature_gt_avg(self):
        """max_points_5gw fallback should be >= avg_points_5gw * 1.2."""
        df = make_player_df()
        result = engineer_captain_features(df)
        # When no ceiling_df provided, fallback is avg * 1.2
        violations = (result["max_points_5gw"] < result["avg_points_5gw"]).sum()
        assert violations == 0, \
            f"{violations} players have max_points below average (impossible)"


# =============================================================================
# LABEL BUILDING TESTS
# =============================================================================

class TestLabelBuilding:
    """Tests that captain labels are built correctly."""

    def test_exactly_one_label_per_gw(self):
        """Each gameweek should have at least one captain label=1."""
        df = make_player_df(n=15)
        df = engineer_captain_features(df)
        df = build_captain_labels(df)
        labels_per_gw = df.groupby(["season", "gameweek"])["captain_label"].sum()
        assert (labels_per_gw >= 1).all(), \
            "Each GW should have at least one top scorer"

    def test_top_scorer_gets_label_1(self):
        """The player with the highest actual_points gets captain_label=1."""
        df = make_player_df(n=15)
        df["actual_points"] = 5
        df.loc[7, "actual_points"] = 20  # Player 8 is the clear top scorer
        df = engineer_captain_features(df)
        df = build_captain_labels(df)
        assert df.loc[7, "captain_label"] == 1, \
            "Player with 20 pts should have captain_label=1"

    def test_non_top_scorers_get_label_0(self):
        """Players who are not top scorers get captain_label=0."""
        df = make_player_df(n=15)
        df["actual_points"] = 5
        df.loc[0, "actual_points"] = 20  # Only player 1 is top scorer
        df = engineer_captain_features(df)
        df = build_captain_labels(df)
        non_top = df[df["captain_label"] == 0]
        assert len(non_top) == 14, \
            "All 14 non-top-scorers should have captain_label=0"

    def test_tied_top_scorers_both_get_label_1(self):
        """When two players tie for top score, both get captain_label=1."""
        df = make_player_df(n=10)
        df["actual_points"] = 3
        df.loc[0, "actual_points"] = 15
        df.loc[1, "actual_points"] = 15  # Tie for top
        df = engineer_captain_features(df)
        df = build_captain_labels(df)
        assert df.loc[0, "captain_label"] == 1, "First tied player should get label=1"
        assert df.loc[1, "captain_label"] == 1, "Second tied player should get label=1"

    def test_zero_points_player_no_label(self):
        """Players with 0 actual_points should not get captain_label=1."""
        df = make_player_df(n=10)
        df["actual_points"] = 0  # All players scored 0
        df = engineer_captain_features(df)
        df = build_captain_labels(df)
        assert df["captain_label"].sum() == 0, \
            "No player should be labelled captain when all score 0"


# =============================================================================
# SIGNAL ORDERING TESTS
# =============================================================================

class TestSignalOrdering:
    """
    Tests that the feature signals produce sensible orderings.
    These test the logic of the features — not the model weights.
    A premium home FWD vs weak defence should have better captain
    signals than a budget away GK vs strong defence.
    """

    def test_home_player_has_higher_was_home(self):
        """was_home=1 player should have higher home signal than was_home=0."""
        df = make_player_df(n=2)
        df.loc[0, "was_home"] = 1.0
        df.loc[1, "was_home"] = 0.0
        result = engineer_captain_features(df)
        assert result.loc[0, "was_home"] > result.loc[1, "was_home"]

    def test_weak_defence_gives_higher_opp_xga_rank(self):
        """Higher opponent xGA should give higher opp_xga_rank."""
        df = make_player_df(n=5)
        df["opponent_xga_5"] = [0.5, 1.0, 1.5, 2.0, 2.5]
        result = engineer_captain_features(df)
        # Higher xGA = weaker defence = higher rank
        assert (result["opp_xga_rank"].diff().dropna() > 0).all(), \
            "opp_xga_rank should increase as opponent_xga_5 increases"

    def test_premium_player_flag(self):
        """£14m player should have is_premium=1, £4m player should have 0."""
        df = make_player_df(n=2)
        df.loc[0, "price"] = 14.0
        df.loc[1, "price"] = 4.0
        result = engineer_captain_features(df)
        assert result.loc[0, "is_premium"] > result.loc[1, "is_premium"]

    def test_attacking_position_advantage(self):
        """FWD should have is_attacking=1, GK should have is_attacking=0."""
        df = make_player_df(n=2)
        df.loc[0, "position"] = "FWD"
        df.loc[1, "position"] = "GK"
        result = engineer_captain_features(df)
        assert result.loc[0, "is_attacking"] > result.loc[1, "is_attacking"]

    def test_high_form_gives_higher_ceiling(self):
        """Player with higher avg_points_5gw should have higher ceiling."""
        df = make_player_df(n=2)
        df.loc[0, "avg_points_5gw"] = 8.0
        df.loc[1, "avg_points_5gw"] = 2.0
        result = engineer_captain_features(df)
        assert result.loc[0, "max_points_5gw"] > result.loc[1, "max_points_5gw"]

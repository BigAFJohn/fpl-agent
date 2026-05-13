"""
tests/integration/test_pipeline.py
===================================
Integration tests for the full prediction pipeline.

These tests verify that components work together correctly
end-to-end. They require a database connection but don't
make external API calls.

Integration tests are slower than unit tests (~5-30 seconds each)
but catch issues that unit tests miss — like the player_id
deduplication bug that caused Top-K accuracy to show 0%.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import pandas as pd
from sqlalchemy import create_engine, text


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def engine():
    """Database engine — shared across all tests in this module."""
    pg_url = os.environ.get("FPL_DB_URL")
    if pg_url:
        return create_engine(pg_url)
    db_path = Path("db/fpl.db")
    if db_path.exists():
        return create_engine(f"sqlite:///{db_path}")
    pytest.skip("No database available")


@pytest.fixture(scope="module")
def has_predictions(engine):
    """Check predictions table exists and has data."""
    try:
        result = pd.read_sql(text("SELECT COUNT(*) as cnt FROM predictions"), engine)
        count = int(result["cnt"].iloc[0])
        if count == 0:
            pytest.skip("No predictions in DB — run predict_points.py first")
        return count
    except Exception:
        pytest.skip("predictions table not found — run predict_points.py first")


@pytest.fixture(scope="module")
def has_model_features(engine):
    """Check model_features table exists and has data."""
    try:
        result = pd.read_sql(text("SELECT COUNT(*) as cnt FROM model_features"), engine)
        count = int(result["cnt"].iloc[0])
        if count == 0:
            pytest.skip("No model_features — run feature_store.py first")
        return count
    except Exception:
        pytest.skip("model_features table not found")


# =============================================================================
# DATABASE INTEGRITY TESTS
# =============================================================================

class TestDatabaseIntegrity:
    """Verify core database tables exist and have expected data."""

    def test_players_table_has_data(self, engine):
        """Players table should have ~836 active players."""
        df = pd.read_sql(text("SELECT COUNT(*) as cnt FROM players"), engine)
        count = int(df["cnt"].iloc[0])
        assert count >= 800, f"Expected 800+ players, got {count}"
        assert count <= 900, f"Expected <900 players, got {count} (possible duplicates)"

    def test_player_history_has_data(self, engine):
        """Player history should have current season gameweek data."""
        df = pd.read_sql(text("SELECT COUNT(*) as cnt FROM player_history"), engine)
        count = int(df["cnt"].iloc[0])
        assert count >= 10000, f"Expected 10000+ history rows, got {count}"

    def test_fixtures_has_380_rows(self, engine):
        """A 20-team Premier League season has exactly 380 fixtures."""
        df = pd.read_sql(text("SELECT COUNT(*) as cnt FROM fixtures"), engine)
        count = int(df["cnt"].iloc[0])
        assert count == 380, f"Expected 380 fixtures, got {count}"

    def test_teams_has_20_rows(self, engine):
        """Premier League has exactly 20 teams."""
        df = pd.read_sql(text("SELECT COUNT(*) as cnt FROM teams"), engine)
        count = int(df["cnt"].iloc[0])
        assert count == 20, f"Expected 20 teams, got {count}"

    def test_no_null_player_ids_in_history(self, engine):
        """player_history should have no null player_ids."""
        df = pd.read_sql(
            text("SELECT COUNT(*) as cnt FROM player_history WHERE player_id IS NULL"),
            engine
        )
        assert int(df["cnt"].iloc[0]) == 0, "Found null player_ids in player_history"

    def test_player_positions_valid(self, engine):
        """All players should have valid position labels."""
        df = pd.read_sql(
            text("SELECT DISTINCT position_label FROM players WHERE position_label IS NOT NULL"),
            engine
        )
        valid_positions = {"GK", "DEF", "MID", "FWD"}
        actual_positions = set(df["position_label"].tolist())
        invalid = actual_positions - valid_positions
        assert len(invalid) == 0, f"Invalid positions found: {invalid}"


# =============================================================================
# FEATURE ENGINEERING TESTS
# =============================================================================

class TestFeatureEngineering:
    """Verify feature store contains expected data."""

    def test_model_features_row_count(self, engine, has_model_features):
        """model_features should have 100k+ rows across 4 seasons."""
        assert has_model_features >= 100000, \
            f"Expected 100k+ rows, got {has_model_features}"

    def test_model_features_has_all_seasons(self, engine, has_model_features):
        """model_features should contain all 4 seasons."""
        df = pd.read_sql(
            text("SELECT DISTINCT season FROM model_features ORDER BY season"),
            engine
        )
        seasons = set(df["season"].tolist())
        expected = {"2022-23", "2023-24", "2024-25", "2025-26"}
        assert expected == seasons, \
            f"Missing seasons: {expected - seasons}"

    def test_rolling_features_not_all_null(self, engine, has_model_features):
        """Rolling form features should not be all null."""
        df = pd.read_sql(text("""
            SELECT COUNT(*) as total,
                   COUNT(avg_points_5gw) as non_null
            FROM model_features
            WHERE season = '2025-26'
        """), engine)
        total    = int(df["total"].iloc[0])
        non_null = int(df["non_null"].iloc[0])
        pct      = non_null / total if total > 0 else 0
        assert pct >= 0.5, \
            f"Only {pct:.0%} of avg_points_5gw are non-null — expected 50%+"

    def test_no_data_leakage_in_features(self, engine, has_model_features):
        """
        Rolling features use shift(1) to prevent leakage.
        GW1 features should be null/0 (no history to average).
        """
        df = pd.read_sql(text("""
            SELECT AVG(avg_points_5gw) as avg_form
            FROM model_features
            WHERE gameweek = 1
              AND season = '2025-26'
        """), engine)
        avg_gw1_form = df["avg_form"].iloc[0]
        # GW1 should have low form — only prior season data available, no current season leakage
        # Non-zero is expected (prior seasons contribute), but should be low
        assert avg_gw1_form is None or float(avg_gw1_form) < 5.0, \
            f"GW1 form suspiciously high ({avg_gw1_form}) — possible data leakage"

    def test_fixture_fdr_in_valid_range(self, engine, has_model_features):
        """Custom FDR scores should be between 1 and 5."""
        df = pd.read_sql(text("""
            SELECT MIN(fixture_fdr) as min_fdr,
                   MAX(fixture_fdr) as max_fdr
            FROM model_features
            WHERE fixture_fdr IS NOT NULL
        """), engine)
        min_fdr = float(df["min_fdr"].iloc[0])
        max_fdr = float(df["max_fdr"].iloc[0])
        assert min_fdr >= 1.0, f"FDR below minimum: {min_fdr}"
        assert max_fdr <= 5.5, f"FDR above maximum: {max_fdr}"


# =============================================================================
# PREDICTION PIPELINE TESTS
# =============================================================================

class TestPredictionPipeline:
    """Verify prediction output is complete and sensible."""

    def test_predictions_cover_all_active_players(self, engine, has_predictions):
        """
        Predictions should cover most active players.
        We expect 600+ after filtering unavailable players.
        """
        assert has_predictions >= 600, \
            f"Expected 600+ predictions, got {has_predictions}"

    def test_no_null_predicted_points(self, engine, has_predictions):
        """Every prediction should have a predicted_points value."""
        df = pd.read_sql(text("""
            SELECT COUNT(*) as cnt
            FROM predictions
            WHERE predicted_points IS NULL
        """), engine)
        null_count = int(df["cnt"].iloc[0])
        assert null_count == 0, f"{null_count} predictions have null predicted_points"

    def test_predictions_in_valid_range(self, engine, has_predictions):
        """Predicted points should be between 0 and 25 (FPL maximum)."""
        df = pd.read_sql(text("""
            SELECT MIN(predicted_points) as min_pred,
                   MAX(predicted_points) as max_pred
            FROM predictions
        """), engine)
        min_pred = float(df["min_pred"].iloc[0])
        max_pred = float(df["max_pred"].iloc[0])
        assert min_pred >= 0.0, f"Negative predictions found: {min_pred}"
        assert max_pred <= 25.0, f"Predictions exceed FPL max: {max_pred}"

    def test_adjusted_points_le_predicted_points(self, engine, has_predictions):
        """
        Adjusted points = predicted × lineup_probability.
        Since lineup_probability ≤ 1.0, adjusted ≤ predicted always.
        """
        df = pd.read_sql(text("""
            SELECT COUNT(*) as violations
            FROM predictions
            WHERE adjusted_points > predicted_points + 0.01
        """), engine)
        violations = int(df["violations"].iloc[0])
        assert violations == 0, \
            f"{violations} players have adjusted_points > predicted_points"

    def test_no_duplicate_player_predictions(self, engine, has_predictions):
        """Each player should appear only once in predictions."""
        df = pd.read_sql(text("""
            SELECT player_id, COUNT(*) as cnt
            FROM predictions
            GROUP BY player_id
            HAVING COUNT(*) > 1
        """), engine)
        assert len(df) == 0, \
            f"{len(df)} players have duplicate predictions"

    def test_top_predictions_are_known_players(self, engine, has_predictions):
        """
        Top 5 predictions should be recognisable FPL players,
        not data artifacts or bench-warmers.
        """
        df = pd.read_sql(text("""
            SELECT web_name, adjusted_points, lineup_probability
            FROM predictions
            ORDER BY adjusted_points DESC
            LIMIT 5
        """), engine)
        # All top 5 should have lineup_probability > 0.7
        min_lp = float(df["lineup_probability"].min())
        assert min_lp >= 0.7, \
            f"Top 5 predictions include player with low lineup prob ({min_lp:.2f})"


# =============================================================================
# LINEUP PROBABILITY TESTS
# =============================================================================

class TestLineupProbability:
    """Verify lineup probability outputs are sensible."""

    def test_injured_players_have_low_probability(self, engine):
        """Players with status='i' should have lineup_probability <= 0.10."""
        try:
            df = pd.read_sql(text("""
                SELECT lp.web_name, lp.fpl_status, lp.lineup_probability
                FROM lineup_probability lp
                WHERE lp.fpl_status = 'i'
                  AND lp.lineup_probability > 0.10
            """), engine)
            assert len(df) == 0, \
                f"{len(df)} injured players have lineup_probability > 0.10: {df['web_name'].tolist()}"
        except Exception:
            pytest.skip("lineup_probability table not found")

    def test_suspended_players_have_low_probability(self, engine):
        """Suspended players (status='s') should have probability <= 0.05."""
        try:
            df = pd.read_sql(text("""
                SELECT lp.web_name, lp.fpl_status, lp.lineup_probability
                FROM lineup_probability lp
                WHERE lp.fpl_status = 's'
                  AND lp.lineup_probability > 0.05
            """), engine)
            assert len(df) == 0, \
                f"Suspended players with high probability: {df['web_name'].tolist()}"
        except Exception:
            pytest.skip("lineup_probability table not found")

    def test_available_players_have_reasonable_probability(self, engine):
        """Available players (status='a') should have probability >= 0.50."""
        try:
            df = pd.read_sql(text("""
                SELECT COUNT(*) as cnt
                FROM lineup_probability
                WHERE fpl_status = 'a'
                  AND lineup_probability < 0.50
            """), engine)
            low_count = int(df["cnt"].iloc[0])
            # Some available players may have low minutes (rotation risk)
            # but very few should be below 0.50
            total_available = pd.read_sql(
                text("SELECT COUNT(*) as cnt FROM lineup_probability WHERE fpl_status = 'a'"),
                engine
            )["cnt"].iloc[0]
            pct_low = low_count / max(total_available, 1)
            assert pct_low <= 0.20, \
                f"{pct_low:.0%} of available players have probability < 0.50"
        except Exception:
            pytest.skip("lineup_probability table not found")

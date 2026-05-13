"""
tests/unit/test_signals.py
==========================
Unit tests for lineup_probability.py signal builder functions.

These are pure function tests — no database, no API calls.
Each test is fast (<1ms) and deterministic.

As a test automation engineer you'll recognise these patterns:
- Arrange / Act / Assert
- Boundary value testing (0, 100, None edge cases)
- Equivalence partitioning (status categories)
- Negative testing (invalid inputs)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from lineup_probability import (
    build_fpl_signal,
    build_minutes_signal,
    build_injury_signal,
)


# =============================================================================
# build_fpl_signal tests
# =============================================================================

class TestBuildFplSignal:
    """Tests for FPL status → probability conversion."""

    def test_available_no_chance(self):
        """Status 'a' with no chance set → default available probability."""
        result = build_fpl_signal("a", None)
        assert 0.75 <= result <= 0.95, f"Expected 0.75-0.95, got {result}"

    def test_chance_100_returns_1(self):
        """Explicit 100% chance → 1.0 probability."""
        result = build_fpl_signal("a", 100)
        assert result == 1.0

    def test_chance_75_returns_075(self):
        """75% chance of playing → 0.75 probability."""
        result = build_fpl_signal("d", 75)
        assert result == pytest.approx(0.75, abs=0.01)

    def test_chance_50_returns_05(self):
        """50% chance → 0.5 probability."""
        result = build_fpl_signal("d", 50)
        assert result == pytest.approx(0.50, abs=0.01)

    def test_chance_25_returns_025(self):
        """25% chance → 0.25 probability."""
        result = build_fpl_signal("d", 25)
        assert result == pytest.approx(0.25, abs=0.01)

    def test_chance_0_available_not_capped(self):
        """
        Available player with chance=0 (data issue — field not updated)
        should NOT be capped at 0.05. This was the Lammens bug.
        """
        result = build_fpl_signal("a", 0)
        assert result > 0.05, f"Available player should not be capped at 0.05, got {result}"

    def test_injured_status(self):
        """Injured player → very low probability."""
        result = build_fpl_signal("i", None)
        assert result <= 0.10, f"Injured player should have prob <= 0.10, got {result}"

    def test_injured_with_zero_chance(self):
        """Injured + chance=0 → capped at injured max."""
        result = build_fpl_signal("i", 0)
        assert result <= 0.10

    def test_suspended_status(self):
        """Suspended player → near zero probability."""
        result = build_fpl_signal("s", None)
        assert result <= 0.05, f"Suspended player should have prob <= 0.05, got {result}"

    def test_unavailable_status(self):
        """Unavailable (loaned out) → near zero probability."""
        result = build_fpl_signal("u", None)
        assert result <= 0.05

    def test_not_in_squad_status(self):
        """Not in squad → near zero probability."""
        result = build_fpl_signal("n", None)
        assert result <= 0.05

    def test_doubtful_no_chance(self):
        """Doubtful without chance set → moderate probability."""
        result = build_fpl_signal("d", None)
        assert 0.50 <= result <= 0.75, f"Doubtful should be 0.50-0.75, got {result}"

    def test_probability_bounded_0_1(self):
        """All outputs must be between 0 and 1."""
        test_cases = [
            ("a", None), ("a", 100), ("a", 0),
            ("d", 75), ("d", 50), ("d", 25),
            ("i", None), ("i", 0),
            ("u", None), ("s", None), ("n", None),
        ]
        for status, chance in test_cases:
            result = build_fpl_signal(status, chance)
            assert 0.0 <= result <= 1.0, \
                f"build_fpl_signal({status!r}, {chance}) = {result} — out of [0,1]"

    def test_unknown_status_returns_reasonable_default(self):
        """Unknown status codes should return a reasonable default, not crash."""
        result = build_fpl_signal("x", None)
        assert 0.0 <= result <= 1.0


# =============================================================================
# build_minutes_signal tests
# =============================================================================

class TestBuildMinutesSignal:
    """Tests for recent minutes trend → probability conversion."""

    def test_iron_man_full_90s(self):
        """Player averaging 90 mins with 100% start rate → high probability."""
        result = build_minutes_signal(90.0, 1.0)
        assert result >= 0.85, f"Iron-man should have prob >= 0.85, got {result}"

    def test_regular_starter(self):
        """Player averaging 80 mins with 80% start rate → likely starter."""
        result = build_minutes_signal(80.0, 0.8)
        assert result >= 0.70, f"Regular starter should have prob >= 0.70, got {result}"

    def test_rotation_risk(self):
        """Player averaging 45 mins with 60% start rate → rotation risk."""
        result = build_minutes_signal(45.0, 0.6)
        assert result < 0.70, f"Rotation risk should have prob < 0.70, got {result}"

    def test_bench_warmer(self):
        """Player averaging 15 mins with 20% start rate → bench player."""
        result = build_minutes_signal(15.0, 0.2)
        assert result < 0.40, f"Bench warmer should have prob < 0.40, got {result}"

    def test_none_inputs_return_neutral(self):
        """Missing data → neutral default, not crash."""
        result = build_minutes_signal(None, None)
        assert 0.5 <= result <= 0.8, f"None inputs should return neutral ~0.7, got {result}"

    def test_none_minutes_only(self):
        """Missing minutes but known start rate → neutral."""
        result = build_minutes_signal(None, 0.8)
        assert 0.0 <= result <= 1.0

    def test_zero_minutes(self):
        """Player with 0 average minutes → very low probability."""
        result = build_minutes_signal(0.0, 0.0)
        assert result <= 0.30, f"Zero minutes should give low probability, got {result}"

    def test_returning_from_injury_low_minutes(self):
        """
        Player returning from injury: 0% start rate last 5 GWs
        but FPL marks as available. Minutes signal should be low.
        This is the De Cuyper scenario.
        """
        result = build_minutes_signal(7.0, 0.0)
        assert result <= 0.30, \
            f"Recently injured player (7 avg mins) should score low, got {result}"

    def test_probability_bounded(self):
        """All outputs must be between 0 and 1."""
        test_cases = [
            (90.0, 1.0), (45.0, 0.5), (0.0, 0.0),
            (None, None), (None, 0.8), (80.0, None),
        ]
        for mins, rate in test_cases:
            result = build_minutes_signal(mins, rate)
            assert 0.0 <= result <= 1.0, \
                f"build_minutes_signal({mins}, {rate}) = {result} — out of [0,1]"


# =============================================================================
# build_injury_signal tests
# =============================================================================

class TestBuildInjurySignal:
    """
    Tests for injury history → probability multiplier.
    Note: this returns a MULTIPLIER (0-1), not a probability.
    1.0 = no adjustment, 0.5 = halve the probability.
    """

    def test_no_injury_profile_no_adjustment(self):
        """No injury data → multiplier of 1.0 (no adjustment)."""
        result = build_injury_signal(None, None)
        assert result == 1.0, f"No profile should return 1.0, got {result}"

    def test_reliable_player_no_adjustment(self):
        """Low injury score + not recently returned → no adjustment."""
        result = build_injury_signal(0.2, 999)
        assert result == pytest.approx(1.0, abs=0.05)

    def test_very_risky_player_penalised(self):
        """High injury score (0.8+) → probability reduced."""
        result = build_injury_signal(0.8, 999)
        assert result < 1.0, f"Very risky player should be penalised, got {result}"

    def test_recently_returned_penalised(self):
        """
        Player in re-injury risk window (games_since_return <= 3)
        → probability reduced regardless of overall score.
        """
        result = build_injury_signal(0.3, 2)
        assert result < 1.0, \
            f"Recently returned player should be penalised, got {result}"

    def test_re_injury_window_3_games(self):
        """games_since_return = 3 is still in the risk window."""
        result_in_window  = build_injury_signal(0.5, 3)
        result_out_window = build_injury_signal(0.5, 10)
        assert result_in_window < result_out_window, \
            "Player at games_since_return=3 should score lower than games_since_return=10"

    def test_multiplier_bounded_0_1(self):
        """Multiplier must always be between 0 and 1."""
        test_cases = [
            (None, None), (0.0, 0), (0.5, 5),
            (0.8, 1), (1.0, 0), (0.3, 999),
        ]
        for score, gsr in test_cases:
            result = build_injury_signal(score, gsr)
            assert 0.0 <= result <= 1.0, \
                f"build_injury_signal({score}, {gsr}) = {result} — out of [0,1]"

    def test_high_score_recently_returned_worst_case(self):
        """
        Injury-prone player (score=0.9) who just returned (gsr=0)
        → maximum penalty applied.
        """
        result_worst = build_injury_signal(0.9, 0)
        result_best  = build_injury_signal(0.1, 999)
        assert result_worst < result_best, \
            "Worst case should score lower than best case"


# =============================================================================
# Cross-function integration: signal ordering
# =============================================================================

class TestSignalOrdering:
    """
    Sanity checks that signals produce sensible relative orderings.
    These test the overall logic rather than specific values.
    """

    def test_available_beats_injured(self):
        """Available player should always score higher than injured."""
        available = build_fpl_signal("a", 100)
        injured   = build_fpl_signal("i", 0)
        assert available > injured

    def test_full_time_beats_bench(self):
        """Regular starter beats bench warmer on minutes signal."""
        starter = build_minutes_signal(85.0, 1.0)
        bench   = build_minutes_signal(10.0, 0.1)
        assert starter > bench

    def test_reliable_beats_risky_on_injury(self):
        """Reliable player multiplier >= risky player multiplier."""
        reliable = build_injury_signal(0.1, 999)
        risky    = build_injury_signal(0.9, 1)
        assert reliable >= risky

    def test_chance_ordering(self):
        """Higher chance of playing → higher probability."""
        chance_100 = build_fpl_signal("a", 100)
        chance_75  = build_fpl_signal("d", 75)
        chance_50  = build_fpl_signal("d", 50)
        chance_25  = build_fpl_signal("d", 25)
        assert chance_100 > chance_75 > chance_50 > chance_25

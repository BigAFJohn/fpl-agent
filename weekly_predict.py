"""
weekly_predict.py  —  Phase 4, Component 3: Weekly Prediction Runner
=====================================================================

CONCEPTS TO UNDERSTAND BEFORE READING THIS SCRIPT
--------------------------------------------------

1. What this script does
   This is the single entry point for generating weekly FPL predictions.
   It orchestrates the full prediction pipeline in the correct order:

   Data refresh → Features → Predictions → Team Selection → Report

   Run this every week before the FPL deadline to get your recommended
   team for the upcoming gameweek.

2. When to run it
   - Thursday/Friday after the previous gameweek completes
   - After any major injury news (re-run with --no-collect flag)
   - Before the deadline (usually Friday 11:30am or Saturday 11:30am)

   The script checks which gameweek is current and targets predictions
   at the next gameweek automatically.

3. The pipeline steps
   Step 1: Data collection (optional, ~8 mins)
     - Fetches latest FPL API data: player prices, injuries, form
     - Fetches Understat xG data for current season
     - Runs ETL to clean and validate
     - Runs news monitor for injury signals
     - Rebuilds injury profiles

   Step 2: Feature engineering (~45s)
     - Rebuilds rolling form features from fresh data
     - Recomputes fixture difficulty from latest xGA
     - Recomputes lineup probabilities from latest signals
     - Consolidates into model_features table

   Step 3: Prediction (~3s)
     - Retrains XGBoost on latest data
     - Generates point predictions for upcoming GW
     - Applies lineup probability adjustment

   Step 4: Team selection (~2s)
     - Runs LP optimisation for optimal 15-player squad
     - Outputs starting XI, bench, captain

   Step 5: Report
     - Prints full recommended team
     - Shows top picks by position
     - Highlights injury concerns
     - Shows key fixture matchups

4. Flags
   --no-collect   Skip data collection (use existing data)
   --no-train     Skip model retraining (use saved model)
   --gameweek N   Target a specific gameweek (default: current+1)
   --budget N     Override budget in £m (default: 100.0)
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"predict_{datetime.now().strftime('%Y%m%d_%H%M')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )
    return logging.getLogger("weekly_predict")


# =============================================================================
# PIPELINE STEPS
# =============================================================================

def step_collect(logger):
    """Refreshes all data from FPL API, Understat, and news sources."""
    logger.info("=" * 50)
    logger.info("Step 1/5: Data collection")
    logger.info("=" * 50)

    steps = [
        ("FPL collector",    "collectors.fpl_collector",    "run_collection",      {"max_players": None}),
        ("Understat scraper","collectors.understat_scraper", "run_understat_collection", {}),
        ("ETL pipeline",     "etl_pipeline",                "run_etl",             {}),
        ("News monitor",     "collectors.news_monitor",     "run_news_monitor",    {}),
        ("Injury tracker",   "collectors.injury_tracker",   "run_injury_tracker",  {}),
    ]

    for name, module_path, func_name, kwargs in steps:
        try:
            import importlib
            module = importlib.import_module(module_path)
            func   = getattr(module, func_name)
            func(**kwargs)
            logger.info(f"  ✓ {name}")
        except Exception as e:
            logger.error(f"  ✗ {name} failed: {e}")
            return False

    return True


def step_features(logger):
    """Rebuilds all feature tables from fresh data."""
    logger.info("=" * 50)
    logger.info("Step 2/5: Feature engineering")
    logger.info("=" * 50)

    steps = [
        ("Form features",      "feature_engineering", "run_feature_engineering"),
        ("Fixture difficulty", "fixture_difficulty",  "run_fixture_difficulty"),
        ("Lineup probability", "lineup_probability",  "run_lineup_probability"),
        ("Feature store",      "feature_store",       "run_feature_store"),
    ]

    for name, module_path, func_name in steps:
        try:
            import importlib
            module = importlib.import_module(module_path)
            func   = getattr(module, func_name)
            func()
            logger.info(f"  ✓ {name}")
        except Exception as e:
            logger.error(f"  ✗ {name} failed: {e}")
            return False

    return True


def step_predict(logger):
    """Trains XGBoost model and generates predictions."""
    logger.info("=" * 50)
    logger.info("Step 3/5: Generating predictions")
    logger.info("=" * 50)

    try:
        from predict_points import run_predict_points
        run_predict_points()
        logger.info("  ✓ Predictions generated")
        return True
    except Exception as e:
        logger.error(f"  ✗ Prediction failed: {e}")
        return False


def step_select(logger):
    """Runs LP optimisation to select optimal team."""
    logger.info("=" * 50)
    logger.info("Step 4/5: Team selection")
    logger.info("=" * 50)

    try:
        from team_selector import run_team_selector
        run_team_selector()
        logger.info("  ✓ Team selected")
        return True
    except Exception as e:
        logger.error(f"  ✗ Team selection failed: {e}")
        return False


def step_report(logger):
    """Generates the weekly prediction report."""
    logger.info("=" * 50)
    logger.info("Step 5/5: Weekly report")
    logger.info("=" * 50)

    try:
        import pandas as pd
        from sqlalchemy import create_engine, text

        pg_url = os.environ.get("FPL_DB_URL")
        engine = create_engine(pg_url) if pg_url else create_engine("sqlite:///db/fpl.db")

        # Current gameweek
        gw_df = pd.read_sql(text("SELECT id FROM gameweeks WHERE is_current = TRUE LIMIT 1"), engine)
        current_gw = int(gw_df["id"].iloc[0]) if not gw_df.empty else "?"

        print("\n")
        print("╔" + "═" * 68 + "╗")
        print(f"║  FPL PREDICTIONS — GW{current_gw}{'':>44}║")
        print(f"║  Generated {datetime.now().strftime('%a %d %b %Y %H:%M')}{'':>34}║")
        print("╚" + "═" * 68 + "╝")

        # Top 10 overall picks
        top = pd.read_sql(text("""
            SELECT web_name, position, team_name, price,
                   predicted_points, adjusted_points,
                   lineup_probability, fixture_fdr, opponent_name
            FROM predictions
            ORDER BY adjusted_points DESC
            LIMIT 10
        """), engine)

        print("\n  TOP 10 PICKS THIS WEEK")
        print(f"  {'#':<3} {'Player':<20} {'Pos':<4} {'£':>5} {'Adj Pts':>8} {'LinP':>5} {'vs':<20} {'FDR':>4}")
        print(f"  {'-'*74}")
        for rank, (_, r) in enumerate(top.iterrows(), 1):
            print(
                f"  {rank:<3} {str(r['web_name']):<20} {str(r['position']):<4} "
                f"£{float(r['price'] or 0):>4.1f} "
                f"{float(r['adjusted_points'] or 0):>8.2f} "
                f"{float(r['lineup_probability'] or 0):>5.2f} "
                f"{str(r['opponent_name'] or ''):<20} "
                f"{float(r['fixture_fdr'] or 0):>4.1f}"
            )

        # Injury watchlist — high form players with low lineup probability
        print("\n  ⚠ INJURY WATCHLIST (high form, availability concern)")
        watchlist = pd.read_sql(text("""
            SELECT p.web_name, lp.team_name, p.status, p.news,
                   p.chance_of_playing_this_round,
                   pf.avg_points_5gw,
                   lp.lineup_probability
            FROM players p
            JOIN player_features pf ON p.id = pf.player_id
              AND pf.season = '2025-26'
              AND pf.gameweek = (SELECT MAX(gameweek) FROM player_features WHERE season = '2025-26')
            JOIN lineup_probability lp ON p.id = lp.player_id
            WHERE lp.lineup_probability < 0.70
              AND pf.avg_points_5gw > 4.0
              AND p.status != 'u'
            ORDER BY pf.avg_points_5gw DESC
            LIMIT 8
        """), engine)

        if not watchlist.empty:
            print(f"  {'Player':<22} {'Team':<20} {'St':>3} {'LinP':>5} {'Form':>5}  News")
            print(f"  {'-'*80}")
            for _, r in watchlist.iterrows():
                news = str(r['news'] or '')[:35]
                print(
                    f"  {str(r['web_name']):<22} {str(r['team_name']):<20} "
                    f"{str(r['status'] or ''):>3} "
                    f"{float(r['lineup_probability'] or 0):>5.2f} "
                    f"{float(r['avg_points_5gw'] or 0):>5.1f}  {news}"
                )

        # Best fixture matchups this week
        print("\n  BEST FIXTURE MATCHUPS (easiest fixtures)")
        fixtures = pd.read_sql(text("""
            SELECT home_team, away_team, gameweek,
                   home_attack_fdr, away_attack_fdr,
                   home_opp_xga_5, away_opp_xga_5
            FROM fixture_difficulty
            WHERE finished = FALSE
              AND gameweek IS NOT NULL
            ORDER BY LEAST(home_attack_fdr, away_attack_fdr) ASC
            LIMIT 8
        """), engine)

        if not fixtures.empty:
            print(f"  {'Home':<22} {'Away':<22} {'H-FDR':>6} {'A-FDR':>6}")
            print(f"  {'-'*60}")
            for _, r in fixtures.iterrows():
                print(
                    f"  {str(r['home_team']):<22} {str(r['away_team']):<22} "
                    f"{float(r['home_attack_fdr'] or 0):>6.2f} "
                    f"{float(r['away_attack_fdr'] or 0):>6.2f}"
                )

        # Model stats
        metrics = pd.read_sql(text("""
            SELECT COUNT(*) as total,
                   ROUND(AVG(predicted_points)::numeric, 2) as avg_pred,
                   ROUND(MAX(predicted_points)::numeric, 2) as max_pred
            FROM predictions
        """), engine)
        if not metrics.empty:
            m = metrics.iloc[0]
            print(f"\n  Model: {int(m['total'])} players scored | "
                  f"avg predicted {float(m['avg_pred'] or 0):.2f} pts | "
                  f"max {float(m['max_pred'] or 0):.2f} pts")

        print()
        logger.info("  ✓ Report complete")
        return True

    except Exception as e:
        logger.error(f"  ✗ Report failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FPL Weekly Prediction Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python weekly_predict.py                    # Full pipeline
  python weekly_predict.py --no-collect       # Skip data refresh
  python weekly_predict.py --no-collect --no-train  # Just reselect team
        """
    )
    parser.add_argument("--no-collect", action="store_true",
                        help="Skip data collection (use existing data)")
    parser.add_argument("--no-train",   action="store_true",
                        help="Skip model retraining (use saved model)")
    parser.add_argument("--report-only", action="store_true",
                        help="Only generate report from existing predictions")

    args = parser.parse_args()
    logger = setup_logging()

    start = datetime.now()
    logger.info("FPL Weekly Predictor starting")

    results = {}

    if args.report_only:
        results["report"] = step_report(logger)
    else:
        if not args.no_collect:
            results["collect"] = step_collect(logger)
            if not results["collect"]:
                logger.error("Data collection failed — aborting")
                return

        results["features"] = step_features(logger)
        if not results["features"]:
            logger.error("Feature engineering failed — aborting")
            return

        if not args.no_train:
            results["predict"] = step_predict(logger)
            if not results["predict"]:
                logger.error("Prediction failed — aborting")
                return

        results["select"] = step_select(logger)
        results["report"] = step_report(logger)

    duration = (datetime.now() - start).total_seconds()

    print(f"\n  {'─'*50}")
    for step, ok in results.items():
        print(f"  {'✓' if ok else '✗'} {step}")
    print(f"  Total time: {duration:.0f}s")
    print(f"  {'─'*50}\n")


if __name__ == "__main__":
    main()

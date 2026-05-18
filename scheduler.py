"""
scheduler.py  —  Phase 2, Component 3: Pipeline Orchestrator
=============================================================

CONCEPTS TO UNDERSTAND BEFORE READING THIS SCRIPT
--------------------------------------------------

1. Why APScheduler instead of Airflow?
   Airflow is designed for teams running hundreds of pipelines on
   dedicated servers. For a personal project running 5 scripts nightly,
   it introduces massive dependency conflicts and operational overhead.
   APScheduler is a lightweight Python library that does exactly what
   we need: run functions on a schedule, with retry logic, in the
   background. Zero extra infrastructure, zero conflicts.

2. Job scheduling strategies
   We use two schedule types:
   - CronTrigger: run at specific times (e.g. every day at 2am)
   - IntervalTrigger: run every N minutes/hours
   The FPL pipeline runs nightly. The news monitor runs more frequently
   on matchday weeks when injury signals matter most.

3. Pipeline dependency ordering
   Tasks must run in the right order:
   1. FPL collector (get raw data)
   2. Understat scraper (get xG data)
   3. ETL pipeline (transform + validate + upsert)
   4. News monitor (get injury signals)
   5. Injury tracker (rebuild profiles from clean data)
   6. [3am] Feature engineering → Predictions → Team selection → Agent
   We enforce this with sequential execution — each step waits for
   the previous one to complete successfully.

4. Logging — why it matters for unattended runs
   When the scheduler runs at 2am while you sleep, you need to know
   what happened. We write structured logs to logs/ directory with
   timestamps, durations, and any errors. Check the log file in the
   morning to see what ran.

5. The run modes
   run      = run full data pipeline once now (~15 mins)
   schedule = start background scheduler (runs forever)
   news     = run news monitor only (fast)
   predict  = run full prediction pipeline on existing data (~2 mins)
"""

import os
import sys
import logging
import importlib
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging():
    """
    Configures logging to write to both console and a dated log file.
    Log files go in logs/ directory — one file per day.
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )
    return logging.getLogger("fpl_pipeline")


# =============================================================================
# DATA PIPELINE TASKS
# =============================================================================

def task_fpl_collector(logger):
    """Fetches current season player stats, fixtures, and gameweek history."""
    logger.info("Starting FPL collector...")
    start = datetime.now()
    try:
        from collectors.fpl_collector import run_collection
        run_collection(max_players=None)
        duration = (datetime.now() - start).total_seconds()
        logger.info(f"FPL collector complete ({duration:.0f}s)")
        return True
    except Exception as e:
        logger.error(f"FPL collector failed: {e}")
        return False


def task_understat_scraper(logger):
    """Collects xG/xA data from Understat."""
    logger.info("Starting Understat scraper...")
    start = datetime.now()
    try:
        from collectors.understat_scraper import run_understat_collection
        run_understat_collection()
        duration = (datetime.now() - start).total_seconds()
        logger.info(f"Understat scraper complete ({duration:.0f}s)")
        return True
    except Exception as e:
        logger.error(f"Understat scraper failed: {e}")
        return False


def task_etl(logger):
    """Validates, transforms, and upserts all tables."""
    logger.info("Starting ETL pipeline...")
    start = datetime.now()
    try:
        from pipeline.etl_pipeline import run_etl
        run_etl()
        duration = (datetime.now() - start).total_seconds()
        logger.info(f"ETL pipeline complete ({duration:.0f}s)")
        return True
    except Exception as e:
        logger.error(f"ETL pipeline failed: {e}")
        return False


def task_news_monitor(logger):
    """Collects FPL injury signals and parses RSS articles with Claude."""
    logger.info("Starting news monitor...")
    start = datetime.now()
    try:
        from collectors.news_monitor import run_news_monitor
        run_news_monitor()
        duration = (datetime.now() - start).total_seconds()
        logger.info(f"News monitor complete ({duration:.0f}s)")
        return True
    except Exception as e:
        logger.error(f"News monitor failed: {e}")
        return False


def task_injury_tracker(logger):
    """Rebuilds injury profiles from updated player history."""
    logger.info("Starting injury tracker...")
    start = datetime.now()
    try:
        from collectors.injury_tracker import run_injury_tracker
        run_injury_tracker()
        duration = (datetime.now() - start).total_seconds()
        logger.info(f"Injury tracker complete ({duration:.0f}s)")
        return True
    except Exception as e:
        logger.error(f"Injury tracker failed: {e}")
        return False


# =============================================================================
# PREDICTION PIPELINE TASKS
# =============================================================================

def task_feature_engineering(logger):
    """Rebuilds rolling form features from fresh data."""
    logger.info("Starting feature engineering...")
    start = datetime.now()
    try:
        for module_name, func_name in [
            ("pipeline.feature_engineering", "run_feature_engineering"),
            ("pipeline.fixture_difficulty",  "run_fixture_difficulty"),
            ("pipeline.lineup_probability",  "run_lineup_probability"),
            ("pipeline.feature_store",       "run_feature_store"),
        ]:
            module = importlib.import_module(module_name)
            getattr(module, func_name)()
        duration = (datetime.now() - start).total_seconds()
        logger.info(f"Feature engineering complete ({duration:.0f}s)")
        return True
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        return False


def task_predictions(logger):
    """Retrains XGBoost and generates point predictions."""
    logger.info("Starting predictions...")
    start = datetime.now()
    try:
        from prediction.predict_points import run_predict_points
        run_predict_points()
        duration = (datetime.now() - start).total_seconds()
        logger.info(f"Predictions complete ({duration:.0f}s)")
        return True
    except Exception as e:
        logger.error(f"Predictions failed: {e}")
        return False


def task_team_selection(logger):
    """Runs LP optimisation to select optimal team."""
    logger.info("Starting team selection...")
    start = datetime.now()
    try:
        from prediction.team_selector import run_team_selector
        run_team_selector()
        duration = (datetime.now() - start).total_seconds()
        logger.info(f"Team selection complete ({duration:.0f}s)")
        return True
    except Exception as e:
        logger.error(f"Team selection failed: {e}")
        return False


def task_agent_briefing(logger):
    """Generates Claude-powered weekly briefing."""
    logger.info("Starting agent briefing...")
    start = datetime.now()
    try:
        from agent.fpl_agent import run_fpl_agent
        run_fpl_agent(use_cache=False)  # Always fresh on scheduled runs
        duration = (datetime.now() - start).total_seconds()
        logger.info(f"Agent briefing complete ({duration:.0f}s)")
        return True
    except Exception as e:
        logger.error(f"Agent briefing failed: {e}")
        return False


# =============================================================================
# COMPOSITE RUNNERS
# =============================================================================

def run_full_pipeline(logger=None):
    """
    Runs the complete data pipeline in dependency order:
    FPL → Understat → ETL → News → Injuries

    Each step must succeed before the next runs.
    Returns True if all steps complete, False if any step fails.
    """
    if logger is None:
        logger = setup_logging()

    logger.info("=" * 50)
    logger.info("FPL Data Pipeline starting")
    logger.info("=" * 50)

    pipeline_start = datetime.now()
    results = {}

    tasks = [
        ("fpl_collector",  task_fpl_collector),
        ("understat",      task_understat_scraper),
        ("etl",            task_etl),
        ("news_monitor",   task_news_monitor),
        ("injury_tracker", task_injury_tracker),
    ]

    for task_name, task_fn in tasks:
        success = task_fn(logger)
        results[task_name] = success
        if not success:
            logger.error(f"Pipeline halted at {task_name}")
            break

    duration = (datetime.now() - pipeline_start).total_seconds()
    all_passed = all(results.values())

    logger.info("=" * 50)
    logger.info(f"Pipeline {'complete' if all_passed else 'FAILED'} in {duration:.0f}s")
    for name, ok in results.items():
        logger.info(f"  {'✓' if ok else '✗'} {name}")
    logger.info("=" * 50)

    return all_passed


def run_full_predictions(logger=None):
    """
    Runs the full prediction pipeline on existing data:
    Features → Predictions → Team Selection → Agent Briefing

    Runs at 3am after the data pipeline completes at 2am.
    Can also be triggered manually: python scheduler.py predict
    """
    if logger is None:
        logger = setup_logging()

    logger.info("=" * 50)
    logger.info("FPL Prediction Pipeline starting")
    logger.info("=" * 50)

    pipeline_start = datetime.now()
    results = {}

    tasks = [
        ("features",  task_feature_engineering),
        ("predict",   task_predictions),
        ("select",    task_team_selection),
        ("agent",     task_agent_briefing),
    ]

    for task_name, task_fn in tasks:
        success = task_fn(logger)
        results[task_name] = success
        if not success:
            logger.error(f"Prediction pipeline halted at {task_name}")
            break

    duration = (datetime.now() - pipeline_start).total_seconds()
    all_passed = all(results.values())

    logger.info("=" * 50)
    logger.info(f"Prediction pipeline {'complete' if all_passed else 'FAILED'} in {duration:.0f}s")
    for name, ok in results.items():
        logger.info(f"  {'✓' if ok else '✗'} {name}")
    logger.info("=" * 50)

    return all_passed


def run_news_only(logger=None):
    """
    Runs only the news monitor — used for frequent matchday checks.
    Faster than the full pipeline (no API calls or ETL).
    """
    if logger is None:
        logger = setup_logging()
    logger.info("News-only run starting...")
    task_news_monitor(logger)


# =============================================================================
# SCHEDULER
# =============================================================================

def start_scheduler():
    """
    Starts the APScheduler with three jobs:

    1. Full data pipeline — every night at 2:00am
       Refreshes all data: FPL stats, xG, ETL transforms, injury profiles

    2. Prediction pipeline — every night at 3:00am
       Runs after data pipeline: features, XGBoost, team selection, agent

    3. News monitor — every 6 hours
       Keeps injury signals fresh without the overhead of a full pipeline

    Run this in a terminal and leave it running.
    Press Ctrl+C to stop.
    """
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    logger = setup_logging()
    scheduler = BlockingScheduler(timezone="Europe/London")

    # Full data pipeline — nightly at 2am
    scheduler.add_job(
        func=run_full_pipeline,
        trigger=CronTrigger(hour=2, minute=0),
        id="full_pipeline",
        name="Full FPL Data Pipeline",
        kwargs={"logger": logger},
        max_instances=1,
        misfire_grace_time=3600,
        coalesce=True,
    )

    # Prediction pipeline — nightly at 3am (after data pipeline)
    scheduler.add_job(
        func=run_full_predictions,
        trigger=CronTrigger(hour=3, minute=0),
        id="predictions",
        name="FPL Prediction Pipeline",
        kwargs={"logger": logger},
        max_instances=1,
        misfire_grace_time=3600,
        coalesce=True,
    )

    # News monitor — every 6 hours
    scheduler.add_job(
        func=run_news_only,
        trigger=CronTrigger(hour="6,12,18,22"),
        id="news_monitor",
        name="News Monitor",
        kwargs={"logger": logger},
        max_instances=1,
        misfire_grace_time=1800,
        coalesce=True,
    )

    logger.info("Scheduler started")
    logger.info("  Data pipeline  : daily at 02:00 London time")
    logger.info("  Predictions    : daily at 03:00 London time")
    logger.info("  News monitor   : 06:00, 12:00, 18:00, 22:00")
    logger.info("Press Ctrl+C to stop")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped")
        scheduler.shutdown()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FPL Pipeline Scheduler")
    parser.add_argument(
        "mode",
        choices=["run", "schedule", "news", "predict"],
        help=(
            "run      = run full data pipeline once now (~15 mins)\n"
            "schedule = start background scheduler (runs forever)\n"
            "news     = run news monitor only (fast)\n"
            "predict  = run prediction pipeline on existing data (~2 mins)"
        )
    )
    args = parser.parse_args()

    logger = setup_logging()

    if args.mode == "run":
        run_full_pipeline(logger)
    elif args.mode == "schedule":
        start_scheduler()
    elif args.mode == "news":
        run_news_only(logger)
    elif args.mode == "predict":
        run_full_predictions(logger)

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
   2. Historical loader (one-time, skip if already done)
   3. Understat scraper (get xG data)
   4. ETL pipeline (transform + validate + upsert)
   5. News monitor (get injury signals)
   6. Injury tracker (rebuild profiles from clean data)
   We enforce this with sequential execution — each step waits for
   the previous one to complete successfully.

4. Logging — why it matters for unattended runs
   When the scheduler runs at 2am while you sleep, you need to know
   what happened. We write structured logs to airflow/logs/ (reusing
   the folder Airflow created) with timestamps, durations, and any
   errors. Check the log file in the morning to see what ran.

5. The run_once vs scheduled modes
   run_pipeline_once() runs everything immediately — useful for testing
   and manual refreshes. start_scheduler() runs everything on a timer
   indefinitely. We always test with run_once first.
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path so we can import our modules
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
            logging.StreamHandler(),  # Also print to terminal
        ]
    )
    return logging.getLogger("fpl_pipeline")


# =============================================================================
# PIPELINE TASKS
# =============================================================================

def task_fpl_collector(logger):
    """
    Runs the FPL API collector.
    Fetches current season player stats, fixtures, and gameweek history.
    Runs with max_players=None for a full 832-player refresh.
    """
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
    """
    Runs the Understat xG/xA scraper.
    Collects current season data only (2025-26) on scheduled runs
    to avoid re-scraping all 4 seasons every night.
    """
    logger.info("Starting Understat scraper...")
    start = datetime.now()
    try:
        from collectors.understat_scraper import run_understat_collection
        # Only refresh current season on nightly runs
        run_understat_collection()
        duration = (datetime.now() - start).total_seconds()
        logger.info(f"Understat scraper complete ({duration:.0f}s)")
        return True
    except Exception as e:
        logger.error(f"Understat scraper failed: {e}")
        return False


def task_etl(logger):
    """
    Runs the ETL pipeline — validates, transforms, and upserts all tables.
    Must run AFTER collectors so it processes fresh data.
    """
    logger.info("Starting ETL pipeline...")
    start = datetime.now()
    try:
        from etl_pipeline import run_etl
        run_etl()
        duration = (datetime.now() - start).total_seconds()
        logger.info(f"ETL pipeline complete ({duration:.0f}s)")
        return True
    except Exception as e:
        logger.error(f"ETL pipeline failed: {e}")
        return False


def task_news_monitor(logger):
    """
    Runs the news monitor — collects FPL injury signals and parses
    RSS articles with Claude for lineup signals.
    """
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
    """
    Rebuilds injury profiles from the updated player history.
    Must run AFTER ETL so it uses the latest cleaned data.
    """
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
# PIPELINE RUNNER
# =============================================================================

def run_full_pipeline(logger=None):
    """
    Runs the complete pipeline in dependency order:
    FPL → Understat → ETL → News → Injuries

    Each step must succeed before the next runs.
    Returns True if all steps complete, False if any step fails.
    """
    if logger is None:
        logger = setup_logging()

    logger.info("=" * 50)
    logger.info("FPL Pipeline starting")
    logger.info("=" * 50)

    pipeline_start = datetime.now()
    results = {}

    tasks = [
        ("fpl_collector",    task_fpl_collector),
        ("understat",        task_understat_scraper),
        ("etl",              task_etl),
        ("news_monitor",     task_news_monitor),
        ("injury_tracker",   task_injury_tracker),
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
    Starts the APScheduler with two jobs:

    1. Full pipeline — every night at 2:00am
       Refreshes all data: FPL stats, xG, ETL transforms, injury profiles

    2. News monitor — every 6 hours during matchday weeks
       Keeps injury signals fresh without the overhead of a full pipeline

    Run this in a terminal and leave it running.
    Press Ctrl+C to stop.
    """
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    logger = setup_logging()
    scheduler = BlockingScheduler(timezone="Europe/London")

    # Full pipeline — nightly at 2am
    scheduler.add_job(
        func=run_full_pipeline,
        trigger=CronTrigger(hour=2, minute=0),
        id="full_pipeline",
        name="Full FPL Pipeline",
        kwargs={"logger": logger},
        max_instances=1,            # Never run two instances simultaneously
        misfire_grace_time=3600,    # If missed, run within 1hr or skip
        coalesce=True,              # If multiple missed, run once not many
    )

    # News monitor — every 6 hours (more frequent on matchday weeks)
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
    logger.info("  Full pipeline: daily at 02:00 London time")
    logger.info("  News monitor:  06:00, 12:00, 18:00, 22:00")
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
        choices=["run", "schedule", "news"],
        help=(
            "run      = run full pipeline once now\n"
            "schedule = start scheduled background runner\n"
            "news     = run news monitor only"
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

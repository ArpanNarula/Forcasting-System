"""
scheduler.py
------------
Optional weekly retraining scheduler using APScheduler.
"""

import logging
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "retrain.log"

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("retrain_scheduler")

_scheduler = None


def retrain_job():
    """Run the full preprocessing and training pipeline."""
    logger.info("Weekly retraining started.")
    try:
        from src.preprocessing import run_pipeline
        from src.model_selector import train_all_states

        state_data = run_pipeline()
        train_all_states(state_data, n_jobs=1)
        logger.info("Weekly retraining completed successfully.")
    except Exception:
        logger.exception("Weekly retraining failed.")
        raise


def get_scheduler():
    """Create or return the singleton background scheduler."""
    global _scheduler
    if _scheduler is not None:
        return _scheduler

    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError as exc:
        raise ImportError("Install apscheduler to enable weekly auto-retraining.") from exc

    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(
        retrain_job,
        CronTrigger(day_of_week="sun", hour=2, minute=0),
        id="weekly_retrain",
        replace_existing=True,
    )
    _scheduler = scheduler
    return scheduler


def start_scheduler():
    scheduler = get_scheduler()
    if not scheduler.running:
        scheduler.start()
        logger.info("Retraining scheduler started.")
    return scheduler


def scheduler_status() -> dict:
    try:
        scheduler = get_scheduler()
    except ImportError as exc:
        return {
            "available": False,
            "running": False,
            "message": str(exc),
            "log_path": str(LOG_PATH),
        }

    jobs = []
    for job in scheduler.get_jobs():
        jobs.append(
            {
                "id": job.id,
                "next_run_time": str(job.next_run_time) if job.next_run_time else None,
            }
        )

    return {
        "available": True,
        "running": scheduler.running,
        "jobs": jobs,
        "log_path": str(LOG_PATH),
    }

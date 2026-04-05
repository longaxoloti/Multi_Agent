from __future__ import annotations
import os
import sys
from datetime import datetime, timedelta
import requests
from main.config import TELEGRAM_BOT_TOKEN, TELEGRAM_USER_ID

from airflow.decorators import dag, task
from airflow.operators.python import get_current_context

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main.config import (
    AIRFLOW_DAILY_REPORT_CRON,
    AIRFLOW_REPORT_CATCHUP,
    AIRFLOW_REPORT_CHAT_ID,
    AIRFLOW_REPORT_CATEGORIES,
    AIRFLOW_REPORT_DAGRUN_TIMEOUT_MINUTES,
    AIRFLOW_REPORT_MAX_ACTIVE_RUNS,
    AIRFLOW_REPORT_RETRIES,
    AIRFLOW_REPORT_RETRY_DELAY_MINUTES,
)
from pipelines.reporting import build_daily_knowledge_report_text
from storage.trusted_db import TrustedDBRepository


def _send_telegram_message(message: str) -> None:

    if not TELEGRAM_BOT_TOKEN or TELEGRAM_USER_ID == 0:
        return

    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        json={
            "chat_id": TELEGRAM_USER_ID,
            "text": message[:4000],
            "disable_web_page_preview": True,
        },
        timeout=20,
    )


@dag(
    dag_id="daily_user_knowledge_report",
    start_date=datetime(2026, 1, 1),
    schedule=AIRFLOW_DAILY_REPORT_CRON,
    catchup=AIRFLOW_REPORT_CATCHUP,
    max_active_runs=AIRFLOW_REPORT_MAX_ACTIVE_RUNS,
    dagrun_timeout=timedelta(minutes=AIRFLOW_REPORT_DAGRUN_TIMEOUT_MINUTES),
    default_args={
        "retries": AIRFLOW_REPORT_RETRIES,
        "retry_delay": timedelta(minutes=AIRFLOW_REPORT_RETRY_DELAY_MINUTES),
    },
    tags=["agent", "reporting", "knowledge-db"],
)
def daily_user_knowledge_report_dag():
    @task
    def gather_manual_knowledge_records() -> list[dict]:
        context = get_current_context()
        interval_start = context.get("data_interval_start")
        interval_end = context.get("data_interval_end")

        if interval_start is not None:
            start_dt = interval_start.replace(tzinfo=None)
        else:
            start_dt = datetime.utcnow() - timedelta(hours=24)
        if interval_end is not None:
            end_dt = interval_end.replace(tzinfo=None)
        else:
            end_dt = datetime.utcnow()

        repo = TrustedDBRepository()
        repo.initialize()
        items = repo.list_knowledge_records_between(
            start=start_dt,
            end=end_dt,
            chat_id=AIRFLOW_REPORT_CHAT_ID or None,
            categories=AIRFLOW_REPORT_CATEGORIES,
            limit=300,
        )
        return [
            {
                "id": x.id,
                "chat_id": x.chat_id,
                "category": x.category,
                "title": x.title,
                "content": x.content,
                "tags": x.tags,
                "metadata": x.metadata,
                "created_at": x.created_at.isoformat(),
                "updated_at": x.updated_at.isoformat(),
            }
            for x in items
        ]

    @task
    def summarize(records: list[dict]) -> str:
        from storage.trusted_db import UserKnowledgeRecord

        normalized = [
            UserKnowledgeRecord(
                id=i["id"],
                chat_id=i["chat_id"],
                category=i["category"],
                title=i.get("title", ""),
                content=i.get("content", ""),
                tags=i.get("tags", []),
                metadata=i.get("metadata", {}),
                created_at=datetime.fromisoformat(i["created_at"]),
                updated_at=datetime.fromisoformat(i["updated_at"]),
            )
            for i in records
        ]
        return build_daily_knowledge_report_text(normalized)

    @task
    def notify(report_text: str) -> None:
        _send_telegram_message(report_text)

    notify(summarize(gather_manual_knowledge_records()))

dag = daily_user_knowledge_report_dag()
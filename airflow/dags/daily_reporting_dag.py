from __future__ import annotations
import asyncio
import os
import sys
from datetime import datetime, timedelta
import requests
import pendulum
from main.config import TELEGRAM_BOT_TOKEN, TELEGRAM_USER_ID
from langchain_core.messages import HumanMessage
from tools.web_search.crawl4ai_client import crawl_url_to_markdown

from airflow.decorators import dag, task
from airflow.operators.python import get_current_context

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main.config import (
    AIRFLOW_DAILY_REPORT_CRON,
    AIRFLOW_REPORT_CATCHUP,
    AIRFLOW_REPORT_CHAT_ID,
    AIRFLOW_REPORT_DAGRUN_TIMEOUT_MINUTES,
    AIRFLOW_REPORT_MAX_ACTIVE_RUNS,
    AIRFLOW_REPORT_RETRIES,
    AIRFLOW_REPORT_RETRY_DELAY_MINUTES,
    AIRFLOW_TIMEZONE,
)
from graph.workflow import build_workflow
from storage.knowledge_service import KnowledgeService


REPORT_TIMEZONE = pendulum.timezone(AIRFLOW_TIMEZONE)


def _resolve_report_chat_id() -> str:
    chat_id = AIRFLOW_REPORT_CHAT_ID or (str(TELEGRAM_USER_ID) if TELEGRAM_USER_ID else "")
    if not chat_id:
        raise RuntimeError("AIRFLOW_REPORT_CHAT_ID or TELEGRAM_USER_ID must be configured")
    return str(chat_id)


def _send_telegram_message(chat_id: str, message: str) -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN must be configured for daily reports")
    if not chat_id:
        raise RuntimeError("Daily report chat_id is empty")

    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        json={
            "chat_id": chat_id,
            "text": message[:4000],
            "disable_web_page_preview": True,
        },
        timeout=20,
    )


def _build_scheduled_research_prompt(*, start_dt: datetime, end_dt: datetime) -> str:
    return (
        "Research and summarize the latest and most notable world news.\n"
        f"Time window: {start_dt.strftime('%Y-%m-%d %H:%M UTC')} -> {end_dt.strftime('%Y-%m-%d %H:%M UTC')}\n"
        "Requirements:\n"
        "1) Focus on high-impact stories covered by multiple sources.\n"
        "2) Include clear source citations (URL) for each key point.\n"
        "3) If uncertainty exists, explicitly state confidence level.\n"
        "4) Return a concise Telegram-friendly briefing."
    )


@dag(
    dag_id="daily_user_knowledge_report",
    start_date=pendulum.datetime(2026, 1, 1, tz=REPORT_TIMEZONE),
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
    def run_scheduled_research() -> dict:
        context = get_current_context()
        interval_start = context.get("data_interval_start")
        interval_end = context.get("data_interval_end")
        report_chat_id = _resolve_report_chat_id()

        if interval_start is not None:
            start_dt = interval_start.in_timezone("UTC").naive()
        else:
            start_dt = datetime.utcnow() - timedelta(hours=24)
        if interval_end is not None:
            end_dt = interval_end.in_timezone("UTC").naive()
        else:
            end_dt = datetime.utcnow()

        prompt = _build_scheduled_research_prompt(start_dt=start_dt, end_dt=end_dt)
        session_id = f"airflow_research_{end_dt.strftime('%Y%m%d%H%M')}"

        workflow = build_workflow()
        result = asyncio.run(
            workflow.ainvoke(
                {
                    "messages": [HumanMessage(content=prompt)],
                    "chat_id": report_chat_id,
                    "session_id": session_id,
                    "persist_research_to_db": False,
                    "intent": "",
                    "memory_context": "",
                    "verification_summary": "",
                }
            )
        )

        final_messages = result.get("messages", [])
        workflow_summary = ""
        if final_messages and getattr(final_messages[-1], "type", "") == "ai":
            workflow_summary = str(final_messages[-1].content or "").strip()

        sources: list[str] = []
        for item in result.get("task_results", []) or []:
            urls = item.get("sources") or []
            if isinstance(urls, list):
                for url in urls:
                    if isinstance(url, str) and url.strip() and url not in sources:
                        sources.append(url)

        return {
            "chat_id": report_chat_id,
            "session_id": session_id,
            "workflow_summary": workflow_summary,
            "sources": sources,
        }

    @task
    def notify_user(run_meta: dict) -> None:
        workflow_summary = str(run_meta.get("workflow_summary") or "").strip()
        message = workflow_summary or "Research completed, but no summary text was returned."
        _send_telegram_message(_resolve_report_chat_id(), message)

    @task
    def persist_web_news(run_meta: dict) -> dict:
        report_chat_id = str(run_meta.get("chat_id") or _resolve_report_chat_id())
        sources = run_meta.get("sources") or []
        if not isinstance(sources, list):
            return {"saved": 0, "deduplicated": 0, "failed": 0}

        service = KnowledgeService()
        counters = {"saved": 0, "deduplicated": 0, "failed": 0}

        for source_url in sources:
            if not isinstance(source_url, str) or not source_url.strip():
                continue
            markdown = asyncio.run(crawl_url_to_markdown(source_url.strip()))
            if not markdown:
                counters["failed"] += 1
                continue

            try:
                result = service.save_deduplicated(
                    chat_id=report_chat_id,
                    content=markdown,
                    category="web_news",
                    title=source_url.strip()[:200],
                    metadata={
                        "source_url": source_url.strip(),
                        "ingestion_pipeline": "airflow_report_parallel_persist",
                    },
                    source_url=source_url.strip(),
                )
                if result.get("deduplicated"):
                    counters["deduplicated"] += 1
                else:
                    counters["saved"] += 1
            except Exception:
                counters["failed"] += 1

        return counters

    run_meta = run_scheduled_research()
    notify_user(run_meta)
    persist_web_news(run_meta)

dag = daily_user_knowledge_report_dag()
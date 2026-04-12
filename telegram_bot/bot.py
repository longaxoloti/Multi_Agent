import logging
import time
from datetime import datetime, timedelta, time as dt_time
from zoneinfo import ZoneInfo
from telegram import Update
from telegram.error import Conflict, NetworkError, RetryAfter, TimedOut
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from langchain_core.messages import HumanMessage
from graph.workflow import build_workflow
from main.config import (
    AIRFLOW_DAILY_REPORT_CRON,
    AIRFLOW_REPORT_CHAT_ID,
    AIRFLOW_TIMEZONE,
    BRIEFING_HOUR,
    BRIEFING_MINUTE,
    TELEGRAM_USER_ID,
)

logger = logging.getLogger(__name__)

class TelegramBot:
    def __init__(self, token: str):
        self.token = token
        self.workflow = build_workflow()
        
    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Hello! I am your AI assistant. Send me a message, ask for research, or request a daily briefing.")

    async def _daily_now_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return

        chat_id = str(update.message.chat_id)
        await update.message.reply_text("Running daily report now. Please wait...")

        try:
            await self._run_sequential_topic_reports(
                chat_id=chat_id,
                session_prefix="tg_daily_manual",
                bot=context.bot,
            )
            logger.info("Manual /daily_now report sent to chat_id=%s", chat_id)
        except Exception as exc:
            logger.error("Manual /daily_now failed: %s", exc, exc_info=True)
            await update.message.reply_text("Daily report failed. Check server logs for details.")

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.text:
            return
            
        message = update.message.text.strip()
        chat_id = str(update.message.chat_id)
        user_id = update.message.from_user.id
        
        logger.info("Received message from %s (%d chars): %s", user_id, len(message), message)
        
        try:
            result = await self.workflow.ainvoke({
                "messages": [HumanMessage(content=message)],
                "chat_id": chat_id,
                "intent": "",
                "memory_context": "",
                "verification_summary": ""
            })
            
            final_messages = result.get("messages", [])
            if final_messages and final_messages[-1].type == "ai":
                await update.message.reply_text(final_messages[-1].content)
            else:
                logger.warning("No AI response found in workflow result.")
                await update.message.reply_text("I processed your message, but have no response to provide.")
                
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            await update.message.reply_text("Sorry, I encountered an error while processing your request.")

    async def _on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        logger.error("Telegram update handling error: %s", context.error, exc_info=context.error)

    @staticmethod
    def _resolve_report_chat_id() -> str:
        chat_id = AIRFLOW_REPORT_CHAT_ID or (str(TELEGRAM_USER_ID) if TELEGRAM_USER_ID else "")
        return str(chat_id).strip()

    @staticmethod
    def _parse_daily_times() -> list[dt_time]:
        expr = (AIRFLOW_DAILY_REPORT_CRON or "").strip()
        parts = expr.split()
        if len(parts) == 5 and parts[2] == "*" and parts[3] == "*" and parts[4] == "*":
            minute_raw, hour_raw = parts[0], parts[1]
            if minute_raw.isdigit():
                minute = int(minute_raw)
                hours: list[int] = []
                for token in hour_raw.split(","):
                    token = token.strip()
                    if token.isdigit():
                        value = int(token)
                        if 0 <= value <= 23:
                            hours.append(value)
                if 0 <= minute <= 59 and hours:
                    unique_hours = sorted(set(hours))
                    return [dt_time(hour=h, minute=minute, tzinfo=ZoneInfo(AIRFLOW_TIMEZONE)) for h in unique_hours]

        # Fallback to legacy single-run config if cron parsing fails.
        return [dt_time(hour=BRIEFING_HOUR, minute=BRIEFING_MINUTE, tzinfo=ZoneInfo(AIRFLOW_TIMEZONE))]

    @staticmethod
    def _build_scheduled_prompt(now_utc: datetime) -> str:
        start_utc = now_utc - timedelta(hours=12)
        return (
            "Research and summarize the latest and most notable world news.\n"
            f"Time window: {start_utc.strftime('%Y-%m-%d %H:%M UTC')} -> {now_utc.strftime('%Y-%m-%d %H:%M UTC')}\n"
            "Requirements:\n"
            "1) Focus on high-impact stories covered by multiple sources.\n"
            "2) Include clear source citations (URL) for each key point.\n"
            "3) If uncertainty exists, explicitly state confidence level.\n"
            "4) Return a concise Telegram-friendly briefing."
        )

    @staticmethod
    def _build_topic_prompt(now_utc: datetime, topic_label: str, topic_query: str) -> str:
        start_utc = now_utc - timedelta(hours=12)
        return (
            f"Research and summarize the latest and most notable updates for topic: {topic_label}.\n"
            f"Topic focus: {topic_query}\n"
            f"Time window: {start_utc.strftime('%Y-%m-%d %H:%M UTC')} -> {now_utc.strftime('%Y-%m-%d %H:%M UTC')}\n"
            "Requirements:\n"
            "1) Focus on high-impact updates and major developments.\n"
            "2) Include clear source citations (URL).\n"
            "3) Highlight key implications briefly.\n"
            "4) Keep the output concise and factual.\n"
            "5) Return a flexible number of news items based on reliable information available in the selected window.\n"
            "6) Do not force placeholders or invented items when evidence is limited.\n"
            "7) Use the exact format below for each item:\n"
            "** <headline>\n"
            "  - Description: <brief description>\n"
            "  - Source: <full URL>"
        )

    @staticmethod
    def _report_topics() -> list[tuple[str, str]]:
        return [
            ("Tin tức nổi bật", "Latest and most notable updates across major areas"),
            ("Tài chính - Kinh tế", "Financial markets, macroeconomics, and economic volatility"),
            ("Công nghệ thông tin", "Information technology, software, AI, and cybersecurity"),
        ]

    async def _send_long_message(self, bot, chat_id: str, text: str) -> None:
        content = (text or "").strip()
        if not content:
            return

        max_len = 3900
        if len(content) <= max_len:
            await bot.send_message(chat_id=chat_id, text=content, disable_web_page_preview=True)
            return

        parts: list[str] = []
        remaining = content
        while remaining:
            if len(remaining) <= max_len:
                parts.append(remaining)
                break
            split_at = remaining.rfind("\n", 0, max_len)
            if split_at < 400:
                split_at = max_len
            parts.append(remaining[:split_at].strip())
            remaining = remaining[split_at:].strip()

        for idx, part in enumerate(parts, 1):
            prefix = f"[{idx}/{len(parts)}]\n" if len(parts) > 1 else ""
            await bot.send_message(
                chat_id=chat_id,
                text=(prefix + part)[:4000],
                disable_web_page_preview=True,
            )

    async def _run_single_topic_daily_report(
        self,
        *,
        chat_id: str,
        session_prefix: str,
        topic_index: int,
        topic_label: str,
        topic_query: str,
        now_utc: datetime,
    ) -> str:
        prompt = self._build_topic_prompt(now_utc, topic_label=topic_label, topic_query=topic_query)
        session_id = f"{session_prefix}_{topic_index}_{now_utc.strftime('%Y%m%d%H%M%S')}"
        logger.info(
            "Running daily report topic=%s chat_id=%s session_id=%s",
            topic_label,
            chat_id,
            session_id,
        )

        result = await self.workflow.ainvoke(
            {
                "messages": [HumanMessage(content=prompt)],
                "chat_id": chat_id,
                "session_id": session_id,
                "intent": "",
                "memory_context": "",
                "verification_summary": "",
            }
        )
        final_messages = result.get("messages", [])
        if final_messages and final_messages[-1].type == "ai":
            topic_text = str(final_messages[-1].content or "").strip()
        else:
            topic_text = "No response produced for this topic."

        logger.info(
            "Daily report topic completed topic=%s chat_id=%s session_id=%s chars=%s",
            topic_label,
            chat_id,
            session_id,
            len(topic_text),
        )
        return topic_text

    async def _send_topic_report_message(
        self,
        *,
        bot,
        chat_id: str,
        topic_index: int,
        topic_label: str,
        topic_text: str,
    ) -> None:
        message_text = f"## {topic_index}. {topic_label}\n{topic_text}"
        await self._send_long_message(bot, chat_id=chat_id, text=message_text)
        logger.info(
            "Daily report topic message sent topic=%s chat_id=%s chars=%s",
            topic_label,
            chat_id,
            len(message_text),
        )

    async def _run_sequential_topic_reports(self, *, chat_id: str, session_prefix: str, bot) -> None:
        now_utc = datetime.utcnow()

        for idx, (topic_label, topic_query) in enumerate(self._report_topics(), 1):
            try:
                topic_text = await self._run_single_topic_daily_report(
                    chat_id=chat_id,
                    session_prefix=session_prefix,
                    topic_index=idx,
                    topic_label=topic_label,
                    topic_query=topic_query,
                    now_utc=now_utc,
                )
            except Exception as exc:
                logger.error("Daily report topic failed (%s): %s", topic_label, exc, exc_info=True)
                topic_text = "Topic processing failed. Check logs."

            await self._send_topic_report_message(
                bot=bot,
                chat_id=chat_id,
                topic_index=idx,
                topic_label=topic_label,
                topic_text=topic_text,
            )

        logger.info("Daily multi-topic report finished chat_id=%s", chat_id)

    async def _scheduled_daily_report(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = self._resolve_report_chat_id()
        if not chat_id:
            logger.warning("Scheduled daily report skipped: AIRFLOW_REPORT_CHAT_ID/TELEGRAM_USER_ID is empty")
            return

        logger.info("Running scheduled daily multi-topic report for chat_id=%s", chat_id)
        try:
            await self._run_sequential_topic_reports(
                chat_id=chat_id,
                session_prefix="tg_daily",
                bot=context.bot,
            )
            logger.info("Scheduled daily report sent successfully to chat_id=%s", chat_id)
        except Exception as exc:
            logger.error("Scheduled daily report failed: %s", exc, exc_info=True)

    def _register_scheduled_jobs(self, app: Application) -> None:
        if app.job_queue is None:
            logger.warning("Job queue unavailable; scheduled reports will not run")
            return

        chat_id = self._resolve_report_chat_id()
        if not chat_id:
            logger.warning("Skipping scheduled report setup: AIRFLOW_REPORT_CHAT_ID/TELEGRAM_USER_ID is empty")
            return

        times = self._parse_daily_times()
        if not times:
            logger.warning("Skipping scheduled report setup: no valid schedule times parsed")
            return

        for run_time in times:
            app.job_queue.run_daily(
                self._scheduled_daily_report,
                time=run_time,
                name=f"daily_report_{run_time.hour:02d}{run_time.minute:02d}",
            )
            logger.info(
                "Scheduled daily report registered at %02d:%02d (%s)",
                run_time.hour,
                run_time.minute,
                AIRFLOW_TIMEZONE,
            )

    def run(self):
        if not self.token:
            logger.error("No Telegram token provided.")
            return

        max_retries = 10
        base_delay = 3

        for attempt in range(1, max_retries + 1):
            app = Application.builder().token(self.token).build()

            app.add_handler(CommandHandler("start", self._start_command))
            app.add_handler(CommandHandler("daily_now", self._daily_now_command))
            app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))
            app.add_error_handler(self._on_error)
            self._register_scheduled_jobs(app)

            try:
                logger.info("Starting Telegram bot polling (attempt %s/%s)...", attempt, max_retries)
                app.run_polling(
                    drop_pending_updates=False,
                    allowed_updates=Update.ALL_TYPES,
                    close_loop=False,
                )
                return
            except Conflict:
                wait_seconds = min(base_delay * attempt, 30)
                logger.warning(
                    "Telegram polling conflict detected (another getUpdates consumer exists). Retrying in %ss...",
                    wait_seconds,
                )
                try:
                    app.bot.delete_webhook(drop_pending_updates=False)
                except Exception as webhook_error:
                    logger.warning("delete_webhook failed during conflict recovery: %s", webhook_error)
                time.sleep(wait_seconds)
            except (TimedOut, RetryAfter, NetworkError) as network_error:
                wait_seconds = min(base_delay * attempt, 20)
                logger.warning("Transient Telegram network error: %s. Retrying in %ss...", network_error, wait_seconds)
                time.sleep(wait_seconds)
            except Exception as e:
                logger.error("Failed to start Telegram polling: %s", e, exc_info=True)
                raise

        logger.error("Telegram polling could not be stabilized after %s attempts.", max_retries)

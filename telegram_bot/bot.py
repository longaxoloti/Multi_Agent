import logging
import time
from telegram import Update
from telegram.error import Conflict, NetworkError, RetryAfter, TimedOut
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from langchain_core.messages import HumanMessage
from graph.workflow import build_workflow

logger = logging.getLogger(__name__)

class TelegramBot:
    def __init__(self, token: str):
        self.token = token
        self.workflow = build_workflow()
        
    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Hello! I am your AI assistant. Send me a message, ask for research, or request a daily briefing.")

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.text:
            return
            
        message = update.message.text.strip()
        chat_id = str(update.message.chat_id)
        user_id = update.message.from_user.id
        
        logger.info("Received message from %s: %s", user_id, message[:100])
        
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

    def run(self):
        if not self.token:
            logger.error("No Telegram token provided.")
            return

        max_retries = 10
        base_delay = 3

        for attempt in range(1, max_retries + 1):
            app = Application.builder().token(self.token).build()

            app.add_handler(CommandHandler("start", self._start_command))
            app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))
            app.add_error_handler(self._on_error)

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

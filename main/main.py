import fcntl
import logging
import os
import signal
import sys
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.workspace_priming import get_workspace_priming_context
from telegram_bot.bot import TelegramBot

DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = DATA_DIR / "logs"

for d in [DATA_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOGS_DIR / "agent.log"), mode="a"),
    ],
)
logger = logging.getLogger("main")

for noisy in ["httpx", "httpcore"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

_BOT_LOCK_FILE_HANDLE = None

def acquire_single_instance_lock() -> bool:
    global _BOT_LOCK_FILE_HANDLE
    lock_path = LOGS_DIR / "bot.instance.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        owner = handle.read().strip() or "unknown"
        logger.error("xxx Another instance is already running (pid=%s). xxx", owner)
        handle.close()
        return False
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    _BOT_LOCK_FILE_HANDLE = handle
    return True


def main():
    logger.info("=" * 60)
    logger.info("STARTING: Tesla is starting!!!")
    logger.info("=" * 60)

    if not acquire_single_instance_lock():
        sys.exit(1)
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not telegram_token:
        logger.warning("!!! TELEGRAM_BOT_TOKEN not set. Telegram channel will be unavailable.")
    has_llm = any([
        os.getenv("GEMINI_API_KEY"),
        os.getenv("OLLAMA_ENABLED", "true").lower() in {"1", "true", "yes"},
    ])
    if not has_llm:
        logger.error("xxx No LLM provider configured. Enable Ollama or add GEMINI_API_KEY to .env. xxx")
        sys.exit(1)

    logger.info("   Workspace: %s", PROJECT_ROOT)
    priming = get_workspace_priming_context()
    if priming:
        logger.info("   Workspace priming: loaded (%d chars)", len(priming))
    else:
        logger.info("   Workspace priming: not loaded (disabled or files missing)")
    logger.info("=" * 60)
    logger.info("===== Tesla agent is ready =====")
    logger.info("Send /start to your bot on Telegram to begin!")
    logger.info("=" * 60)

    try:
        bot = TelegramBot(token=telegram_token)
        bot.run()
    except Exception as e:
        logger.error(f"Failed to start Telegram Bot: {e}", exc_info=True)


if __name__ == "__main__":
    main()
# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SESSIONS_DIR = DATA_DIR / "sessions"
MEMORY_DIR = DATA_DIR / "memory"
LOGS_DIR = DATA_DIR / "logs"
WORKSPACE_DIR = PROJECT_ROOT / "workspace"

for d in [DATA_DIR, SESSIONS_DIR, MEMORY_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_first(names: list[str], default: str) -> str:
    """Return the first non-empty env value from *names*, else *default*."""
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return default

# API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Local Host
OLLAMA_ENABLED = _env_bool("OLLAMA_ENABLED", True)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "{your_ollama_base_url}")

YOUR_MODEL = _env_first(["YOUR_MODEL_ALIAS"], "{your_model_name}")

# Setup default roles 
OLLAMA_MODEL = _env_first(["OLLAMA_MODEL"], YOUR_MODEL)
OLLAMA_ORCHESTRATOR_MODEL = _env_first(
    ["OLLAMA_ORCHESTRATOR_MODEL", "YOUR_MODEL_ALIAS"],
    YOUR_MODEL,
)
OLLAMA_CLASSIFIER_MODEL = _env_first(
    ["OLLAMA_CLASSIFIER_MODEL", "YOUR_MODEL_ALIAS"],
    YOUR_MODEL,
)
OLLAMA_RESEARCH_MODEL = _env_first(
    ["OLLAMA_RESEARCH_MODEL", "YOUR_MODEL_ALIAS"],
    YOUR_MODEL,
)
OLLAMA_CODER_MODEL = _env_first(
    ["OLLAMA_CODER_MODEL", "YOUR_MODEL_ALIAS"],
    YOUR_MODEL,
)

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_USER_ID = int(os.getenv("TELEGRAM_USER_ID", "0"))

# Scheduler
BRIEFING_HOUR = int(os.getenv("BRIEFING_HOUR", "7"))
BRIEFING_MINUTE = int(os.getenv("BRIEFING_MINUTE", "0"))
AIRFLOW_DAILY_REPORT_CRON = os.getenv("AIRFLOW_DAILY_REPORT_CRON", "0 7 * * *")

# Database
TRUSTED_DB_URL = os.getenv(
    "TRUSTED_DB_URL",
    "postgresql+psycopg://agent:agent@localhost:5432/agent_ai",
)
TRUSTED_DB_REQUIRED = _env_bool("TRUSTED_DB_REQUIRED", False)

# Knowledge persistence (hybrid: PostgreSQL source of truth + ChromaDB semantic index)
KNOWLEDGE_DB_ENABLED = _env_bool("KNOWLEDGE_DB_ENABLED", True)
KNOWLEDGE_DB_REQUIRED = _env_bool("KNOWLEDGE_DB_REQUIRED", False)
KNOWLEDGE_ALLOW_NATURAL_LANGUAGE_COMMANDS = _env_bool(
    "KNOWLEDGE_ALLOW_NATURAL_LANGUAGE_COMMANDS", True
)
KNOWLEDGE_MAX_SEARCH_RESULTS = int(os.getenv("KNOWLEDGE_MAX_SEARCH_RESULTS", "5"))
KNOWLEDGE_MAX_CONTENT_CHARS = int(os.getenv("KNOWLEDGE_MAX_CONTENT_CHARS", "12000"))
KNOWLEDGE_MAX_RECENT_ITEMS = int(os.getenv("KNOWLEDGE_MAX_RECENT_ITEMS", "10"))

# Verifier
VERIFIER_MIN_SOURCES = int(os.getenv("VERIFIER_MIN_SOURCES", "2"))
VERIFIER_MIN_CONFIDENCE = float(os.getenv("VERIFIER_MIN_CONFIDENCE", "0.6"))
VERIFIER_MAX_SEARCH_RESULTS = int(os.getenv("VERIFIER_MAX_SEARCH_RESULTS", "8"))
VERIFIER_SOURCE_WEIGHTS_JSON = os.getenv("VERIFIER_SOURCE_WEIGHTS_JSON", "")
VERIFIER_ALLOWED_DOMAINS = [
    item.strip().lower() for item in os.getenv("VERIFIER_ALLOWED_DOMAINS", "").split(",") if item.strip()
]
VERIFIER_DENIED_DOMAINS = [
    item.strip().lower() for item in os.getenv("VERIFIER_DENIED_DOMAINS", "").split(",") if item.strip()
]

# Trusted claim dedupe
TRUSTED_CLAIM_SIMILARITY_THRESHOLD = float(
    os.getenv("TRUSTED_CLAIM_SIMILARITY_THRESHOLD", "0.92")
)
TRUSTED_CLAIM_SEMANTIC_SIMILARITY_THRESHOLD = float(
    os.getenv("TRUSTED_CLAIM_SEMANTIC_SIMILARITY_THRESHOLD", "0.88")
)
TRUSTED_CLAIM_ENABLE_SEMANTIC_DEDUPE = _env_bool(
    "TRUSTED_CLAIM_ENABLE_SEMANTIC_DEDUPE", True
)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")


# Airflow runtime policy
AIRFLOW_REPORT_CATCHUP = _env_bool("AIRFLOW_REPORT_CATCHUP", True)
AIRFLOW_REPORT_RETRIES = int(os.getenv("AIRFLOW_REPORT_RETRIES", "3"))
AIRFLOW_REPORT_RETRY_DELAY_MINUTES = int(os.getenv("AIRFLOW_REPORT_RETRY_DELAY_MINUTES", "10"))
AIRFLOW_REPORT_DAGRUN_TIMEOUT_MINUTES = int(os.getenv("AIRFLOW_REPORT_DAGRUN_TIMEOUT_MINUTES", "60"))
AIRFLOW_REPORT_MAX_ACTIVE_RUNS = int(os.getenv("AIRFLOW_REPORT_MAX_ACTIVE_RUNS", "1"))
AIRFLOW_REPORT_CHAT_ID = os.getenv("AIRFLOW_REPORT_CHAT_ID", "").strip()
AIRFLOW_REPORT_CATEGORIES = [
    item.strip().lower()
    for item in os.getenv("AIRFLOW_REPORT_CATEGORIES", "").split(",")
    if item.strip()
]

# Crawler runtime
CRAWLER_POLL_SECONDS = int(os.getenv("CRAWLER_POLL_SECONDS", "900"))

# Camoufox Server
CAMOUFOX_ENABLED = _env_bool("CAMOUFOX_ENABLED", True)
CAMOUFOX_API_URL = os.getenv("CAMOUFOX_API_URL", "{your_camoufox_api_url}")
CAMOUFOX_STRICT_ONLY = _env_bool("CAMOUFOX_STRICT_ONLY", True)

# Tavily Search (fallback when Camoufox is disabled or unreachable)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_ENABLED = _env_bool("TAVILY_ENABLED", False)

# Crawl4AI
CRAWL4AI_ENABLED = _env_bool("CRAWL4AI_ENABLED", True)
RESEARCH_MAX_SEARCH_QUERIES = int(os.getenv("RESEARCH_MAX_SEARCH_QUERIES", "2"))
RESEARCH_MAX_DISCOVERED_SOURCES = int(os.getenv("RESEARCH_MAX_DISCOVERED_SOURCES", "5"))
RESEARCH_SOURCE_ALLOWLIST = [
    item.strip().lower()
    for item in os.getenv(
        "RESEARCH_SOURCE_ALLOWLIST",
        "reuters.com,apnews.com,bbc.com,bloomberg.com,ft.com,wsj.com,theguardian.com,cnn.com,nytimes.com,vnexpress.net,thanhnien.vn,tuoitre.vn",
    ).split(",")
    if item.strip()
]

# Memory in conversation
MAX_CONVERSATION_HISTORY = 20
CHROMA_COLLECTION_NAME = "agent_stock_memory"
KNOWLEDGE_MEMORY_TYPE = os.getenv("KNOWLEDGE_MEMORY_TYPE", "user_knowledge")

# Load system prompts from markdown files
WORKSPACE_PRIMING_ENABLED = _env_bool("WORKSPACE_PRIMING_ENABLED", True)
WORKSPACE_PRIMING_MAX_CHARS = int(os.getenv("WORKSPACE_PRIMING_MAX_CHARS", "24000"))
_workspace_priming_files_raw = os.getenv(
    "WORKSPACE_PRIMING_FILES",
    "AGENTS.md,BOOTSTRAP.md,IDENTITY.md,SOUL.md,USER.md,TOOLS.md,HEARTBEAT.md",
)
WORKSPACE_PRIMING_FILES = [
    item.strip() for item in _workspace_priming_files_raw.split(",") if item.strip()
]

WORKSPACE_PRIMING_FILE_SETS: dict[str, list[str]] = {
    "orchestrator": [
        "AGENTS.md",
        "BOOTSTRAP.md",
        "HEARTBEAT.md",
        "IDENTITY.md",
        "SOUL.md",
        "TOOLS.md",
        "USER.md",
        "MODELS.md",
    ],
    "classifier": [
        "skills/Classifying/CLASSIFIER.md",
    ],
    "researcher": [
        "skills/Researching/RESEARCHER.md",
    ],
    "coder": [
        "skills/Coding/CODER.md",
    ],
}

# Temp context dir (for model handoff serialisation)
TEMP_CONTEXT_DIR = SESSIONS_DIR

DAILY_BRIEFING_SYSTEM_PROMPT = """You are a Daily Briefing Agent that compiles daily news summaries.
Your job:
1. Search for the most important news of the day across key categories
2. For each story, provide a brief summary and source link
3. Format the briefing in a clean, readable structure

Categories to cover:
- 🌍 World News (top 2-3 stories)
- 💼 Business & Economy
- 🤖 Technology & AI
- 📈 Markets (stocks, crypto highlights)

Keep each summary to 2-3 sentences. Be factual and objective."""
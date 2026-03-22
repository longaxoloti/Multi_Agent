import logging
from importlib.util import find_spec
from typing import Optional
import os
from langchain_ollama import ChatOllama

from main.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    OLLAMA_ENABLED,
    OLLAMA_BASE_URL,
    OLLAMA_ORCHESTRATOR_MODEL,
    OLLAMA_RESEARCH_MODEL,
    OLLAMA_CODER_MODEL,
)

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None

def _module_available(module_name: str) -> bool:
    return find_spec(module_name) is not None

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

MODEL_ROUTING = {
    "research": os.getenv("MODEL_RESEARCH", "ollama"),
    "analysis": os.getenv("MODEL_ANALYSIS", "ollama"),
    "chat": os.getenv("MODEL_CHAT", "ollama"),
    "code": os.getenv("MODEL_CODE", "ollama"),
    "orchestrator": os.getenv("MODEL_ORCHESTRATOR", "ollama"),
}

_OLLAMA_TASK_MODEL: dict[str, str] = {
    "orchestrator": OLLAMA_ORCHESTRATOR_MODEL,
    "chat": OLLAMA_ORCHESTRATOR_MODEL,
    "research": OLLAMA_RESEARCH_MODEL,
    "analysis": OLLAMA_RESEARCH_MODEL,
    "code": OLLAMA_CODER_MODEL,
}

def _build_ollama(temperature: float = 0.7, model_override: Optional[str] = None):
    if not OLLAMA_ENABLED:
        return None
    try:
        model = model_override or OLLAMA_ORCHESTRATOR_MODEL
        return ChatOllama(
            model=model,
            base_url=OLLAMA_BASE_URL,
            temperature=temperature,
        )
    except Exception as e:
        logger.warning("Failed to init Ollama: %s", e)
        return None

def _build_gemini(temperature: float = 0.7):
    if ChatGoogleGenerativeAI is None:
        logger.warning("Gemini provider unavailable: langchain_google_genai is not installed")
        return None
    if not GEMINI_API_KEY:
        return None
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=temperature,
        convert_system_message_to_human=True,
    )

def _build_openai(temperature: float = 0.7):
    if not OPENAI_API_KEY:
        return None
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o",
            api_key=OPENAI_API_KEY,
            temperature=temperature,
        )
    except Exception as e:
        logger.warning("Failed to init OpenAI: %s", e)
        return None

def _build_anthropic(temperature: float = 0.7):
    if not ANTHROPIC_API_KEY:
        return None
    try:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=ANTHROPIC_API_KEY,
            temperature=temperature,
        )
    except Exception as e:
        logger.warning("Failed to init Anthropic: %s", e)
        return None

_PROVIDERS = {
    "ollama": _build_ollama,
    "gemini": _build_gemini,
    "openai": _build_openai,
    "anthropic": _build_anthropic,
}

def get_llm(
    task_type: str = "chat",
    temperature: float = 0.7,
    streaming: bool = True,
    model_override: Optional[str] = None,
):
    preferred = MODEL_ROUTING.get(task_type, "ollama")
    resolved_ollama_model = model_override or _OLLAMA_TASK_MODEL.get(task_type, OLLAMA_ORCHESTRATOR_MODEL)
    if preferred == "ollama":
        llm = _build_ollama(temperature, model_override=resolved_ollama_model)
        if llm:
            logger.debug(
                "Using ollama/%s for task_type=%s", resolved_ollama_model, task_type
            )
            return llm
    else:
        builder = _PROVIDERS.get(preferred)
        if builder:
            llm = builder(temperature)
            if llm:
                logger.debug("Using %s for task_type=%s", preferred, task_type)
                return llm

    fallback_order = ["ollama", "gemini", "openai", "anthropic"]
    for name in fallback_order:
        if name == preferred:
            continue
        try:
            if name == "ollama":
                llm = _build_ollama(temperature, model_override=resolved_ollama_model)
            else:
                llm = _PROVIDERS[name](temperature)
            if llm:
                logger.info("Falling back to %s for task_type=%s", name, task_type)
                return llm
        except Exception as e:
            logger.debug("Provider %s failed: %s", name, e)
            continue

    raise RuntimeError(
        "No LLM provider available. "
        "Ensure Ollama is running, "
        "or configure GEMINI_API_KEY."
    )

def get_available_providers() -> list[str]:
    available = []
    if OLLAMA_ENABLED and _module_available("langchain_ollama"):
        available.append("ollama")
    if GEMINI_API_KEY and ChatGoogleGenerativeAI is not None:
        available.append("gemini")
    if OPENAI_API_KEY and _module_available("langchain_openai"):
        available.append("openai")
    if ANTHROPIC_API_KEY and _module_available("langchain_anthropic"):
        available.append("anthropic")
    return available
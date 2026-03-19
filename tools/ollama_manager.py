import json
import logging
import asyncio
from pathlib import Path
from typing import Any
import httpx
from main.config import OLLAMA_BASE_URL, TEMP_CONTEXT_DIR

logger = logging.getLogger(__name__)

async def unload_model(model_name: str, base_url: str = OLLAMA_BASE_URL) -> bool:
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {"model": model_name, "keep_alive": 0, "prompt": ""}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code < 300:
                logger.info("Unloaded model from RAM: %s", model_name)
                return True
            logger.warning(
                "Ollama unload returned HTTP %s for model %s",
                resp.status_code,
                model_name,
            )
            return False
    except Exception as exc:
        logger.warning("Failed to unload model %s: %s", model_name, exc)
        return False


def unload_model_sync(model_name: str, base_url: str = OLLAMA_BASE_URL) -> bool:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, unload_model(model_name, base_url))
                return future.result(timeout=15)
        return loop.run_until_complete(unload_model(model_name, base_url))
    except Exception as exc:
        logger.warning("unload_model_sync failed for %s: %s", model_name, exc)
        return False

def _ctx_path(session_id: str) -> Path:
    return Path(TEMP_CONTEXT_DIR) / f"{session_id}_ctx.json"


def save_context(context: dict[str, Any], session_id: str) -> None:
    path = _ctx_path(session_id)
    try:
        path.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.debug("Saved context for session %s → %s", session_id, path)
    except Exception as exc:
        logger.error("Failed to save context for session %s: %s", session_id, exc)


def load_context(session_id: str) -> dict[str, Any]:
    path = _ctx_path(session_id)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("Failed to load context for session %s: %s", session_id, exc)
        return {}


def clear_context(session_id: str) -> None:
    path = _ctx_path(session_id)
    try:
        if path.exists():
            path.unlink()
            logger.debug("Cleared context for session %s", session_id)
    except Exception as exc:
        logger.warning("Failed to clear context for session %s: %s", session_id, exc)

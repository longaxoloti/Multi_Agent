from __future__ import annotations

import logging
import time
from typing import Iterable

import httpx

from main.config import (
    KNOWLEDGE_EMBEDDING_DIMS,
    KNOWLEDGE_EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
)

logger = logging.getLogger(__name__)


class OllamaUnavailableError(RuntimeError):
    """Raised when Ollama is temporarily unavailable for embeddings."""

_EMBEDDING_COOLDOWN_UNTIL = 0.0
_EMBEDDING_LAST_ERROR = ""


def _set_embedding_cooldown(seconds: float, reason: str) -> None:
    global _EMBEDDING_COOLDOWN_UNTIL, _EMBEDDING_LAST_ERROR
    _EMBEDDING_COOLDOWN_UNTIL = max(time.time() + max(seconds, 0.0), _EMBEDDING_COOLDOWN_UNTIL)
    _EMBEDDING_LAST_ERROR = (reason or "").strip()


def _is_in_embedding_cooldown() -> bool:
    return time.time() < _EMBEDDING_COOLDOWN_UNTIL


def _normalize_vector(values: Iterable[float], expected_dims: int) -> list[float]:
    vector = [float(v) for v in values]
    if len(vector) < expected_dims:
        vector.extend([0.0] * (expected_dims - len(vector)))
    return vector[:expected_dims]


def embed_text_ollama(
    text: str,
    *,
    model: str = KNOWLEDGE_EMBEDDING_MODEL,
    expected_dims: int = KNOWLEDGE_EMBEDDING_DIMS,
    base_url: str = OLLAMA_BASE_URL,
    timeout_seconds: float = 30.0,
) -> list[float]:
    if _is_in_embedding_cooldown():
        detail = _EMBEDDING_LAST_ERROR or "temporary connection issue"
        raise OllamaUnavailableError(f"Ollama embedding is cooling down: {detail}")

    endpoint = f"{base_url.rstrip('/')}/api/embeddings"
    last_error: Exception | None = None

    payload = {
        "model": model,
        "prompt": (text or "").strip(),
    }

    for attempt in range(3):
        try:
            with httpx.Client(timeout=timeout_seconds) as client:
                response = client.post(endpoint, json=payload)
                response.raise_for_status()
                data = response.json() or {}
            embedding = data.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                raise ValueError("Ollama embedding response is missing 'embedding' list")
            if attempt > 0:
                logger.info("Embedding succeeded after retry %s for model %s", attempt + 1, model)
            return _normalize_vector(embedding, expected_dims)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            last_error = exc
            _set_embedding_cooldown(30.0, str(exc))
            logger.warning("Ollama embedding unavailable at %s: %s", endpoint, exc)
            raise OllamaUnavailableError(str(exc)) from exc
        except httpx.HTTPStatusError as exc:
            last_error = exc
            logger.warning("Ollama embedding HTTP error for model %s (attempt %s/3): %s", model, attempt + 1, exc)
        except Exception as exc:
            last_error = exc
            logger.warning("Failed to generate embedding via Ollama (%s, attempt %s/3): %s", model, attempt + 1, exc)

        if attempt < 2:
            time.sleep(1.5 * (attempt + 1))

    _set_embedding_cooldown(10.0, str(last_error or "unknown embedding error"))
    if last_error is not None:
        raise last_error
    raise OllamaUnavailableError("Embedding generation failed")
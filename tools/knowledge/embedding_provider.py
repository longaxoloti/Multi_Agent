from __future__ import annotations

import logging
from typing import Iterable

import httpx

from main.config import (
    KNOWLEDGE_EMBEDDING_DIMS,
    KNOWLEDGE_EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
)

logger = logging.getLogger(__name__)


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
    payload = {
        "model": model,
        "prompt": (text or "").strip(),
    }
    endpoint = f"{base_url.rstrip('/')}/api/embeddings"
    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.post(endpoint, json=payload)
            response.raise_for_status()
            data = response.json() or {}
    except Exception as exc:
        logger.error("Failed to generate embedding via Ollama (%s): %s", model, exc)
        raise

    embedding = data.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        raise ValueError("Ollama embedding response is missing 'embedding' list")

    return _normalize_vector(embedding, expected_dims)

from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from typing import Optional

from main.config import (
    KNOWLEDGE_ALLOW_NATURAL_LANGUAGE_COMMANDS,
    KNOWLEDGE_DB_ENABLED,
    KNOWLEDGE_DB_REQUIRED,
    KNOWLEDGE_EMBEDDING_DIMS,
    KNOWLEDGE_EMBEDDING_MODEL,
    KNOWLEDGE_EMBEDDING_PROVIDER,
    KNOWLEDGE_MAX_CONTENT_CHARS,
    KNOWLEDGE_MAX_RECENT_ITEMS,
    KNOWLEDGE_MAX_SEARCH_RESULTS,
    KNOWLEDGE_PGVECTOR_REQUIRED,
)
from storage.trusted_db import TrustedDBRepository, UserKnowledgeRecord
from tools.embedding_provider import embed_text_ollama

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeRequest:
    action: str
    content: str = ""
    record_id: str = ""
    query: str = ""
    category: str = ""
    limit: int = KNOWLEDGE_MAX_SEARCH_RESULTS


class KnowledgeService:
    def __init__(
        self,
        *,
        trusted_repo: Optional[TrustedDBRepository] = None,
        db_enabled: bool = KNOWLEDGE_DB_ENABLED,
        db_required: bool = KNOWLEDGE_DB_REQUIRED,
        embedder=None,
    ):
        self._db_enabled = db_enabled
        self._db_required = db_required
        self._pgvector_required = KNOWLEDGE_PGVECTOR_REQUIRED
        self._repo = trusted_repo
        self._repo_ready = False
        self._embedder = embedder or embed_text_ollama
        self._embedding_provider = KNOWLEDGE_EMBEDDING_PROVIDER
        self._embedding_model = KNOWLEDGE_EMBEDDING_MODEL
        self._embedding_dims = int(KNOWLEDGE_EMBEDDING_DIMS)
        self._init_repo()

    def _init_repo(self) -> None:
        if not self._db_enabled:
            return
        try:
            self._repo = self._repo or TrustedDBRepository()
            self._repo.initialize()
            if (
                self._pgvector_required
                and self._repo.engine.dialect.name == "postgresql"
                and not self._repo.is_pgvector_ready()
            ):
                raise RuntimeError("pgvector is required but not ready in PostgreSQL")
            self._repo_ready = True
        except Exception as exc:
            self._repo_ready = False
            logger.error("Knowledge DB unavailable: %s", exc, exc_info=True)
            raise RuntimeError("Knowledge DB/pgvector initialization failed") from exc

    @property
    def can_use_db(self) -> bool:
        return self._db_enabled and self._repo_ready and self._repo is not None

    def save(
        self,
        *,
        chat_id: str,
        content: str,
        category: str = "note",
        title: str = "",
        metadata: Optional[dict] = None,
    ) -> dict:
        final_content = (content or "").strip()[:KNOWLEDGE_MAX_CONTENT_CHARS]
        if not final_content:
            raise ValueError("content is empty")

        final_category = (category or "note").strip().lower()
        final_metadata = metadata or {}

        if not self.can_use_db:
            raise RuntimeError("Knowledge database is not available")

        embedding_vector = self._build_embedding(final_content)

        record_id = self._repo.save_knowledge_record(
            chat_id=str(chat_id),
            category=final_category,
            title=title,
            content=final_content,
            tags=[],
            metadata=final_metadata,
            embedding_model=self._embedding_model,
            embedding_dims=self._embedding_dims,
            embedding=embedding_vector,
        )

        return {
            "record_id": record_id,
            "stored_in_db": bool(record_id),
            "stored_in_vector": bool(record_id),
            "category": final_category,
        }

    def get(self, *, chat_id: str, record_id: str) -> Optional[dict]:
        if not self.can_use_db:
            raise RuntimeError("Knowledge database is not available")

        item = self._repo.get_knowledge_record(record_id=record_id, chat_id=str(chat_id))
        if item:
            return self._serialize_record(item)
        return None

    def search(
        self,
        *,
        chat_id: str,
        query: str,
        limit: int = KNOWLEDGE_MAX_SEARCH_RESULTS,
        category: Optional[str] = None,
    ) -> list[dict]:
        if not self.can_use_db:
            raise RuntimeError("Knowledge database is not available")

        safe_limit = max(1, min(limit, KNOWLEDGE_MAX_SEARCH_RESULTS))
        query_embedding = self._build_embedding((query or "").strip())
        rows = self._repo.search_knowledge_records(
            chat_id=str(chat_id),
            query_embedding=query_embedding,
            limit=safe_limit,
            category=category,
        )

        results: list[dict] = []
        for item in rows:
            meta = item.get("metadata", {}) or {}
            results.append(
                {
                    "id": item.get("id"),
                    "content": item.get("content", ""),
                    "category": item.get("category") or meta.get("category") or "note",
                    "title": item.get("title") or meta.get("title", ""),
                    "distance": item.get("distance"),
                    "metadata": meta,
                    "embedding_model": item.get("embedding_model") or "",
                    "embedding_dims": int(item.get("embedding_dims") or 0),
                }
            )
        return results

    def list_recent(
        self,
        *,
        chat_id: str,
        limit: int = KNOWLEDGE_MAX_RECENT_ITEMS,
        category: Optional[str] = None,
    ) -> list[dict]:
        safe_limit = max(1, min(limit, KNOWLEDGE_MAX_RECENT_ITEMS))
        if not self.can_use_db:
            raise RuntimeError("Knowledge database is not available")

        items = self._repo.list_knowledge_records(
            chat_id=str(chat_id),
            limit=safe_limit,
            category=category,
        )
        return [self._serialize_record(item) for item in items]

    def delete(self, *, chat_id: str, record_id: str) -> bool:
        if not self.can_use_db:
            raise RuntimeError("Knowledge database is not available")
        return self._repo.delete_knowledge_record(record_id=record_id, chat_id=str(chat_id))

    def _build_embedding(self, text: str) -> list[float]:
        if self._embedding_provider != "ollama":
            raise ValueError(
                f"Unsupported embedding provider '{self._embedding_provider}'. Expected 'ollama'."
            )

        vector = self._embedder(
            text,
            model=self._embedding_model,
            expected_dims=self._embedding_dims,
        )
        if len(vector) != self._embedding_dims:
            raise ValueError(
                f"Embedding dimension mismatch: got {len(vector)}, expected {self._embedding_dims}."
            )
        return vector

    @staticmethod
    def _serialize_record(item: UserKnowledgeRecord) -> dict:
        return {
            "id": item.id,
            "chat_id": item.chat_id,
            "category": item.category,
            "title": item.title,
            "content": item.content,
            "metadata": item.metadata,
            "tags": item.tags,
            "embedding_model": item.embedding_model,
            "embedding_dims": item.embedding_dims,
            "embedding": item.embedding,
            "created_at": item.created_at.isoformat(),
            "updated_at": item.updated_at.isoformat(),
        }


def parse_knowledge_request(
    text: str,
    *,
    allow_natural_language: bool = KNOWLEDGE_ALLOW_NATURAL_LANGUAGE_COMMANDS,
) -> Optional[KnowledgeRequest]:
    raw = (text or "").strip()
    if not raw:
        return None

    command_patterns: list[tuple[str, str]] = [
        ("save", r"^/save(?:\s+(note|fact|artifact))?\s+(.+)$"),
        ("get", r"^/get\s+([\w\-]+)$"),
        ("search", r"^/search(?:\s+(note|fact|artifact))?\s+(.+)$"),
        ("list", r"^/list(?:\s+(\d+))?$"),
        ("delete", r"^/delete\s+([\w\-]+)$"),
    ]

    for action, pattern in command_patterns:
        match = re.match(pattern, raw, re.IGNORECASE | re.DOTALL)
        if not match:
            continue

        if action == "save":
            category = (match.group(1) or "note").strip().lower()
            content = (match.group(2) or "").strip()
            return KnowledgeRequest(action="save", content=content, category=category)
        if action == "get":
            return KnowledgeRequest(action="get", record_id=(match.group(1) or "").strip())
        if action == "search":
            category = (match.group(1) or "").strip().lower()
            query = (match.group(2) or "").strip()
            return KnowledgeRequest(action="search", query=query, category=category)
        if action == "list":
            limit = int(match.group(1) or KNOWLEDGE_MAX_RECENT_ITEMS)
            return KnowledgeRequest(action="list", limit=limit)
        if action == "delete":
            return KnowledgeRequest(action="delete", record_id=(match.group(1) or "").strip())

    if not allow_natural_language:
        return None

    natural_save = re.match(r"^(?:lưu|save)\s*[:：]\s*(.+)$", raw, re.IGNORECASE | re.DOTALL)
    if natural_save:
        return KnowledgeRequest(action="save", content=natural_save.group(1).strip(), category="note")

    natural_search = re.match(r"^(?:tìm|search)\s*[:：]\s*(.+)$", raw, re.IGNORECASE | re.DOTALL)
    if natural_search:
        return KnowledgeRequest(action="search", query=natural_search.group(1).strip(), category="")

    natural_get = re.match(r"^(?:lấy|get)\s*[:：]\s*([\w\-]+)$", raw, re.IGNORECASE)
    if natural_get:
        return KnowledgeRequest(action="get", record_id=natural_get.group(1).strip())

    return None

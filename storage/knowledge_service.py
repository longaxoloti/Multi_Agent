from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from typing import Optional

from config import (
    KNOWLEDGE_ALLOW_NATURAL_LANGUAGE_COMMANDS,
    KNOWLEDGE_DB_ENABLED,
    KNOWLEDGE_DB_REQUIRED,
    KNOWLEDGE_MAX_CONTENT_CHARS,
    KNOWLEDGE_MAX_RECENT_ITEMS,
    KNOWLEDGE_MAX_SEARCH_RESULTS,
    KNOWLEDGE_MEMORY_TYPE,
)
from storage.trusted_db import TrustedDBRepository, UserKnowledgeRecord

try:
    from memory.memory_manager import MemoryManager
except ModuleNotFoundError:
    MemoryManager = None

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
        memory_manager: Optional[MemoryManager] = None,
        trusted_repo: Optional[TrustedDBRepository] = None,
        db_enabled: bool = KNOWLEDGE_DB_ENABLED,
        db_required: bool = KNOWLEDGE_DB_REQUIRED,
    ):
        if memory_manager is not None:
            self._mm = memory_manager
        elif MemoryManager is not None:
            self._mm = MemoryManager()
        else:
            self._mm = None

        self._db_enabled = db_enabled
        self._db_required = db_required
        self._repo = trusted_repo
        self._repo_ready = False
        self._init_repo()

    def _init_repo(self) -> None:
        if not self._db_enabled:
            return
        try:
            self._repo = self._repo or TrustedDBRepository()
            self._repo.initialize()
            self._repo_ready = True
        except Exception as exc:
            self._repo_ready = False
            logger.warning("Knowledge DB unavailable. Falling back to Chroma-only mode: %s", exc)
            if self._db_required:
                raise

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

        record_id: Optional[str] = None
        if self.can_use_db:
            record_id = self._repo.save_knowledge_record(
                chat_id=str(chat_id),
                category=final_category,
                title=title,
                content=final_content,
                tags=[],
                metadata=final_metadata,
            )

        memory_id = None
        if self._mm is not None:
            memory_id = self._mm.add_memory(
                content=final_content,
                memory_type=KNOWLEDGE_MEMORY_TYPE,
                memory_id=record_id,
                chat_id=str(chat_id),
                scope="user",
                metadata={
                    "category": final_category,
                    "title": title or "",
                    "source": "knowledge_service",
                    **final_metadata,
                },
            )

        return {
            "record_id": record_id or memory_id,
            "stored_in_db": bool(record_id),
            "stored_in_vector": bool(memory_id),
            "category": final_category,
        }

    def get(self, *, chat_id: str, record_id: str) -> Optional[dict]:
        if self.can_use_db:
            item = self._repo.get_knowledge_record(record_id=record_id, chat_id=str(chat_id))
            if item:
                return self._serialize_record(item)

        if self._mm is None:
            return None

        raw = self._mm.get_memory(record_id)
        if not raw:
            return None
        metadata = raw.get("metadata", {}) or {}
        if metadata.get("chat_id") and str(metadata.get("chat_id")) != str(chat_id):
            return None

        return {
            "id": raw.get("id", record_id),
            "chat_id": str(chat_id),
            "category": metadata.get("category") or metadata.get("type") or "note",
            "title": metadata.get("title", ""),
            "content": raw.get("content", ""),
            "metadata": metadata,
            "created_at": metadata.get("timestamp", ""),
        }

    def search(
        self,
        *,
        chat_id: str,
        query: str,
        limit: int = KNOWLEDGE_MAX_SEARCH_RESULTS,
        category: Optional[str] = None,
    ) -> list[dict]:
        if self._mm is None:
            return []

        safe_limit = max(1, min(limit, KNOWLEDGE_MAX_SEARCH_RESULTS))
        rows = self._mm.search_memory(
            query=query,
            n_results=safe_limit,
            memory_type=KNOWLEDGE_MEMORY_TYPE,
            chat_id=str(chat_id),
            include_global=False,
        )

        normalized_category = (category or "").strip().lower()
        if normalized_category:
            rows = [
                item
                for item in rows
                if (item.get("metadata", {}).get("category", "").strip().lower() == normalized_category)
            ]

        results: list[dict] = []
        for item in rows:
            meta = item.get("metadata", {}) or {}
            results.append(
                {
                    "id": item.get("id"),
                    "content": item.get("content", ""),
                    "category": meta.get("category") or "note",
                    "title": meta.get("title", ""),
                    "distance": item.get("distance"),
                    "metadata": meta,
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
        if self.can_use_db:
            items = self._repo.list_knowledge_records(
                chat_id=str(chat_id),
                limit=safe_limit,
                category=category,
            )
            return [self._serialize_record(item) for item in items]
        return []

    def delete(self, *, chat_id: str, record_id: str) -> bool:
        db_deleted = False
        if self.can_use_db:
            db_deleted = self._repo.delete_knowledge_record(record_id=record_id, chat_id=str(chat_id))
        vector_deleted = self._mm.delete_memory(record_id) if self._mm is not None else False
        return db_deleted or vector_deleted

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

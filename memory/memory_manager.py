"""
MemoryManager — Conversation memory with optional PostgreSQL persistence.

Maintains an in-memory cache for current-session speed, with write-through
persistence to ``system.conversation_sessions`` when a DB session factory is
provided.  Falls back gracefully to pure in-memory if no DB is available.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, text
from sqlalchemy.orm import sessionmaker

from main.config import MAX_CONVERSATION_HISTORY
from rag.chunking import build_document_chunks

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Hybrid in-memory / PostgreSQL conversation memory.

    Usage::

        # Pure in-memory (legacy)
        mm = MemoryManager()

        # With DB persistence
        mm = MemoryManager(session_factory=repo._session_factory)
    """

    def __init__(
        self,
        *,
        session_factory: Optional[sessionmaker] = None,
        is_pg: bool = True,
    ):
        self._conversations: dict[str, list[dict]] = {}
        self._memories: dict[str, dict] = {}
        self._session_factory = session_factory
        self._is_pg = is_pg
        self._db_ready = session_factory is not None and is_pg
        mode = "PostgreSQL-backed" if self._db_ready else "in-memory only"
        logger.info("MemoryManager initialized (%s).", mode)

    # ==================================================================
    # Conversation history
    # ==================================================================

    def add_message(self, chat_id: str, role: str, content: str) -> None:
        """Add a message — cached in memory and persisted to DB if available."""
        if chat_id not in self._conversations:
            self._conversations[chat_id] = []

        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._conversations[chat_id].append(msg)

        # Trim in-memory window
        if len(self._conversations[chat_id]) > MAX_CONVERSATION_HISTORY:
            self._conversations[chat_id] = self._conversations[chat_id][
                -MAX_CONVERSATION_HISTORY:
            ]

        # Write-through to DB
        if self._db_ready:
            self._persist_message(chat_id, role, content, len(self._conversations[chat_id]) - 1)

    def get_conversation_history(self, chat_id: str) -> list[dict]:
        """
        Return recent conversation history.

        Tries in-memory first; if empty and DB is available, loads from DB.
        """
        cached = self._conversations.get(chat_id)
        if cached:
            return cached

        if self._db_ready:
            loaded = self._load_history_from_db(chat_id)
            if loaded:
                self._conversations[chat_id] = loaded
                return loaded

        return []

    def clear_conversation(self, chat_id: str) -> None:
        """Clear in-memory history for a chat. DB records are preserved for audit."""
        self._conversations.pop(chat_id, None)

    # ==================================================================
    # Session flush / load
    # ==================================================================

    def flush_session(self, chat_id: str, session_id: str) -> int:
        """
        Persist all in-memory messages for ``chat_id`` to the database.

        Returns the number of messages flushed.
        """
        if not self._db_ready:
            return 0

        history = self._conversations.get(chat_id, [])
        if not history:
            return 0

        from storage.models import ConversationSessionORM, _new_uuid, _utcnow

        count = 0
        try:
            with self._session_factory() as session:
                for idx, msg in enumerate(history):
                    row = ConversationSessionORM(
                        id=_new_uuid(),
                        chat_id=chat_id,
                        session_id=session_id,
                        role=msg["role"],
                        content=msg["content"],
                        message_index=idx,
                        metadata_json=json.dumps(
                            {"timestamp": msg.get("timestamp", "")},
                            ensure_ascii=False,
                        ),
                    )
                    session.add(row)
                    count += 1
                session.commit()
            logger.info("Flushed %d messages for chat %s / session %s", count, chat_id, session_id)
        except Exception as exc:
            logger.error("Failed to flush session to DB: %s", exc)
        return count

    def load_recent_sessions(
        self,
        chat_id: str,
        *,
        limit: int = MAX_CONVERSATION_HISTORY,
    ) -> list[dict]:
        """
        Load the most recent messages for ``chat_id`` from the database.

        Populates the in-memory cache and returns the messages.
        """
        if not self._db_ready:
            return []

        loaded = self._load_history_from_db(chat_id, limit=limit)
        if loaded:
            self._conversations[chat_id] = loaded
        return loaded

    # ==================================================================
    # Legacy memory API (unchanged)
    # ==================================================================

    def _build_scope_filter(
        self,
        chat_id: Optional[str],
        include_global: bool,
        memory_type: Optional[str],
    ) -> Optional[dict]:
        clauses: list[dict] = []
        if memory_type:
            clauses.append({"type": memory_type})

        if chat_id:
            if include_global:
                clauses.append(
                    {
                        "$or": [
                            {"chat_id": chat_id},
                            {"scope": "global"},
                        ]
                    }
                )
            else:
                clauses.append({"chat_id": chat_id})
        elif include_global:
            pass
        else:
            clauses.append({"scope": "user"})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def add_memory(
        self,
        content: str,
        memory_type: str = "general",
        metadata: Optional[dict] = None,
        memory_id: Optional[str] = None,
        chat_id: Optional[str] = None,
        scope: Optional[str] = None,
    ) -> str:
        if memory_id is None:
            memory_id = f"{memory_type}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        meta = {
            "type": memory_type,
            "timestamp": datetime.now().isoformat(),
            "scope": scope or ("user" if chat_id else "global"),
        }
        if chat_id:
            meta["chat_id"] = str(chat_id)
        if metadata:
            meta.update(metadata)
        self._memories[memory_id] = {
            "id": memory_id,
            "content": content,
            "metadata": meta,
        }
        logger.info("Stored memory '%s' (type=%s, len=%d)", memory_id, memory_type, len(content))
        return memory_id

    def search_memory(
        self,
        query: str,
        n_results: int = 5,
        memory_type: Optional[str] = None,
        chat_id: Optional[str] = None,
        include_global: bool = True,
    ) -> list[dict]:
        query_tokens = [token.strip().lower() for token in (query or "").split() if token.strip()]
        rows: list[dict] = []
        for row in self._memories.values():
            meta = row.get("metadata", {})
            if memory_type and meta.get("type") != memory_type:
                continue

            row_chat_id = str(meta.get("chat_id")) if meta.get("chat_id") is not None else None
            row_scope = meta.get("scope")
            if chat_id:
                if include_global:
                    if row_chat_id != str(chat_id) and row_scope != "global":
                        continue
                elif row_chat_id != str(chat_id):
                    continue
            elif not include_global and row_scope != "user":
                continue

            content = str(row.get("content") or "")
            meta_text = json.dumps(meta, ensure_ascii=False)
            searchable_text = f"{content}\n{meta_text}".lower()
            if query_tokens and not all(token in searchable_text for token in query_tokens):
                continue

            rows.append(
                {
                    "content": content,
                    "metadata": meta,
                    "distance": 0.0,
                    "id": row.get("id"),
                }
            )
        return rows[:max(1, n_results)]

    def get_memory_count(self) -> int:
        return len(self._memories)

    def get_memory(self, memory_id: str) -> Optional[dict]:
        return self._memories.get(memory_id)

    def save_conversation_summary(
        self,
        chat_id: str,
        summary: str,
    ) -> str:
        return self.add_memory(
            content=summary,
            memory_type="conversation",
            metadata={"chat_id": chat_id},
            chat_id=chat_id,
            scope="user",
        )

    def save_crawled_page(
        self,
        url: str,
        title: str,
        content: str,
        chat_id: Optional[str] = None,
        scope: Optional[str] = None,
    ) -> str:
        stored_content = content[:2000]
        return self.add_memory(
            content=f"Page: {title}\nURL: {url}\n\n{stored_content}",
            memory_type="crawled_page",
            metadata={"url": url, "title": title},
            chat_id=chat_id,
            scope=scope,
        )

    def save_crawled_page_chunks(
        self,
        url: str,
        title: str,
        content: str,
        topic: str = "general",
        chat_id: Optional[str] = None,
        scope: Optional[str] = None,
    ) -> list[str]:
        docs = build_document_chunks(
            content=content,
            source_url=url,
            title=title,
            topic=topic,
        )
        stored_ids: list[str] = []
        for doc in docs:
            memory_id = self.add_memory(
                content=doc["content"],
                memory_type="rag_chunk",
                metadata=doc["metadata"],
                chat_id=chat_id,
                scope=scope,
            )
            stored_ids.append(memory_id)
        return stored_ids

    def save_user_preference(
        self,
        preference: str,
        chat_id: Optional[str] = None,
        is_global: bool = False,
    ) -> str:
        return self.add_memory(
            content=preference,
            memory_type="user_preference",
            metadata={},
            chat_id=None if is_global else chat_id,
            scope="global" if is_global else ("user" if chat_id else "global"),
        )

    def get_user_preferences(
        self,
        chat_id: Optional[str] = None,
        include_global: bool = True,
    ) -> list[dict]:
        preferences: list[dict] = []
        for row in self._memories.values():
            meta = row.get("metadata", {})
            if meta.get("type") != "user_preference":
                continue

            row_chat_id = str(meta.get("chat_id")) if meta.get("chat_id") is not None else None
            row_scope = meta.get("scope")

            if chat_id:
                if include_global:
                    if row_chat_id != str(chat_id) and row_scope != "global":
                        continue
                elif row_chat_id != str(chat_id):
                    continue
            elif not include_global and row_scope != "user":
                continue

            preferences.append(
                {
                    "content": row.get("content", ""),
                    "metadata": meta,
                    "id": row.get("id"),
                }
            )
        return preferences

    def delete_memory(self, memory_id: str) -> bool:
        try:
            if memory_id in self._memories:
                del self._memories[memory_id]
            logger.info("Deleted memory '%s'", memory_id)
            return True
        except Exception as e:
            logger.warning("Failed to delete memory '%s': %s", memory_id, e)
            return False

    # ==================================================================
    # Internal DB helpers
    # ==================================================================

    def _persist_message(self, chat_id: str, role: str, content: str, msg_index: int) -> None:
        """Write a single message to the DB (fire-and-forget)."""
        try:
            from storage.models import ConversationSessionORM, _new_uuid

            with self._session_factory() as session:
                row = ConversationSessionORM(
                    id=_new_uuid(),
                    chat_id=chat_id,
                    session_id=chat_id,  # use chat_id as session until explicit session_id
                    role=role,
                    content=content,
                    message_index=msg_index,
                    metadata_json=json.dumps(
                        {"timestamp": datetime.now(timezone.utc).isoformat()},
                        ensure_ascii=False,
                    ),
                )
                session.add(row)
                session.commit()
        except Exception as exc:
            logger.debug("Failed to persist message to DB (non-critical): %s", exc)

    def _load_history_from_db(
        self,
        chat_id: str,
        *,
        limit: int = MAX_CONVERSATION_HISTORY,
    ) -> list[dict]:
        """Load recent messages from the DB."""
        try:
            from storage.models import ConversationSessionORM

            with self._session_factory() as session:
                rows = session.execute(
                    select(ConversationSessionORM)
                    .where(ConversationSessionORM.chat_id == chat_id)
                    .order_by(ConversationSessionORM.created_at.desc())
                    .limit(limit)
                ).scalars().all()

            # Reverse to get chronological order
            rows = list(reversed(rows))
            return [
                {
                    "role": r.role,
                    "content": r.content,
                    "timestamp": r.created_at.isoformat() if r.created_at else "",
                }
                for r in rows
            ]
        except Exception as exc:
            logger.debug("Failed to load history from DB (non-critical): %s", exc)
            return []

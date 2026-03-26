import json
import logging
from datetime import datetime
from typing import Optional

from main.config import MAX_CONVERSATION_HISTORY
from rag.chunking import build_document_chunks

logger = logging.getLogger(__name__)


class MemoryManager:
    def __init__(self):
        self._conversations: dict[str, list[dict]] = {}
        self._memories: dict[str, dict] = {}
        logger.info("MemoryManager initialized in legacy in-memory mode.")

    def add_message(self, chat_id: str, role: str, content: str) -> None:
        """Add a message to the conversation history for a specific chat."""
        if chat_id not in self._conversations:
            self._conversations[chat_id] = []
        self._conversations[chat_id].append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
            }
        )
        if len(self._conversations[chat_id]) > MAX_CONVERSATION_HISTORY:
            self._conversations[chat_id] = self._conversations[chat_id][
                -MAX_CONVERSATION_HISTORY:
            ]

    def get_conversation_history(self, chat_id: str) -> list[dict]:
        return self._conversations.get(chat_id, [])

    def clear_conversation(self, chat_id: str) -> None:
        self._conversations.pop(chat_id, None)

    # RAG
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

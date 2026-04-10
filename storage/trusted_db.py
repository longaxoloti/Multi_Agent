from __future__ import annotations

import difflib
import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine, inspect, select, text
from sqlalchemy.orm import Mapped, mapped_column, sessionmaker

from main.config import (
    DB_UNIFIED_MEMORY_MODE,
    TRUSTED_DB_URL,
)
from storage.models import Base  # shared Base for all schemas

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Legacy ORM models — retained only for backward compatibility
# ---------------------------------------------------------------------------

class TrustedClaimORM(Base):
    __tablename__ = "trusted_claims"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    topic: Mapped[str] = mapped_column(String(120), index=True)
    claim: Mapped[str] = mapped_column(Text)
    normalized_claim: Mapped[str] = mapped_column(String(512), index=True)
    claim_embedding_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    confidence: Mapped[float] = mapped_column(Float)
    sources_json: Mapped[str] = mapped_column(Text)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)
    last_verified_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)


# ---------------------------------------------------------------------------
# Dataclasses (public API)
# ---------------------------------------------------------------------------

@dataclass
class TrustedClaim:
    topic: str
    claim: str
    confidence: float
    sources: list[str]
    first_seen_at: datetime
    last_verified_at: datetime

@dataclass
class UserKnowledgeRecord:
    id: str
    chat_id: str
    category: str
    title: str
    content: str
    tags: list[str]
    metadata: dict
    embedding_model: str
    embedding_dims: int
    embedding: list[float]
    created_at: datetime
    updated_at: datetime


_UNIFIED_SCHEMAS = ["system", "profile", "knowledge", "security"]


class AgentDBRepository:
    """Central database repository for the multi-schema agent database."""

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or TRUSTED_DB_URL
        self._unified_memory_mode = DB_UNIFIED_MEMORY_MODE
        self.engine = create_engine(self.db_url, future=True, pool_pre_ping=True)
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    def initialize(self) -> None:
        """Create all schemas, tables, vector columns, and HNSW indexes."""
        is_pg = self.engine.dialect.name == "postgresql"

        if is_pg:
            with self.engine.begin() as conn:
                # Create pgvector extension
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                # Create PostgreSQL schemas used by the unified architecture
                for schema in _UNIFIED_SCHEMAS:
                    conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
            TrustedClaimORM.__table__.create(self.engine, checkfirst=True)
        else:
            # SQLite: only create public-schema tables (no schema support)
            public_tables = [
                t for t in Base.metadata.sorted_tables
                if t.schema is None
            ]
            Base.metadata.create_all(self.engine, tables=public_tables)

        # Legacy compatibility
        self._ensure_schema_compatibility()

        logger.info("Agent DB schema ready (all schemas initialized)")

    def _get_column_names(self, table_name: str, *, schema: Optional[str] = None) -> set[str]:
        """Return table column names without reflecting pgvector types via SQLAlchemy inspector."""
        if self.engine.dialect.name == "postgresql":
            table_schema = schema or "public"
            with self.engine.begin() as conn:
                rows = conn.execute(
                    text(
                        """
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_schema = :table_schema
                          AND table_name = :table_name
                        """
                    ),
                    {"table_schema": table_schema, "table_name": table_name},
                ).all()
            return {row[0] for row in rows}

        inspector = inspect(self.engine)
        cols = inspector.get_columns(table_name, schema=schema)
        return {col["name"] for col in cols}

    def is_pgvector_ready(self) -> bool:
        if self.engine.dialect.name != "postgresql":
            return False
        try:
            with self.engine.begin() as conn:
                ext_row = conn.execute(
                    text("SELECT 1 FROM pg_extension WHERE extname = 'vector' LIMIT 1")
                ).first()
                if not ext_row:
                    return False

                col_row = conn.execute(
                    text(
                        """
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_schema = 'knowledge'
                          AND table_name = 'memory_embeddings'
                          AND column_name = 'embedding'
                        LIMIT 1
                        """
                    )
                ).first()
                return bool(col_row)
        except Exception:
            return False

    def _upsert_unified_entity(self, conn, *, owner_user_id: str, chat_id: str) -> str:
        entity_ref = str(chat_id)
        row = conn.execute(
            text(
                """
                INSERT INTO knowledge.entities (
                    id,
                    entity_type,
                    entity_ref,
                    owner_user_id,
                    title,
                    source_type,
                    trust_score,
                    status,
                    created_at,
                    updated_at
                )
                VALUES (
                    :id,
                    'chat',
                    :entity_ref,
                    :owner_user_id,
                    :title,
                    'manual_save',
                    0.9,
                    'active',
                    now(),
                    now()
                )
                ON CONFLICT (entity_type, entity_ref, owner_user_id)
                DO UPDATE SET updated_at = now()
                RETURNING id
                """
            ),
            {
                "id": str(uuid.uuid4()),
                "entity_ref": entity_ref,
                "owner_user_id": owner_user_id,
                "title": entity_ref,
            },
        ).first()
        return str(row[0])

    @staticmethod
    def _set_current_user_id(conn, user_id: str) -> None:
        conn.execute(
            text("SELECT set_config('app.current_user_id', :user_id, true)"),
            {"user_id": str(user_id or "")},
        )

    def _ensure_schema_compatibility(self) -> None:
        inspector = inspect(self.engine)
        table_names = inspector.get_table_names()
        if "trusted_claims" not in inspector.get_table_names():
            pass

        dialect = self.engine.dialect.name
        with self.engine.begin() as conn:
            if "trusted_claims" in table_names:
                cols = self._get_column_names("trusted_claims")
                normalized_exists = "normalized_claim" in cols
                claim_embedding_exists = "claim_embedding_json" in cols

                if not normalized_exists:
                    if dialect == "postgresql":
                        conn.execute(text("ALTER TABLE trusted_claims ADD COLUMN IF NOT EXISTS normalized_claim VARCHAR(512)"))
                    elif dialect == "sqlite":
                        conn.execute(text("ALTER TABLE trusted_claims ADD COLUMN normalized_claim VARCHAR(512)"))
                    else:
                        logger.warning("Unsupported dialect for auto-migration: %s", dialect)
                        return

                    conn.execute(text("UPDATE trusted_claims SET normalized_claim = lower(claim) WHERE normalized_claim IS NULL"))

                    if dialect == "postgresql":
                        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_trusted_claims_normalized ON trusted_claims (normalized_claim)"))
                    elif dialect == "sqlite":
                        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_trusted_claims_normalized ON trusted_claims (normalized_claim)"))

                if not claim_embedding_exists:
                    if dialect == "postgresql":
                        conn.execute(text("ALTER TABLE trusted_claims ADD COLUMN IF NOT EXISTS claim_embedding_json TEXT"))
                    elif dialect == "sqlite":
                        conn.execute(text("ALTER TABLE trusted_claims ADD COLUMN claim_embedding_json TEXT"))
                    else:
                        logger.warning("Unsupported dialect for embedding column migration: %s", dialect)
                        return

    @staticmethod
    def _normalize_claim(text: str) -> str:
        normalized = (text or "").lower().strip()
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = re.sub(r"[^\w\s]", "", normalized)
        return normalized[:512]

    @staticmethod
    def _merge_sources(existing: list[str], incoming: list[str]) -> list[str]:
        return list(dict.fromkeys([*existing, *incoming]))

    def _find_similar_claim(
        self,
        session,
        topic: str,
        normalized_claim: str,
        seen_at: datetime,
    ) -> Optional[TrustedClaimORM]:
        window_start = seen_at - timedelta(days=30)
        candidates = session.execute(
            select(TrustedClaimORM).where(
                TrustedClaimORM.topic == topic,
                TrustedClaimORM.last_verified_at >= window_start,
            )
        ).scalars().all()

        best_match = None
        best_score = 0.0
        threshold = 0.85
        for row in candidates:
            existing_norm = row.normalized_claim or self._normalize_claim(row.claim)
            score = difflib.SequenceMatcher(None, existing_norm, normalized_claim).ratio()

            if score > best_score:
                best_score = score
                best_match = row

        if best_match and best_score >= threshold:
            return best_match
        return None

    def add_trusted_claim(
        self,
        topic: str,
        claim: str,
        confidence: float,
        sources: list[str],
        seen_at: Optional[datetime] = None,
    ) -> int:
        claim_id, _ = self.upsert_trusted_claim(
            topic=topic,
            claim=claim,
            confidence=confidence,
            sources=sources,
            seen_at=seen_at,
        )
        return claim_id

    def upsert_trusted_claim(
        self,
        topic: str,
        claim: str,
        confidence: float,
        sources: list[str],
        seen_at: Optional[datetime] = None,
    ) -> tuple[int, bool]:
        now = seen_at or datetime.utcnow()
        normalized_claim = self._normalize_claim(claim)
        with self._session_factory() as session:
            existing = session.execute(
                select(TrustedClaimORM).where(
                    TrustedClaimORM.topic == topic,
                    TrustedClaimORM.normalized_claim == normalized_claim,
                )
            ).scalars().first()

            if not existing:
                existing = self._find_similar_claim(
                    session=session,
                    topic=topic,
                    normalized_claim=normalized_claim,
                    seen_at=now,
                )

            if existing:
                existing.confidence = max(existing.confidence, confidence)
                existing.last_verified_at = now
                existing_sources = []
                try:
                    existing_sources = json.loads(existing.sources_json) if existing.sources_json else []
                except json.JSONDecodeError:
                    existing_sources = []
                merged_sources = self._merge_sources(existing_sources, sources)
                existing.sources_json = json.dumps(merged_sources, ensure_ascii=False)
                session.add(existing)
                session.commit()
                session.refresh(existing)
                return existing.id, False

            item = TrustedClaimORM(
                topic=topic,
                claim=claim,
                normalized_claim=normalized_claim,
                confidence=confidence,
                sources_json=json.dumps(self._merge_sources([], sources), ensure_ascii=False),
                first_seen_at=now,
                last_verified_at=now,
            )
            session.add(item)
            session.commit()
            session.refresh(item)
            return item.id, True

    def list_trusted_claims_since(self, since: datetime) -> list[TrustedClaim]:
        stmt = (
            select(TrustedClaimORM)
            .where(TrustedClaimORM.last_verified_at >= since)
            .order_by(TrustedClaimORM.last_verified_at.desc())
        )
        with self._session_factory() as session:
            rows = session.execute(stmt).scalars().all()

        claims: list[TrustedClaim] = []
        for row in rows:
            try:
                sources = json.loads(row.sources_json) if row.sources_json else []
            except json.JSONDecodeError:
                sources = []
            claims.append(
                TrustedClaim(
                    topic=row.topic,
                    claim=row.claim,
                    confidence=row.confidence,
                    sources=sources,
                    first_seen_at=row.first_seen_at,
                    last_verified_at=row.last_verified_at,
                )
            )
        return claims

    def list_trusted_claims_between(self, start: datetime, end: datetime) -> list[TrustedClaim]:
        stmt = (
            select(TrustedClaimORM)
            .where(TrustedClaimORM.last_verified_at >= start)
            .where(TrustedClaimORM.last_verified_at < end)
            .order_by(TrustedClaimORM.last_verified_at.desc())
        )
        with self._session_factory() as session:
            rows = session.execute(stmt).scalars().all()

        claims: list[TrustedClaim] = []
        for row in rows:
            try:
                sources = json.loads(row.sources_json) if row.sources_json else []
            except json.JSONDecodeError:
                sources = []
            claims.append(
                TrustedClaim(
                    topic=row.topic,
                    claim=row.claim,
                    confidence=row.confidence,
                    sources=sources,
                    first_seen_at=row.first_seen_at,
                    last_verified_at=row.last_verified_at,
                )
            )
        return claims

    def list_last_24h(self) -> list[TrustedClaim]:
        return self.list_trusted_claims_since(datetime.utcnow() - timedelta(hours=24))

    @staticmethod
    def _safe_json_loads(raw: Optional[str], default):
        if not raw:
            return default
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return default

    def save_knowledge_record(
        self,
        *,
        chat_id: str,
        category: str,
        content: str,
        title: str = "",
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
        embedding_model: str = "",
        embedding_dims: int = 0,
        embedding: Optional[list[float]] = None,
        record_id: Optional[str] = None,
    ) -> str:
        if self.engine.dialect.name != "postgresql":
            raise RuntimeError("Knowledge vector storage requires PostgreSQL + pgvector")
        if not self._unified_memory_mode:
            raise RuntimeError("Legacy manual storage mode has been removed")

        now = datetime.utcnow()
        final_id = record_id or str(uuid.uuid4())
        payload = {
            "tags": tags or [],
            "metadata": metadata or {},
        }

        with self.engine.begin() as conn:
            self._set_current_user_id(conn, str(chat_id))
            entity_id = self._upsert_unified_entity(
                conn,
                owner_user_id=str(chat_id),
                chat_id=str(chat_id),
            )

            conn.execute(
                text(
                    """
                    INSERT INTO knowledge.memories (
                        id,
                        entity_id,
                        owner_user_id,
                        memory_type,
                        ingestion_mode,
                        summary,
                        content,
                        confidence,
                        decay_weight,
                        status,
                        valid_from,
                        content_hash,
                        canonical_hash,
                        source_schema,
                        source_record_id,
                        created_at,
                        updated_at,
                        metadata_json
                    ) VALUES (
                        :id,
                        :entity_id,
                        :owner_user_id,
                        :memory_type,
                        'user_pinned',
                        :summary,
                        :content,
                        0.95,
                        1.0,
                        'active',
                        :valid_from,
                        md5(:content),
                        md5(regexp_replace(lower(:content), '\\s+', ' ', 'g')),
                        'knowledge.memories',
                        :source_record_id,
                        :created_at,
                        :updated_at,
                        :metadata_json
                    )
                    """
                ),
                {
                    "id": final_id,
                    "entity_id": entity_id,
                    "owner_user_id": str(chat_id),
                    "memory_type": (category or "note").strip().lower(),
                    "summary": (title or "").strip()[:255],
                    "content": content,
                    "valid_from": now,
                    "source_record_id": final_id,
                    "created_at": now,
                    "updated_at": now,
                    "metadata_json": json.dumps(payload, ensure_ascii=False),
                },
            )

            if embedding:
                conn.execute(
                    text(
                        """
                        INSERT INTO knowledge.memory_embeddings (
                            id,
                            memory_id,
                            embedding,
                            model_name,
                            created_at
                        ) VALUES (
                            :id,
                            :memory_id,
                            CAST(:embedding_literal AS vector),
                            :model_name,
                            :created_at
                        )
                        """
                    ),
                    {
                        "id": str(uuid.uuid4()),
                        "memory_id": final_id,
                        "embedding_literal": self._to_vector_literal(embedding),
                        "model_name": (embedding_model or "").strip() or "bge-m3",
                        "created_at": now,
                    },
                )

            conn.execute(
                text(
                    """
                    INSERT INTO knowledge.access_stats (
                        id,
                        memory_id,
                        access_count,
                        last_accessed_at,
                        last_used_for_task_at,
                        created_at,
                        updated_at
                    ) VALUES (
                        :id,
                        :memory_id,
                        0,
                        NULL,
                        NULL,
                        now(),
                        now()
                    )
                    ON CONFLICT (memory_id) DO NOTHING
                    """
                ),
                {
                    "id": str(uuid.uuid4()),
                    "memory_id": final_id,
                },
            )
        return final_id

    def get_knowledge_record(self, record_id: str, chat_id: Optional[str] = None) -> Optional[UserKnowledgeRecord]:
        if self.engine.dialect.name != "postgresql":
            raise RuntimeError("Knowledge retrieval requires PostgreSQL + pgvector")

        if not self._unified_memory_mode:
            raise RuntimeError("Legacy manual storage mode has been removed")

        with self.engine.begin() as conn:
            self._set_current_user_id(conn, str(chat_id or ""))
            row = conn.execute(
                text(
                    """
                    SELECT
                        m.id,
                        m.owner_user_id AS chat_id,
                        m.memory_type AS category,
                        COALESCE(m.summary, '') AS title,
                        m.content,
                        m.metadata_json,
                        COALESCE(me.model_name, '') AS embedding_model,
                        m.created_at,
                        m.updated_at
                    FROM knowledge.memories m
                    LEFT JOIN knowledge.memory_embeddings me ON me.memory_id = m.id
                    WHERE m.id = :record_id
                      AND m.status = 'active'
                    LIMIT 1
                    """
                ),
                {"record_id": record_id},
            ).mappings().first()

        if not row:
            return None
        if chat_id and str(row["chat_id"]) != str(chat_id):
            return None

        payload = self._safe_json_loads(row.get("metadata_json"), {})
        return UserKnowledgeRecord(
            id=row["id"],
            chat_id=row["chat_id"],
            category=row["category"],
            title=row["title"] or "",
            content=row["content"],
            tags=payload.get("tags", []),
            metadata=payload.get("metadata", payload if isinstance(payload, dict) else {}),
            embedding_model=row["embedding_model"],
            embedding_dims=0,
            embedding=[],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def list_knowledge_records(
        self,
        *,
        chat_id: str,
        limit: int = 10,
        category: Optional[str] = None,
    ) -> list[UserKnowledgeRecord]:
        if self.engine.dialect.name != "postgresql":
            raise RuntimeError("Knowledge retrieval requires PostgreSQL + pgvector")

        if not self._unified_memory_mode:
            raise RuntimeError("Legacy manual storage mode has been removed")

        safe_limit = max(1, min(limit, 100))
        normalized_category = (category or "").strip().lower() or None
        sql = """
            SELECT
                m.id,
                m.owner_user_id AS chat_id,
                m.memory_type AS category,
                COALESCE(m.summary, '') AS title,
                m.content,
                m.metadata_json,
                COALESCE(me.model_name, '') AS embedding_model,
                m.created_at,
                m.updated_at
            FROM knowledge.memories m
            LEFT JOIN knowledge.memory_embeddings me ON me.memory_id = m.id
            WHERE m.owner_user_id = :chat_id
              AND m.status = 'active'
        """
        params: dict = {"chat_id": str(chat_id), "limit": safe_limit}
        if normalized_category:
            sql += " AND m.memory_type = :category"
            params["category"] = normalized_category
        sql += " ORDER BY m.created_at DESC LIMIT :limit"

        with self.engine.begin() as conn:
            self._set_current_user_id(conn, str(chat_id))
            rows = conn.execute(text(sql), params).mappings().all()

        results: list[UserKnowledgeRecord] = []
        for row in rows:
            payload = self._safe_json_loads(row.get("metadata_json"), {})
            results.append(
                UserKnowledgeRecord(
                    id=row["id"],
                    chat_id=row["chat_id"],
                    category=row["category"],
                    title=row["title"] or "",
                    content=row["content"],
                    tags=payload.get("tags", []),
                    metadata=payload.get("metadata", payload if isinstance(payload, dict) else {}),
                    embedding_model=row["embedding_model"],
                    embedding_dims=0,
                    embedding=[],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
            )
        return results

    def list_knowledge_records_between(
        self,
        *,
        start: datetime,
        end: datetime,
        chat_id: Optional[str] = None,
        categories: Optional[list[str]] = None,
        limit: int = 200,
    ) -> list[UserKnowledgeRecord]:
        if self.engine.dialect.name != "postgresql":
            raise RuntimeError("Knowledge retrieval requires PostgreSQL + pgvector")

        if not self._unified_memory_mode:
            raise RuntimeError("Legacy manual storage mode has been removed")

        safe_limit = max(1, min(limit, 1000))
        normalized_categories = [
            item.strip().lower() for item in (categories or []) if item and item.strip()
        ]

        sql = """
            SELECT
                m.id,
                m.owner_user_id AS chat_id,
                m.memory_type AS category,
                COALESCE(m.summary, '') AS title,
                m.content,
                m.metadata_json,
                COALESCE(me.model_name, '') AS embedding_model,
                m.created_at,
                m.updated_at
            FROM knowledge.memories m
            LEFT JOIN knowledge.memory_embeddings me ON me.memory_id = m.id
            WHERE m.created_at >= :start
              AND m.created_at < :end
              AND m.status = 'active'
        """
        params: dict = {
            "start": start,
            "end": end,
            "limit": safe_limit,
        }

        if chat_id:
            sql += " AND m.owner_user_id = :chat_id"
            params["chat_id"] = str(chat_id)

        if normalized_categories:
            sql += " AND m.memory_type = ANY(:categories)"
            params["categories"] = normalized_categories

        sql += " ORDER BY m.created_at DESC LIMIT :limit"

        with self.engine.begin() as conn:
            self._set_current_user_id(conn, str(chat_id or ""))
            rows = conn.execute(text(sql), params).mappings().all()

        results: list[UserKnowledgeRecord] = []
        for row in rows:
            payload = self._safe_json_loads(row.get("metadata_json"), {})
            results.append(
                UserKnowledgeRecord(
                    id=row["id"],
                    chat_id=row["chat_id"],
                    category=row["category"],
                    title=row["title"] or "",
                    content=row["content"],
                    tags=payload.get("tags", []),
                    metadata=payload.get("metadata", payload if isinstance(payload, dict) else {}),
                    embedding_model=row["embedding_model"],
                    embedding_dims=0,
                    embedding=[],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
            )
        return results

    @staticmethod
    def _to_vector_literal(values: list[float]) -> str:
        serialized = ",".join(f"{float(v):.10f}" for v in values)
        return f"[{serialized}]"

    @staticmethod
    def _cosine_distance(a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 1.0
        size = min(len(a), len(b))
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for i in range(size):
            av = float(a[i])
            bv = float(b[i])
            dot += av * bv
            norm_a += av * av
            norm_b += bv * bv
        if norm_a <= 0.0 or norm_b <= 0.0:
            return 1.0
        similarity = dot / ((norm_a ** 0.5) * (norm_b ** 0.5))
        similarity = max(-1.0, min(1.0, similarity))
        return 1.0 - similarity

    def search_knowledge_records(
        self,
        *,
        chat_id: str,
        query_embedding: list[float],
        limit: int = 5,
        category: Optional[str] = None,
    ) -> list[dict]:
        if self.engine.dialect.name != "postgresql":
            raise RuntimeError("Knowledge semantic search requires PostgreSQL + pgvector")

        if not self._unified_memory_mode:
            raise RuntimeError("Legacy manual storage mode has been removed")

        safe_limit = max(1, min(limit, 100))
        normalized_category = (category or "").strip().lower() or None
        query_vector = self._to_vector_literal(query_embedding)

        base_sql = """
            SELECT
                m.id,
                m.owner_user_id AS chat_id,
                m.memory_type AS category,
                COALESCE(m.summary, '') AS title,
                m.content,
                m.metadata_json,
                COALESCE(me.model_name, '') AS embedding_model,
                m.created_at,
                m.updated_at,
                (me.embedding <=> CAST(:query_vector AS vector)) AS distance
            FROM knowledge.memory_embeddings me
            JOIN knowledge.memories m ON m.id = me.memory_id
            WHERE m.owner_user_id = :chat_id
              AND m.status = 'active'
              AND me.embedding IS NOT NULL
        """
        params: dict = {
            "query_vector": query_vector,
            "chat_id": str(chat_id),
            "limit": safe_limit,
        }
        if normalized_category:
            base_sql += " AND m.memory_type = :category "
            params["category"] = normalized_category
        base_sql += " ORDER BY me.embedding <=> CAST(:query_vector AS vector) ASC LIMIT :limit"

        with self.engine.begin() as conn:
            self._set_current_user_id(conn, str(chat_id))
            rows = conn.execute(text(base_sql), params).mappings().all()

        results: list[dict] = []
        for row in rows:
            payload = self._safe_json_loads(row.get("metadata_json"), {})
            results.append(
                {
                    "id": row["id"],
                    "chat_id": row["chat_id"],
                    "category": row["category"],
                    "title": row["title"],
                    "content": row["content"],
                    "metadata": payload.get("metadata", payload if isinstance(payload, dict) else {}),
                    "tags": payload.get("tags", []),
                    "distance": float(row.get("distance") or 0.0),
                    "embedding_model": row.get("embedding_model") or "",
                    "embedding_dims": len(query_embedding),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
            )
        return results

    def delete_knowledge_record(self, record_id: str, chat_id: Optional[str] = None) -> bool:
        if self.engine.dialect.name != "postgresql":
            raise RuntimeError("Knowledge deletion requires PostgreSQL + pgvector")

        if not self._unified_memory_mode:
            raise RuntimeError("Legacy manual storage mode has been removed")

        sql = """
            UPDATE knowledge.memories
            SET status = 'deleted', updated_at = now()
            WHERE id = :record_id
              AND status <> 'deleted'
        """
        params: dict = {"record_id": record_id}
        if chat_id:
            sql += " AND owner_user_id = :chat_id"
            params["chat_id"] = str(chat_id)

        with self.engine.begin() as conn:
            self._set_current_user_id(conn, str(chat_id or ""))
            result = conn.execute(text(sql), params)
            return result.rowcount > 0


# Backward-compatible alias
TrustedDBRepository = AgentDBRepository

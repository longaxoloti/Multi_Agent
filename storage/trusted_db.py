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
    TRUSTED_DB_URL,
)
from storage.models import Base  # shared Base for all schemas

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Legacy ORM models (public schema) — kept for backward compatibility
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


# ---------------------------------------------------------------------------
# PostgreSQL schemas to create
# ---------------------------------------------------------------------------
_PG_SCHEMAS = ["system", "skills", "profile", "projects", "knowledge", "manual", "security"]

# pgvector columns to ensure exist (table, schema, column, dims)
_VECTOR_COLUMNS = [
    ("skill_embeddings",   "skills",    "embedding", 1024),
    ("profile_embeddings", "profile",   "embedding", 1024),
    ("url_embeddings",     "knowledge", "embedding", 1024),
    ("saved_embeddings",   "manual",    "embedding", 1024),
]

_HNSW_INDEXES = [
    ("skills.skill_embeddings",     "embedding", "idx_skill_emb_hnsw"),
    ("profile.profile_embeddings",  "embedding", "idx_profile_emb_hnsw"),
    ("knowledge.url_embeddings",    "embedding", "idx_url_emb_hnsw"),
    ("manual.saved_embeddings",     "embedding", "idx_saved_emb_hnsw"),
]


class AgentDBRepository:
    """Central database repository for the multi-schema agent database."""

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or TRUSTED_DB_URL
        self.engine = create_engine(self.db_url, future=True, pool_pre_ping=True)
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    def initialize(self) -> None:
        """Create all schemas, tables, vector columns, and HNSW indexes."""
        is_pg = self.engine.dialect.name == "postgresql"

        if is_pg:
            with self.engine.begin() as conn:
                # Create pgvector extension
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                # Create PostgreSQL schemas
                for schema in _PG_SCHEMAS:
                    conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))

            # Create all tables across all schemas (PostgreSQL)
            Base.metadata.create_all(self.engine)
        else:
            # SQLite: only create public-schema tables (no schema support)
            public_tables = [
                t for t in Base.metadata.sorted_tables
                if t.schema is None
            ]
            Base.metadata.create_all(self.engine, tables=public_tables)

        # Legacy compatibility
        self._ensure_schema_compatibility()

        if is_pg:
            self._ensure_vector_columns()
            self._ensure_hnsw_indexes()

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

    def _ensure_vector_columns(self) -> None:
        """Add pgvector columns to embedding tables if they don't exist yet."""
        with self.engine.begin() as conn:
            for table_name, schema, col_name, dims in _VECTOR_COLUMNS:
                try:
                    cols = self._get_column_names(table_name, schema=schema)
                except Exception:
                    continue  # table doesn't exist yet
                if col_name not in cols:
                    conn.execute(text(
                        f"ALTER TABLE {schema}.{table_name} "
                        f"ADD COLUMN IF NOT EXISTS {col_name} vector({dims})"
                    ))

    def _ensure_hnsw_indexes(self) -> None:
        """Create HNSW indexes for all vector columns."""
        with self.engine.begin() as conn:
            for fq_table, col_name, idx_name in _HNSW_INDEXES:
                conn.execute(text(
                    f"CREATE INDEX IF NOT EXISTS {idx_name} "
                    f"ON {fq_table} USING hnsw ({col_name} vector_cosine_ops)"
                ))

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
                                                WHERE table_schema = 'manual'
                                                    AND table_name = 'saved_embeddings'
                          AND column_name = 'embedding'
                        LIMIT 1
                        """
                    )
                ).first()
                return bool(col_row)
        except Exception:
            return False

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

        now = datetime.utcnow()
        final_id = record_id or f"k_{uuid.uuid4().hex}"
        source_id = f"s_{uuid.uuid4().hex}"
        embedding_id = f"e_{uuid.uuid4().hex}"
        payload = {
            "tags": tags or [],
            "metadata": metadata or {},
        }

        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO manual.saved_sources (
                        id, command_text, request_id, user_id, chat_id, created_at
                    ) VALUES (
                        :id, :command_text, :request_id, :user_id, :chat_id, :created_at
                    )
                    """
                ),
                {
                    "id": source_id,
                    "command_text": json.dumps(payload, ensure_ascii=False),
                    "request_id": None,
                    "user_id": str(chat_id),
                    "chat_id": str(chat_id),
                    "created_at": now,
                },
            )

            conn.execute(
                text(
                    """
                    INSERT INTO manual.saved_facts (
                        id, saved_source_id, category, title, fact_key,
                        fact_value, status, provenance_type, created_at, updated_at
                    ) VALUES (
                        :id, :saved_source_id, :category, :title, :fact_key,
                        :fact_value, 'active', 'user_direct_save', :created_at, :updated_at
                    )
                    """
                ),
                {
                    "id": final_id,
                    "saved_source_id": source_id,
                    "category": (category or "note").strip().lower(),
                    "title": (title or "").strip()[:255],
                    "fact_key": None,
                    "fact_value": content,
                    "created_at": now,
                    "updated_at": now,
                },
            )

            if embedding:
                conn.execute(
                    text(
                        """
                        INSERT INTO manual.saved_embeddings (
                            id, saved_fact_id, embedding_json, model_name, created_at
                        ) VALUES (
                            :id, :saved_fact_id, :embedding_json, :model_name, :created_at
                        )
                        """
                    ),
                    {
                        "id": embedding_id,
                        "saved_fact_id": final_id,
                        "embedding_json": "[]",
                        "model_name": (embedding_model or "").strip() or "bge-m3",
                        "created_at": now,
                    },
                )
                conn.execute(
                    text(
                        """
                        UPDATE manual.saved_embeddings
                        SET embedding = CAST(:embedding_literal AS vector)
                        WHERE id = :embedding_id
                        """
                    ),
                    {
                        "embedding_literal": self._to_vector_literal(embedding),
                        "embedding_id": embedding_id,
                    },
                )
        return final_id

    def get_knowledge_record(self, record_id: str, chat_id: Optional[str] = None) -> Optional[UserKnowledgeRecord]:
        if self.engine.dialect.name != "postgresql":
            raise RuntimeError("Knowledge retrieval requires PostgreSQL + pgvector")

        with self.engine.begin() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT
                        sf.id,
                        ss.chat_id,
                        sf.category,
                        sf.title,
                        sf.fact_value AS content,
                        ss.command_text,
                        COALESCE(se.model_name, '') AS embedding_model,
                        sf.created_at,
                        sf.updated_at
                    FROM manual.saved_facts sf
                    JOIN manual.saved_sources ss ON ss.id = sf.saved_source_id
                    LEFT JOIN manual.saved_embeddings se ON se.saved_fact_id = sf.id
                    WHERE sf.id = :record_id
                    LIMIT 1
                    """
                ),
                {"record_id": record_id},
            ).mappings().first()

        if not row:
            return None
        if chat_id and str(row["chat_id"]) != str(chat_id):
            return None

        source_payload = self._safe_json_loads(row.get("command_text"), {})
        return UserKnowledgeRecord(
            id=row["id"],
            chat_id=row["chat_id"],
            category=row["category"],
            title=row["title"] or "",
            content=row["content"],
            tags=source_payload.get("tags", []),
            metadata=source_payload.get("metadata", {}),
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

        safe_limit = max(1, min(limit, 100))
        normalized_category = (category or "").strip().lower() or None

        sql = """
            SELECT
                sf.id,
                ss.chat_id,
                sf.category,
                sf.title,
                sf.fact_value AS content,
                ss.command_text,
                COALESCE(se.model_name, '') AS embedding_model,
                sf.created_at,
                sf.updated_at
            FROM manual.saved_facts sf
            JOIN manual.saved_sources ss ON ss.id = sf.saved_source_id
            LEFT JOIN manual.saved_embeddings se ON se.saved_fact_id = sf.id
            WHERE ss.chat_id = :chat_id
              AND sf.status = 'active'
        """
        params: dict = {"chat_id": str(chat_id), "limit": safe_limit}
        if normalized_category:
            sql += " AND sf.category = :category"
            params["category"] = normalized_category
        sql += " ORDER BY sf.created_at DESC LIMIT :limit"

        with self.engine.begin() as conn:
            rows = conn.execute(text(sql), params).mappings().all()

        results: list[UserKnowledgeRecord] = []
        for row in rows:
            source_payload = self._safe_json_loads(row.get("command_text"), {})
            results.append(
                UserKnowledgeRecord(
                    id=row["id"],
                    chat_id=row["chat_id"],
                    category=row["category"],
                    title=row["title"] or "",
                    content=row["content"],
                    tags=source_payload.get("tags", []),
                    metadata=source_payload.get("metadata", {}),
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

        safe_limit = max(1, min(limit, 1000))
        normalized_categories = [
            item.strip().lower()
            for item in (categories or [])
            if item and item.strip()
        ]

        sql = """
            SELECT
                sf.id,
                ss.chat_id,
                sf.category,
                sf.title,
                sf.fact_value AS content,
                ss.command_text,
                COALESCE(se.model_name, '') AS embedding_model,
                sf.created_at,
                sf.updated_at
            FROM manual.saved_facts sf
            JOIN manual.saved_sources ss ON ss.id = sf.saved_source_id
            LEFT JOIN manual.saved_embeddings se ON se.saved_fact_id = sf.id
            WHERE sf.created_at >= :start
              AND sf.created_at < :end
              AND sf.status = 'active'
        """
        params: dict = {
            "start": start,
            "end": end,
            "limit": safe_limit,
        }

        if chat_id:
            sql += " AND ss.chat_id = :chat_id"
            params["chat_id"] = str(chat_id)

        if normalized_categories:
            sql += " AND sf.category = ANY(:categories)"
            params["categories"] = normalized_categories

        sql += " ORDER BY sf.created_at DESC LIMIT :limit"

        with self.engine.begin() as conn:
            rows = conn.execute(text(sql), params).mappings().all()

        results: list[UserKnowledgeRecord] = []
        for row in rows:
            source_payload = self._safe_json_loads(row.get("command_text"), {})
            results.append(
                UserKnowledgeRecord(
                    id=row["id"],
                    chat_id=row["chat_id"],
                    category=row["category"],
                    title=row["title"] or "",
                    content=row["content"],
                    tags=source_payload.get("tags", []),
                    metadata=source_payload.get("metadata", {}),
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

        safe_limit = max(1, min(limit, 100))
        normalized_category = (category or "").strip().lower() or None
        query_vector = self._to_vector_literal(query_embedding)
        base_sql = """
            SELECT
                sf.id,
                ss.chat_id,
                sf.category,
                sf.title,
                sf.fact_value AS content,
                ss.command_text,
                COALESCE(se.model_name, '') AS embedding_model,
                sf.created_at,
                sf.updated_at,
                (se.embedding <=> CAST(:query_vector AS vector)) AS distance
            FROM manual.saved_embeddings se
            JOIN manual.saved_facts sf ON sf.id = se.saved_fact_id
            JOIN manual.saved_sources ss ON ss.id = sf.saved_source_id
            WHERE ss.chat_id = :chat_id
              AND sf.status = 'active'
              AND se.embedding IS NOT NULL
        """
        params: dict = {
            "query_vector": query_vector,
            "chat_id": str(chat_id),
            "limit": safe_limit,
        }
        if normalized_category:
            base_sql += " AND sf.category = :category "
            params["category"] = normalized_category
        base_sql += " ORDER BY se.embedding <=> CAST(:query_vector AS vector) ASC LIMIT :limit"

        with self.engine.begin() as conn:
            rows = conn.execute(text(base_sql), params).mappings().all()

        results: list[dict] = []
        for row in rows:
            source_payload = self._safe_json_loads(row.get("command_text"), {})
            results.append(
                {
                    "id": row["id"],
                    "chat_id": row["chat_id"],
                    "category": row["category"],
                    "title": row["title"],
                    "content": row["content"],
                    "metadata": source_payload.get("metadata", {}),
                    "tags": source_payload.get("tags", []),
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

        sql = """
            DELETE FROM manual.saved_facts sf
            USING manual.saved_sources ss
            WHERE sf.id = :record_id
              AND ss.id = sf.saved_source_id
        """
        params: dict = {"record_id": record_id}
        if chat_id:
            sql += " AND ss.chat_id = :chat_id"
            params["chat_id"] = str(chat_id)

        with self.engine.begin() as conn:
            result = conn.execute(text(sql), params)
            return result.rowcount > 0


# Backward-compatible alias
TrustedDBRepository = AgentDBRepository

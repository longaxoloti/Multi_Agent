from __future__ import annotations

import difflib
import json
import logging
import math
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine, inspect, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from config import (
    EMBEDDING_MODEL,
    GEMINI_API_KEY,
    TRUSTED_CLAIM_ENABLE_SEMANTIC_DEDUPE,
    TRUSTED_CLAIM_SEMANTIC_SIMILARITY_THRESHOLD,
    TRUSTED_CLAIM_SIMILARITY_THRESHOLD,
    TRUSTED_DB_URL,
)

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


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


class UserKnowledgeRecordORM(Base):
    __tablename__ = "user_knowledge_records"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    chat_id: Mapped[str] = mapped_column(String(120), index=True)
    category: Mapped[str] = mapped_column(String(40), index=True)
    title: Mapped[str] = mapped_column(String(255), default="")
    content: Mapped[str] = mapped_column(Text)
    tags_json: Mapped[str] = mapped_column(Text, default="[]")
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)


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
    created_at: datetime
    updated_at: datetime


class TrustedDBRepository:
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or TRUSTED_DB_URL
        self.engine = create_engine(self.db_url, future=True, pool_pre_ping=True)
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)
        self._embedding_client = None
        self._semantic_enabled = TRUSTED_CLAIM_ENABLE_SEMANTIC_DEDUPE and bool(GEMINI_API_KEY)

        if self._semantic_enabled:
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings

                self._embedding_client = GoogleGenerativeAIEmbeddings(
                    model=EMBEDDING_MODEL,
                    google_api_key=GEMINI_API_KEY,
                )
            except Exception as exc:
                logger.warning("Semantic dedupe disabled (embedding client init failed): %s", exc)
                self._semantic_enabled = False

    def initialize(self) -> None:
        Base.metadata.create_all(self.engine)
        self._ensure_schema_compatibility()
        logger.info("Trusted DB schema ready")

    def _ensure_schema_compatibility(self) -> None:
        inspector = inspect(self.engine)
        if "trusted_claims" not in inspector.get_table_names():
            return

        cols = {col["name"] for col in inspector.get_columns("trusted_claims")}
        if "normalized_claim" in cols:
            normalized_exists = True
        else:
            normalized_exists = False

        embedding_exists = "claim_embedding_json" in cols

        if normalized_exists and embedding_exists:
            return

        dialect = self.engine.dialect.name
        with self.engine.begin() as conn:
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

            if not embedding_exists:
                if dialect == "postgresql":
                    conn.execute(text("ALTER TABLE trusted_claims ADD COLUMN IF NOT EXISTS claim_embedding_json TEXT"))
                elif dialect == "sqlite":
                    conn.execute(text("ALTER TABLE trusted_claims ADD COLUMN claim_embedding_json TEXT"))
                else:
                    logger.warning("Unsupported dialect for embedding column migration: %s", dialect)
                    return

    def _embed_text(self, text_value: str) -> Optional[list[float]]:
        if not self._semantic_enabled or not self._embedding_client:
            return None
        try:
            vector = self._embedding_client.embed_query(text_value)
            if not vector:
                return None
            return [float(x) for x in vector]
        except Exception as exc:
            logger.warning("Embedding generation failed, skipping semantic dedupe: %s", exc)
            return None

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

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
        candidate_embedding: Optional[list[float]] = None,
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
        for row in candidates:
            existing_norm = row.normalized_claim or self._normalize_claim(row.claim)
            score = difflib.SequenceMatcher(None, existing_norm, normalized_claim).ratio()

            if candidate_embedding and row.claim_embedding_json:
                try:
                    existing_embedding = json.loads(row.claim_embedding_json)
                    semantic_score = self._cosine_similarity(existing_embedding, candidate_embedding)
                    score = max(score, semantic_score)
                except json.JSONDecodeError:
                    pass

            if score > best_score:
                best_score = score
                best_match = row

        threshold = min(
            TRUSTED_CLAIM_SIMILARITY_THRESHOLD,
            TRUSTED_CLAIM_SEMANTIC_SIMILARITY_THRESHOLD,
        ) if candidate_embedding else TRUSTED_CLAIM_SIMILARITY_THRESHOLD
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
        claim_embedding = self._embed_text(claim)
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
                    candidate_embedding=claim_embedding,
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
                if claim_embedding and not existing.claim_embedding_json:
                    existing.claim_embedding_json = json.dumps(claim_embedding)
                session.add(existing)
                session.commit()
                session.refresh(existing)
                return existing.id, False

            item = TrustedClaimORM(
                topic=topic,
                claim=claim,
                normalized_claim=normalized_claim,
                claim_embedding_json=json.dumps(claim_embedding) if claim_embedding else None,
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
        record_id: Optional[str] = None,
    ) -> str:
        now = datetime.utcnow()
        final_id = record_id or f"k_{uuid.uuid4().hex}"
        item = UserKnowledgeRecordORM(
            id=final_id,
            chat_id=str(chat_id),
            category=(category or "note").strip().lower(),
            title=(title or "").strip()[:255],
            content=content,
            tags_json=json.dumps(tags or [], ensure_ascii=False),
            metadata_json=json.dumps(metadata or {}, ensure_ascii=False),
            created_at=now,
            updated_at=now,
        )
        with self._session_factory() as session:
            session.add(item)
            session.commit()
        return final_id

    def get_knowledge_record(self, record_id: str, chat_id: Optional[str] = None) -> Optional[UserKnowledgeRecord]:
        with self._session_factory() as session:
            row = session.execute(
                select(UserKnowledgeRecordORM).where(UserKnowledgeRecordORM.id == record_id)
            ).scalars().first()
        if not row:
            return None
        if chat_id and str(row.chat_id) != str(chat_id):
            return None
        return UserKnowledgeRecord(
            id=row.id,
            chat_id=row.chat_id,
            category=row.category,
            title=row.title,
            content=row.content,
            tags=self._safe_json_loads(row.tags_json, []),
            metadata=self._safe_json_loads(row.metadata_json, {}),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    def list_knowledge_records(
        self,
        *,
        chat_id: str,
        limit: int = 10,
        category: Optional[str] = None,
    ) -> list[UserKnowledgeRecord]:
        stmt = (
            select(UserKnowledgeRecordORM)
            .where(UserKnowledgeRecordORM.chat_id == str(chat_id))
            .order_by(UserKnowledgeRecordORM.created_at.desc())
            .limit(max(1, min(limit, 100)))
        )
        if category:
            stmt = stmt.where(UserKnowledgeRecordORM.category == category.strip().lower())

        with self._session_factory() as session:
            rows = session.execute(stmt).scalars().all()

        results: list[UserKnowledgeRecord] = []
        for row in rows:
            results.append(
                UserKnowledgeRecord(
                    id=row.id,
                    chat_id=row.chat_id,
                    category=row.category,
                    title=row.title,
                    content=row.content,
                    tags=self._safe_json_loads(row.tags_json, []),
                    metadata=self._safe_json_loads(row.metadata_json, {}),
                    created_at=row.created_at,
                    updated_at=row.updated_at,
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
        stmt = (
            select(UserKnowledgeRecordORM)
            .where(UserKnowledgeRecordORM.created_at >= start)
            .where(UserKnowledgeRecordORM.created_at < end)
            .order_by(UserKnowledgeRecordORM.created_at.desc())
            .limit(max(1, min(limit, 1000)))
        )

        if chat_id:
            stmt = stmt.where(UserKnowledgeRecordORM.chat_id == str(chat_id))

        normalized_categories = [
            item.strip().lower()
            for item in (categories or [])
            if item and item.strip()
        ]
        if normalized_categories:
            stmt = stmt.where(UserKnowledgeRecordORM.category.in_(normalized_categories))

        with self._session_factory() as session:
            rows = session.execute(stmt).scalars().all()

        results: list[UserKnowledgeRecord] = []
        for row in rows:
            results.append(
                UserKnowledgeRecord(
                    id=row.id,
                    chat_id=row.chat_id,
                    category=row.category,
                    title=row.title,
                    content=row.content,
                    tags=self._safe_json_loads(row.tags_json, []),
                    metadata=self._safe_json_loads(row.metadata_json, {}),
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                )
            )
        return results

    def delete_knowledge_record(self, record_id: str, chat_id: Optional[str] = None) -> bool:
        with self._session_factory() as session:
            row = session.execute(
                select(UserKnowledgeRecordORM).where(UserKnowledgeRecordORM.id == record_id)
            ).scalars().first()
            if not row:
                return False
            if chat_id and str(row.chat_id) != str(chat_id):
                return False
            session.delete(row)
            session.commit()
            return True

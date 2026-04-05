"""
UserProfileService — Manages user profile facts with provenance tracking.

Auto-updatable: model ingests USER.md once, then freely updates based on conversations.
Same fact_key can have multiple values in different contexts.
Never overwrites — superseded facts keep status='superseded'.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import select, text
from sqlalchemy.orm import Session, sessionmaker

from storage.models import (
    ProfileEmbeddingORM,
    ProfileFactORM,
    ProfileSourceORM,
    ProfileVersionORM,
    _new_uuid,
    _utcnow,
)
from tools.embedding_provider import embed_text_ollama
from main.config import KNOWLEDGE_EMBEDDING_MODEL, KNOWLEDGE_EMBEDDING_DIMS

logger = logging.getLogger(__name__)


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:64]


class UserProfileService:
    """Service for managing user profile information."""

    def __init__(
        self,
        session_factory: sessionmaker,
        *,
        embedder=None,
        embedding_model: str = KNOWLEDGE_EMBEDDING_MODEL,
        embedding_dims: int = KNOWLEDGE_EMBEDDING_DIMS,
        is_pg: bool = True,
    ):
        self._session_factory = session_factory
        self._embedder = embedder or embed_text_ollama
        self._embedding_model = embedding_model
        self._embedding_dims = embedding_dims
        self._is_pg = is_pg

    # ------------------------------------------------------------------
    # Ingest from USER.md
    # ------------------------------------------------------------------

    def ingest_from_markdown(self, file_path: str, *, user_id: str) -> dict:
        """
        Parse USER.md and extract profile facts.
        Creates a profile source + version, then extracts key-value facts.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Profile file not found: {file_path}")

        raw_content = path.read_text(encoding="utf-8")
        content_hash = _content_hash(raw_content)

        with self._session_factory() as session:
            # Check if we already ingested this exact content
            existing = session.execute(
                select(ProfileSourceORM).where(
                    ProfileSourceORM.source_hash == content_hash,
                    ProfileSourceORM.source_type == "user_md",
                )
            ).scalars().first()

            if existing:
                return {"action": "no_change", "source_id": existing.id}

            # Create new source
            source = ProfileSourceORM(
                id=_new_uuid(),
                source_type="user_md",
                source_hash=content_hash,
            )
            session.add(source)
            session.flush()

            # Extract facts from markdown
            facts = self._parse_user_md(raw_content)
            facts_created = 0

            for key, value in facts.items():
                self._upsert_fact_internal(
                    session,
                    user_id=user_id,
                    fact_key=key,
                    fact_value=value,
                    confidence=1.0,
                    provenance_type="user_md",
                    provenance_id=source.id,
                )
                facts_created += 1

            # Create version snapshot
            latest_no = session.execute(
                select(ProfileVersionORM.version_no).where(
                    ProfileVersionORM.user_id == user_id,
                ).order_by(ProfileVersionORM.version_no.desc())
            ).scalars().first()

            version = ProfileVersionORM(
                id=_new_uuid(),
                profile_source_id=source.id,
                user_id=user_id,
                version_no=(latest_no or 0) + 1,
                canonical_summary=f"Ingested from {path.name}: {facts_created} facts",
                updated_by="user",
            )
            session.add(version)
            session.commit()

            return {
                "action": "ingested",
                "source_id": source.id,
                "version_no": version.version_no,
                "facts_created": facts_created,
            }

    # ------------------------------------------------------------------
    # Upsert fact (agent or user)
    # ------------------------------------------------------------------

    def upsert_fact(
        self,
        *,
        user_id: str,
        fact_key: str,
        fact_value: str,
        confidence: float = 0.8,
        source: str = "inferred",
        is_sensitive: bool = False,
    ) -> dict:
        """
        Update or create a profile fact.
        If same key+value exists (active), just update confidence.
        If same key but different value, keep both (different contexts).
        If explicitly replacing, mark old fact as 'superseded'.
        """
        with self._session_factory() as session:
            fact_id = self._upsert_fact_internal(
                session,
                user_id=user_id,
                fact_key=fact_key,
                fact_value=fact_value,
                confidence=confidence,
                provenance_type=source,
                is_sensitive=is_sensitive,
            )
            session.commit()
            return {"fact_id": fact_id, "fact_key": fact_key}

    def _upsert_fact_internal(
        self,
        session: Session,
        *,
        user_id: str,
        fact_key: str,
        fact_value: str,
        confidence: float,
        provenance_type: str,
        provenance_id: Optional[str] = None,
        is_sensitive: bool = False,
    ) -> str:
        """Internal upsert logic within an existing session."""
        normalized_key = fact_key.strip().lower()
        normalized_value = fact_value.strip()

        # Check if exact key+value pair already exists and is active
        existing = session.execute(
            select(ProfileFactORM).where(
                ProfileFactORM.user_id == user_id,
                ProfileFactORM.fact_key == normalized_key,
                ProfileFactORM.fact_value == normalized_value,
                ProfileFactORM.status == "active",
            )
        ).scalars().first()

        if existing:
            # Same fact exists — update confidence if higher
            if confidence > existing.confidence:
                existing.confidence = confidence
                existing.updated_at = _utcnow()
                session.add(existing)
            return existing.id

        # New fact — create it (same key can have multiple values)
        fact = ProfileFactORM(
            id=_new_uuid(),
            user_id=user_id,
            fact_key=normalized_key,
            fact_value=normalized_value,
            confidence=confidence,
            is_sensitive=is_sensitive,
            status="active",
            provenance_type=provenance_type,
            provenance_id=provenance_id,
            last_verified_at=_utcnow(),
        )
        session.add(fact)
        session.flush()

        # Embed the fact for semantic search
        embed_text = f"{normalized_key}: {normalized_value}"
        try:
            vector = self._embedder(
                embed_text,
                model=self._embedding_model,
                expected_dims=self._embedding_dims,
            )
        except Exception as exc:
            logger.warning("Embedding failed for profile fact '%s': %s", normalized_key, exc)
            vector = []

        embedding = ProfileEmbeddingORM(
            id=_new_uuid(),
            fact_id=fact.id,
            embedding_json="[]",
            model_name=self._embedding_model,
        )
        session.add(embedding)
        session.flush()

        if self._is_pg and vector:
            vec_literal = _to_vector_literal(vector)
            session.execute(
                text("UPDATE profile.profile_embeddings SET embedding = CAST(:vec AS vector) WHERE id = :rid"),
                {"vec": vec_literal, "rid": embedding.id},
            )

        return fact.id

    # ------------------------------------------------------------------
    # Get profile
    # ------------------------------------------------------------------

    def get_profile(self, user_id: str, *, include_superseded: bool = False) -> list[dict]:
        """Get all profile facts for a user."""
        with self._session_factory() as session:
            stmt = select(ProfileFactORM).where(
                ProfileFactORM.user_id == user_id,
            ).order_by(ProfileFactORM.fact_key, ProfileFactORM.created_at.desc())

            if not include_superseded:
                stmt = stmt.where(ProfileFactORM.status == "active")

            rows = session.execute(stmt).scalars().all()
            return [
                {
                    "id": r.id,
                    "fact_key": r.fact_key,
                    "fact_value": r.fact_value,
                    "confidence": r.confidence,
                    "status": r.status,
                    "provenance_type": r.provenance_type,
                    "is_sensitive": r.is_sensitive,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "updated_at": r.updated_at.isoformat() if r.updated_at else None,
                }
                for r in rows
            ]

    # ------------------------------------------------------------------
    # Search profile (semantic)
    # ------------------------------------------------------------------

    def search_profile(
        self,
        user_id: str,
        query: str,
        *,
        limit: int = 5,
    ) -> list[dict]:
        """Semantic search over a user's profile facts."""
        if not self._is_pg:
            raise RuntimeError("Profile search requires PostgreSQL + pgvector")

        try:
            query_vector = self._embedder(
                query, model=self._embedding_model, expected_dims=self._embedding_dims
            )
        except Exception as exc:
            logger.error("Failed to embed query for profile search: %s", exc, exc_info=True)
            raise RuntimeError("Profile query embedding failed") from exc

        vector_literal = _to_vector_literal(query_vector)
        sql = """
            SELECT
                pf.id, pf.fact_key, pf.fact_value, pf.confidence,
                pf.status, pf.provenance_type,
                (pe.embedding <=> CAST(:query_vector AS vector)) AS distance
            FROM profile.profile_embeddings pe
            JOIN profile.profile_facts pf ON pf.id = pe.fact_id
            WHERE pf.user_id = :user_id
              AND pf.status = 'active'
              AND pe.embedding IS NOT NULL
            ORDER BY pe.embedding <=> CAST(:query_vector AS vector) ASC
            LIMIT :limit
        """
        with self._session_factory() as session:
            rows = session.execute(
                text(sql),
                {"query_vector": vector_literal, "user_id": user_id, "limit": limit},
            ).mappings().all()

        return [
            {
                "id": row["id"],
                "fact_key": row["fact_key"],
                "fact_value": row["fact_value"],
                "confidence": float(row["confidence"]),
                "distance": float(row["distance"]),
            }
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Mark fact as superseded
    # ------------------------------------------------------------------

    def supersede_fact(self, fact_id: str, *, user_id: str) -> bool:
        """Mark a specific fact as superseded."""
        with self._session_factory() as session:
            fact = session.execute(
                select(ProfileFactORM).where(
                    ProfileFactORM.id == fact_id,
                    ProfileFactORM.user_id == user_id,
                )
            ).scalars().first()
            if not fact:
                return False
            fact.status = "superseded"
            fact.updated_at = _utcnow()
            session.add(fact)
            session.commit()
            return True

    # ------------------------------------------------------------------
    # Parser
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_user_md(content: str) -> dict[str, str]:
        """Extract key-value facts from USER.md markdown format."""
        facts: dict[str, str] = {}
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Match lines like: - **Key:** Value  or  - **Key**: Value
            match = re.match(r"^-\s*\*\*(.+?)\*\*\s*:?\s*(.+)$", line)
            if match:
                key = match.group(1).strip().rstrip(":").lower().replace(" ", "_")
                value = match.group(2).strip()
                if value:
                    facts[key] = value
                continue

            # Match lines like: - Key: value
            match = re.match(r"^-\s*(.+?):\s+(.+)$", line)
            if match:
                key = match.group(1).strip().rstrip(":").lower().replace(" ", "_")
                value = match.group(2).strip()
                if value:
                    facts[key] = value

        return facts


def _to_vector_literal(values: list[float]) -> str:
    serialized = ",".join(f"{float(v):.10f}" for v in values)
    return f"[{serialized}]"

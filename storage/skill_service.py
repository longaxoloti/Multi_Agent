"""
SkillService — Manages skill prompts with versioning, chunking, and vector search.

Auto-updatable: model can ingest from markdown, optimize skills, and create new versions.
Never overwrites existing versions.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import select, text
from sqlalchemy.orm import Session, sessionmaker

from storage.models import (
    SkillChunkORM,
    SkillEmbeddingORM,
    SkillSourceORM,
    SkillTagORM,
    SkillVersionORM,
    _new_uuid,
    _utcnow,
)
from rag.chunking import chunk_text
from tools.embedding_provider import embed_text_ollama
from main.config import KNOWLEDGE_EMBEDDING_MODEL, KNOWLEDGE_EMBEDDING_DIMS

logger = logging.getLogger(__name__)


def _file_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:64]


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for mixed EN/VI text."""
    return max(1, len(text) // 4)


class SkillService:
    """Service for managing tool-usage skill prompts."""

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
    # Ingest from markdown file
    # ------------------------------------------------------------------

    def ingest_from_markdown(
        self,
        file_path: str,
        *,
        title: Optional[str] = None,
        tags: Optional[list[str]] = None,
        updated_by: str = "user",
    ) -> dict:
        """
        Read a markdown file, create/update skill source, chunk, embed, and version.

        If same file (by path + hash) already exists, returns existing info.
        If file content changed, creates a new version.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Skill file not found: {file_path}")

        raw_content = path.read_text(encoding="utf-8")
        content_hash = _file_hash(raw_content)
        skill_title = title or path.stem.replace("_", " ").replace("-", " ").title()

        with self._session_factory() as session:
            # Check if source already exists for this path
            existing_source = session.execute(
                select(SkillSourceORM).where(
                    SkillSourceORM.source_path == str(path),
                )
            ).scalars().first()

            if existing_source:
                # Check if content changed
                if existing_source.source_hash == content_hash:
                    # No change — return existing
                    active_version = session.execute(
                        select(SkillVersionORM).where(
                            SkillVersionORM.skill_source_id == existing_source.id,
                            SkillVersionORM.status == "active",
                        ).order_by(SkillVersionORM.version_no.desc())
                    ).scalars().first()
                    return {
                        "action": "no_change",
                        "source_id": existing_source.id,
                        "version_id": active_version.id if active_version else None,
                        "version_no": active_version.version_no if active_version else 0,
                    }

                # Content changed — update source, create new version
                existing_source.raw_content = raw_content
                existing_source.source_hash = content_hash
                existing_source.updated_at = _utcnow()
                session.add(existing_source)
                source_id = existing_source.id

                # Deprecate all old active versions
                old_versions = session.execute(
                    select(SkillVersionORM).where(
                        SkillVersionORM.skill_source_id == source_id,
                        SkillVersionORM.status == "active",
                    )
                ).scalars().all()
                for v in old_versions:
                    v.status = "deprecated"
                    session.add(v)

                # Get latest version number
                latest = session.execute(
                    select(SkillVersionORM.version_no).where(
                        SkillVersionORM.skill_source_id == source_id,
                    ).order_by(SkillVersionORM.version_no.desc())
                ).scalars().first()
                new_version_no = (latest or 0) + 1
                action = "updated"
            else:
                # New source
                source = SkillSourceORM(
                    id=_new_uuid(),
                    source_type="markdown_file",
                    source_path=str(path),
                    source_hash=content_hash,
                    title=skill_title,
                    raw_content=raw_content,
                )
                session.add(source)
                session.flush()
                source_id = source.id
                new_version_no = 1
                action = "created"

            # Create new version
            version = SkillVersionORM(
                id=_new_uuid(),
                skill_source_id=source_id,
                version_no=new_version_no,
                canonical_content=raw_content,
                summary=skill_title,
                status="active",
                confidence=1.0,
                updated_by=updated_by,
            )
            session.add(version)
            session.flush()

            # Create chunks + embeddings
            chunks = chunk_text(raw_content, chunk_size=800, chunk_overlap=100)
            chunk_ids = []
            for idx, chunk_content in enumerate(chunks):
                chunk = SkillChunkORM(
                    id=_new_uuid(),
                    skill_version_id=version.id,
                    chunk_index=idx,
                    chunk_text=chunk_content,
                    token_count=_estimate_tokens(chunk_content),
                )
                session.add(chunk)
                session.flush()
                chunk_ids.append(chunk.id)

                # Embed and store
                try:
                    vector = self._embedder(
                        chunk_content,
                        model=self._embedding_model,
                        expected_dims=self._embedding_dims,
                    )
                except Exception as exc:
                    logger.warning("Embedding failed for chunk %d of skill %s: %s", idx, skill_title, exc)
                    vector = []

                embedding = SkillEmbeddingORM(
                    id=_new_uuid(),
                    chunk_id=chunk.id,
                    embedding_json="[]",  # stored via raw SQL for pgvector
                    model_name=self._embedding_model,
                )
                session.add(embedding)
                session.flush()

                # Write vector via raw SQL if PostgreSQL
                if self._is_pg and vector:
                    self._write_vector(
                        session, "skills.skill_embeddings", embedding.id, vector
                    )

            # Create tags
            for tag_text in (tags or []):
                tag = SkillTagORM(
                    id=_new_uuid(),
                    skill_version_id=version.id,
                    tag=tag_text.strip().lower(),
                )
                session.add(tag)

            session.commit()

            return {
                "action": action,
                "source_id": source_id,
                "version_id": version.id,
                "version_no": new_version_no,
                "chunks_created": len(chunks),
            }

    # ------------------------------------------------------------------
    # Model self-optimization
    # ------------------------------------------------------------------

    def optimize_skill(
        self,
        source_id: str,
        new_content: str,
        *,
        summary: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> dict:
        """
        Agent creates an optimized version of a skill.
        Old version stays as 'deprecated', new version becomes 'active'.
        """
        with self._session_factory() as session:
            source = session.execute(
                select(SkillSourceORM).where(SkillSourceORM.id == source_id)
            ).scalars().first()
            if not source:
                raise ValueError(f"Skill source not found: {source_id}")

            # Deprecate old active versions
            old_versions = session.execute(
                select(SkillVersionORM).where(
                    SkillVersionORM.skill_source_id == source_id,
                    SkillVersionORM.status == "active",
                )
            ).scalars().all()
            for v in old_versions:
                v.status = "deprecated"
                session.add(v)

            # Get next version number
            latest = session.execute(
                select(SkillVersionORM.version_no).where(
                    SkillVersionORM.skill_source_id == source_id,
                ).order_by(SkillVersionORM.version_no.desc())
            ).scalars().first()

            new_version_no = (latest or 0) + 1
            version = SkillVersionORM(
                id=_new_uuid(),
                skill_source_id=source_id,
                version_no=new_version_no,
                canonical_content=new_content,
                summary=summary or f"Optimized version {new_version_no}",
                status="active",
                confidence=0.9,
                updated_by="agent",
            )
            session.add(version)
            session.flush()

            # Chunk and embed the new content
            chunks = chunk_text(new_content, chunk_size=800, chunk_overlap=100)
            for idx, chunk_content in enumerate(chunks):
                chunk = SkillChunkORM(
                    id=_new_uuid(),
                    skill_version_id=version.id,
                    chunk_index=idx,
                    chunk_text=chunk_content,
                    token_count=_estimate_tokens(chunk_content),
                )
                session.add(chunk)
                session.flush()

                try:
                    vector = self._embedder(
                        chunk_content,
                        model=self._embedding_model,
                        expected_dims=self._embedding_dims,
                    )
                except Exception as exc:
                    logger.warning("Embedding failed for optimized chunk %d: %s", idx, exc)
                    vector = []

                embedding = SkillEmbeddingORM(
                    id=_new_uuid(),
                    chunk_id=chunk.id,
                    embedding_json="[]",
                    model_name=self._embedding_model,
                )
                session.add(embedding)
                session.flush()

                if self._is_pg and vector:
                    self._write_vector(session, "skills.skill_embeddings", embedding.id, vector)

            for tag_text in (tags or []):
                tag = SkillTagORM(
                    id=_new_uuid(),
                    skill_version_id=version.id,
                    tag=tag_text.strip().lower(),
                )
                session.add(tag)

            session.commit()
            return {
                "version_id": version.id,
                "version_no": new_version_no,
                "chunks_created": len(chunks),
            }

    # ------------------------------------------------------------------
    # Search (RAG)
    # ------------------------------------------------------------------

    def search_skills(
        self,
        query: str,
        *,
        limit: int = 5,
        status: str = "active",
    ) -> list[dict]:
        """Semantic search over active skill chunks using pgvector."""
        if not self._is_pg:
            return self._search_skills_fallback(query, limit=limit, status=status)

        try:
            query_vector = self._embedder(
                query, model=self._embedding_model, expected_dims=self._embedding_dims
            )
        except Exception as exc:
            logger.error("Failed to embed query for skill search: %s", exc)
            return []

        vector_literal = _to_vector_literal(query_vector)
        sql = """
            SELECT
                se.id AS embedding_id,
                sc.chunk_text,
                sc.chunk_index,
                sv.id AS version_id,
                sv.summary,
                sv.version_no,
                ss.id AS source_id,
                ss.title,
                ss.source_path,
                (se.embedding <=> CAST(:query_vector AS vector)) AS distance
            FROM skills.skill_embeddings se
            JOIN skills.skill_chunks sc ON sc.id = se.chunk_id
            JOIN skills.skill_versions sv ON sv.id = sc.skill_version_id
            JOIN skills.skill_sources ss ON ss.id = sv.skill_source_id
            WHERE sv.status = :status
              AND se.embedding IS NOT NULL
            ORDER BY se.embedding <=> CAST(:query_vector AS vector) ASC
            LIMIT :limit
        """
        with self._session_factory() as session:
            rows = session.execute(
                text(sql),
                {"query_vector": vector_literal, "status": status, "limit": limit},
            ).mappings().all()

        return [
            {
                "source_id": row["source_id"],
                "version_id": row["version_id"],
                "title": row["title"],
                "source_path": row["source_path"],
                "summary": row["summary"],
                "version_no": row["version_no"],
                "chunk_text": row["chunk_text"],
                "chunk_index": row["chunk_index"],
                "distance": float(row["distance"]),
            }
            for row in rows
        ]

    def _search_skills_fallback(
        self, query: str, *, limit: int = 5, status: str = "active"
    ) -> list[dict]:
        """Fallback keyword search for non-pgvector backends."""
        query_lower = query.lower()
        with self._session_factory() as session:
            versions = session.execute(
                select(SkillVersionORM).where(SkillVersionORM.status == status)
            ).scalars().all()

            results = []
            for v in versions:
                if query_lower in v.canonical_content.lower():
                    source = session.execute(
                        select(SkillSourceORM).where(SkillSourceORM.id == v.skill_source_id)
                    ).scalars().first()
                    results.append({
                        "source_id": v.skill_source_id,
                        "version_id": v.id,
                        "title": source.title if source else "",
                        "source_path": source.source_path if source else "",
                        "summary": v.summary,
                        "version_no": v.version_no,
                        "chunk_text": v.canonical_content[:500],
                        "chunk_index": 0,
                        "distance": 0.0,
                    })
            return results[:limit]

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get_active_skills(self) -> list[dict]:
        """List all active skill versions."""
        with self._session_factory() as session:
            rows = session.execute(
                select(SkillVersionORM, SkillSourceORM).join(
                    SkillSourceORM, SkillVersionORM.skill_source_id == SkillSourceORM.id
                ).where(SkillVersionORM.status == "active")
                .order_by(SkillSourceORM.title)
            ).all()

            return [
                {
                    "source_id": source.id,
                    "version_id": version.id,
                    "title": source.title,
                    "source_path": source.source_path,
                    "version_no": version.version_no,
                    "summary": version.summary,
                    "confidence": version.confidence,
                    "updated_by": version.updated_by,
                    "created_at": version.created_at.isoformat() if version.created_at else None,
                }
                for version, source in rows
            ]

    def get_skill_content(self, source_id: str, *, version: Optional[int] = None) -> Optional[dict]:
        """Get full content of a skill by source ID, optionally specific version."""
        with self._session_factory() as session:
            source = session.execute(
                select(SkillSourceORM).where(SkillSourceORM.id == source_id)
            ).scalars().first()
            if not source:
                return None

            if version is not None:
                ver = session.execute(
                    select(SkillVersionORM).where(
                        SkillVersionORM.skill_source_id == source_id,
                        SkillVersionORM.version_no == version,
                    )
                ).scalars().first()
            else:
                ver = session.execute(
                    select(SkillVersionORM).where(
                        SkillVersionORM.skill_source_id == source_id,
                        SkillVersionORM.status == "active",
                    ).order_by(SkillVersionORM.version_no.desc())
                ).scalars().first()

            if not ver:
                return None

            return {
                "source_id": source.id,
                "title": source.title,
                "source_path": source.source_path,
                "version_id": ver.id,
                "version_no": ver.version_no,
                "content": ver.canonical_content,
                "summary": ver.summary,
                "status": ver.status,
                "confidence": ver.confidence,
                "updated_by": ver.updated_by,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_vector(self, session: Session, table: str, record_id: str, vector: list[float]) -> None:
        """Write a vector to a pgvector column via raw SQL."""
        vec_literal = _to_vector_literal(vector)
        session.execute(
            text(f"UPDATE {table} SET embedding = CAST(:vec AS vector) WHERE id = :rid"),
            {"vec": vec_literal, "rid": record_id},
        )


def _to_vector_literal(values: list[float]) -> str:
    serialized = ",".join(f"{float(v):.10f}" for v in values)
    return f"[{serialized}]"

"""
BookmarkService — Manages URL registry, access stats, and trust scoring.

Auto-updatable: model auto-tracks URLs during research.
Trusted vs frequent are separate flags (a URL can be frequent but untrustworthy).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

from sqlalchemy import select, text
from sqlalchemy.orm import sessionmaker

from storage.models import (
    UrlEmbeddingORM,
    UrlRegistryORM,
    UrlStatsORM,
    _new_uuid,
    _utcnow,
)
from tools.embedding_provider import embed_text_ollama
from main.config import KNOWLEDGE_EMBEDDING_MODEL, KNOWLEDGE_EMBEDDING_DIMS

logger = logging.getLogger(__name__)


class BookmarkService:
    """Service for managing URL bookmarks and research links."""

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
    # Track URL
    # ------------------------------------------------------------------

    def track_url(
        self,
        *,
        url: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        source_type: str = "research",
        trust_score: float = 0.5,
    ) -> dict:
        """
        Register a URL or update its access stats if already tracked.
        Called automatically during research.
        """
        parsed = urlparse(url)
        domain = parsed.netloc or ""

        with self._session_factory() as session:
            existing = session.execute(
                select(UrlRegistryORM).where(UrlRegistryORM.url == url)
            ).scalars().first()

            if existing:
                # Update metadata
                existing.last_seen_at = _utcnow()
                if title and not existing.title:
                    existing.title = title
                if description and not existing.description:
                    existing.description = description
                session.add(existing)

                # Increment access stats
                self._increment_access(session, existing.id)
                session.commit()
                return {"action": "updated", "url_id": existing.id}

            # New URL
            entry = UrlRegistryORM(
                id=_new_uuid(),
                url=url,
                domain=domain,
                title=title,
                description=description,
                source_type=source_type,
                trust_score=trust_score,
            )
            session.add(entry)
            session.flush()

            # Create stats row
            stats = UrlStatsORM(
                id=_new_uuid(),
                url_id=entry.id,
                access_count=1,
                last_accessed_at=_utcnow(),
            )
            session.add(stats)

            # Embed the URL description if available
            if description or title:
                embed_text = f"{title or ''}: {description or url}"
                try:
                    vector = self._embedder(
                        embed_text,
                        model=self._embedding_model,
                        expected_dims=self._embedding_dims,
                    )
                except Exception as exc:
                    logger.warning("Embedding failed for URL %s: %s", url[:80], exc)
                    vector = []

                emb = UrlEmbeddingORM(
                    id=_new_uuid(),
                    url_id=entry.id,
                    content_summary=embed_text[:2000],
                    embedding_json="[]",
                    model_name=self._embedding_model,
                )
                session.add(emb)
                session.flush()

                if self._is_pg and vector:
                    vec_literal = _to_vector_literal(vector)
                    session.execute(
                        text("UPDATE knowledge.url_embeddings SET embedding = CAST(:vec AS vector) WHERE id = :rid"),
                        {"vec": vec_literal, "rid": emb.id},
                    )

            session.commit()
            return {"action": "created", "url_id": entry.id}

    def _increment_access(self, session, url_id: str) -> None:
        """Increment access count for a URL."""
        stats = session.execute(
            select(UrlStatsORM).where(UrlStatsORM.url_id == url_id)
        ).scalars().first()

        if stats:
            stats.access_count += 1
            stats.last_accessed_at = _utcnow()
            session.add(stats)
        else:
            stats = UrlStatsORM(
                id=_new_uuid(),
                url_id=url_id,
                access_count=1,
                last_accessed_at=_utcnow(),
            )
            session.add(stats)

    # ------------------------------------------------------------------
    # Record task usage
    # ------------------------------------------------------------------

    def record_task_usage(self, url_id: str) -> None:
        """Mark a URL as used for an agent task."""
        with self._session_factory() as session:
            stats = session.execute(
                select(UrlStatsORM).where(UrlStatsORM.url_id == url_id)
            ).scalars().first()
            if stats:
                stats.last_used_for_task_at = _utcnow()
                session.add(stats)
                session.commit()

    # ------------------------------------------------------------------
    # Set trust score
    # ------------------------------------------------------------------

    def set_trust_score(self, url_id: str, trust_score: float) -> bool:
        """Update trust score for a URL (0.0 to 1.0)."""
        with self._session_factory() as session:
            entry = session.execute(
                select(UrlRegistryORM).where(UrlRegistryORM.id == url_id)
            ).scalars().first()
            if not entry:
                return False
            entry.trust_score = max(0.0, min(1.0, trust_score))
            session.add(entry)
            session.commit()
            return True

    # ------------------------------------------------------------------
    # Search bookmarks (semantic)
    # ------------------------------------------------------------------

    def search_bookmarks(self, query: str, *, limit: int = 5) -> list[dict]:
        """Semantic search over bookmarked URLs."""
        if not self._is_pg:
            return self._search_fallback(query, limit=limit)

        try:
            query_vector = self._embedder(
                query, model=self._embedding_model, expected_dims=self._embedding_dims
            )
        except Exception as exc:
            logger.error("Failed to embed query for bookmark search: %s", exc)
            return []

        vector_literal = _to_vector_literal(query_vector)
        sql = """
            SELECT
                ur.id, ur.url, ur.domain, ur.title, ur.description,
                ur.source_type, ur.trust_score,
                us.access_count, us.last_accessed_at,
                (ue.embedding <=> CAST(:query_vector AS vector)) AS distance
            FROM knowledge.url_embeddings ue
            JOIN knowledge.url_registry ur ON ur.id = ue.url_id
            LEFT JOIN knowledge.url_stats us ON us.url_id = ur.id
            WHERE ue.embedding IS NOT NULL
            ORDER BY ue.embedding <=> CAST(:query_vector AS vector) ASC
            LIMIT :limit
        """
        with self._session_factory() as session:
            rows = session.execute(
                text(sql), {"query_vector": vector_literal, "limit": limit}
            ).mappings().all()

        return [
            {
                "id": row["id"],
                "url": row["url"],
                "domain": row["domain"],
                "title": row["title"],
                "description": row["description"],
                "source_type": row["source_type"],
                "trust_score": float(row["trust_score"] or 0),
                "access_count": int(row["access_count"] or 0),
                "distance": float(row["distance"]),
            }
            for row in rows
        ]

    def _search_fallback(self, query: str, *, limit: int = 5) -> list[dict]:
        """Keyword fallback for non-pgvector backends."""
        query_lower = query.lower()
        with self._session_factory() as session:
            rows = session.execute(select(UrlRegistryORM)).scalars().all()
            results = []
            for r in rows:
                searchable = f"{r.url} {r.title or ''} {r.description or ''}".lower()
                if query_lower in searchable:
                    results.append({
                        "id": r.id,
                        "url": r.url,
                        "domain": r.domain,
                        "title": r.title,
                        "description": r.description,
                        "source_type": r.source_type,
                        "trust_score": r.trust_score,
                        "access_count": 0,
                        "distance": 0.0,
                    })
            return results[:limit]

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get_most_accessed(self, *, limit: int = 10) -> list[dict]:
        """Get the most frequently accessed URLs."""
        with self._session_factory() as session:
            rows = session.execute(
                select(UrlRegistryORM, UrlStatsORM).join(
                    UrlStatsORM, UrlStatsORM.url_id == UrlRegistryORM.id
                ).order_by(UrlStatsORM.access_count.desc()).limit(limit)
            ).all()

            return [
                {
                    "id": entry.id,
                    "url": entry.url,
                    "domain": entry.domain,
                    "title": entry.title,
                    "trust_score": entry.trust_score,
                    "access_count": stats.access_count,
                    "last_accessed_at": stats.last_accessed_at.isoformat() if stats.last_accessed_at else None,
                }
                for entry, stats in rows
            ]

    def get_trusted_sources(self, *, category: Optional[str] = None, min_trust: float = 0.7) -> list[dict]:
        """Get URLs with trust score >= threshold."""
        with self._session_factory() as session:
            stmt = select(UrlRegistryORM).where(
                UrlRegistryORM.trust_score >= min_trust
            ).order_by(UrlRegistryORM.trust_score.desc())

            if category:
                stmt = stmt.where(UrlRegistryORM.source_type == category)

            rows = session.execute(stmt).scalars().all()
            return [
                {
                    "id": r.id,
                    "url": r.url,
                    "domain": r.domain,
                    "title": r.title,
                    "trust_score": r.trust_score,
                    "source_type": r.source_type,
                }
                for r in rows
            ]


def _to_vector_literal(values: list[float]) -> str:
    serialized = ",".join(f"{float(v):.10f}" for v in values)
    return f"[{serialized}]"

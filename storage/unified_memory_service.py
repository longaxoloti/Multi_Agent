"""Unified memory service for the redesigned knowledge schema.

This service is intentionally additive and can run in parallel with legacy
services during cutover. It provides:
1. Auto context ingest with similarity-band lifecycle updates.
2. Blended retrieval (vector + full-text + confidence + recency + type weight).
3. Lightweight edge creation between related memories.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import text

from main.config import KNOWLEDGE_EMBEDDING_DIMS, KNOWLEDGE_EMBEDDING_MODEL
from storage.trusted_db import AgentDBRepository
from tools.embedding_provider import embed_text_ollama


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _vector_literal(values: list[float]) -> str:
    serialized = ",".join(f"{float(v):.10f}" for v in values)
    return f"[{serialized}]"


def _canonical_hash(content: str) -> str:
    normalized = " ".join((content or "").strip().lower().split())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def _content_hash(content: str) -> str:
    return hashlib.md5((content or "").encode("utf-8")).hexdigest()


@dataclass
class UnifiedMemoryResult:
    memory_id: str
    entity_id: str
    action: str
    similarity: float
    band: str


class UnifiedMemoryService:
    """RAG and lifecycle service for unified `knowledge.*` memory tables."""

    _TYPE_WEIGHTS = {
        "decision": 1.15,
        "preference": 1.1,
        "error_pattern": 1.08,
        "project_fact": 1.04,
        "url_summary": 1.02,
        "note": 1.0,
        "fact": 1.0,
        "skill_summary": 0.98,
    }

    def __init__(
        self,
        repo: Optional[AgentDBRepository] = None,
        embedder=None,
    ):
        self._repo = repo or AgentDBRepository()
        self._embedder = embedder or embed_text_ollama
        self._embedding_model = KNOWLEDGE_EMBEDDING_MODEL
        self._embedding_dims = int(KNOWLEDGE_EMBEDDING_DIMS)

    @staticmethod
    def _set_current_user_id(conn, owner_user_id: str) -> None:
        conn.execute(
            text("SELECT set_config('app.current_user_id', :user_id, true)"),
            {"user_id": str(owner_user_id or "")},
        )

    @staticmethod
    def _band_for_similarity(similarity: float) -> str:
        if similarity < 0.55:
            return "insert_new"
        if similarity < 0.75:
            return "link_related"
        if similarity < 0.90:
            return "update_evidence"
        if similarity < 0.95:
            return "merge_versioned"
        return "dedup_hard"

    def _build_embedding(self, text: str) -> list[float]:
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

    def _upsert_entity(
        self,
        *,
        owner_user_id: str,
        entity_type: str,
        entity_ref: str,
        title: str,
        description: str,
        source_type: str,
        trust_score: float,
    ) -> str:
        entity_id = str(uuid.uuid4())
        with self._repo.engine.begin() as conn:
            self._set_current_user_id(conn, owner_user_id)
            row = conn.execute(
                text(
                    """
                    INSERT INTO knowledge.entities (
                        id,
                        entity_type,
                        entity_ref,
                        owner_user_id,
                        title,
                        description,
                        source_type,
                        trust_score,
                        status,
                        created_at,
                        updated_at
                    ) VALUES (
                        :id,
                        :entity_type,
                        :entity_ref,
                        :owner_user_id,
                        :title,
                        :description,
                        :source_type,
                        :trust_score,
                        'active',
                        now(),
                        now()
                    )
                    ON CONFLICT (entity_type, entity_ref, owner_user_id)
                    DO UPDATE
                    SET
                        title = EXCLUDED.title,
                        description = EXCLUDED.description,
                        source_type = EXCLUDED.source_type,
                        trust_score = GREATEST(knowledge.entities.trust_score, EXCLUDED.trust_score),
                        updated_at = now()
                    RETURNING id
                    """
                ),
                {
                    "id": entity_id,
                    "entity_type": entity_type,
                    "entity_ref": entity_ref,
                    "owner_user_id": owner_user_id,
                    "title": title,
                    "description": description,
                    "source_type": source_type,
                    "trust_score": trust_score,
                },
            ).first()
        return str(row[0])

    def _find_best_match(
        self,
        *,
        owner_user_id: str,
        embedding: list[float],
        limit: int = 1,
    ) -> tuple[Optional[str], float]:
        if not embedding:
            return None, 0.0

        literal = _vector_literal(embedding)
        with self._repo.engine.begin() as conn:
            self._set_current_user_id(conn, owner_user_id)
            row = conn.execute(
                text(
                    """
                    SELECT
                        m.id,
                        (1 - (me.embedding <=> CAST(:query_vec AS vector))) AS similarity
                    FROM knowledge.memory_embeddings me
                    JOIN knowledge.memories m ON m.id = me.memory_id
                    WHERE m.status = 'active'
                      AND (m.owner_user_id = :owner_user_id OR m.owner_user_id = 'system')
                    ORDER BY me.embedding <=> CAST(:query_vec AS vector)
                    LIMIT :limit
                    """
                ),
                {
                    "query_vec": literal,
                    "owner_user_id": owner_user_id,
                    "limit": limit,
                },
            ).first()
        if not row:
            return None, 0.0
        return str(row[0]), float(row[1] or 0.0)

    def ingest_context(
        self,
        *,
        owner_user_id: str,
        content: str,
        embedding: list[float],
        entity_type: str = "session_fact",
        entity_ref: str = "default",
        memory_type: str = "fact",
        confidence: float = 0.75,
        trust_score: float = 0.7,
        title: str = "",
        metadata: Optional[dict] = None,
    ) -> UnifiedMemoryResult:
        """Store or merge context by similarity bands.

        Band policy:
        - <0.55: insert memory
        - 0.55-0.75: insert + related edge
        - 0.75-0.90: update evidence on best match
        - 0.90-0.95: merge into best match with version record
        - >0.95: hard dedup (touch access stats)
        """

        payload = (content or "").strip()
        if not payload:
            raise ValueError("content is empty")
        if not embedding:
            raise ValueError("embedding is required")

        entity_id = self._upsert_entity(
            owner_user_id=owner_user_id,
            entity_type=entity_type,
            entity_ref=entity_ref,
            title=title,
            description=(title or payload[:180]),
            source_type="auto_capture",
            trust_score=trust_score,
        )

        best_memory_id, similarity = self._find_best_match(
            owner_user_id=owner_user_id,
            embedding=embedding,
        )
        band = self._band_for_similarity(similarity)

        with self._repo.engine.begin() as conn:
            self._set_current_user_id(conn, owner_user_id)
            if band in ("insert_new", "link_related"):
                memory_id = str(uuid.uuid4())
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
                            created_at,
                            updated_at,
                            metadata_json
                        ) VALUES (
                            :id,
                            :entity_id,
                            :owner_user_id,
                            :memory_type,
                            'auto',
                            :summary,
                            :content,
                            :confidence,
                            1.0,
                            'active',
                            now(),
                            :content_hash,
                            :canonical_hash,
                            now(),
                            now(),
                            :metadata_json
                        )
                        """
                    ),
                    {
                        "id": memory_id,
                        "entity_id": entity_id,
                        "owner_user_id": owner_user_id,
                        "memory_type": memory_type,
                        "summary": title or payload[:180],
                        "content": payload,
                        "confidence": confidence,
                        "content_hash": _content_hash(payload),
                        "canonical_hash": _canonical_hash(payload),
                        "metadata_json": json.dumps(metadata or {}, ensure_ascii=True),
                    },
                )
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
                            'bge-m3',
                            now()
                        )
                        """
                    ),
                    {
                        "id": str(uuid.uuid4()),
                        "memory_id": memory_id,
                        "embedding_literal": _vector_literal(embedding),
                    },
                )

                if band == "link_related" and best_memory_id:
                    conn.execute(
                        text(
                            """
                            INSERT INTO knowledge.memory_edges (
                                id,
                                source_memory_id,
                                target_memory_id,
                                edge_type,
                                weight,
                                metadata_json,
                                created_at
                            ) VALUES (
                                :id,
                                :source_memory_id,
                                :target_memory_id,
                                'related_to',
                                :weight,
                                :metadata_json,
                                now()
                            )
                            ON CONFLICT (source_memory_id, target_memory_id, edge_type)
                            DO NOTHING
                            """
                        ),
                        {
                            "id": str(uuid.uuid4()),
                            "source_memory_id": memory_id,
                            "target_memory_id": best_memory_id,
                            "weight": max(0.55, min(similarity, 0.95)),
                            "metadata_json": json.dumps(
                                {"policy": "similarity_band", "band": band},
                                ensure_ascii=True,
                            ),
                        },
                    )

                action = "inserted" if band == "insert_new" else "inserted_linked"
                return UnifiedMemoryResult(
                    memory_id=memory_id,
                    entity_id=entity_id,
                    action=action,
                    similarity=similarity,
                    band=band,
                )

            if not best_memory_id:
                raise RuntimeError("best memory expected for non-insert band")

            if band == "update_evidence":
                conn.execute(
                    text(
                        """
                        UPDATE knowledge.memories
                        SET
                            confidence = GREATEST(confidence, :confidence),
                            updated_at = now(),
                            metadata_json = :metadata_json
                        WHERE id = :memory_id
                        """
                    ),
                    {
                        "confidence": confidence,
                        "metadata_json": json.dumps(metadata or {}, ensure_ascii=True),
                        "memory_id": best_memory_id,
                    },
                )
                return UnifiedMemoryResult(
                    memory_id=best_memory_id,
                    entity_id=entity_id,
                    action="updated_evidence",
                    similarity=similarity,
                    band=band,
                )

            if band == "merge_versioned":
                previous = conn.execute(
                    text("SELECT content FROM knowledge.memories WHERE id = :memory_id"),
                    {"memory_id": best_memory_id},
                ).first()
                previous_content = previous[0] if previous else ""
                merged_content = f"{previous_content}\n\n[MERGED]\n{payload}".strip()

                conn.execute(
                    text(
                        """
                        UPDATE knowledge.memories
                        SET
                            content = :content,
                            canonical_hash = :canonical_hash,
                            updated_at = now(),
                            confidence = GREATEST(confidence, :confidence)
                        WHERE id = :memory_id
                        """
                    ),
                    {
                        "content": merged_content,
                        "canonical_hash": _canonical_hash(merged_content),
                        "confidence": confidence,
                        "memory_id": best_memory_id,
                    },
                )

                version_no = conn.execute(
                    text(
                        """
                        SELECT COALESCE(MAX(version_no), 0) + 1
                        FROM knowledge.memory_versions
                        WHERE memory_id = :memory_id
                        """
                    ),
                    {"memory_id": best_memory_id},
                ).scalar_one()

                conn.execute(
                    text(
                        """
                        INSERT INTO knowledge.memory_versions (
                            id,
                            memory_id,
                            version_no,
                            reason,
                            previous_content,
                            new_content,
                            changed_by,
                            created_at
                        ) VALUES (
                            :id,
                            :memory_id,
                            :version_no,
                            'similarity_merge',
                            :previous_content,
                            :new_content,
                            'agent',
                            now()
                        )
                        """
                    ),
                    {
                        "id": str(uuid.uuid4()),
                        "memory_id": best_memory_id,
                        "version_no": int(version_no),
                        "previous_content": previous_content,
                        "new_content": merged_content,
                    },
                )

                return UnifiedMemoryResult(
                    memory_id=best_memory_id,
                    entity_id=entity_id,
                    action="merged_versioned",
                    similarity=similarity,
                    band=band,
                )

            # dedup_hard
            conn.execute(
                text(
                    """
                    UPDATE knowledge.memories
                    SET
                        access_count = access_count + 1,
                        last_accessed_at = now(),
                        updated_at = now()
                    WHERE id = :memory_id
                    """
                ),
                {"memory_id": best_memory_id},
            )
            return UnifiedMemoryResult(
                memory_id=best_memory_id,
                entity_id=entity_id,
                action="deduped",
                similarity=similarity,
                band=band,
            )

    def search(
        self,
        *,
        owner_user_id: str,
        query_text: str,
        query_embedding: list[float],
        limit: int = 10,
        memory_types: Optional[list[str]] = None,
        graph_hops: int = 1,
    ) -> list[dict]:
        """Search unified memories with blended ranking and event logging."""

        if not query_text.strip():
            return []
        safe_limit = max(1, min(limit, 50))
        types_filter = [t for t in (memory_types or []) if t]

        vector_results: dict[str, dict] = {}
        text_results: dict[str, dict] = {}

        with self._repo.engine.begin() as conn:
            self._set_current_user_id(conn, owner_user_id)
            if query_embedding:
                vector_rows = conn.execute(
                    text(
                        """
                        SELECT
                            m.id,
                            m.entity_id,
                            m.memory_type,
                            m.summary,
                            m.content,
                            m.confidence,
                            m.updated_at,
                            (1 - (me.embedding <=> CAST(:query_vec AS vector))) AS vector_score
                        FROM knowledge.memory_embeddings me
                        JOIN knowledge.memories m ON m.id = me.memory_id
                        WHERE m.status = 'active'
                          AND (m.owner_user_id = :owner_user_id OR m.owner_user_id = 'system')
                          AND (m.valid_until IS NULL OR m.valid_until > now())
                        ORDER BY me.embedding <=> CAST(:query_vec AS vector)
                        LIMIT :limit
                        """
                    ),
                    {
                        "query_vec": _vector_literal(query_embedding),
                        "owner_user_id": owner_user_id,
                        "limit": safe_limit * 4,
                    },
                ).mappings().all()
                for row in vector_rows:
                    vector_results[str(row["id"])] = dict(row)

            text_rows = conn.execute(
                text(
                    """
                    SELECT
                        m.id,
                        m.entity_id,
                        m.memory_type,
                        m.summary,
                        m.content,
                        m.confidence,
                        m.updated_at,
                        ts_rank(m.search_tsv, plainto_tsquery('simple', :query_text)) AS text_score
                    FROM knowledge.memories m
                    WHERE m.status = 'active'
                      AND (m.owner_user_id = :owner_user_id OR m.owner_user_id = 'system')
                      AND (m.valid_until IS NULL OR m.valid_until > now())
                      AND m.search_tsv @@ plainto_tsquery('simple', :query_text)
                    ORDER BY text_score DESC
                    LIMIT :limit
                    """
                ),
                {
                    "query_text": query_text,
                    "owner_user_id": owner_user_id,
                    "limit": safe_limit * 4,
                },
            ).mappings().all()
            for row in text_rows:
                text_results[str(row["id"])] = dict(row)

            merged_ids = set(vector_results.keys()) | set(text_results.keys())
            scored: list[dict] = []
            now = _utcnow()

            for memory_id in merged_ids:
                vec = float(vector_results.get(memory_id, {}).get("vector_score") or 0.0)
                txt = float(text_results.get(memory_id, {}).get("text_score") or 0.0)
                base = vector_results.get(memory_id) or text_results.get(memory_id)
                if not base:
                    continue

                mem_type = str(base.get("memory_type") or "fact")
                if types_filter and mem_type not in types_filter:
                    continue

                updated_at = base.get("updated_at")
                age_days = 365.0
                if isinstance(updated_at, datetime):
                    age_days = max((now - updated_at).total_seconds() / 86400.0, 0.0)
                recency = math.exp(-age_days / 30.0)

                confidence = float(base.get("confidence") or 0.5)
                type_score = float(self._TYPE_WEIGHTS.get(mem_type, 1.0))

                final_score = (
                    0.45 * vec
                    + 0.20 * txt
                    + 0.10 * confidence
                    + 0.10 * recency
                    + 0.15 * type_score
                )

                scored.append(
                    {
                        "id": memory_id,
                        "entity_id": base.get("entity_id"),
                        "memory_type": mem_type,
                        "summary": base.get("summary") or "",
                        "content": base.get("content") or "",
                        "vector_score": vec,
                        "text_score": txt,
                        "confidence_score": confidence,
                        "recency_score": recency,
                        "type_score": type_score,
                        "final_score": final_score,
                    }
                )

            scored.sort(key=lambda item: item["final_score"], reverse=True)
            top = scored[:safe_limit]

            # One-hop graph expansion from the top seeds.
            if graph_hops > 0 and top:
                seed_ids = [item["id"] for item in top]
                edge_rows = conn.execute(
                    text(
                        """
                        SELECT
                            e.source_memory_id,
                            e.target_memory_id,
                            e.edge_type,
                            e.weight
                        FROM knowledge.memory_edges e
                        WHERE e.source_memory_id = ANY(:seed_ids)
                          AND e.edge_type IN ('related_to', 'supports', 'derived_from', 'supersedes')
                        LIMIT :limit
                        """
                    ),
                    {
                        "seed_ids": seed_ids,
                        "limit": safe_limit * 6,
                    },
                ).mappings().all()

                seed_score_map = {item["id"]: item["final_score"] for item in top}
                existing_ids = set(seed_score_map.keys())
                expanded_candidates: dict[str, dict] = {}

                for edge in edge_rows:
                    src_id = str(edge["source_memory_id"])
                    tgt_id = str(edge["target_memory_id"])
                    if tgt_id in existing_ids:
                        continue

                    seed_score = seed_score_map.get(src_id, 0.0)
                    edge_weight = float(edge["weight"] or 0.5)
                    if seed_score <= 0.0:
                        continue

                    graph_bonus = seed_score * max(min(edge_weight, 1.0), 0.1) * 0.25
                    if graph_bonus <= 0:
                        continue

                    current = expanded_candidates.get(tgt_id)
                    if current is None or graph_bonus > current["graph_bonus"]:
                        expanded_candidates[tgt_id] = {
                            "memory_id": tgt_id,
                            "graph_bonus": graph_bonus,
                            "from_memory_id": src_id,
                            "edge_type": str(edge["edge_type"]),
                        }

                if expanded_candidates:
                    expanded_rows = conn.execute(
                        text(
                            """
                            SELECT
                                m.id,
                                m.entity_id,
                                m.memory_type,
                                m.summary,
                                m.content,
                                m.confidence,
                                m.updated_at
                            FROM knowledge.memories m
                            WHERE m.id = ANY(:target_ids)
                              AND m.status = 'active'
                              AND (m.owner_user_id = :owner_user_id OR m.owner_user_id = 'system')
                              AND (m.valid_until IS NULL OR m.valid_until > now())
                            """
                        ),
                        {
                            "target_ids": list(expanded_candidates.keys()),
                            "owner_user_id": owner_user_id,
                        },
                    ).mappings().all()

                    for row in expanded_rows:
                        memory_id = str(row["id"])
                        candidate = expanded_candidates.get(memory_id)
                        if not candidate:
                            continue
                        mem_type = str(row["memory_type"] or "fact")
                        if types_filter and mem_type not in types_filter:
                            continue

                        updated_at = row.get("updated_at")
                        age_days = 365.0
                        if isinstance(updated_at, datetime):
                            age_days = max((now - updated_at).total_seconds() / 86400.0, 0.0)
                        recency = math.exp(-age_days / 30.0)
                        confidence = float(row.get("confidence") or 0.5)
                        type_score = float(self._TYPE_WEIGHTS.get(mem_type, 1.0))

                        final_score = (
                            candidate["graph_bonus"]
                            + 0.10 * confidence
                            + 0.10 * recency
                            + 0.10 * type_score
                        )

                        top.append(
                            {
                                "id": memory_id,
                                "entity_id": row.get("entity_id"),
                                "memory_type": mem_type,
                                "summary": row.get("summary") or "",
                                "content": row.get("content") or "",
                                "vector_score": 0.0,
                                "text_score": 0.0,
                                "confidence_score": confidence,
                                "recency_score": recency,
                                "type_score": type_score,
                                "final_score": final_score,
                                "graph_bonus": candidate["graph_bonus"],
                                "expanded_from": candidate["from_memory_id"],
                                "expanded_via": candidate["edge_type"],
                            }
                        )

                    top.sort(key=lambda item: item["final_score"], reverse=True)
                    top = top[:safe_limit]

            query_hash = hashlib.md5(query_text.strip().lower().encode("utf-8")).hexdigest()
            for rank, row in enumerate(top, start=1):
                conn.execute(
                    text(
                        """
                        INSERT INTO knowledge.retrieval_events (
                            id,
                            owner_user_id,
                            query_text,
                            query_hash,
                            memory_id,
                            vector_score,
                            text_score,
                            type_score,
                            recency_score,
                            confidence_score,
                            final_score,
                            rank_position,
                            created_at
                        ) VALUES (
                            :id,
                            :owner_user_id,
                            :query_text,
                            :query_hash,
                            :memory_id,
                            :vector_score,
                            :text_score,
                            :type_score,
                            :recency_score,
                            :confidence_score,
                            :final_score,
                            :rank_position,
                            now()
                        )
                        """
                    ),
                    {
                        "id": str(uuid.uuid4()),
                        "owner_user_id": owner_user_id,
                        "query_text": query_text,
                        "query_hash": query_hash,
                        "memory_id": row["id"],
                        "vector_score": row["vector_score"],
                        "text_score": row["text_score"],
                        "type_score": row["type_score"],
                        "recency_score": row["recency_score"],
                        "confidence_score": row["confidence_score"],
                        "final_score": row["final_score"],
                        "rank_position": rank,
                    },
                )

            return top

    def search_by_text(
        self,
        *,
        owner_user_id: str,
        query_text: str,
        limit: int = 10,
        memory_types: Optional[list[str]] = None,
        graph_hops: int = 1,
    ) -> list[dict]:
        query = (query_text or "").strip()
        if not query:
            return []

        query_embedding = self._build_embedding(query)
        return self.search(
            owner_user_id=owner_user_id,
            query_text=query,
            query_embedding=query_embedding,
            limit=limit,
            memory_types=memory_types,
            graph_hops=graph_hops,
        )
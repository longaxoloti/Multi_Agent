"""
ProjectService — Manages project knowledge with verification lifecycle.

Auto-updatable: model extracts facts from coding sessions; periodic verification
compares stored facts against actual project state on disk.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select, text
from sqlalchemy.orm import Session, sessionmaker

from storage.models import (
    ProjectEntityORM,
    ProjectFactORM,
    ProjectSnapshotORM,
    ProjectSourceORM,
    ProjectVerificationORM,
    _new_uuid,
    _utcnow,
)

logger = logging.getLogger(__name__)


class ProjectService:
    """Service for managing project analysis knowledge."""

    def __init__(self, session_factory: sessionmaker):
        self._session_factory = session_factory

    # ------------------------------------------------------------------
    # Register project
    # ------------------------------------------------------------------

    def register_project(
        self,
        *,
        project_name: str,
        repo_path: str,
        language: Optional[str] = None,
    ) -> dict:
        """Register a new project entity or return existing one."""
        with self._session_factory() as session:
            existing = session.execute(
                select(ProjectEntityORM).where(
                    ProjectEntityORM.repo_path == repo_path
                )
            ).scalars().first()

            if existing:
                return {
                    "action": "exists",
                    "project_id": existing.id,
                    "project_name": existing.project_name,
                }

            entity = ProjectEntityORM(
                id=_new_uuid(),
                project_name=project_name,
                repo_path=repo_path,
                language=language,
                status="active",
            )
            session.add(entity)
            session.commit()

            return {
                "action": "created",
                "project_id": entity.id,
                "project_name": project_name,
            }

    # ------------------------------------------------------------------
    # Save facts from sessions
    # ------------------------------------------------------------------

    def save_fact(
        self,
        *,
        project_id: str,
        fact_key: str,
        fact_value: str,
        session_id: Optional[str] = None,
        source_type: str = "session",
        confidence: float = 0.8,
    ) -> dict:
        """
        Save a project fact from a session or scan.
        If same key+value exists (active), update confidence.
        If same key + different value, keep both.
        """
        with self._session_factory() as session:
            # Create provenance source
            source = ProjectSourceORM(
                id=_new_uuid(),
                source_type=source_type,
                source_hash=hashlib.sha256(
                    f"{fact_key}:{fact_value}".encode()
                ).hexdigest()[:64],
            )
            session.add(source)
            session.flush()

            # Check for existing identical fact
            existing = session.execute(
                select(ProjectFactORM).where(
                    ProjectFactORM.project_entity_id == project_id,
                    ProjectFactORM.fact_key == fact_key.strip().lower(),
                    ProjectFactORM.fact_value == fact_value.strip(),
                    ProjectFactORM.status == "active",
                )
            ).scalars().first()

            if existing:
                if confidence > existing.confidence:
                    existing.confidence = confidence
                    existing.updated_at = _utcnow()
                    session.add(existing)
                session.commit()
                return {"action": "updated_confidence", "fact_id": existing.id}

            fact = ProjectFactORM(
                id=_new_uuid(),
                project_entity_id=project_id,
                fact_key=fact_key.strip().lower(),
                fact_value=fact_value.strip(),
                confidence=confidence,
                status="active",
                provenance_type=source_type,
                provenance_id=source.id,
                verified_at=_utcnow(),
            )
            session.add(fact)
            session.commit()

            return {"action": "created", "fact_id": fact.id}

    # ------------------------------------------------------------------
    # Get project facts
    # ------------------------------------------------------------------

    def get_project_facts(
        self,
        project_id: str,
        *,
        status: Optional[str] = "active",
        aspect: Optional[str] = None,
    ) -> list[dict]:
        """Get facts for a project, optionally filtered by status and aspect (fact_key prefix)."""
        with self._session_factory() as session:
            stmt = select(ProjectFactORM).where(
                ProjectFactORM.project_entity_id == project_id,
            ).order_by(ProjectFactORM.fact_key)

            if status:
                stmt = stmt.where(ProjectFactORM.status == status)

            if aspect:
                stmt = stmt.where(ProjectFactORM.fact_key.startswith(aspect.lower()))

            rows = session.execute(stmt).scalars().all()
            return [
                {
                    "id": r.id,
                    "fact_key": r.fact_key,
                    "fact_value": r.fact_value,
                    "confidence": r.confidence,
                    "status": r.status,
                    "provenance_type": r.provenance_type,
                    "verified_at": r.verified_at.isoformat() if r.verified_at else None,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]

    def get_project_by_path(self, repo_path: str) -> Optional[dict]:
        """Find project by repository path."""
        with self._session_factory() as session:
            entity = session.execute(
                select(ProjectEntityORM).where(
                    ProjectEntityORM.repo_path == repo_path
                )
            ).scalars().first()

            if not entity:
                return None

            return {
                "id": entity.id,
                "project_name": entity.project_name,
                "repo_path": entity.repo_path,
                "language": entity.language,
                "status": entity.status,
            }

    def list_projects(self, *, status: str = "active") -> list[dict]:
        """List all registered projects."""
        with self._session_factory() as session:
            rows = session.execute(
                select(ProjectEntityORM).where(
                    ProjectEntityORM.status == status
                ).order_by(ProjectEntityORM.project_name)
            ).scalars().all()

            return [
                {
                    "id": r.id,
                    "project_name": r.project_name,
                    "repo_path": r.repo_path,
                    "language": r.language,
                    "status": r.status,
                }
                for r in rows
            ]

    # ------------------------------------------------------------------
    # Verification lifecycle
    # ------------------------------------------------------------------

    def record_verification(
        self,
        *,
        project_id: str,
        result: str,
        details: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> str:
        """Record a verification run result."""
        with self._session_factory() as session:
            verification = ProjectVerificationORM(
                id=_new_uuid(),
                project_entity_id=project_id,
                verification_run_id=run_id or _new_uuid(),
                result=result,
                details=details,
            )
            session.add(verification)
            session.commit()
            return verification.id

    def mark_facts_stale(self, project_id: str, *, fact_keys: Optional[list[str]] = None) -> int:
        """Mark facts as stale (needing re-verification)."""
        with self._session_factory() as session:
            stmt = select(ProjectFactORM).where(
                ProjectFactORM.project_entity_id == project_id,
                ProjectFactORM.status == "active",
            )
            if fact_keys:
                stmt = stmt.where(ProjectFactORM.fact_key.in_([k.lower() for k in fact_keys]))

            rows = session.execute(stmt).scalars().all()
            for r in rows:
                r.status = "needs_verification"
                r.updated_at = _utcnow()
                session.add(r)

            session.commit()
            return len(rows)

    def get_stale_facts(self, *, days: int = 7) -> list[dict]:
        """Get facts that haven't been verified for N days across all projects."""
        cutoff = _utcnow() - timedelta(days=days)
        with self._session_factory() as session:
            rows = session.execute(
                select(ProjectFactORM, ProjectEntityORM).join(
                    ProjectEntityORM,
                    ProjectFactORM.project_entity_id == ProjectEntityORM.id,
                ).where(
                    ProjectFactORM.status.in_(["active", "needs_verification"]),
                    ProjectFactORM.verified_at < cutoff,
                ).order_by(ProjectFactORM.verified_at)
            ).all()

            return [
                {
                    "fact_id": fact.id,
                    "project_name": entity.project_name,
                    "repo_path": entity.repo_path,
                    "fact_key": fact.fact_key,
                    "fact_value": fact.fact_value,
                    "status": fact.status,
                    "verified_at": fact.verified_at.isoformat() if fact.verified_at else None,
                }
                for fact, entity in rows
            ]

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def save_snapshot(
        self,
        *,
        project_id: str,
        summary: str,
        snapshot_hash: Optional[str] = None,
        diff_from_previous: Optional[str] = None,
    ) -> str:
        """Save a periodic snapshot of a project's state."""
        with self._session_factory() as session:
            snapshot = ProjectSnapshotORM(
                id=_new_uuid(),
                project_entity_id=project_id,
                snapshot_hash=snapshot_hash,
                summary=summary,
                diff_from_previous=diff_from_previous,
            )
            session.add(snapshot)
            session.commit()
            return snapshot.id

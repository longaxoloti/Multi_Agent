"""
SecurityService — Manages secret references and access policies.

NEVER stores plaintext credentials. Only metadata + encrypted payload references.
Agent cannot write to manual.* or security.* without explicit user command.
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from storage.models import (
    AccessPolicyORM,
    SecretRefORM,
    _new_uuid,
    _utcnow,
)

logger = logging.getLogger(__name__)

# Default access policies
_DEFAULT_POLICIES = [
    # Agent can write to auto-updatable schemas
    ("skills", "write", "agent", True),
    ("skills", "update", "agent", True),
    ("profile", "write", "agent", True),
    ("profile", "update", "agent", True),
    ("projects", "write", "agent", True),
    ("projects", "update", "agent", True),
    ("knowledge", "write", "agent", True),
    ("knowledge", "update", "agent", True),
    ("system", "write", "agent", True),
    ("system", "write", "system", True),
    # Agent CANNOT write to manual or security
    ("manual", "write", "agent", False),
    ("manual", "update", "agent", False),
    ("manual", "delete", "agent", False),
    ("security", "write", "agent", False),
    ("security", "update", "agent", False),
    ("security", "delete", "agent", False),
    # User can write to everything
    ("manual", "write", "user", True),
    ("manual", "update", "user", True),
    ("manual", "delete", "user", True),
    ("security", "write", "user", True),
    ("security", "update", "user", True),
    # System can write to system
    ("system", "write", "system", True),
    ("system", "update", "system", True),
]


class SecurityService:
    """Service for managing secret references and enforcing access policies."""

    def __init__(self, session_factory: sessionmaker):
        self._session_factory = session_factory

    # ------------------------------------------------------------------
    # Initialize default policies
    # ------------------------------------------------------------------

    def initialize_default_policies(self) -> int:
        """Seed default access policies if not already present."""
        created = 0
        with self._session_factory() as session:
            for schema_name, action, actor, allowed in _DEFAULT_POLICIES:
                existing = session.execute(
                    select(AccessPolicyORM).where(
                        AccessPolicyORM.schema_name == schema_name,
                        AccessPolicyORM.action == action,
                        AccessPolicyORM.actor == actor,
                    )
                ).scalars().first()

                if not existing:
                    policy = AccessPolicyORM(
                        id=_new_uuid(),
                        schema_name=schema_name,
                        action=action,
                        actor=actor,
                        allowed=allowed,
                    )
                    session.add(policy)
                    created += 1

            session.commit()

        if created:
            logger.info("Created %d default access policies", created)
        return created

    # ------------------------------------------------------------------
    # Policy checking
    # ------------------------------------------------------------------

    def check_policy(self, *, schema_name: str, action: str, actor: str) -> bool:
        """
        Check if an actor is allowed to perform an action on a schema.
        Returns False if no matching policy found (deny by default).
        """
        with self._session_factory() as session:
            policy = session.execute(
                select(AccessPolicyORM).where(
                    AccessPolicyORM.schema_name == schema_name,
                    AccessPolicyORM.action == action,
                    AccessPolicyORM.actor == actor,
                )
            ).scalars().first()

            if policy is None:
                logger.warning(
                    "No policy found for %s.%s by %s — denying by default",
                    schema_name, action, actor,
                )
                return False

            return policy.allowed

    # ------------------------------------------------------------------
    # Secret references (NEVER plaintext)
    # ------------------------------------------------------------------

    def store_secret_ref(
        self,
        *,
        secret_name: str,
        secret_type: str,
        storage_backend: str = "env_var",
        encrypted_payload_ref: Optional[str] = None,
    ) -> dict:
        """
        Store a reference to a secret. The actual secret is stored elsewhere
        (env var, vault, encrypted file). We only store metadata + a reference.
        """
        with self._session_factory() as session:
            existing = session.execute(
                select(SecretRefORM).where(SecretRefORM.secret_name == secret_name)
            ).scalars().first()

            if existing:
                existing.secret_type = secret_type
                existing.storage_backend = storage_backend
                existing.encrypted_payload_ref = encrypted_payload_ref
                existing.updated_at = _utcnow()
                session.add(existing)
                session.commit()
                return {"action": "updated", "secret_id": existing.id}

            ref = SecretRefORM(
                id=_new_uuid(),
                secret_name=secret_name,
                secret_type=secret_type,
                storage_backend=storage_backend,
                encrypted_payload_ref=encrypted_payload_ref,
                rotation_status="current",
            )
            session.add(ref)
            session.commit()
            return {"action": "created", "secret_id": ref.id}

    def get_secret_ref(self, secret_name: str) -> Optional[dict]:
        """Get a secret reference by name (NOT the actual secret)."""
        with self._session_factory() as session:
            ref = session.execute(
                select(SecretRefORM).where(SecretRefORM.secret_name == secret_name)
            ).scalars().first()

            if not ref:
                return None

            return {
                "id": ref.id,
                "secret_name": ref.secret_name,
                "secret_type": ref.secret_type,
                "storage_backend": ref.storage_backend,
                "encrypted_payload_ref": ref.encrypted_payload_ref,
                "rotation_status": ref.rotation_status,
            }

    def list_secret_refs(self) -> list[dict]:
        """List all secret references (metadata only, no payloads)."""
        with self._session_factory() as session:
            rows = session.execute(
                select(SecretRefORM).order_by(SecretRefORM.secret_name)
            ).scalars().all()

            return [
                {
                    "id": r.id,
                    "secret_name": r.secret_name,
                    "secret_type": r.secret_type,
                    "storage_backend": r.storage_backend,
                    "rotation_status": r.rotation_status,
                }
                for r in rows
            ]

    def mark_secret_expired(self, secret_name: str) -> bool:
        """Mark a secret as expired (needs rotation)."""
        with self._session_factory() as session:
            ref = session.execute(
                select(SecretRefORM).where(SecretRefORM.secret_name == secret_name)
            ).scalars().first()
            if not ref:
                return False
            ref.rotation_status = "expired"
            ref.updated_at = _utcnow()
            session.add(ref)
            session.commit()
            return True

"""
SQLAlchemy ORM models for Multi_Agent database.

The live database now uses the unified schema set:
    system, profile, knowledge, security
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


# ===========================================================================
# SYSTEM SCHEMA
# ===========================================================================

class IngestionJobORM(Base):
    __tablename__ = "ingestion_jobs"
    __table_args__ = {"schema": "system"}

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    job_type: Mapped[str] = mapped_column(String(40), nullable=False)
    target_schema: Mapped[str] = mapped_column(String(40), nullable=False)
    target_ref: Mapped[Optional[str]] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending", index=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)


class AuditLogORM(Base):
    __tablename__ = "audit_logs"
    __table_args__ = {"schema": "system"}

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    schema_name: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    table_name: Mapped[str] = mapped_column(String(80), nullable=False)
    record_id: Mapped[str] = mapped_column(String(64), nullable=False)
    action: Mapped[str] = mapped_column(String(20), nullable=False)
    actor: Mapped[str] = mapped_column(String(40), nullable=False)
    details_json: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)


class ConversationSessionORM(Base):
    __tablename__ = "conversation_sessions"
    __table_args__ = {"schema": "system"}

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    chat_id: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    message_index: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)


# ===========================================================================
# PROFILE SCHEMA
# ===========================================================================

class ProfileSourceORM(Base):
    __tablename__ = "profile_sources"
    __table_args__ = {"schema": "profile"}

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    source_type: Mapped[str] = mapped_column(String(40), nullable=False)
    source_hash: Mapped[Optional[str]] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class ProfileFactORM(Base):
    __tablename__ = "profile_facts"
    __table_args__ = {"schema": "profile"}

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    user_id: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    fact_key: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    fact_value: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    is_sensitive: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active", index=True)
    provenance_type: Mapped[str] = mapped_column(String(40), nullable=False)
    provenance_id: Mapped[Optional[str]] = mapped_column(String(64))
    last_verified_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    embeddings: Mapped[list["ProfileEmbeddingORM"]] = relationship(back_populates="fact", cascade="all, delete-orphan")


class ProfileVersionORM(Base):
    __tablename__ = "profile_versions"
    __table_args__ = {"schema": "profile"}

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    profile_source_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("profile.profile_sources.id", ondelete="SET NULL")
    )
    user_id: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    version_no: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    canonical_summary: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_by: Mapped[str] = mapped_column(String(40), nullable=False, default="system")


class ProfileEmbeddingORM(Base):
    __tablename__ = "profile_embeddings"
    __table_args__ = {"schema": "profile"}

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    fact_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("profile.profile_facts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    embedding_json: Mapped[str] = mapped_column(Text, default="[]")
    model_name: Mapped[str] = mapped_column(String(80), nullable=False, default="bge-m3")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    fact: Mapped["ProfileFactORM"] = relationship(back_populates="embeddings")


# ===========================================================================
# SECURITY SCHEMA
# ===========================================================================

class SecretRefORM(Base):
    __tablename__ = "secret_refs"
    __table_args__ = {"schema": "security"}

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    secret_name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    secret_type: Mapped[str] = mapped_column(String(40), nullable=False)
    storage_backend: Mapped[str] = mapped_column(String(40), nullable=False, default="env_var")
    encrypted_payload_ref: Mapped[Optional[str]] = mapped_column(String(512))
    rotation_status: Mapped[str] = mapped_column(String(20), default="current")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


class AccessPolicyORM(Base):
    __tablename__ = "access_policies"
    __table_args__ = {"schema": "security"}

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    schema_name: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(20), nullable=False)
    actor: Mapped[str] = mapped_column(String(40), nullable=False)
    allowed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    conditions_json: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

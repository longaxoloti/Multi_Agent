"""
SQLAlchemy ORM models for Multi_Agent database.

7 PostgreSQL schemas:
  system   — ingestion jobs, audit logs, conversation sessions
  skills   — tool usage prompts with versioning + chunked embeddings
  profile  — user profile facts with provenance
  projects — project knowledge with verification
  knowledge — URL registry and bookmarks
  manual   — user-saved data (only via /save command)
  security — secret references + access policies
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
# SKILLS SCHEMA
# ===========================================================================

class SkillSourceORM(Base):
    __tablename__ = "skill_sources"
    __table_args__ = {"schema": "skills"}

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    source_type: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    source_path: Mapped[Optional[str]] = mapped_column(String(512))
    source_hash: Mapped[Optional[str]] = mapped_column(String(128))
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    raw_content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    versions: Mapped[list["SkillVersionORM"]] = relationship(back_populates="source", cascade="all, delete-orphan")


class SkillVersionORM(Base):
    __tablename__ = "skill_versions"
    __table_args__ = (
        UniqueConstraint("skill_source_id", "version_no", name="uq_skill_version"),
        {"schema": "skills"},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    skill_source_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("skills.skill_sources.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version_no: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    canonical_content: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active", index=True)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    updated_by: Mapped[str] = mapped_column(String(40), nullable=False, default="system")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    source: Mapped["SkillSourceORM"] = relationship(back_populates="versions")
    chunks: Mapped[list["SkillChunkORM"]] = relationship(back_populates="version", cascade="all, delete-orphan")
    tags: Mapped[list["SkillTagORM"]] = relationship(back_populates="version", cascade="all, delete-orphan")


class SkillChunkORM(Base):
    __tablename__ = "skill_chunks"
    __table_args__ = (
        UniqueConstraint("skill_version_id", "chunk_index", name="uq_skill_chunk"),
        {"schema": "skills"},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    skill_version_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("skills.skill_versions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0)

    version: Mapped["SkillVersionORM"] = relationship(back_populates="chunks")
    embeddings: Mapped[list["SkillEmbeddingORM"]] = relationship(back_populates="chunk", cascade="all, delete-orphan")


class SkillEmbeddingORM(Base):
    __tablename__ = "skill_embeddings"
    __table_args__ = {"schema": "skills"}

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    chunk_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("skills.skill_chunks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # NOTE: The `embedding` vector column is created via raw SQL in initialize(),
    # because SQLAlchemy doesn't natively support pgvector types.
    embedding_json: Mapped[str] = mapped_column(Text, default="[]")
    model_name: Mapped[str] = mapped_column(String(80), nullable=False, default="bge-m3")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    chunk: Mapped["SkillChunkORM"] = relationship(back_populates="embeddings")


class SkillTagORM(Base):
    __tablename__ = "skill_tags"
    __table_args__ = (
        UniqueConstraint("skill_version_id", "tag", name="uq_skill_tag"),
        {"schema": "skills"},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    skill_version_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("skills.skill_versions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    tag: Mapped[str] = mapped_column(String(80), nullable=False)

    version: Mapped["SkillVersionORM"] = relationship(back_populates="tags")


# ===========================================================================
# PROFILE SCHEMA
# ===========================================================================

class ProfileSourceORM(Base):
    __tablename__ = "profile_sources"
    __table_args__ = {"schema": "profile"}

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    source_type: Mapped[str] = mapped_column(String(40), nullable=False)
    source_hash: Mapped[Optional[str]] = mapped_column(String(128))
    raw_content: Mapped[str] = mapped_column(Text, nullable=False)
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
# PROJECTS SCHEMA
# ===========================================================================

class ProjectSourceORM(Base):
    __tablename__ = "project_sources"
    __table_args__ = {"schema": "projects"}

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    source_type: Mapped[str] = mapped_column(String(40), nullable=False)
    source_path: Mapped[Optional[str]] = mapped_column(String(512))
    source_hash: Mapped[Optional[str]] = mapped_column(String(128))
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class ProjectEntityORM(Base):
    __tablename__ = "project_entities"
    __table_args__ = {"schema": "projects"}

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    project_name: Mapped[str] = mapped_column(String(255), nullable=False)
    repo_path: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    language: Mapped[Optional[str]] = mapped_column(String(80))
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    facts: Mapped[list["ProjectFactORM"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    snapshots: Mapped[list["ProjectSnapshotORM"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    verifications: Mapped[list["ProjectVerificationORM"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )


class ProjectFactORM(Base):
    __tablename__ = "project_facts"
    __table_args__ = {"schema": "projects"}

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    project_entity_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("projects.project_entities.id", ondelete="CASCADE"), nullable=False, index=True
    )
    fact_key: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    fact_value: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="active", index=True)
    provenance_type: Mapped[str] = mapped_column(String(40), nullable=False)
    provenance_id: Mapped[Optional[str]] = mapped_column(String(64))
    verified_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    project: Mapped["ProjectEntityORM"] = relationship(back_populates="facts")


class ProjectSnapshotORM(Base):
    __tablename__ = "project_snapshots"
    __table_args__ = {"schema": "projects"}

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    project_entity_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("projects.project_entities.id", ondelete="CASCADE"), nullable=False, index=True
    )
    snapshot_hash: Mapped[Optional[str]] = mapped_column(String(128))
    snapshot_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    summary: Mapped[Optional[str]] = mapped_column(Text)
    diff_from_previous: Mapped[Optional[str]] = mapped_column(Text)

    project: Mapped["ProjectEntityORM"] = relationship(back_populates="snapshots")


class ProjectVerificationORM(Base):
    __tablename__ = "project_verifications"
    __table_args__ = {"schema": "projects"}

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    project_entity_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("projects.project_entities.id", ondelete="CASCADE"), nullable=False, index=True
    )
    verification_run_id: Mapped[Optional[str]] = mapped_column(String(120))
    result: Mapped[str] = mapped_column(String(20), nullable=False)
    details: Mapped[Optional[str]] = mapped_column(Text)
    verified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    project: Mapped["ProjectEntityORM"] = relationship(back_populates="verifications")


# ===========================================================================
# KNOWLEDGE SCHEMA
# ===========================================================================

class UrlRegistryORM(Base):
    __tablename__ = "url_registry"
    __table_args__ = {"schema": "knowledge"}

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    url: Mapped[str] = mapped_column(String(2048), nullable=False, unique=True)
    domain: Mapped[Optional[str]] = mapped_column(String(255), index=True)
    title: Mapped[Optional[str]] = mapped_column(String(255))
    description: Mapped[Optional[str]] = mapped_column(Text)
    source_type: Mapped[str] = mapped_column(String(40), nullable=False, default="research", index=True)
    trust_score: Mapped[float] = mapped_column(Float, default=0.5, index=True)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    stats: Mapped[Optional["UrlStatsORM"]] = relationship(back_populates="url_entry", uselist=False, cascade="all, delete-orphan")
    embeddings: Mapped[list["UrlEmbeddingORM"]] = relationship(back_populates="url_entry", cascade="all, delete-orphan")


class UrlStatsORM(Base):
    __tablename__ = "url_stats"
    __table_args__ = {"schema": "knowledge"}

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    url_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("knowledge.url_registry.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    access_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_accessed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_used_for_task_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    url_entry: Mapped["UrlRegistryORM"] = relationship(back_populates="stats")


class UrlEmbeddingORM(Base):
    __tablename__ = "url_embeddings"
    __table_args__ = {"schema": "knowledge"}

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    url_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("knowledge.url_registry.id", ondelete="CASCADE"), nullable=False
    )
    content_summary: Mapped[Optional[str]] = mapped_column(Text)
    embedding_json: Mapped[str] = mapped_column(Text, default="[]")
    model_name: Mapped[str] = mapped_column(String(80), nullable=False, default="bge-m3")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    url_entry: Mapped["UrlRegistryORM"] = relationship(back_populates="embeddings")


# ===========================================================================
# MANUAL SCHEMA (/save command only)
# ===========================================================================

class SavedSourceORM(Base):
    __tablename__ = "saved_sources"
    __table_args__ = {"schema": "manual"}

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    command_text: Mapped[str] = mapped_column(Text, nullable=False)
    request_id: Mapped[Optional[str]] = mapped_column(String(120))
    user_id: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    chat_id: Mapped[Optional[str]] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    facts: Mapped[list["SavedFactORM"]] = relationship(back_populates="source", cascade="all, delete-orphan")


class SavedFactORM(Base):
    __tablename__ = "saved_facts"
    __table_args__ = {"schema": "manual"}

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    saved_source_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("manual.saved_sources.id", ondelete="CASCADE"), nullable=False, index=True
    )
    category: Mapped[str] = mapped_column(String(40), nullable=False, default="note", index=True)
    title: Mapped[Optional[str]] = mapped_column(String(255))
    fact_key: Mapped[Optional[str]] = mapped_column(String(255))
    fact_value: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active", index=True)
    provenance_type: Mapped[str] = mapped_column(String(40), nullable=False, default="user_direct_save")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    source: Mapped["SavedSourceORM"] = relationship(back_populates="facts")
    versions: Mapped[list["SavedVersionORM"]] = relationship(back_populates="fact", cascade="all, delete-orphan")
    embeddings: Mapped[list["SavedEmbeddingORM"]] = relationship(back_populates="fact", cascade="all, delete-orphan")


class SavedVersionORM(Base):
    __tablename__ = "saved_versions"
    __table_args__ = (
        UniqueConstraint("saved_fact_id", "version_no", name="uq_saved_version"),
        {"schema": "manual"},
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    saved_fact_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("manual.saved_facts.id", ondelete="CASCADE"), nullable=False
    )
    version_no: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    changed_by: Mapped[str] = mapped_column(String(40), nullable=False)
    change_reason: Mapped[Optional[str]] = mapped_column(Text)
    previous_value: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    fact: Mapped["SavedFactORM"] = relationship(back_populates="versions")


class SavedEmbeddingORM(Base):
    __tablename__ = "saved_embeddings"
    __table_args__ = {"schema": "manual"}

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    saved_fact_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("manual.saved_facts.id", ondelete="CASCADE"), nullable=False
    )
    embedding_json: Mapped[str] = mapped_column(Text, default="[]")
    model_name: Mapped[str] = mapped_column(String(80), nullable=False, default="bge-m3")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    fact: Mapped["SavedFactORM"] = relationship(back_populates="embeddings")


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

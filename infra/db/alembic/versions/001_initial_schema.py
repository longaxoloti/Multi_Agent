"""initial schema

Revision ID: 001_initial
Revises: 
Create Date: 2024-04-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '001_initial'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.execute('''
-- Extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Schemas
CREATE SCHEMA IF NOT EXISTS system;
CREATE SCHEMA IF NOT EXISTS skills;
CREATE SCHEMA IF NOT EXISTS profile;
CREATE SCHEMA IF NOT EXISTS projects;
CREATE SCHEMA IF NOT EXISTS knowledge;
CREATE SCHEMA IF NOT EXISTS manual;
CREATE SCHEMA IF NOT EXISTS security;

-- SYSTEM SCHEMA
CREATE TABLE IF NOT EXISTS system.ingestion_jobs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_type        VARCHAR(40)  NOT NULL,
    target_schema   VARCHAR(40)  NOT NULL,
    target_ref      VARCHAR(512),
    status          VARCHAR(20)  NOT NULL DEFAULT 'pending',
    error_message   TEXT,
    started_at      TIMESTAMPTZ,
    finished_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status ON system.ingestion_jobs (status);
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_created ON system.ingestion_jobs (created_at);

CREATE TABLE IF NOT EXISTS system.audit_logs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    schema_name     VARCHAR(40)  NOT NULL,
    table_name      VARCHAR(80)  NOT NULL,
    record_id       VARCHAR(64)  NOT NULL,
    action          VARCHAR(20)  NOT NULL,
    actor           VARCHAR(40)  NOT NULL,
    details_json    TEXT,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_audit_logs_schema ON system.audit_logs (schema_name, table_name);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created ON system.audit_logs (created_at);

CREATE TABLE IF NOT EXISTS system.conversation_sessions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chat_id         VARCHAR(120) NOT NULL,
    session_id      VARCHAR(120) NOT NULL,
    role            VARCHAR(20)  NOT NULL,
    content         TEXT         NOT NULL,
    message_index   INTEGER      NOT NULL,
    metadata_json   TEXT         DEFAULT '{}',
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_convsess_chat ON system.conversation_sessions (chat_id);
CREATE INDEX IF NOT EXISTS idx_convsess_session ON system.conversation_sessions (session_id);
CREATE INDEX IF NOT EXISTS idx_convsess_created ON system.conversation_sessions (created_at);

-- SKILLS SCHEMA
CREATE TABLE IF NOT EXISTS skills.skill_sources (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_type     VARCHAR(40)  NOT NULL,
    source_path     VARCHAR(512),
    source_hash     VARCHAR(128),
    title           VARCHAR(255) NOT NULL,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_skill_sources_type ON skills.skill_sources (source_type);

CREATE TABLE IF NOT EXISTS skills.skill_versions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    skill_source_id UUID         NOT NULL REFERENCES skills.skill_sources(id) ON DELETE CASCADE,
    version_no      INTEGER      NOT NULL DEFAULT 1,
    summary         TEXT,
    status          VARCHAR(20)  NOT NULL DEFAULT 'active',
    confidence      FLOAT        DEFAULT 1.0,
    updated_by      VARCHAR(40)  NOT NULL DEFAULT 'system',
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    UNIQUE (skill_source_id, version_no)
);
CREATE INDEX IF NOT EXISTS idx_skill_versions_source ON skills.skill_versions (skill_source_id);
CREATE INDEX IF NOT EXISTS idx_skill_versions_status ON skills.skill_versions (status);

CREATE TABLE IF NOT EXISTS skills.skill_chunks (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    skill_version_id UUID        NOT NULL REFERENCES skills.skill_versions(id) ON DELETE CASCADE,
    chunk_index     INTEGER      NOT NULL,
    chunk_text      TEXT         NOT NULL,
    token_count     INTEGER      DEFAULT 0,
    UNIQUE (skill_version_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_skill_chunks_version ON skills.skill_chunks (skill_version_id);

CREATE TABLE IF NOT EXISTS skills.skill_embeddings (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chunk_id        UUID         NOT NULL REFERENCES skills.skill_chunks(id) ON DELETE CASCADE,
    embedding       vector(1024),
    model_name      VARCHAR(80)  NOT NULL DEFAULT 'bge-m3',
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_skill_embeddings_chunk ON skills.skill_embeddings (chunk_id);
CREATE INDEX IF NOT EXISTS idx_skill_embeddings_hnsw
    ON skills.skill_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS skills.skill_tags (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    skill_version_id UUID        NOT NULL REFERENCES skills.skill_versions(id) ON DELETE CASCADE,
    tag             VARCHAR(80)  NOT NULL,
    UNIQUE (skill_version_id, tag)
);
CREATE INDEX IF NOT EXISTS idx_skill_tags_version ON skills.skill_tags (skill_version_id);

-- PROFILE SCHEMA
CREATE TABLE IF NOT EXISTS profile.profile_sources (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_type     VARCHAR(40)  NOT NULL,
    source_hash     VARCHAR(128),
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS profile.profile_facts (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         VARCHAR(120) NOT NULL,
    fact_key        VARCHAR(255) NOT NULL,
    fact_value      TEXT         NOT NULL,
    confidence      FLOAT        DEFAULT 1.0,
    is_sensitive    BOOLEAN      DEFAULT false,
    status          VARCHAR(20)  NOT NULL DEFAULT 'active',
    provenance_type VARCHAR(40)  NOT NULL,
    provenance_id   UUID,
    last_verified_at TIMESTAMPTZ,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_profile_facts_user ON profile.profile_facts (user_id);
CREATE INDEX IF NOT EXISTS idx_profile_facts_key ON profile.profile_facts (fact_key);
CREATE INDEX IF NOT EXISTS idx_profile_facts_status ON profile.profile_facts (status);

CREATE TABLE IF NOT EXISTS profile.profile_versions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    profile_source_id UUID      REFERENCES profile.profile_sources(id) ON DELETE SET NULL,
    user_id         VARCHAR(120) NOT NULL,
    version_no      INTEGER      NOT NULL DEFAULT 1,
    canonical_summary TEXT,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_by      VARCHAR(40)  NOT NULL DEFAULT 'system'
);
CREATE INDEX IF NOT EXISTS idx_profile_versions_user ON profile.profile_versions (user_id);

CREATE TABLE IF NOT EXISTS profile.profile_embeddings (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    fact_id         UUID         NOT NULL REFERENCES profile.profile_facts(id) ON DELETE CASCADE,
    embedding       vector(1024),
    model_name      VARCHAR(80)  NOT NULL DEFAULT 'bge-m3',
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_profile_embeddings_fact ON profile.profile_embeddings (fact_id);
CREATE INDEX IF NOT EXISTS idx_profile_embeddings_hnsw
    ON profile.profile_embeddings USING hnsw (embedding vector_cosine_ops);

-- PROJECTS SCHEMA
CREATE TABLE IF NOT EXISTS projects.project_sources (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_type     VARCHAR(40)  NOT NULL,
    source_path     VARCHAR(512),
    source_hash     VARCHAR(128),
    captured_at     TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS projects.project_entities (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_name    VARCHAR(255) NOT NULL,
    repo_path       VARCHAR(512) NOT NULL UNIQUE,
    language        VARCHAR(80),
    status          VARCHAR(20)  NOT NULL DEFAULT 'active',
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_project_entities_status ON projects.project_entities (status);

CREATE TABLE IF NOT EXISTS projects.project_facts (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_entity_id UUID      NOT NULL REFERENCES projects.project_entities(id) ON DELETE CASCADE,
    fact_key        VARCHAR(255) NOT NULL,
    fact_value      TEXT         NOT NULL,
    confidence      FLOAT        DEFAULT 1.0,
    status          VARCHAR(30)  NOT NULL DEFAULT 'active',
    provenance_type VARCHAR(40)  NOT NULL,
    provenance_id   UUID,
    verified_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_project_facts_entity ON projects.project_facts (project_entity_id);
CREATE INDEX IF NOT EXISTS idx_project_facts_status ON projects.project_facts (status);
CREATE INDEX IF NOT EXISTS idx_project_facts_key ON projects.project_facts (fact_key);

CREATE TABLE IF NOT EXISTS projects.project_snapshots (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_entity_id UUID      NOT NULL REFERENCES projects.project_entities(id) ON DELETE CASCADE,
    snapshot_hash   VARCHAR(128),
    snapshot_at     TIMESTAMPTZ  NOT NULL DEFAULT now(),
    summary         TEXT,
    diff_from_previous TEXT
);
CREATE INDEX IF NOT EXISTS idx_project_snapshots_entity ON projects.project_snapshots (project_entity_id);
CREATE INDEX IF NOT EXISTS idx_project_snapshots_time ON projects.project_snapshots (snapshot_at);

CREATE TABLE IF NOT EXISTS projects.project_verifications (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_entity_id UUID      NOT NULL REFERENCES projects.project_entities(id) ON DELETE CASCADE,
    verification_run_id VARCHAR(120),
    result          VARCHAR(20)  NOT NULL,
    details         TEXT,
    verified_at     TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_project_verifications_entity ON projects.project_verifications (project_entity_id);

-- KNOWLEDGE SCHEMA
CREATE TABLE IF NOT EXISTS knowledge.url_registry (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    url             VARCHAR(2048) NOT NULL UNIQUE,
    domain          VARCHAR(255),
    title           VARCHAR(255),
    description     TEXT,
    source_type     VARCHAR(40)  NOT NULL DEFAULT 'research',
    trust_score     FLOAT        DEFAULT 0.5,
    first_seen_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),
    last_seen_at    TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_url_registry_domain ON knowledge.url_registry (domain);
CREATE INDEX IF NOT EXISTS idx_url_registry_source ON knowledge.url_registry (source_type);
CREATE INDEX IF NOT EXISTS idx_url_registry_trust ON knowledge.url_registry (trust_score);

CREATE TABLE IF NOT EXISTS knowledge.url_stats (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    url_id          UUID         NOT NULL REFERENCES knowledge.url_registry(id) ON DELETE CASCADE UNIQUE,
    access_count    INTEGER      NOT NULL DEFAULT 0,
    last_accessed_at TIMESTAMPTZ,
    last_used_for_task_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS knowledge.url_embeddings (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    url_id          UUID         NOT NULL REFERENCES knowledge.url_registry(id) ON DELETE CASCADE,
    content_summary TEXT,
    embedding       vector(1024),
    model_name      VARCHAR(80)  NOT NULL DEFAULT 'bge-m3',
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_url_embeddings_hnsw
    ON knowledge.url_embeddings USING hnsw (embedding vector_cosine_ops);

-- MANUAL SCHEMA
CREATE TABLE IF NOT EXISTS manual.saved_sources (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    command_text    TEXT         NOT NULL,
    request_id      VARCHAR(120),
    user_id         VARCHAR(120) NOT NULL,
    chat_id         VARCHAR(120),
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_saved_sources_user ON manual.saved_sources (user_id);

CREATE TABLE IF NOT EXISTS manual.saved_facts (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    saved_source_id UUID         NOT NULL REFERENCES manual.saved_sources(id) ON DELETE CASCADE,
    category        VARCHAR(40)  NOT NULL DEFAULT 'note',
    title           VARCHAR(255),
    fact_key        VARCHAR(255),
    fact_value      TEXT         NOT NULL,
    status          VARCHAR(20)  NOT NULL DEFAULT 'active',
    provenance_type VARCHAR(40)  NOT NULL DEFAULT 'user_direct_save',
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_saved_facts_source ON manual.saved_facts (saved_source_id);
CREATE INDEX IF NOT EXISTS idx_saved_facts_category ON manual.saved_facts (category);
CREATE INDEX IF NOT EXISTS idx_saved_facts_status ON manual.saved_facts (status);

CREATE TABLE IF NOT EXISTS manual.saved_versions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    saved_fact_id   UUID         NOT NULL REFERENCES manual.saved_facts(id) ON DELETE CASCADE,
    version_no      INTEGER      NOT NULL DEFAULT 1,
    changed_by      VARCHAR(40)  NOT NULL,
    change_reason   TEXT,
    previous_value  TEXT,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    UNIQUE (saved_fact_id, version_no)
);

CREATE TABLE IF NOT EXISTS manual.saved_embeddings (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    saved_fact_id   UUID         NOT NULL REFERENCES manual.saved_facts(id) ON DELETE CASCADE,
    embedding       vector(1024),
    model_name      VARCHAR(80)  NOT NULL DEFAULT 'bge-m3',
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_saved_embeddings_hnsw
    ON manual.saved_embeddings USING hnsw (embedding vector_cosine_ops);

-- SECURITY SCHEMA
CREATE TABLE IF NOT EXISTS security.secret_refs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    secret_name     VARCHAR(255) NOT NULL UNIQUE,
    secret_type     VARCHAR(40)  NOT NULL,
    storage_backend VARCHAR(40)  NOT NULL DEFAULT 'env_var',
    encrypted_payload_ref VARCHAR(512),
    rotation_status VARCHAR(20)  DEFAULT 'current',
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS security.access_policies (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    schema_name     VARCHAR(40)  NOT NULL,
    action          VARCHAR(20)  NOT NULL,
    actor           VARCHAR(40)  NOT NULL,
    allowed         BOOLEAN      NOT NULL DEFAULT false,
    conditions_json TEXT,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_access_policies_schema ON security.access_policies (schema_name, action, actor);
    ''')

def downgrade() -> None:
    # Not implementing downgrade for initial bootstrap to prevent accidental data loss
    pass

"""initial unified schema

Revision ID: 001_initial
Revises:
Create Date: 2026-04-09 01:00:00.000000
"""

from alembic import op


# revision identifiers, used by Alembic.
revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
-- Extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Schemas (no skills/manual/projects app schema)
CREATE SCHEMA IF NOT EXISTS system;
CREATE SCHEMA IF NOT EXISTS profile;
CREATE SCHEMA IF NOT EXISTS knowledge;
CREATE SCHEMA IF NOT EXISTS security;

-- -------------------------------------------------------------------------
-- SYSTEM SCHEMA
-- -------------------------------------------------------------------------
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

-- -------------------------------------------------------------------------
-- PROFILE SCHEMA
-- -------------------------------------------------------------------------
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

-- -------------------------------------------------------------------------
-- KNOWLEDGE SCHEMA (unified project + knowledge + auto context)
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS knowledge.entities (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_type     VARCHAR(40)  NOT NULL,
    entity_ref      VARCHAR(2048) NOT NULL,
    owner_user_id   VARCHAR(120) NOT NULL,
    title           VARCHAR(255),
    description     TEXT,
    source_type     VARCHAR(40) NOT NULL DEFAULT 'auto',
    trust_score     FLOAT NOT NULL DEFAULT 0.5,
    status          VARCHAR(20) NOT NULL DEFAULT 'active',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (entity_type, entity_ref, owner_user_id)
);
CREATE INDEX IF NOT EXISTS idx_k_entities_owner ON knowledge.entities (owner_user_id);
CREATE INDEX IF NOT EXISTS idx_k_entities_status ON knowledge.entities (status);

CREATE TABLE IF NOT EXISTS knowledge.memories (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_id           UUID NOT NULL REFERENCES knowledge.entities(id) ON DELETE CASCADE,
    owner_user_id       VARCHAR(120) NOT NULL,
    memory_type         VARCHAR(40) NOT NULL DEFAULT 'fact',
    ingestion_mode      VARCHAR(20) NOT NULL DEFAULT 'auto',
    summary             VARCHAR(255),
    content             TEXT NOT NULL,
    confidence          FLOAT NOT NULL DEFAULT 0.7,
    decay_weight        FLOAT NOT NULL DEFAULT 1.0,
    status              VARCHAR(20) NOT NULL DEFAULT 'active',
    valid_from          TIMESTAMPTZ NOT NULL DEFAULT now(),
    valid_until         TIMESTAMPTZ,
    content_hash        VARCHAR(64) NOT NULL,
    canonical_hash      VARCHAR(64) NOT NULL,
    merged_into         UUID,
    superseded_by       UUID,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_accessed_at    TIMESTAMPTZ,
    access_count        INTEGER NOT NULL DEFAULT 0,
    metadata_json       TEXT NOT NULL DEFAULT '{}',
    search_tsv          tsvector
);
CREATE INDEX IF NOT EXISTS idx_k_memories_entity ON knowledge.memories (entity_id);
CREATE INDEX IF NOT EXISTS idx_k_memories_owner ON knowledge.memories (owner_user_id);
CREATE INDEX IF NOT EXISTS idx_k_memories_status ON knowledge.memories (status);
CREATE INDEX IF NOT EXISTS idx_k_memories_type ON knowledge.memories (memory_type);
CREATE INDEX IF NOT EXISTS idx_k_memories_content_hash ON knowledge.memories (content_hash);
CREATE INDEX IF NOT EXISTS idx_k_memories_canonical_hash ON knowledge.memories (canonical_hash);
CREATE INDEX IF NOT EXISTS idx_k_memories_tsv ON knowledge.memories USING GIN (search_tsv);

CREATE TABLE IF NOT EXISTS knowledge.memory_chunks (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    memory_id       UUID NOT NULL REFERENCES knowledge.memories(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    chunk_text      TEXT NOT NULL,
    token_count     INTEGER NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    search_tsv      tsvector,
    UNIQUE (memory_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_k_chunks_tsv ON knowledge.memory_chunks USING GIN (search_tsv);

CREATE TABLE IF NOT EXISTS knowledge.memory_embeddings (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    memory_id       UUID NOT NULL REFERENCES knowledge.memories(id) ON DELETE CASCADE,
    chunk_id        UUID REFERENCES knowledge.memory_chunks(id) ON DELETE CASCADE,
    embedding       vector(1024),
    model_name      VARCHAR(80) NOT NULL DEFAULT 'bge-m3',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_k_mem_embeddings_memory ON knowledge.memory_embeddings (memory_id);
CREATE INDEX IF NOT EXISTS idx_k_mem_embeddings_chunk ON knowledge.memory_embeddings (chunk_id);
CREATE INDEX IF NOT EXISTS idx_k_mem_embeddings_hnsw
ON knowledge.memory_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS knowledge.memory_edges (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_memory_id UUID NOT NULL REFERENCES knowledge.memories(id) ON DELETE CASCADE,
    target_memory_id UUID NOT NULL REFERENCES knowledge.memories(id) ON DELETE CASCADE,
    edge_type        VARCHAR(40) NOT NULL,
    weight           FLOAT NOT NULL DEFAULT 1.0,
    metadata_json    TEXT NOT NULL DEFAULT '{}',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source_memory_id, target_memory_id, edge_type)
);
CREATE INDEX IF NOT EXISTS idx_k_edges_source ON knowledge.memory_edges (source_memory_id, edge_type);
CREATE INDEX IF NOT EXISTS idx_k_edges_target ON knowledge.memory_edges (target_memory_id, edge_type);

CREATE TABLE IF NOT EXISTS knowledge.memory_versions (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    memory_id           UUID NOT NULL REFERENCES knowledge.memories(id) ON DELETE CASCADE,
    version_no          INTEGER NOT NULL,
    reason              VARCHAR(80) NOT NULL,
    previous_content    TEXT,
    new_content         TEXT,
    changed_by          VARCHAR(80) NOT NULL DEFAULT 'system',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (memory_id, version_no)
);

CREATE TABLE IF NOT EXISTS knowledge.memory_claims (
    id                   UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    memory_id            UUID NOT NULL REFERENCES knowledge.memories(id) ON DELETE CASCADE,
    claim_type           VARCHAR(20) NOT NULL,
    claim_text           TEXT NOT NULL,
    normalized_claim     VARCHAR(512),
    confidence           FLOAT NOT NULL DEFAULT 0.7,
    contradicts_claim_id UUID,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_k_claims_memory ON knowledge.memory_claims (memory_id, claim_type);
CREATE INDEX IF NOT EXISTS idx_k_claims_norm ON knowledge.memory_claims (normalized_claim);

CREATE TABLE IF NOT EXISTS knowledge.access_stats (
    id                    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    memory_id             UUID NOT NULL REFERENCES knowledge.memories(id) ON DELETE CASCADE UNIQUE,
    access_count          INTEGER NOT NULL DEFAULT 0,
    last_accessed_at      TIMESTAMPTZ,
    last_used_for_task_at TIMESTAMPTZ,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS knowledge.retrieval_events (
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    owner_user_id     VARCHAR(120) NOT NULL,
    query_text        TEXT NOT NULL,
    query_hash        VARCHAR(64) NOT NULL,
    memory_id         UUID REFERENCES knowledge.memories(id) ON DELETE SET NULL,
    vector_score      FLOAT,
    text_score        FLOAT,
    type_score        FLOAT,
    recency_score     FLOAT,
    confidence_score  FLOAT,
    final_score       FLOAT,
    rank_position     INTEGER,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_k_retrieval_owner_created
ON knowledge.retrieval_events (owner_user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_k_retrieval_query_hash ON knowledge.retrieval_events (query_hash);

CREATE OR REPLACE FUNCTION knowledge.update_memories_tsv()
RETURNS trigger AS $$
BEGIN
  NEW.search_tsv := to_tsvector('simple', COALESCE(NEW.summary, '') || ' ' || COALESCE(NEW.content, ''));
  RETURN NEW;
END
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION knowledge.update_chunks_tsv()
RETURNS trigger AS $$
BEGIN
  NEW.search_tsv := to_tsvector('simple', COALESCE(NEW.chunk_text, ''));
  RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_k_memories_tsv ON knowledge.memories;
CREATE TRIGGER trg_k_memories_tsv
BEFORE INSERT OR UPDATE OF summary, content ON knowledge.memories
FOR EACH ROW EXECUTE FUNCTION knowledge.update_memories_tsv();

DROP TRIGGER IF EXISTS trg_k_chunks_tsv ON knowledge.memory_chunks;
CREATE TRIGGER trg_k_chunks_tsv
BEFORE INSERT OR UPDATE OF chunk_text ON knowledge.memory_chunks
FOR EACH ROW EXECUTE FUNCTION knowledge.update_chunks_tsv();

-- -------------------------------------------------------------------------
-- SECURITY SCHEMA (user-scoped)
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS security.users (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_ref        VARCHAR(120) NOT NULL UNIQUE,
    display_name    VARCHAR(255),
    status          VARCHAR(20) NOT NULL DEFAULT 'active',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS security.roles (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    role_name       VARCHAR(80) NOT NULL UNIQUE,
    description     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS security.user_roles (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID NOT NULL REFERENCES security.users(id) ON DELETE CASCADE,
    role_id         UUID NOT NULL REFERENCES security.roles(id) ON DELETE CASCADE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, role_id)
);

CREATE TABLE IF NOT EXISTS security.resource_policies (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    resource_type   VARCHAR(40) NOT NULL,
    resource_ref    VARCHAR(255) NOT NULL,
    action          VARCHAR(20) NOT NULL,
    actor_type      VARCHAR(20) NOT NULL,
    actor_ref       VARCHAR(120),
    allowed         BOOLEAN NOT NULL DEFAULT false,
    conditions_json TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (resource_type, resource_ref, action, actor_type, actor_ref)
);
CREATE INDEX IF NOT EXISTS idx_sec_policy_lookup
ON security.resource_policies (resource_type, resource_ref, action, actor_type, actor_ref);

CREATE TABLE IF NOT EXISTS security.memory_acl (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    memory_id       UUID NOT NULL REFERENCES knowledge.memories(id) ON DELETE CASCADE,
    principal_type  VARCHAR(20) NOT NULL,
    principal_ref   VARCHAR(120) NOT NULL,
    can_read        BOOLEAN NOT NULL DEFAULT true,
    can_write       BOOLEAN NOT NULL DEFAULT false,
    can_delete      BOOLEAN NOT NULL DEFAULT false,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (memory_id, principal_type, principal_ref)
);

CREATE TABLE IF NOT EXISTS security.secret_refs (
    id                   UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    owner_user_ref       VARCHAR(120) NOT NULL DEFAULT 'system',
    secret_name          VARCHAR(255) NOT NULL,
    secret_type          VARCHAR(40)  NOT NULL,
    storage_backend      VARCHAR(40)  NOT NULL DEFAULT 'env_var',
    encrypted_payload_ref VARCHAR(512),
    rotation_status      VARCHAR(20)  DEFAULT 'current',
    created_at           TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at           TIMESTAMPTZ  NOT NULL DEFAULT now(),
    UNIQUE (owner_user_ref, secret_name)
);

CREATE TABLE IF NOT EXISTS security.audit_security_events (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    actor_ref       VARCHAR(120) NOT NULL,
    action          VARCHAR(40) NOT NULL,
    resource_type   VARCHAR(40) NOT NULL,
    resource_ref    VARCHAR(255),
    allowed         BOOLEAN NOT NULL,
    details_json    TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- -------------------------------------------------------------------------
-- RLS policies
-- -------------------------------------------------------------------------
ALTER TABLE knowledge.entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledge.memories ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledge.memory_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledge.memory_embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledge.memory_edges ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledge.memory_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledge.memory_claims ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledge.access_stats ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledge.retrieval_events ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS p_k_entities_select ON knowledge.entities;
CREATE POLICY p_k_entities_select ON knowledge.entities
FOR SELECT
USING (
  owner_user_id = 'system'
  OR owner_user_id = current_setting('app.current_user_id', true)
);

DROP POLICY IF EXISTS p_k_entities_modify ON knowledge.entities;
CREATE POLICY p_k_entities_modify ON knowledge.entities
FOR ALL
USING (owner_user_id = current_setting('app.current_user_id', true))
WITH CHECK (owner_user_id = current_setting('app.current_user_id', true));

DROP POLICY IF EXISTS p_k_memories_select ON knowledge.memories;
CREATE POLICY p_k_memories_select ON knowledge.memories
FOR SELECT
USING (
  owner_user_id = 'system'
  OR owner_user_id = current_setting('app.current_user_id', true)
);

DROP POLICY IF EXISTS p_k_memories_modify ON knowledge.memories;
CREATE POLICY p_k_memories_modify ON knowledge.memories
FOR ALL
USING (owner_user_id = current_setting('app.current_user_id', true))
WITH CHECK (owner_user_id = current_setting('app.current_user_id', true));

DROP POLICY IF EXISTS p_k_chunks_select ON knowledge.memory_chunks;
CREATE POLICY p_k_chunks_select ON knowledge.memory_chunks
FOR SELECT
USING (
  EXISTS (
    SELECT 1 FROM knowledge.memories m
    WHERE m.id = knowledge.memory_chunks.memory_id
      AND (m.owner_user_id = 'system' OR m.owner_user_id = current_setting('app.current_user_id', true))
  )
);

DROP POLICY IF EXISTS p_k_chunks_modify ON knowledge.memory_chunks;
CREATE POLICY p_k_chunks_modify ON knowledge.memory_chunks
FOR ALL
USING (
  EXISTS (
    SELECT 1 FROM knowledge.memories m
    WHERE m.id = knowledge.memory_chunks.memory_id
      AND m.owner_user_id = current_setting('app.current_user_id', true)
  )
)
WITH CHECK (
  EXISTS (
    SELECT 1 FROM knowledge.memories m
    WHERE m.id = knowledge.memory_chunks.memory_id
      AND m.owner_user_id = current_setting('app.current_user_id', true)
  )
);

DROP POLICY IF EXISTS p_k_embeddings_all ON knowledge.memory_embeddings;
CREATE POLICY p_k_embeddings_all ON knowledge.memory_embeddings
FOR ALL
USING (
  EXISTS (
    SELECT 1 FROM knowledge.memories m
    WHERE m.id = knowledge.memory_embeddings.memory_id
      AND (m.owner_user_id = 'system' OR m.owner_user_id = current_setting('app.current_user_id', true))
  )
)
WITH CHECK (
  EXISTS (
    SELECT 1 FROM knowledge.memories m
    WHERE m.id = knowledge.memory_embeddings.memory_id
      AND m.owner_user_id = current_setting('app.current_user_id', true)
  )
);

DROP POLICY IF EXISTS p_k_edges_all ON knowledge.memory_edges;
CREATE POLICY p_k_edges_all ON knowledge.memory_edges
FOR ALL
USING (
  EXISTS (
    SELECT 1 FROM knowledge.memories s
    WHERE s.id = knowledge.memory_edges.source_memory_id
      AND (s.owner_user_id = 'system' OR s.owner_user_id = current_setting('app.current_user_id', true))
  )
)
WITH CHECK (
  EXISTS (
    SELECT 1 FROM knowledge.memories s
    WHERE s.id = knowledge.memory_edges.source_memory_id
      AND s.owner_user_id = current_setting('app.current_user_id', true)
  )
);

DROP POLICY IF EXISTS p_k_versions_all ON knowledge.memory_versions;
CREATE POLICY p_k_versions_all ON knowledge.memory_versions
FOR ALL
USING (
  EXISTS (
    SELECT 1 FROM knowledge.memories m
    WHERE m.id = knowledge.memory_versions.memory_id
      AND (m.owner_user_id = 'system' OR m.owner_user_id = current_setting('app.current_user_id', true))
  )
)
WITH CHECK (
  EXISTS (
    SELECT 1 FROM knowledge.memories m
    WHERE m.id = knowledge.memory_versions.memory_id
      AND m.owner_user_id = current_setting('app.current_user_id', true)
  )
);

DROP POLICY IF EXISTS p_k_claims_all ON knowledge.memory_claims;
CREATE POLICY p_k_claims_all ON knowledge.memory_claims
FOR ALL
USING (
  EXISTS (
    SELECT 1 FROM knowledge.memories m
    WHERE m.id = knowledge.memory_claims.memory_id
      AND (m.owner_user_id = 'system' OR m.owner_user_id = current_setting('app.current_user_id', true))
  )
)
WITH CHECK (
  EXISTS (
    SELECT 1 FROM knowledge.memories m
    WHERE m.id = knowledge.memory_claims.memory_id
      AND m.owner_user_id = current_setting('app.current_user_id', true)
  )
);

DROP POLICY IF EXISTS p_k_access_stats_all ON knowledge.access_stats;
CREATE POLICY p_k_access_stats_all ON knowledge.access_stats
FOR ALL
USING (
  EXISTS (
    SELECT 1 FROM knowledge.memories m
    WHERE m.id = knowledge.access_stats.memory_id
      AND (m.owner_user_id = 'system' OR m.owner_user_id = current_setting('app.current_user_id', true))
  )
)
WITH CHECK (
  EXISTS (
    SELECT 1 FROM knowledge.memories m
    WHERE m.id = knowledge.access_stats.memory_id
      AND m.owner_user_id = current_setting('app.current_user_id', true)
  )
);

DROP POLICY IF EXISTS p_k_retrieval_events_all ON knowledge.retrieval_events;
CREATE POLICY p_k_retrieval_events_all ON knowledge.retrieval_events
FOR ALL
USING (owner_user_id = current_setting('app.current_user_id', true))
WITH CHECK (owner_user_id = current_setting('app.current_user_id', true));
        """
    )


def downgrade() -> None:
    op.execute(
        """
DROP SCHEMA IF EXISTS security CASCADE;
DROP SCHEMA IF EXISTS knowledge CASCADE;
DROP SCHEMA IF EXISTS profile CASCADE;
DROP SCHEMA IF EXISTS system CASCADE;
        """
    )

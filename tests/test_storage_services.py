"""
Tests for the new multi-schema storage services.

Uses SQLite in-memory for portability (skipping pgvector-specific features).
Tests versioning, provenance, policy gate, and CRUD operations.
"""

import os
import sys
import tempfile
import uuid

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.trusted_db import AgentDBRepository


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _fake_embedder(text: str, *, model: str, expected_dims: int) -> list[float]:
    """Deterministic fake embedder for testing."""
    seed = sum(ord(ch) for ch in (text or "")) % 97
    base = float(seed) / 100.0
    return [base for _ in range(expected_dims)]


def _make_pg_session_factory():
    """Create a PostgreSQL session factory with all schemas."""
    repo = AgentDBRepository()
    repo.initialize()
    return repo._session_factory, repo.engine


def _get_session_factory():
    """Use PostgreSQL only for strict multi-schema architecture tests."""
    try:
        sf, engine = _make_pg_session_factory()
    except Exception as exc:
        pytest.skip(f"PostgreSQL required for strict storage tests: {exc}")
    return sf, engine, True


# ---------------------------------------------------------------------------
# Skill Service Tests
# ---------------------------------------------------------------------------

def test_skill_ingest_from_markdown():
    """Test ingesting a skill from a markdown file."""
    from storage.skill_service import SkillService

    sf, engine, is_pg = _get_session_factory()
    service = SkillService(
        sf, embedder=_fake_embedder, is_pg=is_pg
    )

    # Create a temporary markdown file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("# Web Search Skill\n\nUse Google to search for information.\n\n## Steps\n\n1. Formulate query\n2. Execute search\n3. Parse results")
        f.flush()
        temp_path = f.name

    try:
        # First ingest
        result = service.ingest_from_markdown(temp_path, tags=["research", "web"])
        assert result["action"] == "created"
        assert result["version_no"] == 1
        assert result["chunks_created"] >= 1
        source_id = result["source_id"]

        # Re-ingest same content — should be no_change
        result2 = service.ingest_from_markdown(temp_path)
        assert result2["action"] == "no_change"
        assert result2["source_id"] == source_id

        # Modify file and re-ingest — should create new version
        with open(temp_path, "a") as f:
            f.write("\n\n## Advanced\n\nUse advanced search operators for better results.")

        result3 = service.ingest_from_markdown(temp_path)
        assert result3["action"] == "updated"
        assert result3["version_no"] == 2
        assert result3["source_id"] == source_id

        # List active skills
        active = service.get_active_skills()
        assert len(active) >= 1
        assert any(s["source_id"] == source_id for s in active)
        # Only version 2 should be active
        active_versions = [s for s in active if s["source_id"] == source_id]
        assert all(s["version_no"] == 2 for s in active_versions)

    finally:
        os.unlink(temp_path)


def test_skill_optimize():
    """Test agent self-optimization creating new version."""
    from storage.skill_service import SkillService

    sf, engine, is_pg = _get_session_factory()
    service = SkillService(sf, embedder=_fake_embedder, is_pg=is_pg)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("# Code Review\n\nReview code for bugs and style issues.")
        f.flush()
        temp_path = f.name

    try:
        result = service.ingest_from_markdown(temp_path)
        source_id = result["source_id"]

        # Agent optimizes
        opt_result = service.optimize_skill(
            source_id,
            "# Code Review v2\n\nReview code with focus on: bugs, performance, and security.",
            summary="Optimized for security awareness",
        )
        assert opt_result["version_no"] == 2
        assert opt_result["chunks_created"] >= 1

        # Old version should be deprecated, new should be active
        content = service.get_skill_content(source_id)
        assert content is not None
        assert content["version_no"] == 2
        assert "security" in content["content"].lower()

        # Can still access version 1
        v1 = service.get_skill_content(source_id, version=1)
        assert v1 is not None
        assert v1["version_no"] == 1

    finally:
        os.unlink(temp_path)


# ---------------------------------------------------------------------------
# User Profile Tests
# ---------------------------------------------------------------------------

def test_profile_ingest_from_markdown():
    """Test ingesting user profile from USER.md."""
    from storage.user_profile_service import UserProfileService

    sf, engine, is_pg = _get_session_factory()
    service = UserProfileService(sf, embedder=_fake_embedder, is_pg=is_pg)

    # Use unique content each run to avoid hash collision
    run_id = uuid.uuid4().hex[:8]
    user_id = f"test_user_{run_id}"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(
            f"# USER.md (run {run_id})\n\n"
            "- **Name:** Worldwide Handsome\n"
            "- **Timezone:** Asia/Saigon\n"
            "- **Pronouns:** Sir\n"
        )
        f.flush()
        temp_path = f.name

    try:
        result = service.ingest_from_markdown(temp_path, user_id=user_id)
        assert result["action"] == "ingested"
        assert result["facts_created"] == 3

        # Get profile
        profile = service.get_profile(user_id)
        assert len(profile) == 3
        keys = {f["fact_key"] for f in profile}
        assert "name" in keys
        assert "timezone" in keys

        # Re-ingest same content — no change
        result2 = service.ingest_from_markdown(temp_path, user_id=user_id)
        assert result2["action"] == "no_change"

    finally:
        os.unlink(temp_path)


def test_profile_upsert_and_supersede():
    """Test fact upsert with provenance and supersede."""
    from storage.user_profile_service import UserProfileService

    sf, engine, is_pg = _get_session_factory()
    service = UserProfileService(sf, embedder=_fake_embedder, is_pg=is_pg)

    user_id = f"test_user_{uuid.uuid4().hex[:8]}"

    # Upsert a fact
    result = service.upsert_fact(
        user_id=user_id,
        fact_key="preferred_language",
        fact_value="Vietnamese",
        confidence=0.9,
        source="inferred",
    )
    assert result["fact_id"]
    fact_id = result["fact_id"]

    # Same key, same value — should not create duplicate
    result2 = service.upsert_fact(
        user_id=user_id,
        fact_key="preferred_language",
        fact_value="Vietnamese",
        confidence=0.95,
        source="user_stated",
    )
    assert result2["fact_id"] == fact_id  # same fact, just updated confidence

    # Same key, different value — should create new fact (coexist)
    result3 = service.upsert_fact(
        user_id=user_id,
        fact_key="preferred_language",
        fact_value="English (for code)",
        confidence=0.8,
        source="inferred",
    )
    assert result3["fact_id"] != fact_id

    # Both facts should be active
    profile = service.get_profile(user_id)
    lang_facts = [f for f in profile if f["fact_key"] == "preferred_language"]
    assert len(lang_facts) == 2

    # Supersede the old one
    assert service.supersede_fact(fact_id, user_id=user_id)
    profile2 = service.get_profile(user_id)
    lang_facts2 = [f for f in profile2 if f["fact_key"] == "preferred_language"]
    assert len(lang_facts2) == 1
    assert lang_facts2[0]["fact_value"] == "English (for code)"


# ---------------------------------------------------------------------------
# Project Service Tests
# ---------------------------------------------------------------------------

def test_project_register_and_facts():
    """Test project registration and fact management."""
    from storage.project_service import ProjectService

    sf, engine, is_pg = _get_session_factory()
    service = ProjectService(sf)

    run_id = uuid.uuid4().hex[:8]
    repo_path = f"/tmp/test_project_{run_id}"

    # Register project
    result = service.register_project(
        project_name=f"TestProject_{run_id}",
        repo_path=repo_path,
        language="Python",
    )
    assert result["action"] == "created"
    project_id = result["project_id"]

    # Register same path again — should return exists
    result2 = service.register_project(
        project_name=f"TestProject_{run_id}",
        repo_path=repo_path,
    )
    assert result2["action"] == "exists"
    assert result2["project_id"] == project_id

    # Save facts
    r1 = service.save_fact(
        project_id=project_id,
        fact_key="framework",
        fact_value="LangGraph",
    )
    assert r1["action"] == "created"

    r2 = service.save_fact(
        project_id=project_id,
        fact_key="database",
        fact_value="PostgreSQL + pgvector",
    )
    assert r2["action"] == "created"

    # Get facts
    facts = service.get_project_facts(project_id)
    assert len(facts) >= 2
    keys = {f["fact_key"] for f in facts}
    assert "framework" in keys
    assert "database" in keys

    # Save snapshot
    snap_id = service.save_snapshot(
        project_id=project_id,
        summary="Initial project state with 7 PostgreSQL schemas",
    )
    assert snap_id

    # Mark all facts stale
    stale_count = service.mark_facts_stale(project_id)
    assert stale_count >= 2

    # Get stale facts
    stale = service.get_stale_facts(days=0)
    assert len(stale) >= 2


def test_project_verification():
    """Test project verification recording."""
    from storage.project_service import ProjectService

    sf, engine, is_pg = _get_session_factory()
    service = ProjectService(sf)

    run_id = uuid.uuid4().hex[:8]
    result = service.register_project(
        project_name=f"VerifyProject_{run_id}",
        repo_path=f"/tmp/test_project_verify_{run_id}",
        language="Python",
    )
    project_id = result["project_id"]

    # Record verification
    ver_id = service.record_verification(
        project_id=project_id,
        result="match",
        details="All facts match current project state",
    )
    assert ver_id


# ---------------------------------------------------------------------------
# Security Service Tests
# ---------------------------------------------------------------------------

def test_security_policy_gate():
    """Test that policy gate enforces manual/security schema rules."""
    from storage.security_service import SecurityService

    sf, engine, is_pg = _get_session_factory()
    service = SecurityService(sf)

    # Initialize default policies (may already exist from previous runs)
    created = service.initialize_default_policies()
    # created >= 0 is fine (0 if already seeded)

    # Agent should be ALLOWED to write to skills
    assert service.check_policy(schema_name="skills", action="write", actor="agent") is True

    # Agent should be DENIED writing to manual
    assert service.check_policy(schema_name="manual", action="write", actor="agent") is False

    # Agent should be DENIED writing to security
    assert service.check_policy(schema_name="security", action="write", actor="agent") is False

    # User should be ALLOWED to write to manual
    assert service.check_policy(schema_name="manual", action="write", actor="user") is True


def test_security_secret_refs():
    """Test secret reference storage (no plaintext)."""
    from storage.security_service import SecurityService

    sf, engine, is_pg = _get_session_factory()
    service = SecurityService(sf)

    run_id = uuid.uuid4().hex[:8]
    secret_name = f"TEST_KEY_{run_id}"

    # Store a secret reference
    result = service.store_secret_ref(
        secret_name=secret_name,
        secret_type="api_key",
        storage_backend="env_var",
        encrypted_payload_ref="TEST_KEY_REF",
    )
    assert result["action"] == "created"

    # Retrieve it
    ref = service.get_secret_ref(secret_name)
    assert ref is not None
    assert ref["secret_type"] == "api_key"
    assert ref["storage_backend"] == "env_var"

    # List all refs
    refs = service.list_secret_refs()
    assert len(refs) >= 1

    # Mark as expired
    assert service.mark_secret_expired(secret_name)
    ref2 = service.get_secret_ref(secret_name)
    assert ref2["rotation_status"] == "expired"


# ---------------------------------------------------------------------------
# Backward Compatibility Tests
# ---------------------------------------------------------------------------

def test_backward_compat_trusted_db():
    """Test that TrustedDBRepository alias still works."""
    from storage.trusted_db import TrustedDBRepository, AgentDBRepository
    assert TrustedDBRepository is AgentDBRepository

    from storage import TrustedDBRepository as TDB
    assert TDB is AgentDBRepository


def test_backward_compat_knowledge_service():
    """Test that KnowledgeService works on PostgreSQL strict mode."""
    from storage.knowledge_service import KnowledgeService

    repo = AgentDBRepository()
    repo.initialize()
    service = KnowledgeService(
        trusted_repo=repo,
        db_enabled=True,
        db_required=True,
        embedder=_fake_embedder,
    )

    saved = service.save(
        chat_id="compat_test",
        content="Backward compatibility test data",
        category="note",
    )
    assert saved["record_id"]

    fetched = service.get(chat_id="compat_test", record_id=saved["record_id"])
    assert fetched is not None
    assert "Backward" in fetched["content"]


def test_knowledge_service_rejects_sqlite_strict_mode():
    """Strict mode must reject SQLite knowledge persistence/search paths."""
    from storage.knowledge_service import KnowledgeService

    sqlite_repo = AgentDBRepository(db_url="sqlite+pysqlite:///:memory:")
    service = KnowledgeService(
        trusted_repo=sqlite_repo,
        db_enabled=True,
        db_required=False,
        embedder=_fake_embedder,
    )

    with pytest.raises(RuntimeError, match=r"PostgreSQL \+ pgvector"):
        service.save(
            chat_id="sqlite_strict",
            content="This must fail in strict mode",
            category="note",
        )

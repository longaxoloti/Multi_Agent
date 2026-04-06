from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

from pipelines.reporting import build_daily_knowledge_report_text, build_daily_report_text
from storage.trusted_db import TrustedClaim, TrustedDBRepository, UserKnowledgeRecord


def test_trusted_db_upsert_merges_sources_and_confidence(tmp_path: Path):
    db_path = tmp_path / "trusted.db"
    repo = TrustedDBRepository(db_url=f"sqlite:///{db_path}")
    repo.initialize()

    claim_text = "New sanctions package approved by multiple governments."
    first_id, created = repo.upsert_trusted_claim(
        topic="trade",
        claim=claim_text,
        confidence=0.61,
        sources=["https://reuters.com/a"],
    )
    assert created is True

    second_id, created_again = repo.upsert_trusted_claim(
        topic="trade",
        claim=claim_text,
        confidence=0.84,
        sources=["https://apnews.com/b"],
    )
    assert created_again is False
    assert first_id == second_id

    rows = repo.list_last_24h()
    assert len(rows) == 1
    assert rows[0].confidence == pytest.approx(0.84)
    assert "https://reuters.com/a" in rows[0].sources
    assert "https://apnews.com/b" in rows[0].sources


def test_trusted_db_semantic_dedupe_merges_similar_claims(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "trusted_semantic.db"
    repo = TrustedDBRepository(db_url=f"sqlite:///{db_path}")
    repo.initialize()

    vectors = {
        "Claim A": [1.0, 0.0, 0.0],
        "Claim B": [0.95, 0.05, 0.0],
    }

    def fake_embed(text_value: str):
        return vectors.get(text_value, [0.0, 1.0, 0.0])

    monkeypatch.setattr(repo, "_embed_text", fake_embed)

    id_a, created_a = repo.upsert_trusted_claim(
        topic="world politics",
        claim="Claim A",
        confidence=0.7,
        sources=["https://bbc.com/a"],
    )
    id_b, created_b = repo.upsert_trusted_claim(
        topic="world politics",
        claim="Claim B",
        confidence=0.8,
        sources=["https://ft.com/b"],
    )

    assert created_a is True
    assert created_b is False
    assert id_a == id_b


def test_reporting_pipeline_formats_topics():
    now = datetime(2026, 2, 27, 7, 0, 0)
    claims = [
        TrustedClaim(
            topic="world politics",
            claim="Leaders held emergency summit regarding maritime dispute.",
            confidence=0.91,
            sources=["https://reuters.com/1"],
            first_seen_at=now,
            last_verified_at=now,
        ),
        TrustedClaim(
            topic="trade",
            claim="Regional bloc signed new tariff reduction agreement.",
            confidence=0.86,
            sources=["https://apnews.com/2"],
            first_seen_at=now,
            last_verified_at=now,
        ),
    ]

    text = build_daily_report_text(claims, generated_at=now)
    assert "Daily Trusted Intelligence Report" in text
    assert "## world politics" in text
    assert "## trade" in text
    assert "confidence: 0.91" in text


def test_reporting_pipeline_formats_knowledge_records():
    now = datetime(2026, 2, 27, 7, 0, 0)
    records = [
        UserKnowledgeRecord(
            id="k_001",
            chat_id="chat-1",
            category="fact",
            title="Fed update",
            content="FED giữ nguyên lãi suất sau cuộc họp FOMC.",
            tags=["macro", "fed"],
            metadata={},
            created_at=now,
            updated_at=now,
        )
    ]

    text = build_daily_knowledge_report_text(records, generated_at=now)
    assert "Daily Knowledge Report" in text
    assert "## fact" in text
    assert "Fed update" in text
    assert "chat-1" in text


def test_list_knowledge_records_between_filters(tmp_path: Path):
    db_path = tmp_path / "knowledge_between.db"
    repo = TrustedDBRepository(db_url=f"sqlite:///{db_path}")
    repo.initialize()

    id1 = repo.save_knowledge_record(
        chat_id="chat_a",
        category="fact",
        content="US CPI tăng 3.1%",
        title="CPI",
    )
    _ = repo.save_knowledge_record(
        chat_id="chat_b",
        category="note",
        content="Personal reminder",
        title="Note",
    )

    start = datetime(2020, 1, 1)
    end = datetime(2100, 1, 1)
    rows = repo.list_knowledge_records_between(
        start=start,
        end=end,
        chat_id="chat_a",
        categories=["fact"],
    )

    assert len(rows) == 1
    assert rows[0].id == id1
    assert rows[0].chat_id == "chat_a"
    assert rows[0].category == "fact"


def test_health_check_script_functions(tmp_path: Path):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "health_check.py"
    spec = importlib.util.spec_from_file_location("health_check_module", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    pid_file = tmp_path / "alive.pid"
    pid_file.write_text(str(os.getpid()))
    status = module.read_pid_status(pid_file)
    assert status["present"] is True
    assert status["alive"] is True
    assert status["pid"] == os.getpid()

    invalid_file = tmp_path / "invalid.pid"
    invalid_file.write_text("not-a-pid")
    invalid = module.read_pid_status(invalid_file)
    assert invalid["present"] is True
    assert invalid["alive"] is False

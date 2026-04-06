import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.knowledge_service import KnowledgeService, parse_knowledge_request
from storage.trusted_db import TrustedDBRepository


def _fake_embedder(text: str, *, model: str, expected_dims: int) -> list[float]:
    seed = sum(ord(ch) for ch in (text or "")) % 97
    base = float(seed) / 100.0
    return [base for _ in range(expected_dims)]


def test_parse_command_save_and_search():
    save_req = parse_knowledge_request("/save fact CPI tháng này tăng 3%")
    assert save_req is not None
    assert save_req.action == "save"
    assert save_req.category == "fact"
    assert "CPI" in save_req.content

    search_req = parse_knowledge_request("/search note CPI tháng này")
    assert search_req is not None
    assert search_req.action == "search"
    assert search_req.query == "CPI tháng này"


def test_parse_natural_language_aliases():
    save_req = parse_knowledge_request("lưu: FED giữ nguyên lãi suất")
    assert save_req is not None
    assert save_req.action == "save"

    get_req = parse_knowledge_request("lấy: k_123")
    assert get_req is not None
    assert get_req.action == "get"
    assert get_req.record_id == "k_123"


def test_knowledge_service_save_get_search_delete():
    repo = TrustedDBRepository(db_url="sqlite+pysqlite:///:memory:")
    service = KnowledgeService(
        trusted_repo=repo,
        db_enabled=True,
        db_required=True,
        embedder=_fake_embedder,
    )

    saved = service.save(
        chat_id="chat_test_knowledge",
        content="Nasdaq tăng mạnh sau báo cáo CPI.",
        category="fact",
        title="Market note",
    )
    assert saved["record_id"]
    assert saved["stored_in_vector"] is True
    assert saved["stored_in_db"] is True

    fetched = service.get(chat_id="chat_test_knowledge", record_id=saved["record_id"])
    assert fetched is not None
    assert "Nasdaq" in fetched["content"]
    assert fetched["embedding_model"] == "bge-m3"
    assert fetched["embedding_dims"] == 1024
    assert len(fetched["embedding"]) == 1024

    searched = service.search(
        chat_id="chat_test_knowledge",
        query="CPI Nasdaq",
        category="fact",
        limit=3,
    )
    assert len(searched) >= 1
    assert any("Nasdaq" in item["content"] for item in searched)
    assert all(item.get("embedding_model") == "bge-m3" for item in searched)
    assert all(item.get("embedding_dims") == 1024 for item in searched)

    recent = service.list_recent(chat_id="chat_test_knowledge", limit=5)
    assert len(recent) >= 1

    deleted = service.delete(chat_id="chat_test_knowledge", record_id=saved["record_id"])
    assert deleted is True

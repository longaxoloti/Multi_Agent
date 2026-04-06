"""
Tests for the memory system and tools.
"""

import sys
import os
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMemoryManager:
    """Test the memory manager (short-term + long-term)."""

    def setup_method(self):
        """Create a fresh MemoryManager for each test."""
        from memory.memory_manager import MemoryManager
        self.mm = MemoryManager()

    def test_short_term_add_and_retrieve(self):
        """Test adding and retrieving conversation messages."""
        self.mm.add_message("test_chat", "user", "Hello!")
        self.mm.add_message("test_chat", "assistant", "Hi there!")

        history = self.mm.get_conversation_history("test_chat")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello!"
        assert history[1]["role"] == "assistant"

    def test_short_term_separate_chats(self):
        """Test that different chat IDs have separate histories."""
        self.mm.add_message("chat1", "user", "Message 1")
        self.mm.add_message("chat2", "user", "Message 2")

        assert len(self.mm.get_conversation_history("chat1")) == 1
        assert len(self.mm.get_conversation_history("chat2")) == 1
        assert self.mm.get_conversation_history("chat1")[0]["content"] == "Message 1"

    def test_short_term_clear(self):
        """Test clearing conversation history."""
        self.mm.add_message("test_chat", "user", "Hello!")
        self.mm.clear_conversation("test_chat")

        assert len(self.mm.get_conversation_history("test_chat")) == 0

    def test_long_term_store_and_search(self):
        """Test storing and searching long-term memories."""
        self.mm.add_memory(
            content="Bitcoin reached a new all-time high of $100,000 today.",
            memory_type="research",
            metadata={"query": "bitcoin price"},
            memory_id="test_btc_memory",
        )

        results = self.mm.search_memory("bitcoin price", n_results=1)
        assert len(results) > 0
        assert "Bitcoin" in results[0]["content"]

    def test_long_term_type_filter(self):
        """Test filtering memory search by type."""
        self.mm.add_memory(
            content="Test research content",
            memory_type="research",
            memory_id="test_research_1",
        )
        self.mm.add_memory(
            content="Test briefing content",
            memory_type="briefing",
            memory_id="test_briefing_1",
        )

        research_results = self.mm.search_memory(
            "test content", n_results=5, memory_type="research"
        )
        # All results should be of type 'research'
        for r in research_results:
            assert r["metadata"]["type"] == "research"

    def test_memory_count(self):
        """Test memory count."""
        initial = self.mm.get_memory_count()
        self.mm.add_memory("New memory", memory_id="test_count_1")
        assert self.mm.get_memory_count() >= initial

    def test_scoped_search_user_plus_global(self):
        """Search should include current chat + global, but exclude other chats."""
        self.mm.add_memory(
            content="scope_token global market summary",
            memory_type="research",
            memory_id="test_scope_global_1",
            scope="global",
        )
        self.mm.add_memory(
            content="scope_token chat1 private insight",
            memory_type="research",
            memory_id="test_scope_chat1_1",
            chat_id="chat1",
            scope="user",
        )
        self.mm.add_memory(
            content="scope_token chat2 private insight",
            memory_type="research",
            memory_id="test_scope_chat2_1",
            chat_id="chat2",
            scope="user",
        )

        results = self.mm.search_memory(
            query="scope_token",
            n_results=10,
            memory_type="research",
            chat_id="chat1",
            include_global=True,
        )
        contents = [r["content"] for r in results]
        assert any("global" in c for c in contents)
        assert any("chat1" in c for c in contents)
        assert all("chat2" not in c for c in contents)

    def test_user_preferences_scoped(self):
        """Preferences should return chat-specific + global preferences."""
        self.mm.save_user_preference(
            "Use Vietnamese responses", chat_id="pref_chat1"
        )
        self.mm.save_user_preference(
            "Always add sources", chat_id="pref_chat2"
        )
        self.mm.save_user_preference(
            "Prefer concise output", is_global=True
        )

        prefs_chat1 = self.mm.get_user_preferences(
            chat_id="pref_chat1", include_global=True
        )
        prefs_chat1_text = [p["content"] for p in prefs_chat1]

        assert any("Use Vietnamese responses" in p for p in prefs_chat1_text)
        assert any("Prefer concise output" in p for p in prefs_chat1_text)
        assert all("Always add sources" not in p for p in prefs_chat1_text)


class TestWebCrawler:
    """Test the web crawler tool."""

    def test_crawl_url_success(self):
        """Test crawling a known accessible URL."""
        from tools.web_crawler import crawl_url

        result = crawl_url("https://example.com")
        assert result["success"] is True
        assert result["url"] == "https://example.com"
        assert len(result["content"]) > 0
        assert result["domain"] == "example.com"

    def test_crawl_url_invalid(self):
        """Test crawling an invalid URL."""
        from tools.web_crawler import crawl_url

        result = crawl_url("https://this-domain-does-not-exist-123456.com", timeout=5)
        assert result["success"] is False
        assert "error" in result


class TestWebSearch:
    """Test the web search tool."""

    def test_search_returns_results(self):
        """Test that search returns results."""
        from tools.web_search import search_web

        results = search_web("Python programming language", max_results=3)
        # DuckDuckGo might rate-limit, so we just check the format
        assert isinstance(results, list)
        if results:  # If we got results
            assert "title" in results[0]
            assert "url" in results[0]
            assert "snippet" in results[0]


class TestBrowserAutomation:
    """Test Playwright Chromium browser automation."""

    def test_browser_automation_imports(self):
        """Test that browser automation modules import correctly."""
        try:
            from tools.playwright_browser import PlaywrightBrowserAutomation
            from graph.nodes.browser_automation import browser_automation_node
        except ImportError as e:
            pytest.skip(f"Browser automation modules not available: {e}")

    def test_browser_automation_initialization(self):
        """Test PlaywrightBrowserAutomation initialization."""
        try:
            from tools.playwright_browser import PlaywrightBrowserAutomation
        except ImportError:
            pytest.skip("Playwright not installed")

        browser = PlaywrightBrowserAutomation(headless=True)
        assert browser.headless is True
        assert browser.use_persistent_context is True
        assert browser.action_history == []
        assert browser.current_url == "about:blank"

    def test_crawler_browser_factory(self):
        """Test browser launch factory function."""
        try:
            from tools.web_crawler import _launch_browser
        except ImportError:
            pytest.skip("web_crawler module not available")

        # Just verify the function exists and is callable
        assert callable(_launch_browser)

    def test_browser_session_manager_browser_type(self):
        """Test BrowserSessionManager with different browser types."""
        try:
            from tools.browser_session import BrowserSessionManager
        except ImportError:
            pytest.skip("BrowserSessionManager not available")

        # Test chromium (default)
        manager_chromium = BrowserSessionManager(browser_type="chromium")
        assert manager_chromium.browser_type == "chromium"
        assert "_chromium" not in str(manager_chromium._cookie_path("test"))

        # Test webkit
        manager_webkit = BrowserSessionManager(browser_type="webkit")
        assert manager_webkit.browser_type == "webkit"
        assert "_webkit" in str(manager_webkit._cookie_path("test"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

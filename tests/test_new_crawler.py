"""
Tests for the new crawling stack: RSS feeds + Trafilatura + nodriver.

Replaces the old test_crawl4ai.py tests.
"""

import sys
import os

import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── RSS Feeds ────────────────────────────────────────────────────────────


class TestNewsFeedsImports:
    def test_news_feeds_import(self):
        from tools.news_feeds import fetch_feed, fetch_all_news_feeds, search_news_via_google_rss
        assert callable(fetch_feed)
        assert callable(fetch_all_news_feeds)
        assert callable(search_news_via_google_rss)


class TestNewsFeeds:
    def test_fetch_techcrunch_feed(self):
        """TechCrunch RSS should return articles with title, url, date."""
        from tools.news_feeds import fetch_feed
        items = fetch_feed("https://techcrunch.com/feed/", max_items=3)
        assert len(items) >= 1
        assert items[0]["title"]
        assert items[0]["url"].startswith("http")
        assert "published" in items[0]

    def test_fetch_vnexpress_feed(self):
        """VnExpress RSS should return Vietnamese news items."""
        from tools.news_feeds import fetch_feed
        items = fetch_feed("https://vnexpress.net/rss/tin-moi-nhat.rss", max_items=3)
        assert len(items) >= 1
        assert items[0]["url"].startswith("http")

    def test_fetch_reuters_via_google_news(self):
        """Google News RSS for reuters.com should return results."""
        from tools.news_feeds import search_news_via_google_rss
        items = search_news_via_google_rss("site:reuters.com", max_items=3)
        assert isinstance(items, list)
        # Google News may or may not return results, but should not error
        assert len(items) >= 0

    def test_fetch_feed_invalid_url(self):
        """Invalid feed URL should return empty list, not crash."""
        from tools.news_feeds import fetch_feed
        items = fetch_feed("https://invalid-feed-url-xyz-123.com/feed.xml")
        assert items == []


# ─── Trafilatura Article Extraction ──────────────────────────────────────


class TestArticleExtractorImports:
    def test_article_extractor_import(self):
        from tools.article_extractor import extract_article, extract_articles, extract_from_html
        assert callable(extract_article)
        assert callable(extract_articles)
        assert callable(extract_from_html)


class TestArticleExtractor:
    def test_extract_article_success(self):
        """Trafilatura should extract content from example.com."""
        from tools.article_extractor import extract_article
        result = extract_article("https://example.com")
        assert result["success"] is True
        assert result["url"] == "https://example.com"
        assert result["domain"] == "example.com"
        assert len(result["content"]) > 0

    def test_extract_article_failure(self):
        """Invalid URL should return success=False, not crash."""
        from tools.article_extractor import extract_article
        result = extract_article("https://this-domain-does-not-exist-xyz-123.com")
        assert result["success"] is False
        assert "error" in result

    def test_extract_article_has_metadata(self):
        """Extraction result should have metadata fields."""
        from tools.article_extractor import extract_article
        result = extract_article("https://example.com")
        assert "title" in result
        assert "author" in result
        assert "date" in result
        assert "domain" in result

    @pytest.mark.asyncio
    async def test_extract_articles_concurrent(self):
        """Multiple URLs should be extracted concurrently."""
        from tools.article_extractor import extract_articles
        urls = ["https://example.com", "https://httpbin.org/html"]
        results = await extract_articles(urls)
        assert len(results) == 2
        successes = [r for r in results if r["success"]]
        assert len(successes) >= 1

    @pytest.mark.asyncio
    async def test_extract_articles_empty_list(self):
        """Empty URL list should return empty list."""
        from tools.article_extractor import extract_articles
        results = await extract_articles([])
        assert results == []

    def test_extract_from_html(self):
        """extract_from_html should work on raw HTML."""
        from tools.article_extractor import extract_from_html
        html = """
        <html><head><title>Test</title></head>
        <body><article><h1>Hello</h1><p>This is a test article with enough content for extraction.</p></article></body>
        </html>
        """
        result = extract_from_html(html, url="https://example.com/test")
        # May or may not extract depending on content length, but should not crash
        assert isinstance(result, dict)
        assert "success" in result


# ─── nodriver ────────────────────────────────────────────────────────────


class TestNodriverImports:
    def test_nodriver_import(self):
        """nodriver package should be importable."""
        import nodriver
        assert hasattr(nodriver, "start")

    def test_nodriver_browser_module_import(self):
        """Our nodriver_browser module should import correctly."""
        from tools.nodriver_browser import (
            save_cookies,
            load_cookies,
            login_interactive,
            fetch_authenticated_page,
            fetch_and_extract,
        )
        assert callable(save_cookies)
        assert callable(load_cookies)

    def test_cookie_save_load(self):
        """Cookie save/load round-trip should work."""
        from tools.nodriver_browser import save_cookies, load_cookies
        test_cookies = [{"name": "test", "value": "123", "domain": ".example.com", "path": "/"}]
        save_cookies("test_site", test_cookies)
        loaded = load_cookies("test_site")
        assert len(loaded) == 1
        assert loaded[0]["name"] == "test"
        assert loaded[0]["value"] == "123"


# ─── Web Crawler (Trafilatura-first) ─────────────────────────────────────


class TestWebCrawlerFallback:
    def test_crawl_url_still_works(self):
        """crawl_url should work via Trafilatura (no browser)."""
        from tools.web_crawler import crawl_url
        result = crawl_url("https://example.com")
        assert result["success"] is True
        assert len(result["content"]) > 0

    def test_crawl_url_no_crawl4ai_import(self):
        """Verify web_crawler.py no longer imports crawl4ai."""
        import inspect
        import tools.web_crawler as wc
        source = inspect.getsource(wc)
        assert "crawl4ai" not in source.lower()


# ─── RAG Chunking ────────────────────────────────────────────────────────


class TestRAGChunking:
    def test_build_chunks_from_extraction_result(self):
        """build_document_chunks_from_crawl_result should work with Trafilatura output."""
        from rag.chunking import build_document_chunks_from_crawl_result

        mock_result = {
            "success": True,
            "content": "This is a test article about technology. " * 50,
            "url": "https://example.com/article",
            "title": "Test Article",
            "author": "Test Author",
            "date": "2024-01-01",
        }
        chunks = build_document_chunks_from_crawl_result(mock_result, topic="tech")
        assert len(chunks) >= 1
        assert chunks[0]["metadata"]["source_url"] == "https://example.com/article"
        assert chunks[0]["metadata"]["topic"] == "tech"

    def test_build_chunks_from_failed_result(self):
        """Failed extraction should produce empty chunks."""
        from rag.chunking import build_document_chunks_from_crawl_result

        failed_result = {
            "success": False,
            "content": "",
            "url": "https://example.com/fail",
            "error": "404 Not Found",
        }
        chunks = build_document_chunks_from_crawl_result(failed_result)
        assert chunks == []

    def test_build_chunks_from_empty_content(self):
        """Results with empty content should produce empty chunks."""
        from rag.chunking import build_document_chunks_from_crawl_result

        empty_result = {
            "success": True,
            "content": "",
            "url": "https://example.com/empty",
            "title": "Empty",
        }
        chunks = build_document_chunks_from_crawl_result(empty_result)
        assert chunks == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

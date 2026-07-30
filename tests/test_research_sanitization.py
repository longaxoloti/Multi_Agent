from graph.nodes.research import _extract_crawled_articles, _sanitize_crawled_markdown


def test_sanitize_crawled_markdown_removes_translation_noise_line():
    raw = """
Lead paragraph about a shipping incident near Hormuz.
Tàu đang chở hàng và bị拦截并翻译成中文意思是？
Another factual sentence in Vietnamese.
""".strip()

    cleaned = _sanitize_crawled_markdown(raw)

    assert "翻译成中文" not in cleaned
    assert "Lead paragraph" in cleaned
    assert "Another factual sentence" in cleaned


def test_extract_crawled_articles_applies_content_sanitization():
    context = """
=== CRAWL4AI ARTICLE ===
URL: https://example.com/article-1
DOMAIN: example.com
TITLE_HINT: Incident report
CONTENT:
Main article sentence.
bị拦截并翻译成中文意思是？
Follow-up factual sentence.
""".strip()

    items = _extract_crawled_articles(context)

    assert len(items) == 1
    content = items[0]["content"]
    assert "翻译成中文" not in content
    assert "Main article sentence." in content
    assert "Follow-up factual sentence." in content

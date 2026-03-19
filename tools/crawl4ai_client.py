import asyncio
import logging
from typing import Optional
logger = logging.getLogger(__name__)

try:
    from crawl4ai import AsyncWebCrawler
    from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
    from crawl4ai.content_filter_strategy import PruningContentFilter
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
    _CRAWL4AI_AVAILABLE = True
except ImportError:
    _CRAWL4AI_AVAILABLE = False
    logger.warning(
        "crawl4ai is not installed. Article crawling via Crawl4AI will be disabled. "
        "Run: pip install crawl4ai && crawl4ai-setup"
    )
from main.config import CRAWL4AI_ENABLED

_ARTICLE_CHAR_LIMIT = 15_000

def _build_crawler_config(url: str) -> "CrawlerRunConfig":
    prune_filter = PruningContentFilter(
        threshold=0.40,
        threshold_type="dynamic",
        min_word_threshold=5,
    )
    md_generator = DefaultMarkdownGenerator(
        content_filter=prune_filter,
        options={"ignore_links": False},
    )
    return CrawlerRunConfig(
        markdown_generator=md_generator,
        excluded_tags=["nav", "footer", "header", "aside", "script", "style", "iframe"],
        remove_overlay_elements=True,
        word_count_threshold=5,
        page_timeout=20_000,
        wait_until="domcontentloaded",
        verbose=False,
    )


async def crawl_url_to_markdown(url: str) -> str:
    if not CRAWL4AI_ENABLED:
        logger.debug("CRAWL4AI_ENABLED=false; skipping article crawl for %s", url)
        return ""

    if not _CRAWL4AI_AVAILABLE:
        logger.warning("crawl4ai not installed; skipping %s", url)
        return ""

    browser_config = BrowserConfig(
        browser_type="chromium",
        headless=True,
        verbose=False,
    )
    run_config = _build_crawler_config(url)

    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=run_config)

        if not result.success:
            logger.warning(
                "Crawl4AI failed for %s: status=%s err=%s",
                url, result.status_code, result.error_message
            )
            return ""

        fit = (result.markdown.fit_markdown or "").strip()
        raw = (result.markdown.raw_markdown or "").strip()
        content = fit if len(fit) >= 200 else raw

        if not content:
            logger.debug("Crawl4AI returned empty content for %s", url)
            return ""

        logger.info(
            "Crawl4AI crawled %s — fit=%d chars, raw=%d chars, using=%s",
            url, len(fit), len(raw), "fit" if content is fit else "raw",
        )
        return content[:_ARTICLE_CHAR_LIMIT]

    except asyncio.TimeoutError:
        logger.warning("Crawl4AI timed out for %s", url)
        return ""
    except Exception as exc:
        logger.warning("Crawl4AI exception for %s: %s", url, exc, exc_info=False)
        return ""

import asyncio
import logging
import os
from typing import Optional
logger = logging.getLogger(__name__)

DEBUG_CRAWL4AI = os.getenv("DEBUG_RESEARCH", "").lower() in {"1", "true", "yes"}

try:
    from crawl4ai import AsyncWebCrawler
    from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
    from crawl4ai.content_filter_strategy import PruningContentFilter
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
    _CRAWL4AI_AVAILABLE = True
except ImportError:
    _CRAWL4AI_AVAILABLE = False
    logger.warning(
        "crawl4ai is not installed. Article crawling via Crawl4AI will be disabled."
        "Run: pip install crawl4ai && crawl4ai-setup"
    )
from main.config import CRAWL4AI_ENABLED

_ARTICLE_CHAR_LIMIT = 15_000

def _build_crawler_config(url: str, base_url: Optional[str] = None) -> "CrawlerRunConfig":
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
        base_url=base_url,
        verbose=False,
    )


async def crawl_url_to_markdown(
    url: str,
    html: Optional[str] = None,
    base_url: Optional[str] = None,
    cdp_url: Optional[str] = None,
) -> str:
    if not CRAWL4AI_ENABLED:
        logger.debug("CRAWL4AI_ENABLED=false; skipping article crawl for %s", url)
        return ""

    if not _CRAWL4AI_AVAILABLE:
        logger.warning("crawl4ai not installed; skipping %s", url)
        return ""

    browser_config_kwargs = {
        "browser_type": "chromium",
        "headless": True,
        "verbose": False,
    }
    if cdp_url:
        browser_config_kwargs.update(
            {
                "browser_mode": "custom",
                "cdp_url": cdp_url,
                "cache_cdp_connection": True,
            }
        )

    browser_config = BrowserConfig(**browser_config_kwargs)
    run_config = _build_crawler_config(url, base_url=base_url or (url if html else base_url))

    async def _run_once(crawl_target: str):
        async with AsyncWebCrawler(config=browser_config) as crawler:
            return await crawler.arun(url=crawl_target, config=run_config)

    crawl_target = f"raw://{html}" if html else url
    used_cached_html = bool(html)

    try:
        result = await _run_once(crawl_target)

        # Some sites enforce Trusted Types/CSP and reject document.write during raw:// ingestion.
        # If cached HTML mode fails, retry with direct URL crawling.
        if used_cached_html and (not result.success):
            err_msg = (result.error_message or "").lower()
            if "trustedhtml" in err_msg or "document requires" in err_msg or "set_content" in err_msg:
                logger.info(
                    "Crawl4AI retrying direct URL after cached HTML failure for %s",
                    url,
                )
                result = await _run_once(url)

        if not result.success:
            logger.warning(
                "Crawl4AI failed for %s: status=%s err=%s",
                url, result.status_code, result.error_message
            )
            if DEBUG_CRAWL4AI:
                logger.info("[DEBUG_CRAWL4AI] FAILED url=%s status=%s error=%s", url, result.status_code, result.error_message)
            return ""

        fit = (result.markdown.fit_markdown or "").strip()
        raw = (result.markdown.raw_markdown or "").strip()
        content = fit if len(fit) >= 200 else raw

        if not content:
            logger.debug("Crawl4AI returned empty content for %s", url)
            if DEBUG_CRAWL4AI:
                logger.info("[DEBUG_CRAWL4AI] EMPTY url=%s fit_len=%d raw_len=%d", url, len(fit), len(raw))
            return ""

        logger.info(
            "Crawl4AI crawled %s — fit=%d chars, raw=%d chars, using=%s",
            url, len(fit), len(raw), "fit" if content is fit else "raw",
        )
        if DEBUG_CRAWL4AI:
            logger.info("[DEBUG_CRAWL4AI] SUCCESS url=%s fit_len=%d raw_len=%d content_len=%d", url, len(fit), len(raw), len(content))
            logger.info("[DEBUG_CRAWL4AI] CONTENT_SAMPLE url=%s first_500=%s", url, content[:500])
        return content[:_ARTICLE_CHAR_LIMIT]

    except asyncio.TimeoutError:
        logger.warning("Crawl4AI timed out for %s", url)
        return ""
    except Exception as exc:
        logger.warning("Crawl4AI exception for %s: %s", url, exc, exc_info=False)
        return ""

"""
Crawl4AI-based article crawler.

Replaces Camoufox for individual article pages during research crawls.
Uses PruningContentFilter to strip ads, navigation, sidebars, and thin
promotional blocks — without filtering by topic, so all factual content
on the page is preserved for the research LLM.

Controlled by config.CRAWL4AI_ENABLED (env: CRAWL4AI_ENABLED=true/false).
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import guard — crawl4ai is optional; if not installed the rest of the
# pipeline degrades to empty strings gracefully.
# ---------------------------------------------------------------------------
try:
    from crawl4ai import AsyncWebCrawler
    from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
    from crawl4ai.content_filter_strategy import PruningContentFilter
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
    _CRAWL4AI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CRAWL4AI_AVAILABLE = False
    logger.warning(
        "crawl4ai is not installed. Article crawling via Crawl4AI will be disabled. "
        "Run: pip install crawl4ai && crawl4ai-setup"
    )

# Content cap per article to keep prompt size manageable.
_ARTICLE_CHAR_LIMIT = 15_000


def _build_crawler_config(url: str) -> "CrawlerRunConfig":
    """
    Build a CrawlerRunConfig that:
    - Strips nav/header/footer/aside/script/style tags before any markdown generation.
    - Removes modal/overlay pop-up elements (cookie banners, paywalls).
    - Applies PruningContentFilter to drop thin/ad-heavy text blocks heuristically
      (no topic matching — threshold-only mode keeps ALL factual paragraphs).
    - Enforces a minimum of 5 words per block so stray link labels disappear.
    """
    prune_filter = PruningContentFilter(
        threshold=0.40,            # low threshold → keep more, only drop very thin blocks
        threshold_type="dynamic",  # adapts to tag type + text density
        min_word_threshold=5,      # discard fragments shorter than 5 words
    )
    md_generator = DefaultMarkdownGenerator(
        content_filter=prune_filter,
        options={"ignore_links": False},  # keep inline links for source attribution
    )
    return CrawlerRunConfig(
        markdown_generator=md_generator,
        # ── structural noise removal ────────────────────────────────────────
        excluded_tags=["nav", "footer", "header", "aside", "script", "style", "iframe"],
        remove_overlay_elements=True,   # popups, sticky banners
        word_count_threshold=5,
        # ── behaviour ──────────────────────────────────────────────────────
        page_timeout=20_000,            # 20 s page-load timeout
        wait_until="domcontentloaded",
        verbose=False,
    )


async def crawl_url_to_markdown(url: str) -> str:
    """
    Crawl *url* and return cleaned Fit Markdown (LLM-ready, ads/nav removed).

    Returns empty string on any error so callers can safely skip.
    """
    from config import CRAWL4AI_ENABLED  # imported late to avoid circular imports at module load

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
        content = fit if len(fit) >= 200 else raw  # fallback to raw if fit is too short

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

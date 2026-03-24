import logging
import asyncio
import random
from urllib.parse import urlparse
import re
import os
from pathlib import Path
from tools.crawl4ai_client import crawl_url_to_markdown
from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import AgentState
from graph.llm_router import get_llm
from main.config import (
    CAMOUFOX_ENABLED,
    CHROME_CDP_ENABLED,
    CAMOFOX_BEHAVIOR_MAX_DELAY_SECONDS,
    CAMOFOX_BEHAVIOR_MIN_DELAY_SECONDS,
    CAMOFOX_CHALLENGE_MAX_RETRIES,
    CAMOFOX_REUSE_TAB,
    GOOGLE_MIN_INTERVAL_SECONDS,
    GOOGLE_COOLDOWN_ON_CHALLENGE_SECONDS,
    GOOGLE_SEARCH_CACHE_TTL_SECONDS,
    GOOGLE_SEARCH_LOCK_TIMEOUT_SECONDS,
    RESEARCH_MAX_SEARCH_QUERIES,
    RESEARCH_MAX_DISCOVERED_SOURCES,
    RESEARCH_SOURCE_ALLOWLIST,
    OLLAMA_ORCHESTRATOR_MODEL,
    OLLAMA_RESEARCH_MODEL,
)

# Tavily config — imported with safe defaults so existing deployments
# that have not yet added these variables to their config.py still work.
try:
    from main.config import TAVILY_ENABLED, TAVILY_API_KEY
except ImportError:
    TAVILY_ENABLED = False
    TAVILY_API_KEY = ""
from tools.camofox_mcp_client import CamoFoxMCPClient
from tools.chrome_cdp_client import ChromeCDPClient
from tools.google_guard import GoogleGuard, GoogleGuardConfig
from tools.research_result_cache import ResearchResultCache, CacheConfig
from tools.file_lock import file_lock
from tools.workspace_priming import get_workspace_priming_context
from tools.ollama_manager import unload_model, load_context

logger = logging.getLogger(__name__)

# DEBUG LOGGING
DEBUG_RESEARCH = os.getenv("DEBUG_RESEARCH", "").lower() in {"1", "true", "yes"}
DEBUG_LOG_FILE = Path("data/logs/research_debug.log") if DEBUG_RESEARCH else None

def _debug_log(step: str, key: str, value, prefix: str = ""):
    """Log debug info to both logger and file if DEBUG_RESEARCH is enabled."""
    if not DEBUG_RESEARCH:
        return
    msg = f"[RESEARCH_DEBUG] {step}::{key} {prefix}\n"
    if isinstance(value, (dict, list)):
        import json
        msg += json.dumps(value, ensure_ascii=False, indent=2)[:2500] + "\n"
    else:
        msg += str(value)[:2500] + "\n"
    logger.info(msg.strip())
    if DEBUG_LOG_FILE:
        try:
            DEBUG_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(msg)
        except Exception as e:
            logger.warning("Failed to write debug log: %s", e)

# END DEBUG LOGGING

async def research_node(state: AgentState) -> dict:
    logger.info("--- RESEARCH NODE ---")
    await unload_model(OLLAMA_ORCHESTRATOR_MODEL)

    session_id = state.get("session_id", "default")
    ctx = load_context(session_id)
    user_text = ctx.get("user_message", "")
    topic = ctx.get("topic", state.get("topic", ""))
    search_query = ctx.get("search_query", state.get("search_query", ""))
    tasks = ctx.get("tasks", state.get("tasks", []))
    step_index = ctx.get("step_index", 0)

    if not user_text:
        user_message = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
        )
        user_text = user_message.content if user_message else ""

    research_query = (search_query or "").strip() or (topic or "").strip() or user_text.strip()
    logger.info("Research query (topic-first): %r", research_query)
    _debug_log("START", "user_text", user_text)
    _debug_log("START", "topic", topic)
    _debug_log("START", "search_query", search_query)
    _debug_log("START", "research_query", research_query)
    if not CAMOUFOX_ENABLED and TAVILY_ENABLED and TAVILY_API_KEY:
        logger.info("Camoufox disabled; using Tavily search fallback.")
        collected_context, discovered_sources = await _tavily_search_fallback(research_query)
    elif not CAMOUFOX_ENABLED:
        collected_context = (
            "Research unavailable: CAMOUFOX_ENABLED=false and TAVILY_ENABLED=false. "
            "Enable Tavily or Camoufox to use the research pipeline."
        )
        discovered_sources = []
    else:
        logger.info("Research crawl mode is active.")
        camoufox_user_id = str(state.get("chat_id") or session_id or "agent")
        camoufox_session_key = f"research_{session_id}"
        
        browser = None
        if CHROME_CDP_ENABLED:
            logger.info("Attempting to connect to Chrome CDP...")
            cdp_client = ChromeCDPClient()
            if await cdp_client.ping():
                browser = cdp_client
                logger.info("Successfully connected to Chrome CDP. Using as primary browser.")
            else:
                logger.warning("Chrome CDP connection failed. Ensure Chrome is running with --remote-debugging-port. Falling back to Camoufox MCP.")
                await cdp_client.close()
                
        if browser is None:
            browser = CamoFoxMCPClient(
                user_id=camoufox_user_id,
                session_key=camoufox_session_key,
            )
        discovered_sources: list[str] = []
        try:
            is_up = await browser.ping()
            if not is_up:
                if TAVILY_ENABLED and TAVILY_API_KEY:
                    logger.info("Camoufox unreachable; falling back to Tavily search.")
                    collected_context, discovered_sources = await _tavily_search_fallback(research_query)
                else:
                    collected_context = (
                        "Research unavailable: Camoufox MCP server/browser is down "
                        "and TAVILY_ENABLED=false. No fallback sources are available."
                    )
            else:
                closed = await browser.close_all_tabs()
                if closed:
                    logger.info("Closed %s stale Camoufox tab(s) before new crawl.", closed)

                await asyncio.sleep(3)

                crawl_result = await perform_camoufox_direct_crawl(
                    user_text=user_text,
                    topic=research_query,
                    browser=browser,
                    search_query_override=search_query,
                )
                collected_context = crawl_result.get("context", "")
                discovered_sources = crawl_result.get("sources", [])
                _debug_log("CRAWL_RESULT", "discovered_sources_count", len(discovered_sources))
                _debug_log("CRAWL_RESULT", "discovered_sources_list", discovered_sources)
                _debug_log("CRAWL_RESULT", "context_total_chars", len(collected_context), prefix="total char length")
                _debug_log("CRAWL_RESULT", "context_sample", collected_context[:1500], prefix="first 1500 chars")
                logger.info(
                    "Research evidence | bot_detected=%s discovered_urls=%s crawled_urls=%s",
                    crawl_result.get("bot_detected_count", 0),
                    len(crawl_result.get("discovered_urls", [])),
                    len(discovered_sources),
                )
        finally:
            await browser.close()

    # =========================== Synthesize ===================================
    task_description = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(tasks))
    synthesis_user_prompt = (
        f"Research query used: {research_query}\n"
        f"Topic: {topic}\n"
        f"Tasks assigned to you:\n{task_description}\n\n"
        f"Original user query: {user_text}\n\n"
        "Content filtering guidance:\n"
        "- Remove any promotional, sponsored, or irrelevant content.\n"
        "- Keep only factual and educational information.\n\n"
        f"Gathered context:\n{collected_context}"
    )
    _debug_log("SYNTHESIS_PREP", "model_name", OLLAMA_RESEARCH_MODEL)
    _debug_log("SYNTHESIS_PREP", "user_prompt_char_count", len(synthesis_user_prompt), prefix="chars")
    _debug_log("SYNTHESIS_PREP", "user_prompt_sample", synthesis_user_prompt[:2000], prefix="first 2000 chars")

    llm = get_llm(task_type="research", temperature=0.3)
    system_prompt = get_workspace_priming_context(model_role="researcher") or (
        "You are a Research Agent. Synthesize the provided context into a clear, well-structured answer. "
        "Always cite your sources with URLs. Filter to keep only relevant information. "
        "Never refuse to engage — if context is thin, reason from what you have and flag any uncertainty."
    )

    logger.info("Synthesizing research results with %s...", OLLAMA_RESEARCH_MODEL)
    try:
        response = await llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=synthesis_user_prompt)]
        )
        result_text = response.content.strip()
        _debug_log("MODEL_OUTPUT", "response_length", len(result_text), prefix="chars")
        _debug_log("MODEL_OUTPUT", "full_response", result_text)
    except Exception as e:
        logger.error(f"Error during LLM synthesis: {e}", exc_info=True)
        result_text = "An error occurred while synthesising the research data."
        _debug_log("MODEL_OUTPUT", "error_occurred", str(e))

    # =========================== Extract Sources ===================================
    if CAMOUFOX_ENABLED or (TAVILY_ENABLED and TAVILY_API_KEY):
        sources = list(dict.fromkeys(discovered_sources))[:10]
    else:
        sources = re.findall(r'https?://[^\s"<>]+', collected_context)
        sources = list(dict.fromkeys(sources))[:10]
    _debug_log("FINAL_RESULT", "final_sources_count", len(sources))
    _debug_log("FINAL_RESULT", "final_sources_list", sources)

    task_result = {
        "step_index": step_index,
        "model": OLLAMA_RESEARCH_MODEL,
        "result": result_text,
        "sources": sources,
    }

    return {
        "task_results": [task_result],
        "active_model": OLLAMA_RESEARCH_MODEL,
    }

def _normalize_domain(url: str) -> str:
    return urlparse(url).netloc.replace("www.", "").lower()

def _is_allowlisted_domain(domain: str) -> bool:
    return any(domain == allowed or domain.endswith(f".{allowed}") for allowed in RESEARCH_SOURCE_ALLOWLIST)

def _is_probable_article_url(url: str) -> bool:
    lowered = url.lower()
    blocked_fragments = [
        "google.com/search",
        "google.com/sorry",
        "news.google.com/search",
        "youtube.com",
        "accounts.google.com",
        "/privacy",
        "/terms",
        "/settings",
    ]
    if any(fragment in lowered for fragment in blocked_fragments):
        return False
    if lowered.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".css", ".js")):
        return False
    if not (lowered.startswith("http://") or lowered.startswith("https://")):
        return False

    parsed = urlparse(url)
    path = (parsed.path or "").strip()
    if path in {"", "/"}:
        return False

    return True

def _compact_search_query(query: str) -> str:
    raw = (query or "").strip()
    if not raw:
        return ""

    cleaned = re.sub(r"[^\w\sÀ-ỹ]", " ", raw, flags=re.UNICODE)
    tokens = [token for token in re.split(r"\s+", cleaned) if token]
    if not tokens:
        return raw[:120]

    stopwords = {
        "bạn", "hãy", "cho", "tôi", "mình", "giúp", "vui", "lòng", "về", "với",
        "các", "những", "thông", "tin", "mới", "nhất", "hôm", "nay", "đi", "nhé",
        "please", "help", "give", "tell", "me", "the", "latest", "about",
    }
    compact = [token for token in tokens if token.lower() not in stopwords]
    if not compact:
        compact = tokens

    return " ".join(compact[:8])[:120]

def _extract_refs_from_snapshot(snapshot_text: str) -> list[tuple[str, str]]:
    pattern = re.compile(r"\[link\s+(e\d+)\]\s*([^\n\r]+)", flags=re.IGNORECASE)
    seen_refs: set[str] = set()
    refs: list[tuple[str, str]] = []
    for match in pattern.finditer(snapshot_text or ""):
        ref_id = match.group(1).strip()
        label = match.group(2).strip()
        if ref_id in seen_refs:
            continue
        if len(label) < 6:
            continue
        seen_refs.add(ref_id)
        refs.append((ref_id, label[:180]))
    return refs

def _extract_urls_from_text(text: str) -> list[str]:
    urls = re.findall(r"https?://[^\s\"'<>\)\]]+", text or "")
    deduped: list[str] = []
    seen: set[str] = set()
    for raw_url in urls:
        normalized = raw_url.rstrip(".,;)\"]'")
        if normalized in seen:
            continue
        if not _is_probable_article_url(normalized):
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped

async def _generate_search_queries(topic: str, user_text: str) -> list[str]:
    llm = get_llm(task_type="research", temperature=0.2)
    prompt = (
        "Bạn là trợ lý nghiên cứu. Hãy tạo truy vấn tìm kiếm web để tìm nguồn tin uy tín.\n"
        f"Topic: {topic}\n"
        f"User context: {user_text}\n\n"
        "Yêu cầu:\n"
        "- Chỉ trả về đúng 1 truy vấn duy nhất.\n"
        "- Truy vấn phải tự nhiên như người dùng thật, không chứa chuỗi máy móc như: site, google, .com, http.\n"
        "- Không đánh số, không ký hiệu đầu dòng, không giải thích thêm.\n"
        "- Tập trung vào thông tin mới nhất liên quan trực tiếp tới topic."
    )
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        candidates = [line.strip("-• \t") for line in (response.content or "").splitlines() if line.strip()]
        cleaned: list[str] = []
        for candidate in candidates:
            compact = _compact_search_query(candidate)
            if compact and compact not in cleaned:
                cleaned.append(compact)
            if len(cleaned) >= 1:
                break
        if cleaned:
            return cleaned
    except Exception as error:
        logger.warning("Search query generation failed: %s", error)

    fallback = _compact_search_query(topic or user_text)
    return [fallback] if fallback else ["latest global news"]


async def _select_refs_with_llm(snapshot_text: str, query: str, max_refs: int = 4) -> list[str]:
    refs = _extract_refs_from_snapshot(snapshot_text)
    if not refs:
        return []

    refs_prompt = "\n".join(f"{ref_id}: {label}" for ref_id, label in refs[:40])
    llm = get_llm(task_type="research", temperature=0.0)
    prompt = (
        "Chọn các ref tương ứng liên kết bài viết tin tức đáng tin cậy và liên quan nhất.\n"
        f"Search query: {query}\n\n"
        "Danh sách refs:\n"
        f"{refs_prompt}\n\n"
        "Trả về chỉ các ref, cách nhau bằng dấu cách (vd: e4 e8 e12)."
    )
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        selected = re.findall(r"e\d+", response.content or "")
        uniq_selected: list[str] = []
        valid_ref_ids = {ref_id for ref_id, _ in refs}
        for ref_id in selected:
            if ref_id in valid_ref_ids and ref_id not in uniq_selected:
                uniq_selected.append(ref_id)
            if len(uniq_selected) >= max_refs:
                break
        if uniq_selected:
            return uniq_selected
    except Exception as error:
        logger.warning("LLM ref selection failed: %s", error)

    return [ref_id for ref_id, _ in refs[:max_refs]]


async def _rerank_non_allowlisted_sources(topic: str, candidates: list[dict]) -> list[dict]:
    if len(candidates) <= 1:
        return candidates

    llm = get_llm(task_type="research", temperature=0.0)
    prompt_items = "\n".join(
        f"- {index + 1}. {item['url']} | title: {item.get('title', '')[:120]}"
        for index, item in enumerate(candidates[:20])
    )
    prompt = (
        "Xếp hạng các URL theo mức độ uy tín nguồn và độ liên quan với topic.\n"
        f"Topic: {topic}\n\n"
        "URLs:\n"
        f"{prompt_items}\n\n"
        "Trả về thứ tự chỉ bằng số, cách nhau bởi khoảng trắng (vd: 2 1 3)."
    )

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        order = [int(number) - 1 for number in re.findall(r"\d+", response.content or "")]
        ordered: list[dict] = []
        used_indexes: set[int] = set()
        for index in order:
            if 0 <= index < len(candidates) and index not in used_indexes:
                ordered.append(candidates[index])
                used_indexes.add(index)
        for index, candidate in enumerate(candidates):
            if index not in used_indexes:
                ordered.append(candidate)
        return ordered
    except Exception as error:
        logger.warning("LLM source reranking failed: %s", error)
        return candidates


async def _select_sources_hybrid(topic: str, candidates: list[dict]) -> list[dict]:
    allowlisted = [candidate for candidate in candidates if candidate.get("allowlisted")]
    non_allowlisted = [candidate for candidate in candidates if not candidate.get("allowlisted")]
    ranked_non_allowlisted = await _rerank_non_allowlisted_sources(topic, non_allowlisted)
    combined = allowlisted + ranked_non_allowlisted
    return combined[:RESEARCH_MAX_DISCOVERED_SOURCES]


def _is_low_signal_snapshot(snapshot: str, refs_count: int) -> bool:
    text = (snapshot or "").strip()
    if not text:
        return True
    if len(text) < 120:
        return True
    if refs_count == 0 and len(text) < 500:
        return True
    return False


def _looks_like_bot_challenge(url: str, snapshot_text: str) -> bool:
    text = f"{url}\n{snapshot_text}".lower()
    markers = [
        "google.com/sorry",
        "unusual traffic",
        "verify you are human",
        "i am not a robot",
        "captcha",
        "cloudflare",
        "attention required",
    ]
    return any(marker in text for marker in markers)


async def _behavior_pause(multiplier: float = 1.0) -> None:
    base = random.uniform(CAMOFOX_BEHAVIOR_MIN_DELAY_SECONDS, CAMOFOX_BEHAVIOR_MAX_DELAY_SECONDS)
    await asyncio.sleep(max(0.2, base * max(0.2, multiplier)))


async def _tavily_search_fallback(query: str) -> tuple[str, list[str]]:
    """Execute a Tavily web search and return (context_text, source_urls)."""
    from tavily import AsyncTavilyClient

    logger.info("Tavily fallback search | query=%r", query)
    _debug_log("TAVILY_START", "query", query)

    try:
        client = AsyncTavilyClient(api_key=TAVILY_API_KEY)
        response = await client.search(
            query=query,
            max_results=RESEARCH_MAX_DISCOVERED_SOURCES,
            search_depth="advanced",
            topic="general",
        )
    except Exception as exc:
        logger.error("Tavily search failed: %s", exc, exc_info=True)
        _debug_log("TAVILY_ERROR", "exception", str(exc))
        return (
            "Research unavailable: Tavily search request failed. "
            "Treat context as insufficient and avoid unsupported claims.",
            [],
        )

    results = response.get("results", [])
    _debug_log("TAVILY_RESULT", "result_count", len(results))

    context_parts: list[str] = []
    discovered_sources: list[str] = []

    for item in results:
        url = item.get("url", "")
        title = item.get("title", "")
        content = item.get("content", "")
        if not url or not content:
            continue
        discovered_sources.append(url)
        context_parts.append(
            f"=== TAVILY RESULT ===\n"
            f"URL: {url}\n"
            f"TITLE: {title}\n"
            f"CONTENT:\n{content}"
        )

    if not context_parts:
        return (
            "No results returned from Tavily search. "
            "Treat context as insufficient and avoid unsupported claims.",
            [],
        )

    context_text = "\n\n".join(context_parts)
    _debug_log("TAVILY_RESULT", "total_sources", len(discovered_sources))
    _debug_log("TAVILY_RESULT", "context_chars", len(context_text))
    logger.info("Tavily fallback | sources=%d context_chars=%d", len(discovered_sources), len(context_text))
    return context_text, discovered_sources


async def perform_camoufox_direct_crawl(
    user_text: str,
    topic: str,
    browser,
    search_query_override: str = "",
) -> dict:
    if (search_query_override or "").strip() and (search_query_override or "").strip().upper() != "NONE":
        search_queries = [_compact_search_query(search_query_override)]
    else:
        search_queries = (await _generate_search_queries(topic=topic, user_text=user_text))[:1]
    logger.info("Research evidence | generated_queries=%s max_used=1", search_queries)
    _debug_log("CAMOUFOX_START", "search_queries_generated", search_queries)

    # Persistent controls to avoid self-inflicted Google bot flags.
    google_guard = GoogleGuard(
        config=GoogleGuardConfig(
            min_interval_seconds=GOOGLE_MIN_INTERVAL_SECONDS,
            cooldown_on_challenge_seconds=GOOGLE_COOLDOWN_ON_CHALLENGE_SECONDS,
        )
    )
    cache = ResearchResultCache(config=CacheConfig(ttl_seconds=GOOGLE_SEARCH_CACHE_TTL_SECONDS))

    context_parts: list[str] = []
    discovered_candidates: list[dict] = []
    seen_urls: set[str] = set()
    bot_detected_count = 0
    reusable_tab_id: str | None = None

    for search_query_idx, search_query in enumerate(search_queries):
        if not await browser.ping():
            logger.warning("Camoufox became unavailable during crawl loop.")
            break

        # Cache hit: skip Google entirely.
        cached = cache.get(search_query)
        if cached:
            cached_sources = cached.get("selected_sources") or []
            if isinstance(cached_sources, list) and cached_sources:
                logger.info("Research evidence | google_cache_hit query=%s sources=%d", search_query, len(cached_sources))
                _debug_log("GOOGLE_CACHE", "hit_query", search_query)
                for item in cached_sources:
                    url = (item or {}).get("url")
                    if not isinstance(url, str) or not url:
                        continue
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)
                    discovered_candidates.append(
                        {
                            "url": url,
                            "title": (item or {}).get("title", ""),
                            "domain": (item or {}).get("domain", _normalize_domain(url)),
                            "allowlisted": bool((item or {}).get("allowlisted")),
                        }
                    )
                continue

        try:
            with file_lock(
                "data/state/google_search.lock",
                timeout_seconds=GOOGLE_SEARCH_LOCK_TIMEOUT_SECONDS,
            ):
                ok, reason = google_guard.can_hit_google()
                if not ok:
                    logger.warning("Research evidence | google_blocked reason=%s query=%s", reason, search_query)
                    context_parts.append(
                        "=== GOOGLE SEARCH SKIPPED ===\n"
                        f"QUERY: {search_query}\n"
                        f"REASON: {reason}\n"
                        "NOTE: Using cache/other sources only to reduce bot-detection risk.\n"
                    )
                    continue

                query_handled = False
                for attempt in range(CAMOFOX_CHALLENGE_MAX_RETRIES + 1):
                    tab_id = reusable_tab_id if CAMOFOX_REUSE_TAB and reusable_tab_id else None
                    if not tab_id:
                        tab_id = await browser.create_tab("https://www.google.com")
                        if not tab_id:
                            logger.warning("Camoufox create_tab failed for query: %s", search_query)
                            break
                        if CAMOFOX_REUSE_TAB:
                            reusable_tab_id = tab_id

                    google_guard.mark_google_hit()
                    if not await browser.search_google(tab_id, search_query):
                        logger.warning("Camoufox search_google failed for query: %s", search_query)
                        if CAMOFOX_REUSE_TAB and reusable_tab_id:
                            await browser.close_tab(reusable_tab_id)
                            reusable_tab_id = None
                        await _behavior_pause(multiplier=1.0 + attempt)
                        continue

                    logger.info(
                        "Research evidence | behavior_pause query=%d/%d attempt=%d/%d",
                        search_query_idx + 1,
                        len(search_queries),
                        attempt + 1,
                        CAMOFOX_CHALLENGE_MAX_RETRIES + 1,
                    )
                    await _behavior_pause(multiplier=1.2 + 0.5 * attempt)

                    first_page = await browser.get_snapshot_page(tab_id)
                    if not first_page:
                        logger.warning("No snapshot data from Camoufox for query: %s", search_query)
                        await _behavior_pause(multiplier=1.0 + attempt)
                        continue

                    page_url = str(first_page.get("url") or "")
                    snapshot_text = str(first_page.get("snapshot") or "")
                    refs_count = int(first_page.get("refsCount") or 0)

                    if _looks_like_bot_challenge(page_url, snapshot_text):
                        bot_detected_count += 1
                        google_guard.mark_challenge(reason=f"serp_challenge:{page_url}")
                        logger.info(
                            "Research evidence | bot_detected query=%s url=%s attempt=%d/%d cooldown=%ss",
                            search_query,
                            page_url,
                            attempt + 1,
                            CAMOFOX_CHALLENGE_MAX_RETRIES + 1,
                            int(GOOGLE_COOLDOWN_ON_CHALLENGE_SECONDS),
                        )
                        if CAMOFOX_REUSE_TAB and reusable_tab_id:
                            await browser.close_tab(reusable_tab_id)
                            reusable_tab_id = None
                        await _behavior_pause(multiplier=2.0 + attempt)
                        continue

                    if _is_low_signal_snapshot(snapshot_text, refs_count):
                        logger.info(
                            "Low-signal snapshot detected (refs=%s, len=%s), retry query=%s url=%s attempt=%d/%d",
                            refs_count,
                            len(snapshot_text),
                            search_query,
                            page_url,
                            attempt + 1,
                            CAMOFOX_CHALLENGE_MAX_RETRIES + 1,
                        )
                        await _behavior_pause(multiplier=1.5 + 0.5 * attempt)
                        continue

                    context_parts.append(
                        f"=== CAMOUFOX SEARCH PAGE ===\n"
                        f"QUERY: {search_query}\n"
                        f"URL: {page_url}\n"
                        f"SNAPSHOT:\n{snapshot_text[:7000]}"
                    )

                    for extracted_url in _extract_urls_from_text(snapshot_text):
                        if extracted_url in seen_urls:
                            continue
                        seen_urls.add(extracted_url)
                        domain = _normalize_domain(extracted_url)
                        is_allowlisted = _is_allowlisted_domain(domain)
                        _debug_log("CAMOUFOX_URL_EXTRACTED", "url", extracted_url, prefix=f"allowlisted={is_allowlisted}")
                        discovered_candidates.append(
                            {
                                "url": extracted_url,
                                "title": "",
                                "domain": domain,
                                "allowlisted": is_allowlisted,
                            }
                        )

                    selected_refs = await _select_refs_with_llm(snapshot_text=snapshot_text, query=search_query, max_refs=4)
                    logger.info(
                        "Research evidence | query=%s refs_selected=%s refs_count=%s",
                        search_query,
                        selected_refs,
                        refs_count,
                    )

                    for ref_id in selected_refs:
                        await _behavior_pause(multiplier=0.8)
                        if not await browser.click(tab_id, ref_id):
                            continue
                        await _behavior_pause(multiplier=1.0)
                        landed_page = await browser.get_snapshot_page(tab_id)
                        if not landed_page:
                            await _behavior_pause(multiplier=0.7)
                            continue

                        landed_url = str(landed_page.get("url") or "")
                        if not landed_url or not _is_probable_article_url(landed_url):
                            continue

                        if _looks_like_bot_challenge(landed_url, str(landed_page.get("snapshot") or "")):
                            bot_detected_count += 1
                            google_guard.mark_challenge(reason=f"click_challenge:{landed_url}")
                            logger.info("Research evidence | bot_detected clicked_ref=%s url=%s", ref_id, landed_url)
                            continue

                        if landed_url in seen_urls:
                            continue

                        seen_urls.add(landed_url)
                        domain = _normalize_domain(landed_url)
                        ref_title = next(
                            (label for candidate_ref, label in _extract_refs_from_snapshot(snapshot_text) if candidate_ref == ref_id),
                            "",
                        )
                        discovered_candidates.append(
                            {
                                "url": landed_url,
                                "title": ref_title,
                                "domain": domain,
                                "allowlisted": _is_allowlisted_domain(domain),
                            }
                        )

                    collected_segments = [snapshot_text]
                    next_offset = first_page.get("nextOffset")
                    has_more = bool(first_page.get("hasMore"))

                    for _ in range(2):
                        if not has_more or next_offset is None:
                            break
                        extra_page = await browser.get_snapshot_page(tab_id, offset=int(next_offset))
                        if not extra_page:
                            break
                        collected_segments.append(str(extra_page.get("snapshot") or ""))
                        has_more = bool(extra_page.get("hasMore"))
                        next_offset = extra_page.get("nextOffset")

                    merged_snapshot = "\n\n".join(segment for segment in collected_segments if segment).strip()
                    if merged_snapshot:
                        context_parts.append(
                            f"=== CAMOUFOX SOURCE ===\n"
                            f"URL: {page_url}\n"
                            f"SNAPSHOT:\n{merged_snapshot[:6000]}"
                        )
                    query_handled = True
                    break

        except TimeoutError as error:
            logger.warning("Research evidence | google_lock_timeout query=%s error=%s", search_query, error)
            context_parts.append(
                "=== GOOGLE SEARCH SKIPPED ===\n"
                f"QUERY: {search_query}\n"
                "REASON: google_search_worker_busy (lock timeout)\n"
            )
            continue


        if not query_handled:
            logger.info("Research evidence | query_exhausted_after_retries query=%s", search_query)

    if reusable_tab_id:
        await browser.close_tab(reusable_tab_id)

    logger.info("Research evidence | discovered_url_candidates=%s", len(discovered_candidates))
    _debug_log("CAMOUFOX_URL_DISCOVERY", "all_discovered_candidates_count", len(discovered_candidates))
    _debug_log("CAMOUFOX_URL_DISCOVERY", "all_discovered_candidates", [c["url"] for c in discovered_candidates])

    if not discovered_candidates:
        return {
            "context": (
                "No crawlable sources discovered from Camoufox search flow. "
                "Treat context as insufficient and avoid unsupported claims."
            ),
            "sources": [],
            "discovered_urls": [],
            "bot_detected_count": bot_detected_count,
        }

    selected_sources = await _select_sources_hybrid(topic=topic, candidates=discovered_candidates)
    selected_urls = [source["url"] for source in selected_sources]
    logger.info("Research evidence | selected_urls=%s", selected_urls)
    _debug_log("CAMOUFOX_URL_SELECTION", "selected_sources_count", len(selected_sources))
    _debug_log("CAMOUFOX_URL_SELECTION", "selected_urls", selected_urls)
    for i, source in enumerate(selected_sources):
        _debug_log(
            "CAMOUFOX_URL_SELECTION",
            f"selected_source_{i}",
            {"url": source["url"], "domain": source.get("domain"), "allowlisted": source.get("allowlisted")},
        )

    if selected_sources:
        try:
            cache.set(
                search_queries[0] if search_queries else topic,
                selected_sources=selected_sources,
                meta={"topic": topic, "bot_detected_count": bot_detected_count},
            )
        except Exception as error:
            logger.warning("Failed to write google search cache: %s", error)

    crawled_urls: list[str] = []
    total_crawled_chars = 0
    for source in selected_sources:
        source_url = source["url"]
        source_domain = source.get("domain", "")
        source_title = source.get("title", "")
        logger.info("Crawl4AI → crawling article: %s", source_url)
        _debug_log("CRAWL4AI_FETCH", "source_url", source_url)
        _debug_log("CRAWL4AI_FETCH", "source_domain", source_domain)
        _debug_log("CRAWL4AI_FETCH", "source_title", source_title)
        article_markdown = await crawl_url_to_markdown(source_url)
        if not article_markdown:
            logger.info("Research evidence | crawl_empty url=%s", source_url)
            _debug_log("CRAWL4AI_RESULT", "crawl_failed_empty", source_url)
            continue

        crawled_urls.append(source_url)
        total_crawled_chars += len(article_markdown)
        _debug_log("CRAWL4AI_RESULT", "crawl_successful_url", source_url)
        _debug_log("CRAWL4AI_RESULT", "crawl_content_length", len(article_markdown), prefix="chars")
        _debug_log("CRAWL4AI_RESULT", "crawl_content_sample", article_markdown[:1200], prefix="first 1200 chars")
        context_parts.append(
            f"=== CRAWL4AI ARTICLE ===\n"
            f"URL: {source_url}\n"
            f"DOMAIN: {source_domain}\n"
            f"TITLE_HINT: {source_title}\n"
            f"CONTENT:\n{article_markdown}"
        )

    logger.info(
        "Research evidence | crawled_urls=%s total_crawled_chars=%s",
        len(crawled_urls),
        total_crawled_chars,
    )
    _debug_log("CRAWL4AI_SUMMARY", "total_crawled_urls", len(crawled_urls))
    _debug_log("CRAWL4AI_SUMMARY", "total_crawled_chars", total_crawled_chars)
    _debug_log("CRAWL4AI_SUMMARY", "crawled_urls", crawled_urls)

    if not context_parts:
        context_text = (
            "No crawlable content retrieved from Camoufox + Crawl4AI pipeline. "
            "Treat context as insufficient and avoid unsupported claims."
        )
    else:
        context_text = "\n\n".join(context_parts)

    return {
        "context": context_text,
        "sources": crawled_urls or selected_urls,
        "discovered_urls": selected_urls,
        "bot_detected_count": bot_detected_count,
    }

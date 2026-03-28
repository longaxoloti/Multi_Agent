"""
Request Router — Entry point for all user commands.

Routes:
  /save   → KnowledgeService (manual.* schema only)
  /get    → KnowledgeService (manual.* schema)
  /delete → KnowledgeService (manual.* schema)
  /search → Multi-source retrieval with priority: manual > profile > projects > skills > knowledge
  /list   → KnowledgeService (manual.* schema)
  /skill  → SkillService (skills.* schema)
  /profile → UserProfileService (profile.* schema)
  /project → ProjectService (projects.* schema)

Non-command messages pass through to the orchestrator.
"""

import logging
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage

from graph.state import AgentState
from storage.knowledge_service import KnowledgeRequest, KnowledgeService, parse_knowledge_request

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-initialized service singletons
# ---------------------------------------------------------------------------
_knowledge_service: Optional[KnowledgeService] = None
_skill_service = None
_profile_service = None
_project_service = None
_bookmark_service = None
_repo = None


def _get_repo():
    """Get or create the shared AgentDBRepository."""
    global _repo
    if _repo is None:
        from storage.trusted_db import AgentDBRepository
        _repo = AgentDBRepository()
        _repo.initialize()
    return _repo


def _get_knowledge_service() -> KnowledgeService:
    global _knowledge_service
    if _knowledge_service is None:
        _knowledge_service = KnowledgeService()
    return _knowledge_service


def _get_skill_service():
    global _skill_service
    if _skill_service is None:
        from storage.skill_service import SkillService
        repo = _get_repo()
        _skill_service = SkillService(
            repo._session_factory,
            is_pg=repo.engine.dialect.name == "postgresql",
        )
    return _skill_service


def _get_profile_service():
    global _profile_service
    if _profile_service is None:
        from storage.user_profile_service import UserProfileService
        repo = _get_repo()
        _profile_service = UserProfileService(
            repo._session_factory,
            is_pg=repo.engine.dialect.name == "postgresql",
        )
    return _profile_service


def _get_project_service():
    global _project_service
    if _project_service is None:
        from storage.project_service import ProjectService
        repo = _get_repo()
        _project_service = ProjectService(repo._session_factory)
    return _project_service


def _get_bookmark_service():
    global _bookmark_service
    if _bookmark_service is None:
        from storage.bookmark_service import BookmarkService
        repo = _get_repo()
        _bookmark_service = BookmarkService(
            repo._session_factory,
            is_pg=repo.engine.dialect.name == "postgresql",
        )
    return _bookmark_service


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def _render_item(item: dict) -> str:
    title = item.get("title") or ""
    content = (item.get("content") or item.get("fact_value") or item.get("chunk_text") or "").strip()
    if len(content) > 400:
        content = content[:400] + "..."
    prefix = f"[{item.get('id', 'unknown')}]"
    if title:
        return f"{prefix} {title}\n{content}"
    return f"{prefix} {content}"


def _render_search_result(item: dict, source_label: str) -> str:
    """Render a search result from any source with source label."""
    title = item.get("title") or item.get("fact_key") or ""
    content = (
        item.get("content")
        or item.get("fact_value")
        or item.get("chunk_text")
        or item.get("description")
        or ""
    ).strip()
    if len(content) > 300:
        content = content[:300] + "..."
    distance = item.get("distance")
    dist_str = f" (relevance: {1 - distance:.2f})" if distance is not None else ""

    lines = [f"📌 [{source_label}] {title}{dist_str}"]
    if content:
        lines.append(f"   {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def _handle_knowledge_request(chat_id: str, req: KnowledgeRequest) -> str:
    """Handle /save, /get, /delete, /list — manual schema only."""
    service = _get_knowledge_service()

    if req.action == "save":
        if not req.content:
            return "Not enough content to store. Use: /save <content>"
        saved = service.save(chat_id=chat_id, content=req.content, category=req.category)
        return (
            f"Data saved successfully.\n"
            f"- id: {saved['record_id']}\n"
            f"- category: {saved['category']}\n"
            f"- storage: {'PostgreSQL + pgvector' if saved['stored_in_vector'] else 'PostgreSQL'}"
        )

    if req.action == "get":
        if not req.record_id:
            return "Missing record id. Use: /get <id>"
        item = service.get(chat_id=chat_id, record_id=req.record_id)
        if not item:
            return f"Data not found with id: {req.record_id}"
        return _render_item(item)

    if req.action == "search":
        if not req.query:
            return "Missing search query. Use: /search <query>"
        return _handle_multi_source_search(chat_id=chat_id, query=req.query, limit=req.limit)

    if req.action == "list":
        rows = service.list_recent(chat_id=chat_id, limit=req.limit)
        if not rows:
            return "No recent data found."
        lines = ["Recent data list:"]
        for row in rows:
            lines.append(_render_item(row))
        return "\n\n".join(lines)

    if req.action == "delete":
        if not req.record_id:
            return "Missing record id. Use: /delete <id>"
        deleted = service.delete(chat_id=chat_id, record_id=req.record_id)
        if deleted:
            return f"Data deleted successfully: {req.record_id}"
        return f"Failed to delete or data not found with id: {req.record_id}"

    return "Request not supported."


def _handle_multi_source_search(
    *,
    chat_id: str,
    query: str,
    limit: int = 5,
) -> str:
    """
    Multi-source search with retrieval priority:
      manual > profile > projects > skills > knowledge

    Collects results from each source in order, deduplicates, and caps at
    ``limit`` total results.
    """
    all_results: list[dict] = []
    per_source_limit = max(2, limit)

    # 1. Manual (highest priority — user-saved data)
    try:
        service = _get_knowledge_service()
        manual_rows = service.search(
            chat_id=chat_id, query=query, limit=per_source_limit
        )
        for row in manual_rows:
            row["_source"] = "Saved"
            all_results.append(row)
    except Exception as exc:
        logger.debug("Manual search failed: %s", exc)

    # 2. Profile
    try:
        profile_svc = _get_profile_service()
        # Use 'default' as user_id since we don't have user auth yet
        profile_rows = profile_svc.search_profile(
            "default", query, limit=per_source_limit
        )
        for row in profile_rows:
            row["_source"] = "Profile"
            all_results.append(row)
    except Exception as exc:
        logger.debug("Profile search failed: %s", exc)

    # 3. Projects
    try:
        project_svc = _get_project_service()
        projects = project_svc.list_projects()
        for proj in projects[:3]:  # search top 3 active projects
            facts = project_svc.get_project_facts(proj["id"])
            query_lower = query.lower()
            for fact in facts:
                searchable = f"{fact['fact_key']} {fact['fact_value']}".lower()
                if query_lower in searchable:
                    fact["_source"] = f"Project:{proj['project_name']}"
                    fact["distance"] = 0.1  # keyword match = high relevance
                    all_results.append(fact)
    except Exception as exc:
        logger.debug("Project search failed: %s", exc)

    # 4. Skills
    try:
        skill_svc = _get_skill_service()
        skill_rows = skill_svc.search_skills(query, limit=per_source_limit)
        for row in skill_rows:
            row["_source"] = "Skill"
            all_results.append(row)
    except Exception as exc:
        logger.debug("Skill search failed: %s", exc)

    # 5. Knowledge (URL bookmarks — lowest priority)
    try:
        bookmark_svc = _get_bookmark_service()
        bookmark_rows = bookmark_svc.search_bookmarks(query, limit=per_source_limit)
        for row in bookmark_rows:
            row["_source"] = "Bookmark"
            all_results.append(row)
    except Exception as exc:
        logger.debug("Bookmark search failed: %s", exc)

    if not all_results:
        return "No matching data found across all sources."

    # Sort by distance (lower = more relevant), keep source priority via stable sort
    all_results.sort(key=lambda r: r.get("distance", 0.5))

    # Cap at limit
    display = all_results[:limit]

    lines = [f"🔍 Search results for: **{query}** ({len(display)} of {len(all_results)} total)"]
    for item in display:
        source_label = item.get("_source", "Unknown")
        lines.append(_render_search_result(item, source_label))

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Skill-specific command handler
# ---------------------------------------------------------------------------

def _handle_skill_command(args: str) -> str:
    """Handle /skill <subcommand>."""
    parts = args.strip().split(None, 1)
    if not parts:
        return (
            "Skill commands:\n"
            "  /skill list       — List all active skills\n"
            "  /skill search <q> — Search skills\n"
            "  /skill get <id>   — Get skill content\n"
            "  /skill ingest <path> — Ingest skill from file"
        )

    sub = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""
    svc = _get_skill_service()

    if sub == "list":
        skills = svc.get_active_skills()
        if not skills:
            return "No active skills found."
        lines = [f"📚 Active Skills ({len(skills)}):"]
        for s in skills:
            lines.append(f"  • [{s['source_id'][:8]}] {s['title']} (v{s['version_no']})")
        return "\n".join(lines)

    if sub == "search":
        if not rest:
            return "Usage: /skill search <query>"
        results = svc.search_skills(rest)
        if not results:
            return f"No skills matching: {rest}"
        lines = [f"🔍 Skill search: {rest}"]
        for r in results:
            lines.append(f"  • {r['title']} (v{r['version_no']}) — chunk {r['chunk_index']}")
            chunk_preview = r["chunk_text"][:200] + "..." if len(r["chunk_text"]) > 200 else r["chunk_text"]
            lines.append(f"    {chunk_preview}")
        return "\n".join(lines)

    if sub == "get":
        if not rest:
            return "Usage: /skill get <source_id>"
        content = svc.get_skill_content(rest.strip())
        if not content:
            return f"Skill not found: {rest}"
        return (
            f"📄 {content['title']} (v{content['version_no']})\n"
            f"Status: {content['status']} | Confidence: {content['confidence']}\n"
            f"---\n{content['content'][:2000]}"
        )

    if sub == "ingest":
        if not rest:
            return "Usage: /skill ingest <file_path>"
        try:
            result = svc.ingest_from_markdown(rest.strip())
            return (
                f"Skill ingested: {result['action']}\n"
                f"- source_id: {result['source_id']}\n"
                f"- version: {result.get('version_no', 'N/A')}\n"
                f"- chunks: {result.get('chunks_created', 'N/A')}"
            )
        except FileNotFoundError as e:
            return str(e)

    return f"Unknown skill subcommand: {sub}. Use /skill for help."


# ---------------------------------------------------------------------------
# Profile-specific command handler
# ---------------------------------------------------------------------------

def _handle_profile_command(args: str, chat_id: str) -> str:
    """Handle /profile <subcommand>."""
    parts = args.strip().split(None, 1)
    if not parts:
        return (
            "Profile commands:\n"
            "  /profile show     — Show my profile\n"
            "  /profile search <q> — Search profile facts\n"
            "  /profile ingest <path> — Ingest from USER.md"
        )

    sub = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""
    svc = _get_profile_service()
    user_id = "default"  # placeholder until user auth

    if sub == "show":
        facts = svc.get_profile(user_id)
        if not facts:
            return "No profile facts found."
        lines = [f"👤 Profile ({len(facts)} facts):"]
        for f in facts:
            sensitive = " 🔒" if f.get("is_sensitive") else ""
            lines.append(f"  • {f['fact_key']}: {f['fact_value']}{sensitive}")
        return "\n".join(lines)

    if sub == "search":
        if not rest:
            return "Usage: /profile search <query>"
        results = svc.search_profile(user_id, rest)
        if not results:
            return f"No profile facts matching: {rest}"
        lines = [f"🔍 Profile search: {rest}"]
        for r in results:
            lines.append(f"  • {r['fact_key']}: {r['fact_value']} (confidence: {r['confidence']:.2f})")
        return "\n".join(lines)

    if sub == "ingest":
        if not rest:
            return "Usage: /profile ingest <path>"
        try:
            result = svc.ingest_from_markdown(rest.strip(), user_id=user_id)
            return (
                f"Profile ingested: {result['action']}\n"
                f"- facts: {result.get('facts_created', 'N/A')}\n"
                f"- version: {result.get('version_no', 'N/A')}"
            )
        except FileNotFoundError as e:
            return str(e)

    return f"Unknown profile subcommand: {sub}. Use /profile for help."


# ---------------------------------------------------------------------------
# Project-specific command handler
# ---------------------------------------------------------------------------

def _handle_project_command(args: str) -> str:
    """Handle /project <subcommand>."""
    parts = args.strip().split(None, 1)
    if not parts:
        return (
            "Project commands:\n"
            "  /project list     — List registered projects\n"
            "  /project info <path> — Show project details\n"
            "  /project register <name> <path> — Register new project"
        )

    sub = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""
    svc = _get_project_service()

    if sub == "list":
        projects = svc.list_projects()
        if not projects:
            return "No projects registered."
        lines = [f"📁 Projects ({len(projects)}):"]
        for p in projects:
            lang = f" ({p['language']})" if p.get("language") else ""
            lines.append(f"  • {p['project_name']}{lang} → {p['repo_path']}")
        return "\n".join(lines)

    if sub == "info":
        if not rest:
            return "Usage: /project info <repo_path>"
        proj = svc.get_project_by_path(rest.strip())
        if not proj:
            return f"Project not found at: {rest}"
        facts = svc.get_project_facts(proj["id"])
        lines = [
            f"📁 {proj['project_name']} ({proj.get('language', 'unknown')})",
            f"   Path: {proj['repo_path']}",
            f"   Status: {proj['status']}",
            f"   Facts ({len(facts)}):",
        ]
        for f in facts[:20]:
            lines.append(f"     • {f['fact_key']}: {f['fact_value']}")
        if len(facts) > 20:
            lines.append(f"     ... and {len(facts) - 20} more")
        return "\n".join(lines)

    if sub == "register":
        reg_parts = rest.strip().split(None, 1)
        if len(reg_parts) < 2:
            return "Usage: /project register <name> <path>"
        name, path = reg_parts
        result = svc.register_project(project_name=name, repo_path=path)
        return f"Project {result['action']}: {result['project_name']} ({result['project_id'][:8]})"

    return f"Unknown project subcommand: {sub}. Use /project for help."


# ---------------------------------------------------------------------------
# Extended command parser
# ---------------------------------------------------------------------------

def _parse_extended_command(raw: str) -> Optional[tuple[str, str]]:
    """
    Parse extended commands: /skill, /profile, /project.

    Returns (command, args) or None.
    """
    import re
    match = re.match(r"^/(skill|profile|project)\s*(.*)?$", raw.strip(), re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).lower(), (match.group(2) or "").strip()
    return None


# ---------------------------------------------------------------------------
# Main router node
# ---------------------------------------------------------------------------

async def request_router_node(state: AgentState) -> dict:
    user_message = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    if not user_message:
        return {"knowledge_action": "none"}

    raw_text = user_message.content.strip()
    chat_id = str(state.get("chat_id") or "")

    # 1. Check extended commands first (/skill, /profile, /project)
    ext = _parse_extended_command(raw_text)
    if ext:
        cmd, args = ext
        try:
            if cmd == "skill":
                answer = _handle_skill_command(args)
            elif cmd == "profile":
                answer = _handle_profile_command(args, chat_id)
            elif cmd == "project":
                answer = _handle_project_command(args)
            else:
                answer = "Unsupported command."
            return {
                "messages": [AIMessage(content=answer)],
                "routing_decision": "CHAT",
                "knowledge_action": "handled",
            }
        except Exception as exc:
            logger.error("Extended command handler failed: %s", exc, exc_info=True)
            return {
                "messages": [AIMessage(content=f"Command failed: {exc}")],
                "routing_decision": "CHAT",
                "knowledge_action": "handled",
            }

    # 2. Standard knowledge commands (/save, /get, /search, /list, /delete)
    req = parse_knowledge_request(raw_text)
    if not req:
        return {"knowledge_action": "none"}

    try:
        answer = _handle_knowledge_request(chat_id=chat_id, req=req)
        return {
            "messages": [AIMessage(content=answer)],
            "routing_decision": "CHAT",
            "knowledge_action": "handled",
        }
    except Exception as exc:
        logger.error("Knowledge request handler failed: %s", exc, exc_info=True)
        return {
            "messages": [AIMessage(content="Not able to process the save/retrieve data request at the moment.")],
            "routing_decision": "CHAT",
            "knowledge_action": "handled",
        }

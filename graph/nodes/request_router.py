import logging
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage

from graph.state import AgentState
from storage.knowledge_service import KnowledgeRequest, KnowledgeService, parse_knowledge_request

logger = logging.getLogger(__name__)

_knowledge_service: Optional[KnowledgeService] = None


def _get_knowledge_service() -> KnowledgeService:
    global _knowledge_service
    if _knowledge_service is None:
        _knowledge_service = KnowledgeService()
    return _knowledge_service


def _render_item(item: dict) -> str:
    title = item.get("title") or ""
    content = (item.get("content") or "").strip()
    if len(content) > 400:
        content = content[:400] + "..."
    prefix = f"[{item.get('id', 'unknown')}]"
    if title:
        return f"{prefix} {title}\n{content}"
    return f"{prefix} {content}"


def _handle_knowledge_request(chat_id: str, req: KnowledgeRequest) -> str:
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
        rows = service.search(
            chat_id=chat_id,
            query=req.query,
            limit=req.limit,
            category=req.category,
        )
        if not rows:
            return "No matching data found."
        lines = ["Search results:"]
        for row in rows:
            lines.append(_render_item(row))
        return "\n\n".join(lines)

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


async def request_router_node(state: AgentState) -> dict:
    user_message = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    if not user_message:
        return {"knowledge_action": "none"}

    req = parse_knowledge_request(user_message.content)
    if not req:
        return {"knowledge_action": "none"}

    chat_id = str(state.get("chat_id") or "")
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

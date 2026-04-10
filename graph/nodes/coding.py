import logging
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from main.config import OLLAMA_CODER_MODEL, OLLAMA_ORCHESTRATOR_MODEL
from graph.state import AgentState
from graph.llm_router import get_llm
from tools.ollama_manager import load_context, unload_model
from tools.workspace_priming import build_system_prompt
import re

logger = logging.getLogger(__name__)
_knowledge_service = None


def _get_knowledge_service():
    global _knowledge_service
    if _knowledge_service is None:
        from storage.knowledge_service import KnowledgeService
        _knowledge_service = KnowledgeService()
    return _knowledge_service


def _persist_coding_result(
    *,
    topic: str,
    tasks: list[str],
    result_text: str,
    sources: list[str],
    session_id: str,
) -> int:
    if not result_text.strip():
        return 0

    try:
        knowledge_service = _get_knowledge_service()
    except Exception as exc:
        logger.debug("Knowledge service unavailable; skip coding persistence: %s", exc)
        return 0

    task_preview = " | ".join((tasks or [])[:4])[:400]
    result_preview = (result_text or "").strip()[:1200]
    payload = (
        f"topic={topic}\n"
        f"tasks={task_preview}\n"
        f"sources={', '.join(sources) if sources else 'none'}\n"
        f"result={result_preview}"
    )
    try:
        knowledge_service.save(
            chat_id=session_id,
            content=payload,
            category="coding_output",
            title=topic[:255],
            metadata={"sources": sources, "session_id": session_id},
        )
        return 1
    except Exception as exc:
        logger.debug("Failed to persist coding output to unified knowledge: %s", exc)
        return 0

_CODER_SYSTEM_PROMPT = """\
You are the Coder Agent — a specialist software engineer.
Write correct, well-commented, production-ready code.
Explain your approach briefly before the code block.
List any files you create or modify as sources at the end.
Respond in the same language as the user message."""


async def coding_node(state: AgentState) -> dict:
    logger.info("--- CODING NODE ---")

    await unload_model(OLLAMA_ORCHESTRATOR_MODEL)
    session_id = state.get("session_id", "default")
    ctx = load_context(session_id)
    user_text = ctx.get("user_message", "")
    tasks = ctx.get("tasks", state.get("tasks", []))
    topic = ctx.get("topic", state.get("topic", ""))
    step_index = ctx.get("step_index", 0)

    task_description = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(tasks))
    user_prompt = (
        f"Topic: {topic}\n\n"
        f"Tasks assigned to you:\n{task_description}\n\n"
        f"Original user message:\n{user_text}"
    )

    system_prompt = build_system_prompt(_CODER_SYSTEM_PROMPT, model_role="coder")
    llm = get_llm(task_type="code", temperature=0.1)
    try:
        response = await llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        result_text = response.content.strip()
    except Exception as exc:
        logger.error("Coding node LLM call failed: %s", exc, exc_info=True)
        result_text = "An error occurred while generating the code solution."

    file_refs = re.findall(r"`([^`]+\.[a-zA-Z]{1,6})`", result_text)
    sources = list(dict.fromkeys(file_refs))

    persisted = _persist_coding_result(
        topic=topic,
        tasks=tasks,
        result_text=result_text,
        sources=sources,
        session_id=session_id,
    )
    if persisted:
        logger.info("Persisted coding session output into %d project(s).", persisted)

    task_result = {
        "step_index": step_index,
        "model": OLLAMA_CODER_MODEL,
        "result": result_text,
        "sources": sources,
    }

    return {
        "task_results": [task_result],
        "active_model": OLLAMA_CODER_MODEL,
    }

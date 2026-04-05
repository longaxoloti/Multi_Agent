import logging
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from main.config import OLLAMA_CODER_MODEL, OLLAMA_ORCHESTRATOR_MODEL
from graph.state import AgentState
from graph.llm_router import get_llm
from tools.ollama_manager import load_context, unload_model
from tools.workspace_priming import build_system_prompt
import re

logger = logging.getLogger(__name__)

_repo = None
_project_service = None


def _get_repo():
    global _repo
    if _repo is None:
        from storage.trusted_db import AgentDBRepository
        _repo = AgentDBRepository()
        _repo.initialize()
    return _repo


def _get_project_service():
    global _project_service
    if _project_service is None:
        from storage.project_service import ProjectService
        repo = _get_repo()
        _project_service = ProjectService(repo._session_factory)
    return _project_service


def _pick_target_projects(file_sources: list[str], projects: list[dict]) -> list[dict]:
    if not projects:
        return []

    normalized_sources = [s.lower() for s in file_sources if isinstance(s, str)]
    matched: list[dict] = []
    for project in projects:
        repo_path = (project.get("repo_path") or "").lower().rstrip("/")
        if not repo_path:
            continue
        if any(src.lower().startswith(repo_path) or repo_path in src.lower() for src in normalized_sources):
            matched.append(project)

    if matched:
        return matched[:3]

    # Fallback: if source paths are relative, attach to at most first active project.
    return projects[:1]


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
        project_service = _get_project_service()
        projects = project_service.list_projects(status="active")
    except Exception as exc:
        logger.debug("Project service unavailable; skip coding persistence: %s", exc)
        return 0

    targets = _pick_target_projects(sources, projects)
    if not targets:
        return 0

    persisted = 0
    task_preview = " | ".join((tasks or [])[:4])[:400]
    result_preview = (result_text or "").strip()[:1200]
    for project in targets:
        project_id = project.get("id")
        if not project_id:
            continue
        try:
            project_service.save_fact(
                project_id=project_id,
                fact_key="coding.session_output",
                fact_value=f"topic={topic}\ntasks={task_preview}\nresult={result_preview}",
                session_id=session_id,
                source_type="coding_session",
                confidence=0.78,
            )
            persisted += 1
        except Exception as exc:
            logger.debug("Failed to persist coding fact for project %s: %s", project_id, exc)
    return persisted

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

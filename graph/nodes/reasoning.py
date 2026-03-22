import logging
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from graph.state import AgentState
from graph.llm_router import get_llm
from main.config import (
    OLLAMA_ORCHESTRATOR_MODEL,
    OLLAMA_RESEARCH_MODEL,
)
from tools.workspace_priming import build_system_prompt
from tools.ollama_manager import unload_model, load_context

logger = logging.getLogger(__name__)

_REASONING_SYSTEM_PROMPT = """You are a deep-reasoning specialist assistant.
Approach the question methodically: break it into sub-problems, reason through each one,
then deliver a clear, well-structured answer.
Respond in the same language as the user."""


async def reasoning_node(state: AgentState) -> dict:
    logger.info("--- REASONING NODE ---")

    session_id = state.get("session_id", "default")
    ctx = load_context(session_id)
    user_text = ctx.get("user_message", "")

    if not user_text:
        user_message = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
        )
        user_text = user_message.content if user_message else ""

    logger.info("Routing to REASONING path (%s)", OLLAMA_RESEARCH_MODEL)
    await unload_model(OLLAMA_ORCHESTRATOR_MODEL)

    tasks = ctx.get("tasks", state.get("tasks", [])) or []
    step_index = ctx.get("step_index", 0)

    task_description = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(tasks))
    prompt = (
        f"Tasks:\n{task_description}\n\nUser message:\n{user_text}"
        if task_description
        else user_text
    )

    system_prompt = build_system_prompt(_REASONING_SYSTEM_PROMPT, model_role="researcher")
    llm = get_llm(task_type="research", temperature=0.4)

    try:
        response = await llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=prompt)])
    except Exception as e:
        logger.error(f"Error in reasoning node: {e}", exc_info=True)
        response = AIMessage(content="Sorry, I encountered an error while reasoning.")

    result_text = (response.content or "").strip()
    task_result = {
        "step_index": step_index,
        "model": OLLAMA_RESEARCH_MODEL,
        "result": result_text,
        "sources": [],
    }

    return {
        "task_results": [task_result],
        "active_model": OLLAMA_RESEARCH_MODEL,
    }

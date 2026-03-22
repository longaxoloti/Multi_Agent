import logging
from langchain_core.messages import AIMessage
from graph.state import AgentState
from tools.ollama_manager import load_context
from main.config import OLLAMA_ORCHESTRATOR_MODEL

logger = logging.getLogger(__name__)

async def briefing_node(state: AgentState) -> dict:
    logger.info("--- BRIEFING NODE ---")

    session_id = state.get("session_id", "default")
    ctx = load_context(session_id)
    step_index = ctx.get("step_index", 0)

    response = AIMessage(
        content="[Daily Briefing] The system has been optimized for targeted RSS search queries. This is a placeholder daily briefing response."
    )

    task_result = {
        "step_index": step_index,
        "model": "BRIEFING",
        "result": (response.content or "").strip(),
        "sources": [],
    }

    return {
        "task_results": [task_result],
        "active_model": "",
    }


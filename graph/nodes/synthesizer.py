import logging
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from config import OLLAMA_ORCHESTRATOR_MODEL
from graph.state import AgentState
from graph.llm_router import get_llm
from tools.ollama_manager import clear_context, load_context, unload_model
from tools.workspace_priming import build_system_prompt
from langchain_core.messages import HumanMessage as HM

logger = logging.getLogger(__name__)

_SYNTHESIZER_PROMPT = """\
You are synthesising the results from your Worker Agents into a single coherent reply for the user.

Original user message: {user_message}
Topic: {topic}
Tasks that were executed:
{tasks}

Worker results:
{worker_results}

Instructions:
- Write a clear, well-structured final answer that directly addresses the user's message.
- Incorporate all relevant information from the worker results.
- Include source URLs or file references if provided by workers.
- Prioritize facts present in worker results and sources.
- If evidence is incomplete, mention uncertainty briefly and provide a best-effort synthesis.
- Do NOT mention internal model names, task lists, or implementation details to the user.
- Respond in the same language as the user message.
"""


def _format_worker_results(task_results: list[dict]) -> str:
    parts = []
    for i, tr in enumerate(task_results, 1):
        model = tr.get("model", "unknown")
        result = tr.get("result", "")
        sources = tr.get("sources", [])
        part = f"[Result {i} from {model}]\n{result}"
        if sources:
            part += "\nSources: " + ", ".join(str(s) for s in sources)
        parts.append(part)
    return "\n\n---\n\n".join(parts) if parts else "(no results)"


async def synthesizer_node(state: AgentState) -> dict:
    logger.info("--- SYNTHESIZER NODE ---")

    session_id = state.get("session_id", "default")
    active_model = state.get("active_model", "")
    if active_model and active_model != OLLAMA_ORCHESTRATOR_MODEL:
        await unload_model(active_model)

    ctx = load_context(session_id)
    user_text = ctx.get("user_message", "")
    topic = ctx.get("topic", state.get("topic", ""))
    tasks = ctx.get("tasks", state.get("tasks", []))

    if not user_text:
        hm = next((m for m in reversed(state["messages"]) if isinstance(m, HM)), None)
        user_text = hm.content if hm else ""

    task_results = state.get("task_results", [])
    worker_results_text = _format_worker_results(task_results)
    tasks_text = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(tasks))

    synthesis_prompt = _SYNTHESIZER_PROMPT.format(
        user_message=user_text,
        topic=topic,
        tasks=tasks_text or "(none)",
        worker_results=worker_results_text,
    )

    full_system = build_system_prompt(
        "Follow the workspace instruction pack and synthesize worker outputs for the final user reply.",
        model_role="orchestrator",
    )
    llm = get_llm(task_type="orchestrator", temperature=0.5)

    try:
        response = await llm.ainvoke(
            [SystemMessage(content=full_system), HumanMessage(content=synthesis_prompt)]
        )
        final_reply = response.content.strip()
    except Exception as exc:
        logger.error("Synthesizer LLM call failed: %s", exc, exc_info=True)
        final_reply = worker_results_text

    clear_context(session_id)

    return {
        "messages": [AIMessage(content=final_reply)],
        "active_model": OLLAMA_ORCHESTRATOR_MODEL,
    }

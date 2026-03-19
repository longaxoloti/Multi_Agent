import json
import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from main.config import OLLAMA_CLASSIFIER_MODEL, OLLAMA_ORCHESTRATOR_MODEL
from graph.state import AgentState
from graph.llm_router import get_llm
from tools.ollama_manager import clear_context, save_context, unload_model
from tools.workspace_priming import build_system_prompt

logger = logging.getLogger(__name__)

_ORCHESTRATOR_TASK_PROMPT = """\
You are given:
- intent: {intent}
- user message: {user_message}

Your job:
1. Write a numbered list of concrete sub-tasks needed to fulfil the user's request.
2. On the LAST line, write exactly one of these routing decisions:
   ROUTE: RESEARCH
   ROUTE: CODING
   ROUTE: REASONING
   ROUTE: CHAT

Use RESEARCH if a web lookup is needed.
Use CODING if any coding/scripting work is required.
Use REASONING if the topic is complex but requires no web data.
Use CHAT if it is a simple or conversational request you can answer directly.

Output format:
TASKS:
1. <first task>
2. <second task>
...
ROUTE: <DECISION>

If ROUTE is CHAT, add this section at the end:
ANSWER:
<your final user-facing answer in the same language as the user>
"""


_ORCHESTRATOR_CHAT_PROMPT = """\
You are given:
- intent: CHAT
- user message: {user_message}

Your job:
1. Write a numbered list of concrete sub-tasks needed to fulfil the user's request.
2. On the LAST line, write exactly one of these routing decisions:
    ROUTE: RESEARCH
    ROUTE: CODING
    ROUTE: REASONING
    ROUTE: CHAT

Output format:
TASKS:
1. <first task>
2. <second task>
...
ROUTE: <DECISION>

If ROUTE is CHAT, add this section at the end:
ANSWER:
<your final user-facing answer in the same language as the user>
"""


async def orchestrator_node(state: AgentState) -> dict:
    logger.info("--- ORCHESTRATOR NODE ---")

    # ── 1. Unload classifier to free RAM ─────────────────────────────────
    await unload_model(OLLAMA_CLASSIFIER_MODEL)

    # ── 2. Gather inputs ─────────────────────────────────────────────────
    user_message = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
    )
    user_text = user_message.content if user_message else ""
    intent = state.get("intent", "CHAT")
    topic = state.get("topic", "")
    session_id = state.get("session_id", "default")

    # ── 3. Build prompt ───────────────────────────────────────────────────
    if intent == "CHAT":
        task_prompt = _ORCHESTRATOR_CHAT_PROMPT.format(user_message=user_text)
    else:
        task_prompt = _ORCHESTRATOR_TASK_PROMPT.format(
            intent=intent,
            user_message=user_text,
        )
    full_system = build_system_prompt(
        "Follow the workspace instruction pack and execute orchestration exactly.",
        model_role="orchestrator",
    )

    llm = get_llm(task_type="orchestrator", temperature=0.2)

    try:
        response = await llm.ainvoke(
            [SystemMessage(content=full_system), HumanMessage(content=task_prompt)]
        )
        raw = response.content.strip()
    except Exception as exc:
        logger.error("Orchestrator LLM call failed: %s", exc, exc_info=True)
        raw = "TASKS:\n1. Answer directly.\nROUTE: CHAT"

    # ── 4. Parse tasks and routing decision ───────────────────────────────
    tasks: list[str] = []
    routing_decision = "CHAT"

    for line in raw.splitlines():
        line = line.strip()
        if re.match(r"^\d+\.", line):
            tasks.append(re.sub(r"^\d+\.\s*", "", line))
        match = re.match(r"^ROUTE:\s*(RESEARCH|CODING|REASONING|CHAT|BRIEFING)", line, re.I)
        if match:
            routing_decision = match.group(1).upper()

    if not tasks:
        tasks = ["Respond to user request directly."]

    logger.info(
        "Orchestrator → tasks=%d  routing=%s  topic_forwarded=%r",
        len(tasks),
        routing_decision,
        topic if routing_decision == "RESEARCH" else "",
    )

    # ── 5. Save context snapshot for worker handoff ────────────────────────
    topic_for_workers = topic if routing_decision == "RESEARCH" else ""
    save_context(
        {
            "intent": intent,
            "topic": topic_for_workers,
            "user_message": user_text,
            "tasks": tasks,
            "routing_decision": routing_decision,
        },
        session_id,
    )

    # ── 6. If CHAT, respond immediately and end workflow here ───────────
    result: dict = {
        "tasks": tasks,
        "routing_decision": routing_decision,
        "topic": topic_for_workers,
        "active_model": OLLAMA_ORCHESTRATOR_MODEL,
    }

    if routing_decision == "CHAT":
        answer_text = ""
        if "ANSWER:" in raw:
            answer_text = raw.split("ANSWER:", 1)[1].strip()

        if not answer_text:
            answer_text = "Xin lỗi, tôi chưa thể trả lời ngay lúc này."

        clear_context(session_id)
        result["messages"] = [AIMessage(content=answer_text)]

    return result

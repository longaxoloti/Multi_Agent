"""
Multi-model LangGraph workflow.

classify
    └─► orchestrate
            ├─ RESEARCH  ─► research (qwen2.5:14b-instruct) ─► synthesize (qwen2.5:7b) ─► END
            ├─ CODING    ─► coding   (qwen2.5-coder:14b)    ─► synthesize (qwen2.5:7b) ─► END
            ├─ REASONING ─► reasoning (qwen2.5:14b-instruct, no-web reasoning path)   ─► END
            ├─ BRIEFING  ─► briefing                                                   ─► END
            └─ CHAT      ─► END (orchestrator replies directly)

Context is persisted to a temp JSON file at data/sessions/<session_id>_ctx.json so that
when the orchestrator is reloaded for synthesis, it has full situational awareness.
"""

import logging
import uuid
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from graph.state import AgentState
from graph.nodes.orchestrator import orchestrator_node
from graph.nodes.research import research_node
from graph.nodes.coding import coding_node
from graph.nodes.reasoning import reasoning_node
from graph.nodes.briefing import briefing_node
from graph.nodes.synthesizer import synthesizer_node
from graph.nodes.request_router import request_router_node
from graph.llm_router import get_llm
from tools.workspace_priming import build_system_prompt

logger = logging.getLogger(__name__)


def _parse_classifier_output(raw: str) -> tuple[str, str]:
    intent = ""
    topic = ""
    for line in (raw or "").splitlines():
        m = re.match(r"INTENT:\s*(RESEARCH|CODING|REASONING|CHAT|BRIEFING)", line, re.I)
        if m:
            intent = m.group(1).upper()
        m2 = re.match(r"TOPIC:\s*(.+)", line, re.I)
        if m2:
            topic = m2.group(1).strip()
    return intent, topic


async def classify_node(state: AgentState) -> dict:
    """
    Uses llama3.2:3b to classify the user's intent and extract the main topic.
    Outputs: intent (RESEARCH | CODING | REASONING | CHAT | BRIEFING) and topic.
    Also assigns a session_id if one is not already set.
    """
    logger.info("--- CLASSIFY NODE ---")

    user_message = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
    )
    if not user_message:
        return {
            "intent": "CHAT",
            "topic": "",
            "session_id": state.get("session_id") or str(uuid.uuid4()),
        }

    user_text = user_message.content
    session_id = state.get("session_id") or str(uuid.uuid4())

    system_prompt = build_system_prompt(
        "Follow the classification format exactly as defined in your instructions.",
        model_role="classifier",
    )

    llm = get_llm(task_type="fast", temperature=0.0)

    try:
        response = await llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_text)]
        )
        raw = response.content.strip()
    except Exception as exc:
        logger.error("Classify node LLM call failed: %s", exc, exc_info=True)
        raw = "INTENT: CHAT\nTOPIC: general"

    intent, topic = _parse_classifier_output(raw)

    if not intent or not topic:
        repair_prompt = (
            "Your previous classification output is invalid or incomplete. "
            "Re-read the user message and output exactly two lines in this format:\n"
            "INTENT: <RESEARCH|CODING|REASONING|CHAT|BRIEFING>\n"
            "TOPIC: <non-empty topic phrase in user language>\n\n"
            f"User message: {user_text}\n"
            f"Previous output: {raw}"
        )
        try:
            repair_response = await llm.ainvoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=repair_prompt)]
            )
            repair_raw = repair_response.content.strip()
            intent2, topic2 = _parse_classifier_output(repair_raw)
            if intent2:
                intent = intent2
            if topic2:
                topic = topic2
        except Exception as exc:
            logger.warning("Classifier repair pass failed: %s", exc)

    if not intent:
        intent = "CHAT"
    if not topic:
        topic_prompt = (
            "Extract ONLY the main topic phrase from the user message. "
            "Return just the topic phrase, no labels, no explanation.\n\n"
            f"User message: {user_text}"
        )
        try:
            topic_response = await llm.ainvoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=topic_prompt)]
            )
            topic = (topic_response.content or "").strip()
        except Exception as exc:
            logger.warning("Classifier topic-only pass failed: %s", exc)
            topic = "general"

    if len(topic) > 140:
        topic = topic[:140]

    logger.info("Classified → intent=%s  topic=%r", intent, topic)
    return {"intent": intent, "topic": topic, "session_id": session_id}


def route_after_orchestrator(state: AgentState) -> str:
    """Route to the appropriate worker based on the orchestrator's decision."""
    decision = state.get("routing_decision", "CHAT").upper()
    if decision == "RESEARCH":
        return "research"
    if decision == "CODING":
        return "coding"
    if decision == "REASONING":
        return "reasoning"
    if decision == "BRIEFING":
        return "briefing"
    return "end"


def route_after_request_router(state: AgentState) -> str:
    if (state.get("knowledge_action") or "").lower() == "handled":
        return "end"
    return "classify"


def build_workflow():
    workflow = StateGraph(AgentState)

    workflow.add_node("request_router", request_router_node)
    workflow.add_node("classify", classify_node)
    workflow.add_node("orchestrate", orchestrator_node)
    workflow.add_node("research", research_node)
    workflow.add_node("coding", coding_node)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("briefing", briefing_node)
    workflow.add_node("synthesize", synthesizer_node)

    workflow.set_entry_point("request_router")
    workflow.add_conditional_edges(
        "request_router",
        route_after_request_router,
        {
            "classify": "classify",
            "end": END,
        },
    )
    workflow.add_edge("classify", "orchestrate")

    workflow.add_conditional_edges(
        "orchestrate",
        route_after_orchestrator,
        {
            "research": "research",
            "coding": "coding",
            "reasoning": "reasoning",
            "briefing": "briefing",
            "end": END,
        },
    )
    workflow.add_edge("research", "synthesize")
    workflow.add_edge("coding", "synthesize")

    workflow.add_edge("synthesize", END)
    workflow.add_edge("reasoning", END)
    workflow.add_edge("briefing", END)
    return workflow.compile()
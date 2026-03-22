"""
Multi-model LangGraph workflow.
request_router
    └─► orchestrate (PLAN → [worker loop via PROGRESS] → SYNTHESIZE)
            ├─ RESEARCH  ─► research ─► orchestrate
            ├─ CODING    ─► coding   ─► orchestrate
            ├─ REASONING ─► reasoning ─► orchestrate
            ├─ BRIEFING  ─► briefing ─► orchestrate
            └─ CHAT      ─► END (orchestrator replies directly)
            └─ SYNTHESIZE ─► synthesize ─► END

Context is persisted to a temp JSON file at data/sessions/<session_id>_ctx.json so that
when the orchestrator is reloaded for synthesis, it has full situational awareness.
"""

import logging
from langgraph.graph import StateGraph, END
from graph.state import AgentState
from graph.nodes.orchestrator import orchestrator_node
from graph.nodes.research import research_node
from graph.nodes.coding import coding_node
from graph.nodes.reasoning import reasoning_node
from graph.nodes.briefing import briefing_node
from graph.nodes.synthesizer import synthesizer_node
from graph.nodes.request_router import request_router_node
logger = logging.getLogger(__name__)

def route_after_orchestrator(state: AgentState) -> str:
    decision = state.get("routing_decision", "CHAT").upper()
    if decision == "RESEARCH":
        return "research"
    if decision == "CODING":
        return "coding"
    if decision == "REASONING":
        return "reasoning"
    if decision == "BRIEFING":
        return "briefing"
    if decision == "SYNTHESIZE":
        return "synthesize"
    if decision == "CHAT":
        return "end"
    return "end"

def route_after_request_router(state: AgentState) -> str:
    if (state.get("knowledge_action") or "").lower() == "handled":
        return "end"
    return "orchestrate"

def build_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("request_router", request_router_node)
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
            "orchestrate": "orchestrate",
            "end": END,
        },
    )
    workflow.add_conditional_edges(
        "orchestrate",
        route_after_orchestrator,
        {
            "research": "research",
            "coding": "coding",
            "reasoning": "reasoning",
            "briefing": "briefing",
            "synthesize": "synthesize",
            "end": END,
        },
    )
    # Check progress.
    workflow.add_edge("research", "orchestrate")
    workflow.add_edge("coding", "orchestrate")
    workflow.add_edge("reasoning", "orchestrate")
    workflow.add_edge("briefing", "orchestrate")
    workflow.add_edge("synthesize", END)
    return workflow.compile()
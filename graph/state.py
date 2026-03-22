import operator
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage


class PlanStep(TypedDict):
    worker: str  # RESEARCH | CODING | REASONING | BRIEFING
    tasks: list[str]
    step_goal: str


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    chat_id: str
    session_id: str
    intent: str
    topic: str
    search_query: str

    # Orchestrator output
    phase: str
    plan_steps: list[PlanStep]
    current_step_index: int
    tasks: list[str]
    routing_decision: str
    knowledge_action: str

    # Worker results
    task_results: Annotated[list[dict], operator.add]
    active_model: str

    # Legacy / compatibility
    memory_context: str
    verification_summary: str
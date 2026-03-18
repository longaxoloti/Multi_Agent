import operator
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    # ── Conversation history (append-only) ───────────────────────────────
    messages: Annotated[list[BaseMessage], operator.add]

    # ── Session / routing ─────────────────────────────────────────────────
    chat_id: str          # Telegram chat ID (or any session identifier)
    session_id: str       # Used as key for temp context JSON files

    # ── Classifier output ─────────────────────────────────────────────────
    intent: str           # RESEARCH | CODING | REASONING | CHAT | BRIEFING
    topic: str            # Main subject/topic extracted by the classifier

    # ── Orchestrator output ───────────────────────────────────────────────
    tasks: list[str]      # Ordered task list produced by the Orchestrator
    routing_decision: str # Which worker to call: RESEARCH | CODING | REASONING | CHAT
    knowledge_action: str # handled | none

    # ── Worker results ────────────────────────────────────────────────────
    task_results: list[dict]  # [{model, result, sources}, ...]
    active_model: str          # Name of the last Ollama model used (for unloading)

    # ── Legacy / compatibility ─────────────────────────────────────────────
    memory_context: str
    verification_summary: str


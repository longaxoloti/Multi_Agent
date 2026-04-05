import json
import asyncio
import logging
import re
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from main.config import OLLAMA_ORCHESTRATOR_MODEL
from graph.state import AgentState
from graph.llm_router import get_llm
from tools.ollama_manager import clear_context, load_context, save_context, unload_model
from tools.workspace_priming import build_system_prompt

logger = logging.getLogger(__name__)

_repo = None
_skill_service = None
_profile_service = None


def _get_repo():
    global _repo
    if _repo is None:
        from storage.trusted_db import AgentDBRepository
        _repo = AgentDBRepository()
        _repo.initialize()
    return _repo


def _get_skill_service():
    global _skill_service
    if _skill_service is None:
        from storage.skill_service import SkillService
        repo = _get_repo()
        _skill_service = SkillService(
            repo._session_factory,
            is_pg=repo.engine.dialect.name == "postgresql",
        )
    return _skill_service


def _get_profile_service():
    global _profile_service
    if _profile_service is None:
        from storage.user_profile_service import UserProfileService
        repo = _get_repo()
        _profile_service = UserProfileService(
            repo._session_factory,
            is_pg=repo.engine.dialect.name == "postgresql",
        )
    return _profile_service


def _build_skill_context(user_text: str, *, limit: int = 3) -> str:
    if not user_text.strip():
        return ""
    try:
        rows = _get_skill_service().search_skills(user_text, limit=limit)
    except Exception as exc:
        logger.debug("Skill context unavailable: %s", exc)
        return ""

    if not rows:
        return ""

    lines = ["Relevant skill snippets:"]
    for idx, row in enumerate(rows, 1):
        title = (row.get("title") or "Untitled skill").strip()
        chunk = (row.get("chunk_text") or "").strip().replace("\n", " ")
        if len(chunk) > 240:
            chunk = chunk[:240] + "..."
        lines.append(f"{idx}. {title}: {chunk}")
    return "\n".join(lines)


def _build_profile_context(user_id: str, user_text: str, *, limit: int = 4) -> str:
    if not user_id or not user_text.strip():
        return ""
    try:
        rows = _get_profile_service().search_profile(user_id, user_text, limit=limit)
    except Exception as exc:
        logger.debug("Profile context unavailable: %s", exc)
        return ""

    if not rows:
        return ""

    lines = ["Relevant user profile facts:"]
    for idx, row in enumerate(rows, 1):
        key = (row.get("fact_key") or "fact").strip()
        value = (row.get("fact_value") or "").strip().replace("\n", " ")
        if len(value) > 180:
            value = value[:180] + "..."
        lines.append(f"{idx}. {key}: {value}")
    return "\n".join(lines)

_ORCHESTRATOR_PLAN_PROMPT = """\
You must perform internal step-by-step reasoning (Chain-of-Thought),
but you MUST NOT output any chain-of-thought. Output only strict JSON.

User request:
{user_message}

Your job:
1. Analyze the user request to decide which worker types are needed (research/coding/reasoning/briefing).
2. Identify the main subject/topic.
3. Plan an ordered list of steps (each step assigned to exactly one worker) and the concrete tasks for that worker.
4. Decide the next route for the first step.

Worker routing rules:
- Use RESEARCH when a web lookup is needed (current facts, verification, comparison with sources).
- Use CODING when the user asks to write/fix/explain/review code.
- Use REASONING when the user asks for deep analysis/logic/planning without web lookup.
- Use BRIEFING when the user asks for a digest/brief summary.
- Use CHAT if you can directly answer the user without any worker steps.

Strict JSON schema (output only this JSON, no markdown):
{
  "intent": "RESEARCH|CODING|REASONING|CHAT|BRIEFING",
  "topic": "<non-empty short topic in user's language>",
  "search_query": "<search query if intent=RESEARCH else NONE>",
  "plan_steps": [
    {
      "worker": "RESEARCH|CODING|REASONING|BRIEFING",
      "tasks": ["<task 1>", "<task 2>", "..."],
      "step_goal": "<short goal of this step>"
    }
  ],
  "routing_decision": "RESEARCH|CODING|REASONING|BRIEFING|CHAT",
  "next_step_index": 0,
  "answer": "<only when routing_decision=CHAT; otherwise empty string>"
}
Notes:
- If routing_decision=CHAT, plan_steps may be empty and answer must be provided.
- If routing_decision is not CHAT, plan_steps MUST be non-empty and tasks MUST be non-empty.
"""


_ORCHESTRATOR_PROGRESS_PROMPT = """\
You are the Orchestrator Agent.
You must perform internal step-by-step reasoning (Chain-of-Thought),
but you MUST NOT output any chain-of-thought. Output only strict JSON.

User request:
{user_message}

Plan context:
- intent: {intent}
- topic: {topic}
- search_query: {search_query}
- plan_steps: {plan_steps}

Progress context:
- completed_step_indices: {completed_step_indices}
- expected_next_step_index: {expected_next_step_index}
- last_worker_result_preview: {last_worker_result_preview}

Your job:
1. Check which steps are completed.
2. Decide the next route for the next unfinished step, OR decide SYNTHESIZE if all steps are done.
3. If you choose a worker route, provide tasks_for_worker for that next step.

Strict JSON schema (output only this JSON, no markdown):
{
  "routing_decision": "RESEARCH|CODING|REASONING|BRIEFING|SYNTHESIZE|CHAT",
  "next_step_index": <int or null>,
  "tasks_for_worker": ["<task 1>", "<task 2>", "..."],
  "answer": "<only when routing_decision=CHAT; otherwise empty string>"
}

Rules:
- If all steps are completed, routing_decision MUST be "SYNTHESIZE" and next_step_index MUST be null.
- If not all steps are completed, routing_decision MUST be one of RESEARCH|CODING|REASONING|BRIEFING,
  and next_step_index MUST equal expected_next_step_index.
- tasks_for_worker MUST be a non-empty list when routing_decision is not SYNTHESIZE/CHAT.
"""


def _extract_json(raw: str) -> dict:
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    candidate = raw[start : end + 1]
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


async def orchestrator_node(state: AgentState) -> dict:
    logger.info("--- ORCHESTRATOR NODE ---")

    user_message = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
    )
    user_text = user_message.content if user_message else ""
    session_id = state.get("session_id") or state.get("chat_id") or "default"
    phase = (state.get("phase") or "PLAN").upper()
    if phase not in {"PLAN", "PROGRESS"}:
        phase = "PLAN"

    ctx = load_context(session_id)

    intent = state.get("intent") or ctx.get("intent") or "CHAT"
    topic = state.get("topic") or ctx.get("topic") or ""
    search_query = state.get("search_query") or ctx.get("search_query") or "NONE"

    full_system = build_system_prompt(
        "Follow the workspace instruction pack and execute orchestration exactly.",
        model_role="orchestrator",
    )

    user_id = str(state.get("user_id") or state.get("chat_id") or "default")
    context_blocks = [
        _build_skill_context(user_text),
        _build_profile_context(user_id, user_text),
    ]
    context_blocks = [block for block in context_blocks if block]
    if context_blocks:
        full_system += "\n\n" + "\n\n".join(context_blocks)

    llm = get_llm(task_type="orchestrator", temperature=0.2)

    try:
        if phase == "PLAN":
            task_prompt = _ORCHESTRATOR_PLAN_PROMPT.replace("{user_message}", user_text)
        else:
            task_prompt = None

        if phase == "PROGRESS":
            plan_steps = state.get("plan_steps") or ctx.get("plan_steps") or []
            completed_step_indices: list[int] = []

            task_results = state.get("task_results") or []
            has_step_index = any(isinstance(tr, dict) and "step_index" in tr for tr in task_results)
            if has_step_index:
                for tr in task_results:
                    if not isinstance(tr, dict):
                        continue
                    idx = tr.get("step_index")
                    if isinstance(idx, int):
                        completed_step_indices.append(idx)
            else:
                completed_step_indices = [i for i in range(min(len(task_results), len(plan_steps)))]

            plan_steps_len = len(plan_steps)
            expected_next_step_index = None
            for i in range(plan_steps_len):
                if i not in set(completed_step_indices):
                    expected_next_step_index = i
                    break
            if expected_next_step_index is None:
                expected_next_step_index = plan_steps_len - 1 if plan_steps_len else 0

            last_worker_result = (task_results[-1] if task_results else {}) or {}
            last_worker_result_preview = str(last_worker_result)[:1200]

            task_prompt = _ORCHESTRATOR_PROGRESS_PROMPT
            task_prompt = task_prompt.replace("{user_message}", user_text)
            task_prompt = task_prompt.replace("{intent}", intent)
            task_prompt = task_prompt.replace("{topic}", topic)
            task_prompt = task_prompt.replace("{search_query}", search_query)
            task_prompt = task_prompt.replace("{plan_steps}", str(plan_steps))
            task_prompt = task_prompt.replace(
                "{completed_step_indices}", str(sorted(set(completed_step_indices)))
            )
            task_prompt = task_prompt.replace(
                "{expected_next_step_index}", str(expected_next_step_index)
            )
            task_prompt = task_prompt.replace(
                "{last_worker_result_preview}", last_worker_result_preview
            )

        response = await asyncio.wait_for(
            llm.ainvoke([SystemMessage(content=full_system), HumanMessage(content=task_prompt)]),
            timeout=180,
        )
        raw = response.content.strip()
    except asyncio.TimeoutError:
        logger.error("Orchestrator LLM call timed out")
        if phase == "PLAN":
            raw = json.dumps(
                {
                    "intent": "CHAT",
                    "topic": "general",
                    "search_query": "NONE",
                    "plan_steps": [],
                    "routing_decision": "CHAT",
                    "next_step_index": 0,
                    "answer": "Sorry, I can't provide an answer at this moment.",
                },
                ensure_ascii=False,
            )
        else:
            raw = json.dumps(
                {
                    "routing_decision": "SYNTHESIZE",
                    "next_step_index": None,
                    "tasks_for_worker": [],
                    "answer": "",
                },
                ensure_ascii=False,
            )
    except Exception as exc:
        logger.error("Orchestrator LLM call failed: %s", exc, exc_info=True)
        raw = ""

    data = _extract_json(raw)

    if phase == "PLAN":
        plan_steps = data.get("plan_steps") or []
        routing_decision = (data.get("routing_decision") or "CHAT").upper()
        intent = data.get("intent") or intent or "CHAT"
        topic = (data.get("topic") or topic or "general").strip()[:140]
        search_query = data.get("search_query") or search_query or "NONE"

        if not plan_steps and routing_decision != "CHAT":
            routing_decision = "CHAT"

        if routing_decision == "CHAT":
            answer_text = str(data.get("answer") or "").strip()
            if not answer_text:
                answer_text = "Sorry, I can't provide an answer at this moment."
            clear_context(session_id)
            return {
                "phase": "PLAN",
                "routing_decision": "CHAT",
                "active_model": OLLAMA_ORCHESTRATOR_MODEL,
                "messages": [AIMessage(content=answer_text)],
            }

        if not plan_steps:
            plan_steps = [
                {"worker": "REASONING", "tasks": [user_text], "step_goal": "Direct reasoning"}
            ]

        step0 = plan_steps[0] or {}
        tasks_for_worker = step0.get("tasks") or []
        tasks_for_worker = [str(t) for t in tasks_for_worker if str(t).strip()]
        if not tasks_for_worker:
            tasks_for_worker = [user_text]

        step_index = 0
        worker_type = (step0.get("worker") or routing_decision).upper()
        if worker_type not in {"RESEARCH", "CODING", "REASONING", "BRIEFING"}:
            worker_type = routing_decision if routing_decision in {"RESEARCH", "CODING", "REASONING", "BRIEFING"} else "REASONING"

        search_query_for_workers = search_query if worker_type == "RESEARCH" else "NONE"
        save_context(
            {
                "intent": intent,
                "topic": topic,
                "search_query": search_query_for_workers,
                "user_message": user_text,
                "tasks": tasks_for_worker,
                "routing_decision": worker_type,
                "plan_steps": plan_steps,
                "step_index": step_index,
            },
            session_id,
        )

        return {
            "phase": "PROGRESS",
            "intent": intent,
            "topic": topic,
            "search_query": search_query,
            "plan_steps": plan_steps,
            "current_step_index": step_index,
            "tasks": tasks_for_worker,
            "routing_decision": worker_type,
            "active_model": OLLAMA_ORCHESTRATOR_MODEL,
        }

    plan_steps = state.get("plan_steps") or ctx.get("plan_steps") or []
    task_results = state.get("task_results") or []

    completed_step_indices: set[int] = set()
    has_step_index = any(isinstance(tr, dict) and "step_index" in tr for tr in task_results)
    if has_step_index:
        for tr in task_results:
            if not isinstance(tr, dict):
                continue
            idx = tr.get("step_index")
            if isinstance(idx, int):
                completed_step_indices.add(idx)
    else:
        completed_step_indices = set(range(min(len(task_results), len(plan_steps))))

    expected_next_step_index = None
    for i in range(len(plan_steps)):
        if i not in completed_step_indices:
            expected_next_step_index = i
            break

    # Synthesize
    if expected_next_step_index is None:
        return {
            "phase": "PROGRESS",
            "routing_decision": "SYNTHESIZE",
            "active_model": OLLAMA_ORCHESTRATOR_MODEL,
        }

    expected_step = plan_steps[expected_next_step_index] or {}
    expected_worker = (expected_step.get("worker") or "").upper()
    expected_tasks = expected_step.get("tasks") or []
    expected_tasks = [str(t) for t in expected_tasks if str(t).strip()]
    if not expected_worker:
        expected_worker = "REASONING"
    if not expected_tasks:
        expected_tasks = [user_text]

    routing_decision = (data.get("routing_decision") or expected_worker).upper()
    next_step_index = data.get("next_step_index")
    tasks_for_worker = data.get("tasks_for_worker") or expected_tasks
    tasks_for_worker = [str(t) for t in tasks_for_worker if str(t).strip()]
    if not tasks_for_worker:
        tasks_for_worker = expected_tasks

    if not isinstance(next_step_index, int):
        next_step_index = expected_next_step_index
    if next_step_index != expected_next_step_index:
        logger.warning(
            "Orchestrator PROGRESS: model suggested next_step_index=%r, expected=%r. Using expected.",
            next_step_index,
            expected_next_step_index,
        )
        next_step_index = expected_next_step_index

    if routing_decision == "SYNTHESIZE":
        return {"phase": "PROGRESS", "routing_decision": "SYNTHESIZE", "active_model": OLLAMA_ORCHESTRATOR_MODEL}

    if routing_decision == "CHAT":
        answer_text = str(data.get("answer") or "").strip()
        if not answer_text:
            answer_text = "Sorry, I can't provide an answer at this moment."
        clear_context(session_id)
        return {
            "phase": "PROGRESS",
            "routing_decision": "CHAT",
            "active_model": OLLAMA_ORCHESTRATOR_MODEL,
            "messages": [AIMessage(content=answer_text)],
        }

    if routing_decision not in {"RESEARCH", "CODING", "REASONING", "BRIEFING"}:
        routing_decision = expected_worker

    search_query_for_workers = search_query if routing_decision == "RESEARCH" else "NONE"
    save_context(
        {
            "intent": intent,
            "topic": topic,
            "search_query": search_query_for_workers,
            "user_message": user_text,
            "tasks": tasks_for_worker,
            "routing_decision": routing_decision,
            "plan_steps": plan_steps,
            "step_index": next_step_index,
        },
        session_id,
    )

    return {
        "phase": "PROGRESS",
        "intent": intent,
        "topic": topic,
        "search_query": search_query,
        "plan_steps": plan_steps,
        "current_step_index": next_step_index,
        "tasks": tasks_for_worker,
        "routing_decision": routing_decision,
        "active_model": OLLAMA_ORCHESTRATOR_MODEL,
    }

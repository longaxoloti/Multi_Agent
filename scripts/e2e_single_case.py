import asyncio
import sys
import logging
import os
from langchain_core.messages import HumanMessage

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graph.workflow import build_workflow

CASES = {
    "chat": "Chào bạn, hôm nay bạn khỏe không?",
    "research": "Tóm tắt nhanh tin tức AI mới nhất hôm nay và cho nguồn",
    "coding": "Viết cho tôi một hàm Python kiểm tra số nguyên tố và ví dụ dùng",
}


def base_state(text: str):
    return {
        "messages": [HumanMessage(content=text)],
        "intent": "",
        "topic": "",
        "tasks": [],
        "task_results": [],
        "routing_decision": "",
        "memory_context": "",
        "verification_summary": "",
        "chat_id": "e2e_test",
        "session_id": "",
        "active_model": "",
    }


async def main(case: str, custom_text: str | None = None):
    text = (custom_text or "").strip() or CASES[case]
    wf = build_workflow()
    result = await wf.ainvoke(base_state(text))

    messages = result.get("messages", [])
    final_text = messages[-1].content if messages else ""
    task_results = result.get("task_results", []) or []

    print("=" * 90)
    print(f"CASE: {case.upper()}")
    print("USER:", text)
    print("INTENT:", result.get("intent"))
    print("TOPIC:", result.get("topic"))
    print("ROUTING_DECISION:", result.get("routing_decision"))
    print("ACTIVE_MODEL:", result.get("active_model"))
    print("TASK_RESULTS_COUNT:", len(task_results))

    for idx, tr in enumerate(task_results, 1):
        model = tr.get("model")
        sources = tr.get("sources") or []
        preview = (tr.get("result") or "")[:220].replace("\n", " ")
        print(f"  - worker[{idx}] model={model} sources={len(sources)} preview={preview}")

    preview_final = (final_text or "")[:500].replace("\n", " ")
    print("FINAL_RESPONSE_PREVIEW:", preview_final)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(name)s | %(message)s')
    case_name = (sys.argv[1] if len(sys.argv) > 1 else "chat").strip().lower()
    if case_name not in CASES:
        raise SystemExit(f"Invalid case: {case_name}. Use one of: {', '.join(CASES)}")
    custom_query = " ".join(sys.argv[2:]).strip() if len(sys.argv) > 2 else None
    asyncio.run(main(case_name, custom_query))

import asyncio
from langchain_core.messages import HumanMessage
from graph.workflow import build_workflow

CASES = [
    ("CHAT", "Chào bạn, hôm nay bạn khỏe không?"),
    ("RESEARCH", "Tóm tắt nhanh tin tức AI mới nhất hôm nay và cho nguồn"),
    ("CODING", "Viết cho tôi một hàm Python kiểm tra số nguyên tố và ví dụ dùng"),
]


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


async def run_case(name: str, text: str):
    wf = build_workflow()
    result = await wf.ainvoke(base_state(text))

    messages = result.get("messages", [])
    final_text = messages[-1].content if messages else ""
    task_results = result.get("task_results", []) or []

    print("\n" + "=" * 90)
    print(f"CASE: {name}")
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


async def main():
    for name, text in CASES:
        try:
            await run_case(name, text)
        except Exception as exc:
            print("\n" + "=" * 90)
            print(f"CASE: {name}")
            print("ERROR:", repr(exc))


if __name__ == "__main__":
    asyncio.run(main())

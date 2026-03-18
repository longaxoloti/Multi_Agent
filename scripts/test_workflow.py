import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.workflow import build_workflow
from langchain_core.messages import HumanMessage

async def main():
    workflow = build_workflow()
    print("Testing phrase: 'Tóm tắt sự kiện mới nhất về Vinfast'")
    result = await workflow.ainvoke({
        "messages": [HumanMessage(content="Tóm tắt sự kiện mới nhất về Vinfast")],
        "intent": "",
        "memory_context": "",
        "verification_summary": "",
        "chat_id": "test_123"
    })
    print("\n--- TEST RESULTS ---")
    print("FINAL INTENT:", result.get("intent"))
    print("FINAL VERIFIER SUMMARY:", result.get("verification_summary"))
    print("FINAL MESSAGES:")
    for m in result.get("messages", []):
        print(f"[{m.type}]: {m.content}")

if __name__ == "__main__":
    asyncio.run(main())

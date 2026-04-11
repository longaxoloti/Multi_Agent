#!/usr/bin/env python3
"""
Debug test for Research Node pipeline.

Traces the data flow through: Camoufox → URL extraction → Crawl4AI → Model synthesis

Usage:
    # Set DEBUG_RESEARCH=1 to enable detailed logging
    DEBUG_RESEARCH=1 python tests/test_research_debug.py "your research query here"
    
    # Example:
    DEBUG_RESEARCH=1 python tests/test_research_debug.py "AI news today"

Output:
    - Console logs (filtered by DEBUG_RESEARCH flag)
    - File: data/logs/research_debug.log (full trace)
    - Shows URL discovery, content crawled, and model output
"""

import asyncio
import sys
import os
import logging
from pathlib import Path

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from langchain_core.messages import HumanMessage
from graph.state import AgentState
from graph.nodes.research import research_node
from tools.agent_org.ollama_manager import save_context


async def test_research_node(query: str):
    """Run research node with given query and observe data flow."""
    print("=" * 80)
    print(f"Research Debug Test: {query}")
    print("=" * 80)
    print(f"DEBUG_RESEARCH env: {os.getenv('DEBUG_RESEARCH', 'NOT SET')}")
    print(f"Debug log file will be: {PROJECT_ROOT}/data/logs/research_debug.log")
    print("=" * 80)
    
    # Create a minimal state for research node
    session_id = "debug_test"
    chat_id = "debug_user"
    
    # Save context with the research query
    context = {
        "user_message": query,
        "topic": "",
        "tasks": ["Provide a comprehensive summary of the current research topic"],
    }
    save_context(context, session_id)
    
    state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "intent": "RESEARCH",
        "topic": "",
        "search_query": "",
        "tasks": ["Provide a comprehensive summary"],
        "task_results": [],
        "routing_decision": "RESEARCH",
        "memory_context": "",
        "verification_summary": "",
        "chat_id": chat_id,
        "session_id": session_id,
        "active_model": "",
    }
    
    print(f"\nRunning research for query: {query}")
    print(f"Session: {session_id}, Chat: {chat_id}\n")
    
    try:
        result = await research_node(state)
        
        print("\n" + "=" * 80)
        print("RESEARCH NODE RESULT:")
        print("=" * 80)
        
        if result.get("task_results"):
            for i, task_result in enumerate(result["task_results"]):
                print(f"\nTask Result #{i + 1}:")
                print(f"  Model: {task_result.get('model')}")
                print(f"  Response length: {len(task_result.get('result', ''))} chars")
                print(f"  Sources count: {len(task_result.get('sources', []))}")
                print(f"  Sources: {task_result.get('sources', [])}")
                print(f"\n  Response preview (first 500 chars):")
                print("  " + "\n  ".join(task_result.get('result', '')[:500].split("\n")))
        
        print("\n" + "=" * 80)
        print(f"Check debug log for detailed data flow: {PROJECT_ROOT}/data/logs/research_debug.log")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nERROR during research: {e}")
        import traceback
        traceback.print_exc()


def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Latest AI and technology news today"
    
    if os.getenv("DEBUG_RESEARCH", "").lower() not in {"1", "true", "yes"}:
        print("WARNING: DEBUG_RESEARCH env var is not set")
        print("To see detailed logging, run with: DEBUG_RESEARCH=1 python tests/test_research_debug.py ...")
        print()
    
    asyncio.run(test_research_node(query))


if __name__ == "__main__":
    main()

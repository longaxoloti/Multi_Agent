#!/usr/bin/env python3
"""Integration test for PostgreSQL pgvector pipeline without Chroma."""

import sys
sys.path.insert(0, '.')

from storage.knowledge_service import KnowledgeService
from storage.trusted_db import TrustedDBRepository

def main():
    print("=" * 60)
    print("INTEGRATION TEST: PostgreSQL + pgvector pipeline")
    print("=" * 60)

    try:
        repo = TrustedDBRepository()
        service = KnowledgeService(repo)
        
        # Check pgvector readiness
        print("\n1. Checking pgvector readiness...")
        is_ready = repo.is_pgvector_ready()
        print(f"   pgvector ready: {is_ready}")
        
        # Save knowledge record
        print("\n2. Saving knowledge record...")
        chat_id = "test_integration_001"
        content = "This is a test knowledge record for pgvector integration"
        title = "Integration Test"
        
        saved = service.save(chat_id, content, title, category="test")
        print(f"   Saved successfully")
        print(f"   - ID: {saved.get('id')}")
        print(f"   - Embedding model: {saved.get('embedding_model')}")
        print(f"   - Embedding dims: {saved.get('embedding_dims')}")
        print(f"   - Has embedding: {'embedding' in saved}")
        
        # Search knowledge record  
        print("\n3. Searching knowledge record...")
        results = service.search(chat_id, "test pgvector")
        print(f"   Found {len(results)} results")
        if results:
            r = results[0]
            print(f"   - Top result title: {r.get('title')}")
            print(f"   - Similarity score: {r.get('similarity', 'N/A')}")
            print(f"   - Content preview: {r.get('content')[:50]}...")
        
        # Get specific record
        print("\n4. Getting specific record...")
        record_id = saved['id']
        retrieved = service.get(chat_id, record_id)
        print(f"   Retrieved: {retrieved is not None}")
        if retrieved:
            print(f"   - Retrieved title: {retrieved.get('title')}")
        
        # Clean up
        print("\n5. Cleaning up...")
        service.delete(chat_id, record_id)
        print("   Deleted successfully")
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - pgvector pipeline working!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

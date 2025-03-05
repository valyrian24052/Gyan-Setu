#!/usr/bin/env python3
"""
Basic test for LlamaIndex integration
"""
import os
import sys
from pathlib import Path

# Create test directories if they don't exist
os.makedirs("./knowledge", exist_ok=True)
os.makedirs("./index_storage", exist_ok=True)
os.makedirs("./.cache", exist_ok=True)

# Create a test document if none exists
test_doc_path = Path("./knowledge/test_document.txt")
if not test_doc_path.exists():
    with open(test_doc_path, "w") as f:
        f.write("""
        # Test Document for LlamaIndex
        
        This is a test document to verify that LlamaIndex is working properly.
        
        ## Teaching Strategies
        
        Effective teaching strategies for ESL students include visual aids, 
        group work, and scaffolded instruction. Teachers should provide clear, 
        simple instructions and check for understanding frequently.
        
        ## Classroom Management
        
        Good classroom management involves setting clear expectations, 
        establishing routines, and being consistent with consequences. 
        Positive reinforcement can be more effective than punishment.
        """)

try:
    # Try to import the LlamaIndexKnowledgeManager
    from llama_index_integration import LlamaIndexKnowledgeManager
    
    # Initialize with mock LLM for testing without API keys
    manager = LlamaIndexKnowledgeManager(
        documents_dir="./knowledge",
        index_dir="./index_storage",
        llm_provider="local"  # Use MockLLM for testing
    )
    
    # Load or create the index
    print("Loading or creating index...")
    manager.load_or_create_index()
    print("Index loaded/created successfully!")
    
    # Test a query
    test_query = "What are effective teaching strategies for ESL students?"
    print(f"\nQuerying: {test_query}")
    
    result = manager.query_knowledge(test_query)
    
    print("\nResponse:")
    print(result["response"])
    
    print("\nSources:")
    for i, source in enumerate(result["sources"], 1):
        print(f"\n{i}. Source: {source.get('source', 'Unknown')}")
        print(f"   Score: {source.get('score', 0.0)}")
        print(f"   Text: {source.get('text', '')[:100]}...")
    
    print("\nBasic test completed successfully!")
    
except Exception as e:
    print(f"Error during testing: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 
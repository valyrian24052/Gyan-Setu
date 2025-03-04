import os
import time
import logging
from llama_index_integration import LlamaIndexKnowledgeManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_knowledge_base(llm_provider="openai"):
    """
    Test the LlamaIndex knowledge base with real-world queries.
    
    Args:
        llm_provider: LLM provider to use ('openai', 'anthropic', or 'local')
    """
    print(f"Starting LlamaIndex test with {llm_provider} provider...")
    
    try:
        # Create knowledge manager
        km = LlamaIndexKnowledgeManager(
            documents_dir='knowledge_base/books',
            llm_provider=llm_provider,
            enable_caching=True,
            cache_ttl=3600  # 1 hour cache TTL
        )
        
        # Load index
        print("Loading or creating index...")
        start_time = time.time()
        km.load_or_create_index()
        print(f"Index loaded in {time.time() - start_time:.2f} seconds")
        
        # Test queries
        test_queries = [
            "What are effective classroom management techniques?",
            "How can I help students with dyslexia?",
            "What are best practices for teaching mathematics?"
        ]
        
        for query in test_queries:
            print(f"\n\nTESTING QUERY: {query}")
            try:
                start_time = time.time()
                result = km.query_knowledge(query, top_k=2)
                query_time = time.time() - start_time
                
                # Check for errors
                if "error" in result:
                    print(f"Error: {result['error']}")
                    continue
                
                print(f"Query processed in {query_time:.2f} seconds")
                print(f"Response: {result['response']}")
                print("\nSources:")
                for i, source in enumerate(result["sources"], 1):
                    print(f"Source {i}: {source[:200]}...")
            except Exception as e:
                print(f"Error processing query '{query}': {e}")
                
        # Test cache functionality by repeating a query
        if km.enable_caching:
            print("\n\nTesting cache with repeat query...")
            repeat_query = test_queries[0]
            
            try:
                start_time = time.time()
                result = km.query_knowledge(repeat_query, top_k=2)
                cache_time = time.time() - start_time
                
                print(f"Cached query processed in {cache_time:.2f} seconds")
                print(f"Response: {result['response'][:100]}...")
            except Exception as e:
                print(f"Error testing cache: {e}")
                
        print("\nLlamaIndex test completed successfully!")
        return True
    
    except Exception as e:
        print(f"Error during LlamaIndex test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_multiple_providers():
    """Test the knowledge base with different LLM providers."""
    print("Testing with multiple LLM providers...")
    
    # First test with local (MockLLM) as it's fastest and doesn't require API keys
    print("\n\n========== TESTING WITH LOCAL PROVIDER ==========")
    test_knowledge_base(llm_provider="local")
    
    # Test with OpenAI if API key is available
    if "OPENAI_API_KEY" in os.environ:
        print("\n\n========== TESTING WITH OPENAI PROVIDER ==========")
        test_knowledge_base(llm_provider="openai")
    else:
        print("\nSkipping OpenAI test: API key not found")
    
    # Test with Anthropic if API key is available
    if "ANTHROPIC_API_KEY" in os.environ:
        print("\n\n========== TESTING WITH ANTHROPIC PROVIDER ==========")
        test_knowledge_base(llm_provider="anthropic")
    else:
        print("\nSkipping Anthropic test: API key not found")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LlamaIndex knowledge integration")
    parser.add_argument("--provider", type=str, default="openai", 
                        choices=["openai", "anthropic", "local", "all"],
                        help="LLM provider to use")
    
    args = parser.parse_args()
    
    if args.provider == "all":
        test_with_multiple_providers()
    else:
        test_knowledge_base(llm_provider=args.provider) 
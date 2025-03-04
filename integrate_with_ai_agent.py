#!/usr/bin/env python3
"""
LlamaIndex Integration with AI Agent

This script demonstrates how to integrate the LlamaIndex knowledge system
with the AI agent for educational assistance. It shows how to:

1. Initialize the LlamaIndex knowledge manager
2. Connect it to the AI agent
3. Use it for answering educational queries
4. Handle errors and cache results

Usage:
    python integrate_with_ai_agent.py
"""

import os
import logging
import argparse
from typing import Dict, List, Any, Optional

# Import the AI agent
from ai_agent import EnhancedTeacherTrainingGraph

# Import the LlamaIndex knowledge manager
from llama_index_integration import LlamaIndexKnowledgeManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TeacherAssistantWithKnowledge:
    """
    Enhanced teacher assistant that uses LlamaIndex for knowledge retrieval.
    This class demonstrates how to integrate the AI agent with LlamaIndex.
    """
    
    def __init__(
        self,
        documents_dir: str = "knowledge_base/books",
        llm_provider: str = "openai",
        model_name: str = "gpt-3.5-turbo",
        enable_caching: bool = True,
        cache_ttl: int = 3600  # 1 hour cache TTL
    ):
        """
        Initialize the teacher assistant with knowledge integration.
        
        Args:
            documents_dir: Directory containing educational materials
            llm_provider: LLM provider to use ("openai", "anthropic", or "local")
            model_name: Name of the LLM model to use
            enable_caching: Whether to enable query caching
            cache_ttl: Time-to-live for cached queries in seconds
        """
        # Initialize the knowledge manager
        self.knowledge_manager = LlamaIndexKnowledgeManager(
            documents_dir=documents_dir,
            llm_provider=llm_provider,
            enable_caching=enable_caching,
            cache_ttl=cache_ttl
        )
        
        # Initialize the AI agent
        self.agent = EnhancedTeacherTrainingGraph(model_name=model_name)
        
        # Load or create the knowledge index
        try:
            logger.info("Loading or creating knowledge index...")
            self.knowledge_manager.load_or_create_index()
            logger.info("Knowledge index ready")
        except Exception as e:
            logger.error(f"Error initializing knowledge index: {e}")
            raise
    
    def answer_educational_question(self, question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Answer an educational question using both the LlamaIndex knowledge 
        and the AI agent's capabilities.
        
        Args:
            question: The educational question to answer
            context: Optional context information for the agent
            
        Returns:
            Dictionary containing the response and sources
        """
        logger.info(f"Processing question: {question}")
        
        try:
            # First, retrieve relevant knowledge from LlamaIndex
            knowledge_result = self.knowledge_manager.query_knowledge(question, top_k=3)
            
            # Check if there was an error in knowledge retrieval
            if "error" in knowledge_result:
                logger.warning(f"Knowledge retrieval error: {knowledge_result['error']}")
                # Continue with just the agent response
                knowledge_context = ""
                sources = []
            else:
                # Extract the knowledge context and sources
                knowledge_context = knowledge_result["response"]
                sources = knowledge_result["sources"]
                logger.info(f"Retrieved {len(sources)} relevant sources")
            
            # Create a combined context for the agent
            agent_context = {
                "knowledge_context": knowledge_context,
                "knowledge_sources": sources[:3],  # Limit to top 3 sources
            }
            
            # If additional context was provided, merge it
            if context:
                agent_context.update(context)
            
            # Get the agent's response
            formatted_question = f"""
            Based on educational research and best practices, please answer the following question:
            
            {question}
            """
            agent_response = self.agent.run(formatted_question, agent_context)
            
            # Combine the results
            result = {
                "response": agent_response.get("response", "No response generated"),
                "sources": sources,
                "analysis": agent_response.get("analysis", {}),
                "feedback": agent_response.get("agent_feedback", "")
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                "error": str(e),
                "response": f"I encountered an error while trying to answer your question: {str(e)}",
                "sources": []
            }
    
    def clear_knowledge_cache(self):
        """Clear the knowledge cache."""
        try:
            self.knowledge_manager.clear_cache()
            logger.info("Knowledge cache cleared")
        except Exception as e:
            logger.error(f"Error clearing knowledge cache: {e}")


def main():
    """Run a demonstration of the integrated system."""
    parser = argparse.ArgumentParser(description="Teacher Assistant with LlamaIndex Knowledge Integration")
    parser.add_argument("--query", type=str, default="", help="Educational query to test")
    parser.add_argument("--documents", type=str, default="knowledge_base/books", help="Path to educational documents")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="LLM model to use")
    parser.add_argument("--provider", type=str, default="openai", help="LLM provider (openai, anthropic, local)")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    
    args = parser.parse_args()
    
    # Create the integrated assistant
    try:
        assistant = TeacherAssistantWithKnowledge(
            documents_dir=args.documents,
            llm_provider=args.provider,
            model_name=args.model,
            enable_caching=not args.no_cache
        )
        
        # Process the query if provided
        if args.query:
            print(f"\nProcessing query: {args.query}\n")
            result = assistant.answer_educational_question(args.query)
            
            print("\n==== RESPONSE ====\n")
            print(result["response"])
            
            if result.get("sources"):
                print("\n==== SOURCES ====\n")
                for i, source in enumerate(result["sources"][:3], 1):
                    print(f"Source {i}:")
                    print(f"{source[:300]}...\n")
        else:
            # Demonstration queries
            demo_queries = [
                "What are research-based strategies for teaching fractions to students with math anxiety?",
                "How can I implement differentiated instruction in a classroom with diverse learning needs?",
                "What are effective classroom management techniques for middle school students?"
            ]
            
            for query in demo_queries:
                print(f"\n\n{'='*80}")
                print(f"QUERY: {query}")
                print(f"{'='*80}\n")
                
                result = assistant.answer_educational_question(query)
                
                print("\n==== RESPONSE ====\n")
                print(result["response"])
                
                if result.get("sources"):
                    print("\n==== SOURCES ====\n")
                    for i, source in enumerate(result["sources"][:2], 1):
                        print(f"Source {i}:")
                        print(f"{source[:200]}...\n")
                        
                print(f"{'='*80}")
    
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 
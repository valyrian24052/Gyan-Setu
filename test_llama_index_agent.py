#!/usr/bin/env python3
"""
Test Integration of LlamaIndex with AI Agent

This script demonstrates how to integrate LlamaIndex-based knowledge retrieval
with the existing AI agent system. It shows how to use LlamaIndex's advanced
retrieval capabilities to enhance the agent's responses with educational content.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import agent components
from ai_agent import TeacherAgent
from llama_index_integration import LlamaIndexKnowledgeManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LlamaIndexEnhancedAgent(TeacherAgent):
    """
    An extension of the TeacherAgent that integrates LlamaIndex
    for advanced knowledge retrieval.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the agent with LlamaIndex integration."""
        # Initialize the base agent
        super().__init__(*args, **kwargs)
        
        # Initialize LlamaIndex knowledge manager
        self.llama_index_manager = LlamaIndexKnowledgeManager(
            documents_dir=kwargs.get("documents_dir", "knowledge_base/books"),
            index_dir=kwargs.get("index_dir", "knowledge_base/llama_index"),
            llm_provider=kwargs.get("llm_provider", "local"),
            local_model_path=kwargs.get("local_model_path", None),
            chunk_size=kwargs.get("chunk_size", 500),
            chunk_overlap=kwargs.get("chunk_overlap", 50)
        )
        
        # Load or create the index
        self.llama_index_manager.load_or_create_index()
        logger.info("LlamaIndex knowledge base initialized and ready")
        
    def _enhanced_response_generation(self, query, student_context=None, max_tokens=1000):
        """
        Generate responses using both the base agent and LlamaIndex knowledge.
        
        Args:
            query: The student's query or prompt
            student_context: Optional context about the student
            max_tokens: Maximum response length
            
        Returns:
            An enhanced response with educational content
        """
        # First, get knowledge from LlamaIndex
        knowledge_result = self.llama_index_manager.query_knowledge(query)
        retrieved_knowledge = knowledge_result["response"]
        knowledge_sources = knowledge_result["sources"]
        
        # Create an enhanced context with the retrieved knowledge
        enhanced_context = f"""
I have the following educational knowledge that may be relevant:
{retrieved_knowledge}

Based on this knowledge, I will answer the student's question.
"""
        
        # Generate a response using the base agent's method with enhanced context
        response = self._generate_response(
            query,
            system_context=enhanced_context,
            student_context=student_context,
            max_tokens=max_tokens
        )
        
        return {
            "response": response,
            "knowledge_used": retrieved_knowledge,
            "sources": knowledge_sources[:3]  # Include top 3 sources
        }
    
    def respond_to_student(self, query, student_context=None):
        """
        Override the base respond_to_student method to use enhanced generation.
        
        Args:
            query: The student's query
            student_context: Optional context about the student
            
        Returns:
            The agent's response
        """
        result = self._enhanced_response_generation(query, student_context)
        
        # Optionally log the sources used
        if hasattr(self, 'debug') and self.debug:
            logger.info("Knowledge sources used:")
            for i, source in enumerate(result["sources"], 1):
                logger.info(f"Source {i}: {source[:100]}...")
        
        return result["response"]

def main():
    """Demonstrate the LlamaIndex-enhanced agent."""
    print("Initializing LlamaIndex-enhanced Teacher Agent...")
    
    # Use local model if available, otherwise fall back to OpenAI
    local_model_path = os.path.join("models", "llama3-8b-instruct.Q5_K_M.gguf")
    if not os.path.exists(local_model_path):
        print("Local model not found, using OpenAI API instead")
        llm_provider = "openai"
        local_model_path = None
    else:
        print(f"Using local model at: {local_model_path}")
        llm_provider = "local"
    
    # Create the enhanced agent
    agent = LlamaIndexEnhancedAgent(
        model="gpt-3.5-turbo",  # Base agent model (can be different from LlamaIndex LLM)
        documents_dir="knowledge_base/books",
        llm_provider=llm_provider,
        local_model_path=local_model_path,
        debug=True
    )
    
    # Example student queries
    example_queries = [
        "I'm having trouble understanding fractions. Can you explain them to me?",
        "Why do I need to learn algebra if I want to be an artist?",
        "I get anxious during math tests. What can I do?",
        "How can I remember the difference between mitosis and meiosis?"
    ]
    
    # Test the agent with each query
    for i, query in enumerate(example_queries, 1):
        print(f"\n\n--- Example {i} ---")
        print(f"Student: {query}")
        
        # Create a sample student context
        student_context = {
            "grade": "second grade" if i <= 2 else "high school",
            "learning_style": "visual learner",
            "interests": ["art", "music"] if i % 2 == 0 else ["science", "computers"]
        }
        
        # Get the agent's response
        response = agent.respond_to_student(query, student_context)
        print(f"\nTeacher Agent: {response}")
    
if __name__ == "__main__":
    main() 
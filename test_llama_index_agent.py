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
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import agent components
from ai_agent import TeacherTrainingGraph
from llama_index_integration import LlamaIndexKnowledgeManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LlamaIndexEnhancedGraph(TeacherTrainingGraph):
    """
    An extension of the TeacherTrainingGraph that integrates LlamaIndex
    for advanced knowledge retrieval.
    """
    
    def __init__(self, model_name="gpt-4", **kwargs):
        """Initialize the graph with LlamaIndex integration."""
        # Initialize the base graph
        super().__init__(model_name=model_name)
        
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
        
        # Add debug flag
        self.debug = kwargs.get("debug", False)
        
    def _generate_student_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Override to enhance student response generation with educational content from LlamaIndex.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with enhanced student response
        """
        # First, get the basic student response using the parent method
        state = super()._generate_student_response(state)
        
        # Get the scenario and teaching approach
        scenario = state.get("scenario", {})
        subject = scenario.get("subject", "general education")
        query = f"As a student learning about {subject}, what would I want to know about this topic?"
        
        # Use LlamaIndex to get relevant educational content
        try:
            knowledge_result = self.llama_index_manager.query_knowledge(query)
            retrieved_knowledge = knowledge_result["response"]
            
            # Enhance the student response with educational content
            current_response = state.get("student_responses", [])[-1] if state.get("student_responses") else ""
            
            enhanced_response = (
                f"{current_response}\n\n"
                f"I'm also curious about some of the things I've learned: {retrieved_knowledge[:200]}..."
            )
            
            # Update the student response in the state
            if state.get("student_responses"):
                state["student_responses"][-1] = enhanced_response
            
            # Log sources if debug is enabled
            if self.debug:
                logger.info("Knowledge sources used:")
                for i, source in enumerate(knowledge_result.get("sources", [])[:3], 1):
                    logger.info(f"Source {i}: {source[:100]}...")
        
        except Exception as e:
            logger.error(f"Error enhancing student response with LlamaIndex: {e}")
        
        return state
        
    def run_with_llama_index(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the graph with LlamaIndex enhancements.
        
        Args:
            user_input: The teacher's input
            context: Optional context parameters
            
        Returns:
            The final state after running the graph
        """
        # Use the parent's run method
        result = self.run(user_input, context)
        
        # Add information about LlamaIndex usage
        if hasattr(self, 'debug') and self.debug:
            logger.info("LlamaIndex enhanced response generated")
        
        return result

def main():
    """Demonstrate the LlamaIndex-enhanced agent."""
    print("Initializing LlamaIndex-enhanced Teacher Training Graph...")
    
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
    agent = LlamaIndexEnhancedGraph(
        model_name="gpt-3.5-turbo",  # Base agent model
        documents_dir="knowledge_base/books",
        llm_provider=llm_provider,
        local_model_path=local_model_path,
        debug=True
    )
    
    # Example teacher inputs
    example_inputs = [
        "I would introduce fractions using visual aids like pizza slices",
        "To explain the importance of algebra, I would connect it to artistic concepts like proportion and perspective",
        "For test anxiety, I would teach relaxation techniques and provide practice tests",
        "I would explain cell division using a side-by-side comparison and mnemonic devices"
    ]
    
    # Test the agent with each input
    for i, user_input in enumerate(example_inputs, 1):
        print(f"\n\n--- Example {i} ---")
        print(f"Teacher: {user_input}")
        
        # Create context for the scenario
        context = {
            "subject": "mathematics" if i <= 2 else "biology",
            "difficulty": "beginner" if i % 2 == 0 else "intermediate",
            "student_profile": {
                "grade_level": "elementary" if i <= 2 else "high school",
                "learning_style": ["visual", "hands-on"],
                "challenges": ["abstract concepts"] if i % 2 == 0 else ["test anxiety"],
                "strengths": ["creativity"] if i % 2 == 0 else ["memorization"]
            }
        }
        
        # Run the enhanced agent
        result = agent.run_with_llama_index(user_input, context)
        
        # Display the student response
        student_responses = result.get("student_responses", [])
        if student_responses:
            print(f"\nStudent: {student_responses[-1]}")
        
        # Display the agent's feedback
        agent_feedback = result.get("agent_feedback", "")
        if agent_feedback:
            print(f"\nFeedback: {agent_feedback}")
    
if __name__ == "__main__":
    main() 
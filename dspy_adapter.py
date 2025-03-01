"""
DSPy Adapter Module

This adapter module provides compatibility between the existing application's
LangChain-based code and the new DSPy implementation. It ensures that the
application can use our enhanced DSPy features without requiring modifications
to the existing codebase.

The adapter maintains the same interface expected by the application while
delegating the actual processing to the DSPy implementation.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Union

# Import the DSPy implementation
from dspy_llm_handler import EnhancedDSPyLLMInterface, PedagogicalLanguageProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class LLMInterface:
    """
    Legacy LLMInterface that adapts to the new DSPy implementation.
    
    This class maintains the same interface as the original LLMInterface
    while delegating the actual work to our DSPy implementation.
    """
    
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0.7):
        """Initialize the legacy interface with the DSPy implementation."""
        self.model_name = model_name
        self.temperature = temperature
        self.chat_model = None
        
        # Initialize the DSPy implementation
        logging.info(f"Initializing DSPy LLM Interface with model: {model_name}")
        self.dspy_interface = EnhancedDSPyLLMInterface(model_name=model_name)
    
    def _initialize_model(self, model_name):
        """
        Initialize the appropriate language model.
        This method exists for compatibility with the original interface.
        The actual initialization is handled by the DSPy implementation.
        """
        try:
            # We already initialized the model in __init__, so just log it
            logging.info(f"Model {model_name} already initialized in DSPy implementation")
            return self.dspy_interface.lm
        except Exception as e:
            logging.error(f"Error initializing model: {e}")
            raise e
    
    def get_chat_response(self, messages, stream=False):
        """
        Get a response from the chat model.
        
        Args:
            messages: List of message dictionaries with role and content
            stream: Whether to stream the response (ignored in this implementation)
            
        Returns:
            str: The model's response
        """
        try:
            # Use the DSPy implementation to get the response
            logging.info("Getting chat response using DSPy implementation")
            return self.dspy_interface.get_llm_response(messages)
        except Exception as e:
            logging.error(f"Error getting chat response: {e}")
            return "I'm sorry, I encountered an error while processing your request."
    
    def generate_with_retry(self, messages, max_retries=3):
        """
        Generate a response with retry logic.
        
        Args:
            messages: List of message dictionaries
            max_retries: Maximum number of retries
            
        Returns:
            str: The model's response
        """
        # DSPy implementation already includes retry logic
        return self.dspy_interface.get_llm_response(messages)


class EnhancedLLMInterface(LLMInterface):
    """
    Legacy EnhancedLLMInterface that adapts to the new DSPy implementation.
    
    This class maintains the same interface as the original EnhancedLLMInterface
    while delegating the actual work to our DSPy implementation.
    """
    
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0.7):
        """Initialize the enhanced interface with the DSPy implementation."""
        super().__init__(model_name, temperature)
        
        # Also initialize the pedagogical processor for educational scenarios
        self.pedagogical_processor = PedagogicalLanguageProcessor(model=model_name)
    
    def get_streaming_response(self, messages):
        """
        Get a streaming response from the model.
        
        Note: True streaming is not supported in this implementation,
        but we maintain the method for compatibility.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            str: The complete response (non-streaming)
        """
        logging.warning("Streaming not supported in DSPy adapter; returning complete response")
        return self.get_chat_response(messages, stream=False)
    
    def analyze_teaching_approach(self, teaching_input, context):
        """
        Analyze a teaching approach.
        
        Args:
            teaching_input: The teacher's statement or approach
            context: Contextual information about the scenario
            
        Returns:
            dict: Analysis of the teaching approach
        """
        # Delegate to the pedagogical processor
        return self.pedagogical_processor.analyze_teaching_response(teaching_input, context)
    
    def generate_teaching_recommendation(self, scenario, student_profile):
        """
        Generate teaching recommendations.
        
        Args:
            scenario: Teaching scenario details
            student_profile: Student characteristics and needs
            
        Returns:
            dict: Teaching recommendations
        """
        return self.dspy_interface.generate_teaching_recommendation(scenario, student_profile)
    
    def simulate_student_response(self, teacher_input, student_profile, scenario_context):
        """
        Simulate a student's response to a teacher's input.
        
        Args:
            teacher_input: The teacher's statement or question
            student_profile: Student characteristics
            scenario_context: Additional context about the scenario
            
        Returns:
            str: A simulated student response
        """
        return self.pedagogical_processor.generate_student_reaction(
            teacher_input, student_profile, scenario_context
        )


# Factory function for creating the appropriate interface
def create_llm_interface(model_name="gpt-3.5-turbo", enhanced=True, temperature=0.7):
    """
    Create an LLM interface with the specified parameters.
    
    Args:
        model_name: Name of the model to use
        enhanced: Whether to use the enhanced interface
        temperature: Temperature parameter for generation
        
    Returns:
        LLMInterface or EnhancedLLMInterface: The appropriate interface
    """
    if enhanced:
        return EnhancedLLMInterface(model_name=model_name, temperature=temperature)
    else:
        return LLMInterface(model_name=model_name, temperature=temperature) 
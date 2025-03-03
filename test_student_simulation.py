#!/usr/bin/env python3
"""
Test script to debug the StudentReactionGenerator and student simulation functionality.
This script isolates the student simulation functionality for faster testing and debugging.
"""

import os
import sys
import json
import logging
import dspy
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()

# Make sure we can import from the local directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the required modules
from dspy_adapter import create_llm_interface, EnhancedLLMInterface
from dspy_llm_handler import StudentReactionGenerator, PedagogicalLanguageProcessor, DSPyConfigManager

def test_dspy_config():
    """Test the DSPy configuration"""
    print("\n=== Testing DSPy Configuration ===")
    
    # Create an enhanced LLM interface
    llm = create_llm_interface(model_name="gpt-3.5-turbo", enhanced=True)
    
    # Configure DSPy settings
    success = llm.dspy_interface.configure_dspy_settings()
    
    print(f"DSPy configuration successful: {success}")
    return llm

def test_student_reaction_generator_direct():
    """Test the StudentReactionGenerator class directly"""
    print("\n=== Testing StudentReactionGenerator Directly ===")
    
    # Configure dspy using the DSPyConfigManager
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    # Use the configuration manager
    config_manager = DSPyConfigManager()
    if not config_manager.is_configured:
        # Import the right OpenAI model
        try:
            from dspy.openai import OpenAI
            config_manager.configure_dspy(OpenAI(model="gpt-3.5-turbo", api_key=api_key))
        except ImportError:
            print("Could not import OpenAI from dspy. Creating LLM interface instead.")
            llm = create_llm_interface(model_name="gpt-3.5-turbo", enhanced=True)
            llm.dspy_interface.configure_dspy_settings()
    
    # Create a sample student profile and teacher input
    teacher_input = "Can you explain how you solved that math problem?"
    student_profile = json.dumps({
        "grade_level": "4th grade",
        "learning_style": ["visual", "hands-on"],
        "challenges": ["math anxiety", "attention span"],
        "strengths": ["creativity", "verbal expression"]
    })
    scenario_context = json.dumps({
        "subject": "Mathematics",
        "topic": "Fractions",
        "classroom_setting": "Small group work"
    })
    
    try:
        # Initialize the generator
        generator = StudentReactionGenerator()
        
        # Generate a student reaction
        print("Generating student reaction...")
        result = generator(
            teacher_input=teacher_input,
            student_profile=student_profile,
            scenario_context=scenario_context
        )
        
        print(f"Student reaction: {result.response}")
        return result.response
    except Exception as e:
        print(f"Error testing StudentReactionGenerator: {e}")
        raise

def test_pedagogical_processor():
    """Test the PedagogicalLanguageProcessor's student reaction generation"""
    print("\n=== Testing PedagogicalLanguageProcessor ===")
    
    # Configure dspy using the DSPyConfigManager
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    # Use the configuration manager
    config_manager = DSPyConfigManager()
    if not config_manager.is_configured:
        # Import the right OpenAI model
        try:
            from dspy.openai import OpenAI
            config_manager.configure_dspy(OpenAI(model="gpt-3.5-turbo", api_key=api_key))
        except ImportError:
            print("Could not import OpenAI from dspy. Creating LLM interface instead.")
            llm = create_llm_interface(model_name="gpt-3.5-turbo", enhanced=True)
            llm.dspy_interface.configure_dspy_settings()
    
    # Create a sample student profile and teacher input
    teacher_input = "Why do you think fractions are difficult for you?"
    student_profile = {
        "grade_level": "4th grade",
        "learning_style": ["visual", "hands-on"],
        "challenges": ["math anxiety", "attention span"],
        "strengths": ["creativity", "verbal expression"]
    }
    scenario_context = {
        "subject": "Mathematics",
        "topic": "Fractions",
        "classroom_setting": "Small group work"
    }
    
    try:
        # Initialize the processor
        processor = PedagogicalLanguageProcessor()
        
        # Generate a student reaction
        print("Generating student reaction via processor...")
        reaction = processor.generate_student_reaction(
            teacher_input=teacher_input,
            student_profile=student_profile,
            scenario_context=scenario_context
        )
        
        print(f"Student reaction: {reaction}")
        return reaction
    except Exception as e:
        print(f"Error testing PedagogicalLanguageProcessor: {e}")
        raise

def test_enhanced_interface():
    """Test the EnhancedLLMInterface's student simulation"""
    print("\n=== Testing EnhancedLLMInterface ===")
    
    # Create an enhanced LLM interface
    llm = create_llm_interface(model_name="gpt-3.5-turbo", enhanced=True)
    
    # Configure DSPy settings
    success = llm.dspy_interface.configure_dspy_settings()
    if not success:
        print("Error: Failed to configure DSPy settings")
        return
    
    # Create a sample student profile and teacher input
    teacher_input = "What's one thing you find challenging about learning fractions?"
    student_profile = {
        "grade_level": "4th grade",
        "learning_style": ["visual", "hands-on"],
        "challenges": ["math anxiety", "attention span"],
        "strengths": ["creativity", "verbal expression"]
    }
    scenario_context = {
        "subject": "Mathematics",
        "topic": "Fractions",
        "classroom_setting": "Small group work"
    }
    
    try:
        # Generate a student response
        print("Simulating student response...")
        response = llm.simulate_student_response(
            teacher_input=teacher_input,
            student_profile=student_profile,
            scenario_context=scenario_context
        )
        
        print(f"Student response: {response}")
        return response
    except Exception as e:
        print(f"Error testing EnhancedLLMInterface: {e}")
        raise

def test_conversation_flow():
    """Test a complete conversation flow with multiple exchanges"""
    print("\n=== Testing Conversation Flow ===")
    
    # Create an enhanced LLM interface
    llm = create_llm_interface(model_name="gpt-3.5-turbo", enhanced=True)
    
    # Configure DSPy settings
    success = llm.dspy_interface.configure_dspy_settings()
    if not success:
        print("Error: Failed to configure DSPy settings")
        return
    
    # Create a sample student profile and scenario context
    student_profile = {
        "name": "Alex",
        "grade_level": "4th grade",
        "learning_style": ["visual", "hands-on"],
        "challenges": ["math anxiety", "attention span"],
        "strengths": ["creativity", "verbal expression"]
    }
    scenario_context = {
        "subject": "Mathematics",
        "topic": "Fractions",
        "classroom_setting": "Small group work",
        "scenario_description": "Students are learning to add fractions with the same denominator."
    }
    
    # Simulate a conversation
    conversation = [
        {"role": "teacher", "content": "Alex, can you tell me what 1/4 plus 2/4 equals?"},
        {"role": "student", "content": None},  # Will be filled in
        {"role": "teacher", "content": "That's right! Can you explain how you got that answer?"},
        {"role": "student", "content": None},  # Will be filled in
        {"role": "teacher", "content": "Good explanation. What's one thing you find challenging about fractions?"},
        {"role": "student", "content": None},  # Will be filled in
    ]
    
    # Fill in the student responses
    for i in range(len(conversation)):
        if conversation[i]["role"] == "teacher" and i + 1 < len(conversation) and conversation[i + 1]["role"] == "student":
            teacher_input = conversation[i]["content"]
            print(f"\nTeacher: {teacher_input}")
            
            try:
                # Generate student response
                student_response = llm.simulate_student_response(
                    teacher_input=teacher_input,
                    student_profile=student_profile,
                    scenario_context=scenario_context
                )
                
                # Update the conversation
                conversation[i + 1]["content"] = student_response
                print(f"Student: {student_response}")
                
            except Exception as e:
                print(f"Error generating student response: {e}")
                conversation[i + 1]["content"] = f"[Error: {str(e)}]"
    
    return conversation

if __name__ == "__main__":
    print("===== Testing Student Simulation Functionality =====")
    
    # Run the tests in order
    try:
        # Test DSPy configuration
        llm = test_dspy_config()
        
        # Test StudentReactionGenerator directly
        test_student_reaction_generator_direct()
        
        # Test PedagogicalLanguageProcessor
        test_pedagogical_processor()
        
        # Test EnhancedLLMInterface
        test_enhanced_interface()
        
        # Test conversation flow
        test_conversation_flow()
        
        print("\n===== All tests completed =====")
    except Exception as e:
        print(f"\nTest suite failed: {e}") 
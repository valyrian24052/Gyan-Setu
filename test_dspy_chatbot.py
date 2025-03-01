#!/usr/bin/env python3
"""
Test script for the DSPy-based teaching assistant chatbot.

This script tests the functionality of the DSPy implementation
and demonstrates its capabilities for teaching-related tasks.
"""

import os
import json
import logging
from dspy_adapter import create_llm_interface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_basic_chat():
    """Test basic chatbot functionality"""
    print("=== Testing Basic Chat ===")
    
    # Create the LLM interface
    llm = create_llm_interface(model_name="gpt-4o-mini")
    
    # Test a simple conversation
    messages = [
        {"role": "system", "content": "You are a helpful teaching assistant."},
        {"role": "user", "content": "What are some effective strategies for teaching fractions to 4th graders?"}
    ]
    
    response = llm.get_chat_response(messages)
    print(f"Response:\n{response}\n")
    return response


def test_teaching_analysis():
    """Test teaching analysis functionality"""
    print("=== Testing Teaching Analysis ===")
    
    # Create the enhanced LLM interface
    llm = create_llm_interface(model_name="gpt-4o-mini", enhanced=True)
    
    # Example teaching approach to analyze
    teaching_approach = """
    Today we're going to learn about fractions. A fraction represents part of a whole.
    Think of a pizza divided into 8 slices. If you eat 3 slices, you've eaten 3/8 of the pizza.
    Let's practice by drawing some pictures and dividing them into equal parts.
    """
    
    # Context for the analysis
    context = {
        "grade_level": "4th grade",
        "subject": "Mathematics",
        "topic": "Introduction to Fractions",
        "student_profile": {
            "learning_style": "visual",
            "challenges": ["abstract concepts"],
            "strengths": ["artistic expression"]
        }
    }
    
    # Analyze the teaching approach
    analysis = llm.analyze_teaching_approach(teaching_approach, context)
    print(f"Teaching Analysis:\n{json.dumps(analysis, indent=2)}\n")
    return analysis


def test_student_simulation():
    """Test student simulation functionality"""
    print("=== Testing Student Simulation ===")
    
    # Create the enhanced LLM interface
    llm = create_llm_interface(model_name="gpt-4o-mini", enhanced=True)
    
    # Example teacher input
    teacher_input = """
    Can someone tell me what 1/4 plus 2/4 equals? Think about it like a pizza cut into 4 slices.
    If you have 1 slice and then get 2 more slices, how many slices do you have in total?
    """
    
    # Student profile
    student_profile = {
        "name": "Alex",
        "grade": "4th",
        "learning_style": "kinesthetic",
        "challenges": ["attention span", "number sense"],
        "strengths": ["creative thinking", "pattern recognition"]
    }
    
    # Scenario context
    scenario_context = {
        "subject": "Mathematics",
        "topic": "Adding Fractions with Same Denominator",
        "classroom_setting": "Small group work",
        "previous_topics": ["introduction to fractions", "representing fractions"]
    }
    
    # Generate student response
    student_response = llm.simulate_student_response(
        teacher_input, student_profile, scenario_context
    )
    print(f"Student Response:\n{student_response}\n")
    return student_response


if __name__ == "__main__":
    print("Testing DSPy-based Teaching Assistant Chatbot")
    print("============================================\n")
    
    # Run the tests
    test_basic_chat()
    test_teaching_analysis()
    test_student_simulation()
    
    print("\nAll tests completed!") 
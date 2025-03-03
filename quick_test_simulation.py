#!/usr/bin/env python3
"""
Quick test script for the student simulation functionality.
This script provides a simple command-line interface to test
student responses without running the full web application.
"""

import os
import sys
import json
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the necessary modules
from dspy_adapter import create_llm_interface

def main():
    """Main function to run the quick test"""
    print("=== Quick Test of Student Simulation ===")
    
    # Create the LLM interface with the enhanced version
    print("Initializing LLM interface...")
    llm = create_llm_interface(model_name="gpt-3.5-turbo", enhanced=True)
    
    # Sample student profile
    student_profile = {
        "name": "Alex",
        "grade_level": "4th grade",
        "learning_style": ["visual", "hands-on"],
        "challenges": ["math anxiety", "attention span"],
        "strengths": ["creativity", "verbal expression"]
    }
    
    # Sample scenario context
    scenario_context = {
        "subject": "Mathematics",
        "topic": "Fractions",
        "grade_level": "4th grade",
        "learning_objectives": ["Understand fractions as parts of a whole", 
                                "Add fractions with like denominators"],
        "scenario_description": "Students are working in small groups to solve fraction problems."
    }
    
    print("\nStudent Profile:")
    print(json.dumps(student_profile, indent=2))
    
    print("\nScenario Context:")
    print(json.dumps(scenario_context, indent=2))
    
    print("\n=== Begin Teacher-Student Interaction ===")
    print("(Enter 'quit' to exit)")
    
    # Interactive loop for testing
    while True:
        # Get teacher input
        teacher_input = input("\nTeacher: ")
        if teacher_input.lower() in ['quit', 'exit', 'q']:
            break
        
        # Generate student response
        print("Student is thinking...")
        try:
            student_response = llm.simulate_student_response(
                teacher_input=teacher_input,
                student_profile=student_profile,
                scenario_context=scenario_context
            )
            print(f"Student: {student_response}")
        except Exception as e:
            logger.error(f"Error generating student response: {e}", exc_info=True)
            print(f"Error: {str(e)}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main() 
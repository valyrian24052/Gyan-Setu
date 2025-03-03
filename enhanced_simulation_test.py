#!/usr/bin/env python3
"""
Enhanced simulation test script for the Teacher Training Simulator.
This script implements a more realistic conversation flow where:
1. The teacher receives scenario information first
2. The student initiates the conversation based on the scenario
3. The conversation continues naturally with proper responses
"""

import os
import sys
import json
import logging
import time
from datetime import datetime
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

def generate_student_introduction(llm, student_profile, scenario_context):
    """Generate a student introduction based on the scenario and profile"""
    
    # Create a special prompt for the student introduction
    teacher_input = "Please introduce yourself to the class. Tell us a little about yourself and your feelings about this subject."
    
    try:
        # Generate the student's introduction
        student_intro = llm.simulate_student_response(
            teacher_input=teacher_input,
            student_profile=student_profile,
            scenario_context=scenario_context
        )
        
        return student_intro
    except Exception as e:
        logger.error(f"Error generating student introduction: {e}", exc_info=True)
        return "Hi, I'm a student in this class."

def format_timestamp():
    """Generate a timestamp for chat messages"""
    return datetime.now().strftime("%H:%M:%S")

def format_conversation_history(conversation):
    """Format conversation history in a way that helps the LLM maintain context"""
    formatted_history = []
    
    for entry in conversation:
        if entry["role"] == "teacher":
            formatted_history.append(f"Teacher: {entry['content']}")
        else:
            formatted_history.append(f"Student (Jamie): {entry['content']}")
    
    return "\n".join(formatted_history)

def main():
    """Main function to run the enhanced simulation test"""
    print("\n===== Enhanced Teacher Training Simulator =====")
    print("Improve your teaching skills through realistic simulations")
    
    # Create the LLM interface with the enhanced version
    print("\nInitializing LLM interface...")
    llm = create_llm_interface(model_name="gpt-3.5-turbo", enhanced=True)
    
    # Define a more detailed student profile
    student_profile = {
        "name": "Jamie",
        "grade_level": "7th grade",
        "age": 12,
        "learning_style": ["visual", "hands-on"],
        "challenges": ["math anxiety", "difficulty with abstract concepts", "attention span"],
        "strengths": ["creativity", "verbal expression", "collaborative work"],
        "interests": ["art", "music", "working with friends"],
        "personality": "curious but easily frustrated when concepts seem too abstract"
    }
    
    # Define a more detailed scenario context
    scenario_context = {
        "subject": "Mathematics",
        "topic": "Introduction to Algebra",
        "grade_level": "7th grade",
        "classroom_setting": "Small group work on algebraic expressions",
        "learning_objectives": [
            "Understand variables as representing unknown quantities",
            "Translate word problems into algebraic expressions",
            "Solve simple linear equations"
        ],
        "scenario_description": "Students are working on translating word problems into algebraic expressions. Jamie has been struggling with the concept of variables and is showing signs of frustration. The teacher needs to help Jamie understand the concept while maintaining engagement and confidence.",
        "current_activity": "Students are working on a worksheet with word problems that need to be translated into algebraic expressions."
    }
    
    # Display scenario information for the teacher
    print("\n===== SCENARIO INFORMATION =====")
    print(f"Subject: {scenario_context['subject']} - {scenario_context['topic']}")
    print(f"Grade Level: {scenario_context['grade_level']}")
    print(f"Setting: {scenario_context['classroom_setting']}")
    print("\nLearning Objectives:")
    for i, obj in enumerate(scenario_context["learning_objectives"], 1):
        print(f"  {i}. {obj}")
    
    print("\nScenario Description:")
    print(scenario_context["scenario_description"])
    
    print("\n===== STUDENT PROFILE =====")
    print(f"Name: {student_profile['name']}")
    print(f"Grade: {student_profile['grade_level']} (Age: {student_profile['age']})")
    
    print("\nLearning Style:")
    for style in student_profile["learning_style"]:
        print(f"  - {style.capitalize()}")
    
    print("\nChallenges:")
    for challenge in student_profile["challenges"]:
        print(f"  - {challenge.capitalize()}")
    
    print("\nStrengths:")
    for strength in student_profile["strengths"]:
        print(f"  - {strength.capitalize()}")
    
    print("\nInterests:")
    for interest in student_profile["interests"]:
        print(f"  - {interest.capitalize()}")
    
    # Generate the student's introduction
    print("\n\n===== CONVERSATION =====")
    print("The student will start the conversation. You can then respond as the teacher.")
    print("(Enter 'quit' to exit)")
    
    # Generate the student's introduction
    student_intro = generate_student_introduction(llm, student_profile, scenario_context)
    
    # Start the conversation history
    conversation = [
        {"role": "student", "content": student_intro, "timestamp": format_timestamp()}
    ]
    
    # Display the student's introduction
    print(f"\nStudent ({conversation[0]['timestamp']}):")
    print(f"{student_intro}")
    
    # Keep track of last input and response to prevent repetition
    last_teacher_input = ""
    last_student_response = student_intro
    
    # Interactive loop for the conversation
    while True:
        # Get teacher input
        teacher_input = input(f"\nTeacher ({format_timestamp()}): ")
        if teacher_input.lower() in ['quit', 'exit', 'q']:
            break
        
        # Check if this is a duplicate input
        if teacher_input == last_teacher_input:
            print("Note: You entered the same input again. This may lead to repetitive responses.")
        
        last_teacher_input = teacher_input
        
        # Add teacher input to conversation
        timestamp = format_timestamp()
        conversation.append({"role": "teacher", "content": teacher_input, "timestamp": timestamp})
        
        # Generate student response
        print("Student is thinking...")
        try:
            # Format conversation history as a readable string
            conversation_history = format_conversation_history(conversation)
            
            # Create an augmented scenario context with conversation history
            augmented_context = scenario_context.copy()
            augmented_context["conversation_history"] = conversation_history
            augmented_context["instruction"] = "Remember to respond differently from your previous response. Avoid repetition."
            
            # Create a specific prompt that includes the conversation history
            specific_prompt = (
                f"Teacher's most recent question: {teacher_input}\n\n"
                f"Previous conversation:\n{conversation_history}\n\n"
                f"Respond as Jamie, a 7th grade student with the profile: {json.dumps(student_profile)}\n"
                f"Remember to respond DIFFERENTLY than your previous response, which was: '{last_student_response}'\n"
                f"Make sure your response is unique and moves the conversation forward."
            )
            
            # Use the specific prompt as the teacher input
            student_response = llm.simulate_student_response(
                teacher_input=specific_prompt,
                student_profile=student_profile,
                scenario_context=augmented_context
            )
            
            # Update last response
            last_student_response = student_response
            
            # Add student response to conversation
            timestamp = format_timestamp()
            conversation.append({"role": "student", "content": student_response, "timestamp": timestamp})
            
            # Display the student's response
            print(f"\nStudent ({timestamp}):")
            print(f"{student_response}")
            
        except Exception as e:
            logger.error(f"Error generating student response: {e}", exc_info=True)
            print(f"Error: {str(e)}")
    
    print("\n===== Conversation End =====")
    print(f"Total exchanges: {len(conversation) // 2}")
    
    # Save the conversation if requested
    save = input("\nWould you like to save this conversation? (y/n): ")
    if save.lower() == 'y':
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(conversation, f, indent=2)
        print(f"Conversation saved to {filename}")

if __name__ == "__main__":
    main() 
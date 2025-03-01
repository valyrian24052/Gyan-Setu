"""
Streamlit web application for the Teacher Training Simulator.
Provides an interactive interface for teachers to practice and analyze teaching scenarios.
Uses LangGraph for state management and orchestration.
"""

import streamlit as st
import os
import pandas as pd
import plotly.express as px
from datetime import datetime
import json
from dotenv import load_dotenv
from ai_agent import TeacherTrainingGraph, EnhancedTeacherTrainingGraph
from llm_handler import PedagogicalLanguageProcessor, EnhancedLLMInterface
import subprocess
import getpass

# Load environment variables from .env file
load_dotenv()

# Make sure the API key is set
if not os.environ.get("OPENAI_API_KEY"):
    st.error("OpenAI API key not found! Please make sure it's set in the .env file or as an environment variable.")
    st.stop()

class WebInterface:
    def __init__(self):
        """Initialize the web interface and session state."""
        # Agent will be initialized after getting the token
        if 'agent' not in st.session_state:
            st.session_state.agent = None
        if 'scenario' not in st.session_state:
            st.session_state.scenario = None
        if 'history' not in st.session_state:
            st.session_state.history = []
        if 'analysis' not in st.session_state:
            st.session_state.analysis = None
        if 'strategies' not in st.session_state:
            st.session_state.strategies = []
        if 'teacher_feedback' not in st.session_state:
            st.session_state.teacher_feedback = None
        if 'conversation_id' not in st.session_state:
            st.session_state.conversation_id = None
        if 'student_profile' not in st.session_state:
            st.session_state.student_profile = None
        if 'reflection' not in st.session_state:
            st.session_state.reflection = None

    def setup_page(self):
        """Configure the Streamlit page layout and styling."""
        st.set_page_config(
            page_title="Teacher Training Simulator",
            page_icon="🎓",
            layout="wide"
        )

        # Load custom CSS
        try:
            with open('style.css') as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        except FileNotFoundError:
            st.warning("Custom CSS file not found. Using default styling.")

        # Page header
        st.title("Teacher Training Simulator")
        st.markdown("### Improve your teaching skills through realistic simulations")

    def initialize_agent(self, model_name="gpt-4"):
        """Initialize the LangGraph agent with the specified model."""
        try:
            # Using the enhanced implementation that supports LangGraph
            st.session_state.agent = EnhancedTeacherTrainingGraph(model_name=model_name)
            return True
        except Exception as e:
            st.error(f"Error initializing agent: {str(e)}")
            return False

    def create_scenario(self):
        """Create a teaching scenario with specified parameters."""
        st.markdown("## Create a Teaching Scenario")
        
        with st.form(key="scenario_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                subject = st.selectbox(
                    "Subject", 
                    ["Mathematics", "Science", "Language Arts", "Social Studies", "Art", "Music", "Physical Education"]
                )
                
                grade_level = st.selectbox(
                    "Grade Level",
                    ["Kindergarten", "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th", "12th"]
                )
                
                learning_objectives = st.text_area(
                    "Learning Objectives (one per line)",
                    "Understand fractions as parts of a whole\nCompare fractions with common denominators\nVisualize fractions using diagrams"
                )
            
            with col2:
                learning_styles = st.multiselect(
                    "Student Learning Styles",
                    ["Visual", "Auditory", "Reading/Writing", "Kinesthetic"],
                    ["Visual", "Kinesthetic"]
                )
                
                challenges = st.multiselect(
                    "Student Challenges",
                    ["Attention span", "Abstract concepts", "Reading comprehension", "Math anxiety", 
                     "Organization", "Memory", "Processing speed", "English language learner"],
                    ["Abstract concepts"]
                )
                
                strengths = st.multiselect(
                    "Student Strengths",
                    ["Creativity", "Verbal skills", "Visual-spatial thinking", "Memory", 
                     "Logical reasoning", "Collaboration", "Self-motivation", "Persistence"],
                    ["Creativity", "Visual-spatial thinking"]
                )
            
            submit_button = st.form_submit_button("Create Scenario")
        
        if submit_button:
            if not st.session_state.agent:
                if not self.initialize_agent():
                    st.error("Failed to initialize agent. Please check your API key.")
                    return
            
            try:
                # Format learning objectives as a list
                objectives_list = [obj.strip() for obj in learning_objectives.split('\n') if obj.strip()]
                
                # Create student profile
                student_profile = {
                    "grade_level": grade_level,
                    "learning_style": [style.lower() for style in learning_styles],
                    "challenges": [challenge.lower() for challenge in challenges],
                    "strengths": [strength.lower() for strength in strengths]
                }
                
                # Store the student profile in session state
                st.session_state.student_profile = student_profile
                
                # Create custom scenario using LangGraph agent
                scenario = st.session_state.agent.create_custom_scenario(
                    subject=subject,
                    grade_level=grade_level,
                    learning_objectives=objectives_list,
                    student_characteristics=student_profile
                )
                
                # Store the scenario
                st.session_state.scenario = scenario
                
                # Clear previous history if any
                st.session_state.history = []
                st.session_state.analysis = None
                st.session_state.strategies = []
                st.session_state.teacher_feedback = None
                st.session_state.reflection = None
                
                # Display success message
                st.success("Teaching scenario created successfully!")
                
                # Display the scenario details
                self.display_scenario(scenario)
                
                # Scroll to the simulation section
                js = f"""
                <script>
                    function scroll() {{
                        document.querySelector('h2:contains("Practice Teaching Interaction")').scrollIntoView();
                    }}
                    setTimeout(scroll, 500);
                </script>
                """
                st.markdown(js, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error creating scenario: {str(e)}")
    
    def display_scenario(self, scenario):
        """Display the details of the current teaching scenario."""
        st.markdown("## Scenario Details")
        
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Scenario")
            st.markdown(f"**Subject:** {scenario.get('subject', 'Not specified')}")
            st.markdown(f"**Description:** {scenario.get('description', 'No description available')}")
            
            # Learning objectives
            st.markdown("### Learning Objectives")
            objectives = scenario.get('learning_objectives', [])
            if objectives:
                for i, obj in enumerate(objectives, 1):
                    st.markdown(f"{i}. {obj}")
            else:
                st.markdown("No learning objectives specified")
        
        with col2:
            st.markdown("### Student Profile")
            
            student_profile = st.session_state.student_profile
            if student_profile:
                st.markdown(f"**Grade Level:** {student_profile.get('grade_level', 'Not specified')}")
                
                # Learning styles
                learning_styles = student_profile.get('learning_style', [])
                if learning_styles:
                    st.markdown(f"**Learning Styles:** {', '.join(style.capitalize() for style in learning_styles)}")
                
                # Challenges
                challenges = student_profile.get('challenges', [])
                if challenges:
                    st.markdown("**Challenges:**")
                    for challenge in challenges:
                        st.markdown(f"- {challenge.capitalize()}")
                
                # Strengths
                strengths = student_profile.get('strengths', [])
                if strengths:
                    st.markdown("**Strengths:**")
                    for strength in strengths:
                        st.markdown(f"- {strength.capitalize()}")
            else:
                st.markdown("No student profile available")
            
            # Additional scenario information
            st.markdown("### Potential Challenges")
            challenges = scenario.get('challenges', [])
            if challenges:
                for challenge in challenges:
                    st.markdown(f"- {challenge}")
            else:
                st.markdown("No challenges specified")

    def display_simulation_interface(self):
        """Display the interface for practicing teaching interactions."""
        st.markdown("## Practice Teaching Interaction")
        
        if not st.session_state.scenario:
            st.info("Please create a scenario first to begin the simulation.")
            return
        
        # Display conversation history
        self.display_conversation()
        
        # Input for teacher's response
        teacher_input = st.text_area("Your teaching response:", height=100, 
                                    placeholder="Enter your teaching approach or response...")
        
        # Create a key for the submit button to avoid redundant clicks
        if 'submit_key' not in st.session_state:
            st.session_state.submit_key = 0
        
        # Add a spinner for visual feedback during processing
        submit_button = st.button("Submit Response", key=f"submit_{st.session_state.submit_key}")
        
        if submit_button:
            if not teacher_input.strip():
                st.warning("Please enter a teaching response before submitting.")
                return
            
            # Increment the key to prevent double submission
            st.session_state.submit_key += 1
            
            # Show processing indicator
            with st.spinner("Processing your response..."):
                if not st.session_state.agent:
                    if not self.initialize_agent():
                        st.error("Failed to initialize agent. Please check your API key.")
                        return
                
                try:
                    # Record teacher's message in history
                    st.session_state.history.append({
                        "role": "teacher",
                        "content": teacher_input,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    
                    # Process with LangGraph agent
                    if st.session_state.conversation_id:
                        # Continue existing conversation
                        result = st.session_state.agent.run(teacher_input)
                    else:
                        # Start new conversation with custom scenario
                        result = st.session_state.agent.run_with_custom_scenario(
                            user_input=teacher_input, 
                            scenario=st.session_state.scenario
                        )
                    
                    # Extract information from the response
                    response = result.get("response", "")
                    state = result.get("state", {})
                    st.session_state.conversation_id = result.get("conversation_id")
                    
                    # Update session state with relevant information
                    if "analysis" in state and state["analysis"]:
                        st.session_state.analysis = state["analysis"]
                    
                    # Process feedback and student responses
                    if response:
                        # Handle combined response with both student message and feedback
                        parts = response.split("\n\nFeedback:")
                        
                        # First part is student response if not starting with Feedback
                        if not response.startswith("Feedback:") and len(parts) > 0 and "Student:" in parts[0]:
                            # Store student response
                            st.session_state.history.append({
                                "role": "student",
                                "content": parts[0],
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                            print(f"Added student response to history: {parts[0]}")
                        
                        # Second part or first part if starts with Feedback is feedback
                        feedback_part = parts[1] if len(parts) > 1 else (parts[0] if response.startswith("Feedback:") else None)
                        
                        if feedback_part or "agent_feedback" in state:
                            feedback_content = f"Feedback: {feedback_part}" if feedback_part else f"Feedback: {state.get('agent_feedback', '')}"
                            st.session_state.teacher_feedback = feedback_part if feedback_part else state.get('agent_feedback', '')
                            
                            # Add feedback to history
                            st.session_state.history.append({
                                "role": "system",
                                "content": feedback_content,
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                    
                    # Fallback for student response if not in the message but in state
                    if "student_responses" in state and state["student_responses"] and not any("Student:" in entry["content"] for entry in st.session_state.history[-2:]):
                        # Get the latest student response
                        latest_response = state["student_responses"][-1]
                        
                        # Add student response to history explicitly
                        st.session_state.history.append({
                            "role": "student",
                            "content": f"Student: {latest_response}",
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        })
                        
                        # Log that a student response was added
                        print(f"Added student response to history: {latest_response}")
                    
                    # Extract reflection if available
                    if "messages" in state and state["messages"]:
                        for msg in state["messages"]:
                            if isinstance(msg, dict) and msg.get("content", "").startswith("Reflection:"):
                                st.session_state.reflection = msg["content"]
                    
                    # Refresh display
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing response: {str(e)}")
                    # Log detailed error information
                    import traceback
                    print(f"Detailed error: {traceback.format_exc()}")
                    
                    # Add a generic student response if there was an error
                    fallback_response = "I'm a bit confused. Could you please explain that again?"
                    st.session_state.history.append({
                        "role": "student",
                        "content": f"Student: {fallback_response}",
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    
                    # Add a system message about the error
                    st.session_state.history.append({
                        "role": "system",
                        "content": "Feedback: Try to clarify your teaching approach and be more specific about what you're trying to teach.",
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    
                    # Refresh display
                    st.rerun()
    
    def display_conversation(self):
        """Display the teaching conversation history."""
        if not st.session_state.history:
            st.info("Start the conversation by submitting your teaching approach.")
            return
        
        # Create conversation container
        st.markdown("### Conversation")
        
        # Use st.container with a fixed height to prevent the conversation from taking too much space
        conversation_container = st.container()
        
        with conversation_container:
            # Log the conversation history state for debugging
            print(f"Conversation history: {len(st.session_state.history)} messages")
            
            # Group messages by role for more efficient rendering
            current_role = None
            grouped_messages = []
            current_group = []
            
            for entry in st.session_state.history:
                if entry["role"] != current_role:
                    if current_group:
                        grouped_messages.append((current_role, current_group.copy()))
                        current_group = []
                    current_role = entry["role"]
                current_group.append(entry)
            
            if current_group:
                grouped_messages.append((current_role, current_group))
            
            # Render grouped messages more efficiently
            for role, entries in grouped_messages:
                if role == "teacher":
                    for entry in entries:
                        st.markdown(f'<div class="teacher-message"><span class="timestamp">{entry["timestamp"]}</span><span class="role">Teacher:</span> {entry["content"]}</div>', unsafe_allow_html=True)
                elif role == "student":
                    for entry in entries:
                        # Make sure student messages are prominently displayed
                        content = entry["content"]
                        if not content.startswith("Student:"):
                            content = f"Student: {content}"
                        st.markdown(f'<div class="student-message"><span class="timestamp">{entry["timestamp"]}</span> {content}</div>', unsafe_allow_html=True)
                elif role == "system":
                    for entry in entries:
                        st.markdown(f'<div class="system-message"><span class="timestamp">{entry["timestamp"]}</span> {entry["content"]}</div>', unsafe_allow_html=True)
        
        # Display reflection if available
        if st.session_state.reflection:
            st.markdown("### Reflection")
            st.info(st.session_state.reflection)

    def display_analysis(self):
        """Display the analysis of the teaching interaction."""
        st.markdown("## Teaching Analysis")
        
        if not st.session_state.analysis:
            if st.session_state.history:
                st.info("Teaching analysis will appear here after more interaction.")
            else:
                st.info("Teaching analysis will appear here after you begin the simulation.")
            return
        
        analysis = st.session_state.analysis
        
        # Display overall assessment
        if "overall_assessment" in analysis:
            st.markdown("### Overall Assessment")
            st.write(analysis["overall_assessment"])
        
        # Display detailed analysis if available
        if "detailed" in analysis:
            st.markdown("### Detailed Analysis")
            st.write(analysis["detailed"])
        
        # Display objective assessment if available
        if "objectives_assessment" in analysis:
            st.markdown("### Learning Objectives Assessment")
            st.write(analysis["objectives_assessment"])
        
        # Display visualization if we have numeric scores
        if "effectiveness_score" in analysis:
            score = analysis["effectiveness_score"]
            st.markdown("### Effectiveness Score")
            
            # Create a gauge chart
            fig = px.bar(
                x=["Effectiveness Score"], 
                y=[score], 
                labels={"x": "", "y": "Score"},
                range_y=[0, 1],
                color=["Effectiveness"],
                title="Teaching Effectiveness"
            )
            st.plotly_chart(fig)
        
        # Display strengths and areas for improvement
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Strengths")
            strengths = analysis.get("identified_strengths", [])
            if strengths:
                for strength in strengths:
                    st.markdown(f"- {strength}")
            else:
                st.write("No specific strengths identified yet.")
        
        with col2:
            st.markdown("### Areas for Improvement")
            improvements = analysis.get("improvement_areas", [])
            if improvements:
                for improvement in improvements:
                    st.markdown(f"- {improvement}")
            else:
                st.write("No specific areas for improvement identified yet.")

    def display_resources(self):
        """Display supplementary resources for teachers."""
        st.markdown("## Resources")
        
        # Create tabs for different resource types
        resources_tabs = st.tabs(["Teaching Strategies", "Research", "Tools"])
        
        with resources_tabs[0]:
            st.markdown("### Effective Teaching Strategies")
            strategies = [
                {
                    "name": "Think-Pair-Share",
                    "description": "Students think individually, discuss with a partner, then share with the class."
                },
                {
                    "name": "Concept Mapping",
                    "description": "Visual organization of information showing relationships between concepts."
                },
                {
                    "name": "Jigsaw Method",
                    "description": "Students become experts on one part of an assignment and teach others."
                },
                {
                    "name": "Scaffolding",
                    "description": "Providing temporary support that is gradually removed as students gain proficiency."
                },
                {
                    "name": "Formative Assessment",
                    "description": "Ongoing assessment during learning to monitor progress and adjust instruction."
                }
            ]
            
            for strategy in strategies:
                with st.expander(strategy["name"]):
                    st.write(strategy["description"])
        
        with resources_tabs[1]:
            st.markdown("### Research on Effective Teaching")
            
            research_papers = [
                {
                    "title": "Visible Learning: A Synthesis of Over 800 Meta-Analyses Relating to Achievement",
                    "author": "John Hattie",
                    "year": 2009,
                    "summary": "Comprehensive analysis of factors that influence student achievement."
                },
                {
                    "title": "How People Learn: Brain, Mind, Experience, and School",
                    "author": "National Research Council",
                    "year": 2000,
                    "summary": "Examines the science of learning and its implications for teaching."
                },
                {
                    "title": "The Power of Feedback",
                    "author": "John Hattie & Helen Timperley",
                    "year": 2007,
                    "summary": "Analysis of how different types of feedback affect learning and achievement."
                }
            ]
            
            for paper in research_papers:
                with st.expander(f"{paper['title']} ({paper['year']})"):
                    st.write(f"**Author(s):** {paper['author']}")
                    st.write(f"**Summary:** {paper['summary']}")
        
        with resources_tabs[2]:
            st.markdown("### Educational Tools")
            
            tools = [
                {
                    "name": "Kahoot!",
                    "description": "Game-based learning platform for creating interactive quizzes and assessments."
                },
                {
                    "name": "Padlet",
                    "description": "Digital bulletin board for collaborative projects and discussions."
                },
                {
                    "name": "Nearpod",
                    "description": "Interactive lesson delivery platform with built-in assessment tools."
                },
                {
                    "name": "Flipgrid",
                    "description": "Video discussion platform for student engagement and reflection."
                }
            ]
            
            for tool in tools:
                with st.expander(tool["name"]):
                    st.write(tool["description"])

    def run(self):
        """Main method to run the web application."""
        # Set up the page
        self.setup_page()
        
        # Sidebar for configuration and navigation
        with st.sidebar:
            st.title("Teacher Training Simulator")
            st.markdown("## Configuration")
            
            # Model selection
            model_options = {
                "OpenAI GPT-4": "gpt-4",
                "OpenAI GPT-3.5 Turbo": "gpt-3.5-turbo",
                "Claude 3 Opus": "claude-3-opus-20240229",
                "Claude 3 Sonnet": "claude-3-sonnet-20240229",
                "Claude 3 Haiku": "claude-3-haiku-20240307",
                "Llama 3 8B (Local)": "llama-3-8b",
                "Llama 3 70B (Local)": "llama-3-70b"
            }
            
            selected_model_name = st.selectbox(
                "Select Model",
                options=list(model_options.keys()),
                index=0
            )
            
            selected_model = model_options[selected_model_name]
            
            # If Llama model is selected, show additional configuration
            if "Llama 3" in selected_model_name:
                st.info("""
                Using local Llama 3 model. Make sure you have either:
                1. Ollama installed with the selected model, or
                2. Downloaded the GGUF model file and set the LLAMA_MODEL_PATH environment variable
                """)
                
                model_path = st.text_input(
                    "Model Path (optional)", 
                    value=os.environ.get("LLAMA_MODEL_PATH", "./models/llama-3-8b.gguf"),
                    help="Path to the GGUF model file if not using Ollama"
                )
                
                if model_path:
                    os.environ["LLAMA_MODEL_PATH"] = model_path
            
            # Initialize agent button
            if st.button("Initialize Agent"):
                with st.spinner("Initializing agent..."):
                    if self.initialize_agent(selected_model):
                        st.success("Agent initialized successfully!")
                    else:
                        st.error("Failed to initialize agent.")
            
            # Navigation
            st.markdown("## Navigation")
            nav_options = [
                "Create Scenario",
                "Practice Teaching",
                "View Analysis",
                "Resources"
            ]
            
            nav_selection = st.radio("Go to:", nav_options)
            
            # Reset button
            if st.button("Reset Simulation"):
                st.session_state.scenario = None
                st.session_state.history = []
                st.session_state.analysis = None
                st.session_state.strategies = []
                st.session_state.teacher_feedback = None
                st.session_state.conversation_id = None
                st.session_state.student_profile = None
                st.session_state.reflection = None
                st.success("Simulation reset successfully!")
        
        # Main content area
        if nav_selection == "Create Scenario":
            self.create_scenario()
        
        elif nav_selection == "Practice Teaching":
            self.display_simulation_interface()
        
        elif nav_selection == "View Analysis":
            self.display_analysis()
        
        elif nav_selection == "Resources":
            self.display_resources()

# Run the application
if __name__ == "__main__":
    app = WebInterface()
    app.run() 
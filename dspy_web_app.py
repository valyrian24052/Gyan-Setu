"""
Streamlit web application for the Teacher Training Simulator with DSPy.
Provides an interactive interface for teachers to practice and analyze teaching scenarios.
This version uses DSPy instead of LangChain for more efficient and reliable LLM interactions.
"""

import streamlit as st
import os
import pandas as pd
import plotly.express as px
from datetime import datetime
import json
from dotenv import load_dotenv
import subprocess
import getpass
import logging

# Import our DSPy adapter and implementations
from dspy_adapter import create_llm_interface, EnhancedLLMInterface
from dspy_llm_handler import PedagogicalLanguageProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load environment variables from .env file
load_dotenv()

class DSPyWebInterface:
    def __init__(self):
        """Initialize the web interface and session state."""
        # LLM interface will be initialized after model selection
        if 'llm_interface' not in st.session_state:
            st.session_state.llm_interface = None
        if 'pedagogical_processor' not in st.session_state:
            st.session_state.pedagogical_processor = None
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
            page_title="Teacher Training Simulator (DSPy Edition)",
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
        st.title("Teacher Training Simulator (DSPy Edition)")
        st.markdown("### Improve your teaching skills through realistic simulations")

    def initialize_llm(self, model_name="gpt-3.5-turbo"):
        """Initialize the DSPy LLM interface with the specified model."""
        try:
            logging.info(f"Initializing DSPy LLM interface with model: {model_name}")
            # Create the enhanced interface
            llm_interface = create_llm_interface(model_name=model_name, enhanced=True)
            st.session_state.llm_interface = llm_interface
            
            # Also create a pedagogical processor
            st.session_state.pedagogical_processor = PedagogicalLanguageProcessor(model=model_name)
            
            return True
        except Exception as e:
            logging.error(f"Error initializing LLM interface: {e}")
            st.error(f"Error initializing LLM interface: {str(e)}")
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
            if not st.session_state.llm_interface:
                st.error("Please initialize the LLM interface first.")
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
                
                # Create scenario context
                scenario_context = {
                    "subject": subject,
                    "difficulty": "intermediate",  # Default
                    "grade_level": grade_level,
                    "learning_objectives": objectives_list,
                    "student_profile": student_profile
                }
                
                # Use the pedagogical processor to create a scenario
                with st.spinner("Creating scenario..."):
                    scenario = st.session_state.pedagogical_processor.create_scenario(scenario_context)
                
                # Add additional fields to the scenario
                scenario["subject"] = subject
                scenario["grade_level"] = grade_level
                scenario["learning_objectives"] = objectives_list
                
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
                
            except Exception as e:
                logging.error(f"Error creating scenario: {e}")
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
        if st.session_state.history:
            self.display_conversation()
        
        # Teacher input
        with st.form(key="interaction_form"):
            teacher_input = st.text_area(
                "Your response as the teacher",
                placeholder="Enter your teaching response...",
                height=150
            )
            
            submit = st.form_submit_button("Submit Response")
            
            if submit and teacher_input:
                if not st.session_state.llm_interface:
                    st.error("Please initialize the LLM interface first.")
                    return
                
                try:
                    # Add teacher message to history
                    st.session_state.history.append({
                        "role": "teacher",
                        "content": teacher_input,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    
                    # Generate student response
                    with st.spinner("Student is responding..."):
                        student_response = st.session_state.llm_interface.simulate_student_response(
                            teacher_input=teacher_input,
                            student_profile=st.session_state.student_profile,
                            scenario_context=st.session_state.scenario
                        )
                    
                    # Add student message to history
                    st.session_state.history.append({
                        "role": "student",
                        "content": student_response,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    
                    # Analyze teaching approach after at least one exchange
                    if len(st.session_state.history) >= 2 and not st.session_state.analysis:
                        with st.spinner("Analyzing teaching approach..."):
                            analysis = st.session_state.llm_interface.analyze_teaching_strategies(
                                teacher_input=teacher_input,
                                student_profile=st.session_state.student_profile,
                                scenario_context=st.session_state.scenario
                            )
                            st.session_state.analysis = analysis
                    
                    # Refresh the page to show updated conversation
                    st.rerun()
                    
                except Exception as e:
                    logging.error(f"Error in simulation: {e}")
                    st.error(f"Error: {str(e)}")
        
        # Allow ending the conversation and viewing analysis
        if st.session_state.history:
            if st.button("End Conversation & View Analysis"):
                # If we don't have an analysis yet, generate one
                if not st.session_state.analysis:
                    with st.spinner("Analyzing teaching approach..."):
                        # Get all teacher inputs
                        teacher_inputs = " ".join([
                            msg["content"] for msg in st.session_state.history 
                            if msg["role"] == "teacher"
                        ])
                        
                        analysis = st.session_state.llm_interface.analyze_teaching_strategies(
                            teacher_input=teacher_inputs,
                            student_profile=st.session_state.student_profile,
                            scenario_context=st.session_state.scenario
                        )
                        st.session_state.analysis = analysis
                
                # Navigate to analysis page
                st.session_state.nav_selection = "View Analysis"
                st.rerun()

    def display_conversation(self):
        """Display the conversation history in a chat-like interface."""
        st.markdown("### Conversation")
        
        # Create a container for the conversation
        conversation_container = st.container()
        
        with conversation_container:
            for message in st.session_state.history:
                role = message["role"]
                content = message["content"]
                timestamp = message.get("timestamp", "")
                
                # Use different styles for different roles
                if role == "teacher":
                    st.markdown(
                        f"""
                        <div class="teacher-message">
                            <div class="message-header">
                                <span class="role">Teacher</span>
                                <span class="timestamp">{timestamp}</span>
                            </div>
                            <div class="message-content">{content}</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                elif role == "student":
                    st.markdown(
                        f"""
                        <div class="student-message">
                            <div class="message-header">
                                <span class="role">Student</span>
                                <span class="timestamp">{timestamp}</span>
                            </div>
                            <div class="message-content">{content}</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                # Add a small spacing between messages
                st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)

    def display_analysis(self):
        """Display analysis of the teaching interaction."""
        st.markdown("## Teaching Analysis")
        
        if not st.session_state.history:
            st.info("You need to practice a teaching interaction first.")
            return
        
        if not st.session_state.analysis:
            st.warning("No analysis available. Please complete a teaching interaction first.")
            return
        
        analysis = st.session_state.analysis
        
        # Display overall effectiveness
        overall_score = analysis.get("overall_effectiveness", 0)
        
        st.markdown("### Overall Effectiveness")
        st.progress(overall_score / 10)
        st.markdown(f"**Score: {overall_score}/10**")
        
        # Display rationale
        if "rationale" in analysis:
            st.markdown("### Detailed Assessment")
            st.markdown(analysis["rationale"])
        
        # Display identified strategies
        if "identified_strategies" in analysis:
            st.markdown("### Teaching Strategies Used")
            
            for strategy in analysis["identified_strategies"]:
                strategy_name = strategy.get("strategy", "Unnamed Strategy")
                effectiveness = strategy.get("effectiveness", 5)
                description = strategy.get("description", "")
                rationale = strategy.get("rationale", "")
                
                st.markdown(f"#### {strategy_name}")
                st.markdown(f"**Effectiveness: {effectiveness}/10**")
                
                if description:
                    st.markdown(description)
                
                if rationale:
                    st.markdown(f"*Rationale: {rationale}*")
        
        # Display improvement suggestions
        if "suggested_improvements" in analysis:
            st.markdown("### Suggested Improvements")
            improvements = analysis["suggested_improvements"]
            
            for i, improvement in enumerate(improvements, 1):
                st.markdown(f"{i}. {improvement}")
        
        # Option to generate reflection prompts
        if st.button("Generate Reflection Prompts"):
            with st.spinner("Generating reflection prompts..."):
                # Create a prompt for reflection
                reflection_prompt = f"""
                Based on the teaching interaction and analysis, generate 3-5 reflection prompts
                that will help the teacher think deeply about their approach and how to improve.
                Focus on the specific teaching strategies used and alignment with the student profile.
                """
                
                reflection = st.session_state.llm_interface.get_llm_response([
                    {"role": "system", "content": "You are an educational coach providing reflection prompts."},
                    {"role": "user", "content": reflection_prompt}
                ])
                
                st.session_state.reflection = reflection
        
        # Display reflection prompts if available
        if st.session_state.reflection:
            st.markdown("### Reflection Prompts")
            st.markdown(st.session_state.reflection)

    def display_resources(self):
        """Display educational resources related to the current teaching scenario."""
        st.markdown("## Teaching Resources")
        
        if not st.session_state.scenario:
            st.info("Please create a scenario first to get relevant resources.")
            return
        
        # Get scenario details for resource recommendations
        subject = st.session_state.scenario.get("subject", "")
        grade_level = st.session_state.scenario.get("grade_level", "")
        student_profile = st.session_state.student_profile or {}
        learning_styles = student_profile.get("learning_style", [])
        
        # Generate resource recommendations based on scenario
        if not st.session_state.llm_interface:
            st.error("Please initialize the LLM interface first.")
            return
        
        # Resource recommendations
        if st.button("Get Teaching Resource Recommendations"):
            with st.spinner("Generating resource recommendations..."):
                resource_prompt = f"""
                Recommend teaching resources for a {grade_level} {subject} class.
                The students have these learning styles: {', '.join(learning_styles)}.
                
                Include:
                1. Books and articles
                2. Online teaching platforms
                3. Interactive activities
                4. Assessment tools
                5. Professional development resources
                
                Return your recommendations in markdown format with clear headings and brief descriptions.
                """
                
                resources = st.session_state.llm_interface.get_llm_response([
                    {"role": "system", "content": "You are an educational resource specialist."},
                    {"role": "user", "content": resource_prompt}
                ])
                
                st.session_state.resources = resources
        
        # Display resources
        if hasattr(st.session_state, 'resources') and st.session_state.resources:
            st.markdown(st.session_state.resources)
        else:
            # Default resources
            st.markdown("### General Teaching Resources")
            
            st.markdown("""
            #### Books and Articles
            - **"Visible Learning" by John Hattie** - Research-based approaches to teaching effectiveness
            - **"Why Don't Students Like School?" by Daniel Willingham** - Cognitive principles for teachers
            
            #### Online Platforms
            - **Khan Academy** - Free video lessons and practice exercises
            - **Edutopia** - Evidence-based classroom strategies and resources
            
            #### Interactive Tools
            - **Kahoot!** - Game-based learning platform
            - **Padlet** - Virtual bulletin board for collaboration
            
            #### Assessment Resources
            - **Formative** - Real-time formative assessment tool
            - **Rubric Makers** - Tools for creating assessment rubrics
            
            #### Professional Development
            - **Teaching Channel** - Videos of teaching practices
            - **EdWeb** - Professional learning communities and webinars
            """)

    def run(self):
        """Run the Streamlit application."""
        self.setup_page()
        
        # Sidebar for configuration and navigation
        with st.sidebar:
            st.title("Teacher Training Simulator")
            st.markdown("## Configuration")
            
            # Model selection
            model_options = {
                "OpenAI GPT-4o-mini": "gpt-4o-mini",
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
                index=0  # Default to GPT-4o-mini which is more affordable
            )
            
            selected_model = model_options[selected_model_name]
            
            # Initialize LLM button
            if st.button("Initialize LLM"):
                with st.spinner("Initializing LLM..."):
                    if self.initialize_llm(selected_model):
                        st.success("LLM initialized successfully!")
                    else:
                        st.error("Failed to initialize LLM.")
            
            # Navigation
            st.markdown("## Navigation")
            nav_options = [
                "Create Scenario",
                "Practice Teaching",
                "View Analysis",
                "Resources"
            ]
            
            nav_selection = st.radio("Go to:", nav_options)
            
            # Store navigation selection in session state
            if 'nav_selection' not in st.session_state:
                st.session_state.nav_selection = nav_selection
            else:
                # Update if changed
                if nav_selection != st.session_state.nav_selection:
                    st.session_state.nav_selection = nav_selection
            
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
        current_nav = st.session_state.nav_selection \
            if 'nav_selection' in st.session_state else nav_selection
            
        if current_nav == "Create Scenario":
            self.create_scenario()
        
        elif current_nav == "Practice Teaching":
            self.display_simulation_interface()
        
        elif current_nav == "View Analysis":
            self.display_analysis()
        
        elif current_nav == "Resources":
            self.display_resources()

# Run the application
if __name__ == "__main__":
    app = DSPyWebInterface()
    app.run() 
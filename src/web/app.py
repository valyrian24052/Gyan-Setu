"""
Streamlit web application for the Teacher Training Simulator.
Provides an interactive interface for teachers to practice and analyze teaching scenarios.
Uses DSPy for efficient and reliable LLM interactions.
"""

import streamlit as st
import uuid
import time

# Set page configuration first - this must be the first Streamlit command
st.set_page_config(
    page_title="Utah Teacher Training Assistant",
    page_icon="👨‍🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import pandas as pd
import plotly.express as px
from datetime import datetime
import json
from dotenv import load_dotenv
import subprocess
import getpass
import logging
import random
import time
import traceback
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import our DSPy adapter and implementations
from dspy_adapter import create_llm_interface, EnhancedLLMInterface
from dspy_llm_handler import PedagogicalLanguageProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables from .env file
load_dotenv()

class WebInterface:
    def __init__(self):
        """Initialize the web interface."""
        # Load students and scenarios first
        self.students = self.load_students()
        self.scenarios = self.load_scenarios()
        
        # Initialize session state
        self.init_session_state()
        
        # Apply custom CSS
        self.apply_custom_css()
        
        # Comment out the metrics evaluator initialization
        # Initialize the metrics evaluator if not already done
        # if 'metrics_evaluator' not in st.session_state:
        #     st.session_state.metrics_evaluator = AutomatedMetricsEvaluator()
        
        # DO NOT run the application from init - it will be called explicitly later
        # self.run()

    def apply_custom_css(self):
        """Apply custom CSS styles to the Streamlit app."""
        st.markdown("""
        <style>
            /* Global styling with UVU color palette */
            :root {
                --uvu-green: #275D38;
                --uvu-green-light: #6E9D7D;
                --uvu-green-dark: #1A3F26;
                --uvu-black: #000000;
                --uvu-white: #FFFFFF;
                --neutral-50: #F9FAFB;
                --neutral-100: #F3F4F6;
                --neutral-200: #E5E7EB;
                --neutral-300: #D1D5DB;
                --neutral-700: #374151;
                --neutral-800: #1F2937;
                --neutral-900: #111827;
                --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
                --shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
                --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                --radius-sm: 0.25rem;
                --radius: 0.5rem;
                --radius-md: 0.75rem;
                --radius-lg: 1rem;
            }
            
            /* Main container styling */
            .main .block-container {
                padding-top: 1rem;
                padding-bottom: 1rem;
                max-width: 1200px;
            }
            
            /* App container */
            .app-container {
                display: flex;
                flex-direction: column;
                min-height: calc(100vh - 80px);
                border-radius: var(--radius);
                background-color: var(--uvu-white);
                box-shadow: var(--shadow);
                overflow: hidden;
            }
            
            /* Header styling */
            .app-header {
                padding: 1rem 1.5rem;
                background: linear-gradient(135deg, var(--uvu-green), var(--uvu-green-dark));
                color: var(--uvu-white);
                border-radius: var(--radius) var(--radius) 0 0;
                margin-bottom: 1rem;
            }
            
            .app-header h1 {
                font-size: 1.75rem;
                margin: 0;
                font-weight: 600;
                color: var(--uvu-white);
                text-align: center;
            }
            
            .app-header p {
                margin: 0.5rem 0 0;
                opacity: 0.9;
                font-size: 1rem;
                text-align: center;
            }
            
            h1, h2, h3, h4 {
                color: var(--neutral-900);
                margin-top: 0;
            }
            
            h2 {
                font-size: 1.5rem;
                font-weight: 600;
            }
            
            h3 {
                font-size: 1.25rem;
                font-weight: 600;
                margin-bottom: 0.75rem;
            }
            
            h4 {
                font-size: 1rem;
                font-weight: 600;
                margin-bottom: 0.5rem;
                color: var(--neutral-700);
            }
            
            /* Chat container */
            .chat-container {
                border-radius: var(--radius);
                padding: 1rem;
                margin-bottom: 1rem;
                display: flex;
                flex-direction: column;
                background-color: var(--neutral-50);
                border: 1px solid var(--neutral-200);
                min-height: 200px;
                max-height: 600px;
                overflow-y: auto;
            }
            
            /* Custom chat bubbles */
            .stChatMessage {
                margin-bottom: 0.5rem;
            }
            
            .stChatMessage [data-testid="chatAvatarIcon-user"] {
                background-color: var(--primary-color) !important;
            }
            
            .stChatMessage [data-testid="chatAvatarIcon-assistant"] {
                background-color: var(--secondary-color) !important;
            }
            
            .stChatMessage [data-testid="stChatMessageContent"] {
                border-radius: var(--radius) !important;
                padding: 0.75rem 1rem !important;
                box-shadow: var(--shadow-sm) !important;
                margin: 0 !important;
            }
            
            .stChatMessage.user [data-testid="stChatMessageContent"] {
                background-color: var(--secondary-light) !important;
                border: 1px solid #D1FAE5 !important;
            }
            
            .stChatMessage.assistant [data-testid="stChatMessageContent"] {
                background-color: var(--primary-light) !important;
                border: 1px solid #DDD6FE !important;
            }
            
            /* Info panel container - simplified */
            .info-panel {
                height: 670px;
                overflow-y: auto;
                padding-right: 10px;
                display: flex;
                flex-direction: column;
                gap: 1rem;
                scrollbar-width: thin;
                scrollbar-color: var(--neutral-300) var(--neutral-100);
            }
            
            .info-panel::-webkit-scrollbar {
                width: 6px;
            }
            
            .info-panel::-webkit-scrollbar-track {
                background: var(--neutral-100);
                border-radius: var(--radius);
            }
            
            .info-panel::-webkit-scrollbar-thumb {
                background-color: var(--neutral-300);
                border-radius: var(--radius);
            }
            
            /* Chat container scrollbar styling */
            .chat-container::-webkit-scrollbar {
                width: 6px;
            }
            
            .chat-container::-webkit-scrollbar-track {
                background: var(--neutral-100);
                border-radius: var(--radius);
            }
            
            .chat-container::-webkit-scrollbar-thumb {
                background-color: var(--neutral-300);
                border-radius: var(--radius);
            }
            
            /* Fix for stChatMessage spacing */
            .stChatMessage {
                margin-bottom: 1rem;
            }
            
            /* Fix for chat input positioning */
            .stChatInputContainer {
                position: sticky;
                bottom: 0;
                background-color: white;
                padding: 1rem 0;
                z-index: 100;
                border-top: 1px solid var(--neutral-200);
            }
            
            /* Sidebar styling - simplified */
            [data-testid="stSidebar"] {
                background-color: var(--neutral-50);
                border-right: 1px solid var(--neutral-200);
            }
            
            .sidebar .sidebar-content {
                background-color: var(--neutral-50);
            }
            
            .sidebar .block-container {
                padding-top: 2rem;
            }
            
            /* Sidebar sections */
            .sidebar-section {
                background-color: white;
                border-radius: var(--radius);
                padding: 1rem;
                margin-bottom: 1rem;
                border: 1px solid var(--neutral-200);
                box-shadow: var(--shadow-sm);
            }
            
            .sidebar-title {
                font-size: 1rem;
                font-weight: 600;
                margin-bottom: 0.75rem;
                color: var(--neutral-800);
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            
            /* Cards for scenario information - simplified */
            .scenario-card {
                border-radius: var(--radius);
                padding: 1.25rem;
                margin-bottom: 1rem;
                background-color: white;
                border: 1px solid var(--neutral-200);
                box-shadow: var(--shadow-sm);
                transition: all 0.2s;
            }
            
            .scenario-header {
                position: sticky;
                top: 0;
                background-color: white;
                z-index: 100;
                padding: 1rem;
                margin-bottom: 1rem;
                border-radius: var(--radius);
                border: 1px solid var(--neutral-200);
                box-shadow: var(--shadow-sm);
            }
            
            .conversation-header {
                position: sticky;
                top: 0;
                background-color: white;
                z-index: 50;
                padding: 1rem;
                margin-bottom: 1rem;
                border-radius: var(--radius);
                border: 1px solid var(--neutral-200);
                text-align: center;
            }
            
            .scenario-title {
                font-weight: 600;
                color: var(--neutral-900);
                margin-bottom: 0.5rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            
            .scenario-title-icon {
                color: var(--primary-color);
            }
            
            /* Better list styling */
            .info-list {
                list-style-type: none;
                padding-left: 0.5rem;
                margin-top: 0.5rem;
            }
            
            .info-list li {
                position: relative;
                padding-left: 1.5rem;
                margin-bottom: 0.5rem;
                line-height: 1.5;
            }
            
            .info-list li:before {
                content: "";
                position: absolute;
                left: 0;
                top: 0.6rem;
                height: 0.5rem;
                width: 0.5rem;
                border-radius: 50%;
                background-color: var(--primary-color);
            }
            
            /* Control panel for buttons - simplified */
            .control-panel {
                display: flex;
                flex-direction: column;
                gap: 0.75rem;
                padding: 1.25rem;
                border-radius: var(--radius);
                border: 1px solid var(--neutral-200);
                background-color: white;
                box-shadow: var(--shadow-sm);
            }
            
            /* Button styling */
            .stButton button {
                width: 100%;
                border-radius: var(--radius-sm);
                padding: 0.5rem 1rem;
                font-weight: 500;
                transition: all 0.2s;
                border: 1px solid var(--neutral-200);
            }
            
            .stButton button:first-child {
                background-color: var(--primary-color);
                color: white;
            }
            
            .stButton button:nth-child(2) {
                background-color: var(--neutral-50);
                color: var(--neutral-800);
            }
            
            .stButton button:hover {
                opacity: 0.9;
                box-shadow: var(--shadow);
            }
        </style>
        """, unsafe_allow_html=True)

    def init_session_state(self):
        """Initialize the Streamlit session state."""
        # Default session state for conversation
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        # Initialize the LLM interface
        if "llm_interface" not in st.session_state:
            try:
                st.session_state.llm_interface = create_llm_interface()
            except Exception as e:
                st.error(f"Failed to initialize LLM interface: {str(e)}")
                st.session_state.llm_interface = None
        
        # Set unique conversation ID if not present
        if "conversation_id" not in st.session_state:
            st.session_state.conversation_id = str(uuid.uuid4())
            
        # Student profile and scenario defaults
        if "student_profile" not in st.session_state:
            st.session_state.student_profile = self.students[0] if self.students else None
            
        if "scenario" not in st.session_state:
            st.session_state.scenario = self.scenarios[0] if self.scenarios else None
            
        # Set default app mode to expert feedback
        if "app_mode" not in st.session_state:
            st.session_state.app_mode = "expert_feedback"
            
        # Initialize points system
        if "expert_points" not in st.session_state:
            st.session_state.expert_points = 0
            
        if "expert_level" not in st.session_state:
            st.session_state.expert_level = "Novice Teacher"
            
        # Initialize feedback and example collections
        if "sme_feedback" not in st.session_state:
            st.session_state.sme_feedback = []
            
        if "teaching_examples" not in st.session_state:
            st.session_state.teaching_examples = []
            
        # Initialize expert data collections
        if "expert_examples" not in st.session_state:
            st.session_state.expert_examples = []
            
        if "expert_reviews" not in st.session_state:
            st.session_state.expert_reviews = []
            
        # Initialize streak and badges system
        if "streak_days" not in st.session_state:
            st.session_state.streak_days = 0
            
        if "last_contribution_date" not in st.session_state:
            st.session_state.last_contribution_date = datetime.now().strftime("%Y-%m-%d")
            
        if "badges" not in st.session_state:
            st.session_state.badges = []
            
        if "awards" not in st.session_state:
            st.session_state.awards = {
                "literacy_star": False,
                "math_wizard": False,
                "science_explorer": False,
                "feedback_champion": False
            }
            
        # Initialize announcement banner state
        if "show_announcement" not in st.session_state:
            st.session_state.show_announcement = True
            
    def load_students(self):
        """Load student profiles from data files."""
        # Default student profiles for demonstration
        return [
            {
                "name": "Alex Johnson",
                "grade_level": "8th Grade",
                "learning_style": ["Visual", "Active"],
                "challenges": ["Limited attention span", "Math anxiety"],
                "interests": ["Sports", "Video games", "Science fiction"]
            },
            {
                "name": "Maya Rodriguez",
                "grade_level": "5th Grade",
                "learning_style": ["Auditory", "Reflective"],
                "challenges": ["Reading comprehension", "Shy in groups"],
                "interests": ["Art", "Music", "Animals"]
            },
            {
                "name": "Jamal Washington",
                "grade_level": "11th Grade",
                "learning_style": ["Kinesthetic", "Practical"],
                "challenges": ["Test anxiety", "Difficulty with abstract concepts"],
                "interests": ["Basketball", "Computer programming", "History"]
            },
            {
                "name": "Emma Chen",
                "grade_level": "3rd Grade",
                "learning_style": ["Visual", "Sequential"],
                "challenges": ["English as second language", "Fine motor skills"],
                "interests": ["Reading", "Drawing", "Nature"]
            }
        ]
        
    def load_scenarios(self):
        """Load teaching scenarios from data files."""
        # Default scenarios for demonstration
        return [
            {
                "title": "Classroom Management",
                "description": "Managing student behavior and creating a positive learning environment",
                "teaching_challenge": "Students are disruptive during group work",
                "subject_area": "General"
            },
            {
                "title": "Mathematics Instruction",
                "description": "Teaching mathematical concepts effectively",
                "teaching_challenge": "Students struggling with fractions and decimals",
                "subject_area": "Mathematics"
            },
            {
                "title": "Science Engagement",
                "description": "Making science topics interesting and accessible",
                "teaching_challenge": "Getting students excited about the scientific method",
                "subject_area": "Science"
            },
            {
                "title": "Reading Comprehension",
                "description": "Helping students understand and analyze texts",
                "teaching_challenge": "Students read but don't comprehend deeper meaning",
                "subject_area": "Language Arts"
            },
            {
                "title": "Historical Context",
                "description": "Teaching history with proper context and engagement",
                "teaching_challenge": "Making historical events relevant to modern students",
                "subject_area": "Social Studies"
            }
        ]

    def setup_page(self):
        """Set up the page with header and description."""
        # Header with logo and title side by side
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.image("logo.png", width=150)
            
        with col2:
            st.title("Utah Teacher Training Assistant")
            st.markdown("""
            ### Subject Matter Expert Platform
            Share your teaching expertise to help improve AI-assisted instruction for K-5 education.
            """)
            
        # Display current app version and UVU styling based on color palette
        st.markdown("""
        <style>
            :root {
                --uvu-green: #275D38;
                --uvu-black: #000000;
                --uvu-white: #FFFFFF;
                --uvu-light-green: #E9F2ED;
            }
            
            .stApp {
                background-color: var(--uvu-white);
            }
            
            h1, h2, h3 {
                color: var(--uvu-green);
            }
            
            .stButton button {
                background-color: var(--uvu-green);
                color: var(--uvu-white);
            }
            
            .info-box {
                background-color: var(--uvu-light-green);
                border-left: 5px solid var(--uvu-green);
                padding: 10px;
                margin-bottom: 10px;
            }
        </style>
        <div style="text-align: right; font-size: 0.8em; color: gray;">Version 3.0</div>
        """, unsafe_allow_html=True)

    def initialize_llm(self, model_name="gpt-3.5-turbo"):
        """Initialize the LLM interface with the specified model."""
        try:
            logging.info(f"Initializing DSPy LLM Interface with model: {model_name}")
            
            # Create the LLM interface - this goes directly to the underlying implementation
            st.session_state.llm_interface = create_llm_interface(model_name=model_name)
            
            # Store the model name in session state
            st.session_state.model_name = model_name
            
            logging.info(f"Initialized LLM interface with model: {model_name}")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize LLM interface: {str(e)}")
            st.error(f"Failed to initialize LLM interface: {str(e)}")
            return False
            
    def _format_conversation_history(self, messages, max_messages=10):
        """Format the conversation history for the LLM context.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_messages: Maximum number of recent messages to include
        
        Returns:
            Formatted conversation string
        """
        # Get the most recent messages
        recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages
        
        # Format each message with role and content
        formatted_history = ""
        for msg in recent_messages:
            role = "Teacher" if msg["role"] == "teacher" else "Student"
            formatted_history += f"{role}: {msg['content']}\n\n"
        
        return formatted_history

    def create_scenario(self):
        """Create a teaching scenario with specified parameters."""
        st.markdown("## Create a Teaching Scenario")
        
        with st.form(key="scenario_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                subject = st.selectbox(
                    "Subject", 
                    ["Mathematics", "Science", "Language Arts", "Social Studies", "Art", "Music", "Physical Education"],
                    key="create_scenario_subject_selectbox"
                )
                
                grade_level = st.selectbox(
                    "Grade Level",
                    ["Kindergarten", "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th", "12th"],
                    key="create_scenario_grade_selectbox"
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
                st.session_state.messages = []
                st.session_state.analysis = None
                st.session_state.strategies = []
                st.session_state.teacher_feedback = None
                st.session_state.reflection = None
                st.session_state.classroom_management_strategies = []
                
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
                    st.error(f"Error generating response: {str(e)}")
        
        # Quick tips and reminders
        with st.expander("Teaching Tips"):
            st.markdown("""
            ### Quick Reminders
            - Ask open-ended questions to engage critical thinking
            - Provide specific, constructive feedback
            - Adapt your approach based on the student's learning style
            - Use visual aids for visual learners
            - Break down complex concepts into smaller parts
            - Check for understanding before moving on
            """)
    
    def display_conversation(self):
        """Display the conversation history in a chat-like interface."""
        st.markdown("### Conversation")
        
        # Display conversation header
        st.markdown("""
        <div class="conversation-header">
            <h3>Classroom Conversation</h3>
            <p>Interact with the simulated student below.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Use a container with dynamic height for the conversation
        chat_container = st.container()
        with chat_container:
            # Display the conversation
            for message in st.session_state.messages:
                with st.chat_message(message["role"], avatar="👨‍🎓" if message["role"] == "student" else "👨‍🏫"):
                    st.write(f"{message['timestamp']}")
                    st.markdown(message["content"])
        
        # Get teacher input
        teacher_input = st.chat_input("Type your response as the teacher...")
        
        if teacher_input:
            # Add teacher message to chat history
            st.session_state.messages.append({"role": "teacher", "content": teacher_input, "timestamp": self._get_timestamp()})
            
            # Process student response in the background
            try:
                # Format conversation history for context
                history = self._format_conversation_history(st.session_state.messages)
                
                # Get the last student response to avoid repetition
                last_student_response = ""
                for msg in reversed(st.session_state.messages):
                    if msg["role"] == "student":
                        last_student_response = msg["content"]
                        break
                
                # Add classroom management strategies to the context if any are selected
                strategy_instructions = ""
                if st.session_state.classroom_management_strategies:
                    strategy_instructions = "\n\nThe teacher is using the following classroom management strategies:\n"
                    for strategy in st.session_state.classroom_management_strategies:
                        strategy_instructions += f"- {strategy['name']}: {strategy['description']}\n"
                    strategy_instructions += "\nThe student should respond appropriately to these teaching strategies. If the teacher effectively uses the strategy, the student should show a slightly more positive, engaged, or improved response. If the strategy is not effectively applied, the student can show confusion or maintain their previous state."
                
                # Create an enhanced teacher input with explicit instructions to be conversational
                enhanced_teacher_input = f"""
                The teacher just said: "{teacher_input}"
                
                You are {student_profile.get('name', 'Jamie')}, a {student_profile.get('grade_level', '7th grade')} student.
                Respond naturally as yourself in a brief, realistic way.
                
                KEY INSTRUCTIONS:
                1. DO NOT repeat your introduction or describe yourself again
                2. DO NOT say "As a 7th-grade student who..." or anything like that
                3. Respond directly to what the teacher just said
                4. Keep your response short (1-3 sentences)
                5. Use casual, age-appropriate language
                6. Show your personality through your response
                7. If the teacher just said "hello" or greeted you, respond naturally and maybe mention something about the class or ask a question
                8. NEVER list your traits, challenges or learning style directly - these should only be implied by your response
                9. IMPORTANT: NEVER repeat any of your previous responses verbatim
                10. Each of your responses should be unique and move the conversation forward
                {strategy_instructions}
                
                Your previous response was: "{last_student_response}"
                DO NOT repeat this response - say something completely new and different that makes sense in the context of this conversation.
                """
                
                # Create context with conversation history and last response
                context_with_history = scenario.copy()
                context_with_history['conversation_history'] = history
                context_with_history['previous_response'] = last_student_response
                context_with_history['session_state'] = {'last_student_response': last_student_response}
                context_with_history['instruction'] = "Be brief and conversational, like a real student in class"
                
                # Add classroom management strategies to the context
                if st.session_state.classroom_management_strategies:
                    context_with_history['classroom_management_strategies'] = st.session_state.classroom_management_strategies
                
                # Try to get the student's response with fallbacks
                try:
                    response = st.session_state.llm_interface.simulate_student_response(
                        teacher_input=enhanced_teacher_input,
                        student_profile=student_profile,
                        scenario_context=context_with_history
                    )
                    
                    # Check if response is identical to any previous student response
                    previous_responses = [msg["content"] for msg in st.session_state.messages if msg["role"] == "student"]
                    if response in previous_responses:
                        logging.warning("Generated response is identical to a previous one. Using fallback.")
                        
                        # Random conversational responses that can work in various contexts
                        fallback_options = [
                            "I'm not sure I understand. Could you explain that differently?",
                            "That's interesting. Can we talk more about how this works?",
                            "I've been thinking about what you said. Can you give another example?",
                            "OK, I think I'm starting to get it.",
                            "Let me try to understand this from another angle.",
                            "I'm still confused about part of this.",
                            "So what does that mean for our assignment?",
                            "I hadn't thought about it that way before."
                        ]
                        
                        # Choose a random fallback that's not in previous responses
                        filtered_fallbacks = [opt for opt in fallback_options if opt not in previous_responses]
                        if filtered_fallbacks:
                            response = random.choice(filtered_fallbacks)
                        else:
                            # Create a unique response by adding a small random modifier
                            response = f"I see. {random.choice(['Actually, ', 'Well, ', 'Hmm, ', 'So, '])}I'm still working through this."
                except Exception as e:
                    logging.error(f"Error generating student response: {e}", exc_info=True)
                    # Set up a fallback response
                    response = "I'm not sure how to respond to that."
                
                # Check if response is empty or has an error
                if not response or "Error" in response:
                    # Create a simple fallback response based on the teacher's input
                    if "?" in teacher_input:
                        response = "I'm not really sure about that. Can you explain it differently?"
                    else:
                        response = "Okay, I understand. What should we do next?"
                
                # Check if response seems like a self-description rather than a natural response
                # We'll look for common patterns that indicate the student is describing themselves
                description_indicators = [
                    "as a", "my learning style", "I am a", "my strengths", 
                    "my challenges", "I struggle with", "my teachers say"
                ]
                
                is_description = any(indicator in response.lower() for indicator in description_indicators)
                
                # If it's a description or too long, generate a simpler response
                if is_description or len(response.split()) > 50:
                    logging.info("Detected self-description in student response, generating simpler response")
                    
                    # Generate a very simple, age-appropriate response
                    if "hello" in teacher_input.lower() or "hi" in teacher_input.lower() or "hey" in teacher_input.lower():
                        fallback_options = [
                            f"Hi! Are we starting algebra today?",
                            f"Hey! Do we need our textbooks for today's lesson?",
                            f"Hello! I was just trying to figure out this math problem.",
                            f"Hi there! Is this going to be on the test?",
                            f"Hey teacher, I was wondering what we're covering today.",
                            f"Hi! I've been stuck on problem number 5, can you help me?"
                        ]
                        # Avoid repeating the last student response
                        last_response = ""
                        for msg in reversed(st.session_state.messages):
                            if msg["role"] == "student":
                                last_response = msg["content"]
                                break
                        
                        # Filter out any options that match the previous response
                        filtered_options = [opt for opt in fallback_options if opt != last_response]
                        
                        # If we have options left, choose from them, otherwise use a more generic response
                        if filtered_options:
                            response = random.choice(filtered_options)
                        else:
                            response = "Sorry, I didn't catch what you said. Can you explain what we're doing today?"
                    elif "?" in teacher_input:
                        fallback_options = [
                            "Um, I'm not really sure.",
                            "I don't know, can you explain it again?",
                            "Maybe? I get confused with all these x's and y's.",
                            "I think so... but I'm not 100% sure.",
                            "Could you break that down for me?",
                            "I'm still trying to understand that concept.",
                            "Not completely. Could you show an example?"
                        ]
                        # Avoid repeating the last student response
                        last_response = ""
                        for msg in reversed(st.session_state.messages):
                            if msg["role"] == "student":
                                last_response = msg["content"]
                                break
                        
                        # Filter out any options that match the previous response
                        filtered_options = [opt for opt in fallback_options if opt != last_response]
                        
                        # If we have options left, choose from them, otherwise use a more generic response
                        if filtered_options:
                            response = random.choice(filtered_options)
                        else:
                            response = "I'm confused about that question. Can we try a different approach?"
                    else:
                        fallback_options = [
                            "Okay, I'll try that.",
                            "That makes sense, I think.",
                            "So we're supposed to solve for x?",
                            "Can you show me an example first?",
                            "Math is so confusing sometimes.",
                            "I'll give it my best shot.",
                            "Let me see if I understand the process correctly.",
                            "That's clearer now, thanks!"
                        ]
                        # Avoid repeating the last student response
                        last_response = ""
                        for msg in reversed(st.session_state.messages):
                            if msg["role"] == "student":
                                last_response = msg["content"]
                                break
                        
                        # Filter out any options that match the previous response
                        filtered_options = [opt for opt in fallback_options if opt != last_response]
                        
                        # If we have options left, choose from them, otherwise use a more generic response
                        if filtered_options:
                            response = random.choice(filtered_options)
                        else:
                            response = "I'll work on that. Is there something specific I should focus on?"
                
                # Add the response to messages
                timestamp = self._get_timestamp()
                st.session_state.messages.append({"role": "student", "content": response, "timestamp": timestamp})
                
            except Exception as e:
                error_message = f"Error generating student response: {str(e)}"
                logging.error(f"{error_message}", exc_info=True)
                
                # Provide a fallback response
                fallback_response = f"Sorry, what do you mean?"
                timestamp = self._get_timestamp()
                st.session_state.messages.append({"role": "student", "content": fallback_response, "timestamp": timestamp})
            
            # Rerun the app to display the updated conversation
            st.rerun()
    
    def display_analysis(self):
        """Display analysis of the teaching interaction."""
        st.markdown("## Teaching Analysis")
        
        if not st.session_state.history:
            st.info("Complete a teaching interaction first to see analysis.")
            return
        
        # Get analysis if not already available
        if not st.session_state.analysis and st.session_state.llm_interface:
            with st.spinner("Analyzing teaching approach..."):
                last_teacher_message = None
                for message in reversed(st.session_state.history):
                    if message["role"] == "teacher":
                        last_teacher_message = message["content"]
                        break
                
                if last_teacher_message:
                    # Add classroom management strategies to the analysis context
                    analysis_context = {
                        "student_profile": st.session_state.student_profile,
                        "scenario_context": st.session_state.scenario,
                        "classroom_management_strategies": st.session_state.classroom_management_strategies if hasattr(st.session_state, 'classroom_management_strategies') else []
                    }
                    
                    analysis = st.session_state.llm_interface.analyze_teaching_strategies(
                        teacher_input=last_teacher_message,
                        student_profile=st.session_state.student_profile,
                        scenario_context=st.session_state.scenario
                    )
                    
                    # If classroom management strategies are being used, request specific feedback on them
                    if st.session_state.classroom_management_strategies:
                        strategies_str = "\n".join([f"- {s['name']} ({s['book']}): {s['description']}" for s in st.session_state.classroom_management_strategies])
                        
                        strategy_prompt = f"""
                        Analyze how effectively the teacher has implemented the following classroom management strategies in their teaching:
                        
                        {strategies_str}
                        
                        Teacher's last response: "{last_teacher_message}"
                        
                        For each strategy, provide:
                        1. A rating out of 10
                        2. Specific evidence from the teacher's response
                        3. Constructive feedback on how to better implement the strategy
                        
                        Format your response as JSON with the following structure:
                        {{
                            "strategy_analysis": [
                                {{
                                    "strategy_name": "Strategy name",
                                    "rating": 0-10,
                                    "evidence": "What the teacher did related to this strategy",
                                    "feedback": "Constructive suggestions for improvement"
                                }}
                            ],
                            "overall_strategy_implementation": "Overall assessment of strategy implementation",
                            "next_steps": "What the teacher should focus on next"
                        }}
                        """
                        
                        try:
                            strategy_analysis_response = st.session_state.llm_interface.get_chat_response([
                                {"role": "system", "content": "You are an expert in classroom management strategies and teacher evaluation."},
                                {"role": "user", "content": strategy_prompt}
                            ])
                            
                            # Try to parse the JSON response
                            import json
                            try:
                                # Find JSON in the response - look for content between curly braces
                                import re
                                json_match = re.search(r'\{.*\}', strategy_analysis_response, re.DOTALL)
                                if json_match:
                                    json_str = json_match.group(0)
                                    strategy_analysis = json.loads(json_str)
                                    analysis["strategy_analysis"] = strategy_analysis
                                else:
                                    logging.error("Could not extract JSON from strategy analysis response")
                            except Exception as e:
                                logging.error(f"Error parsing strategy analysis JSON: {e}", exc_info=True)
                        except Exception as e:
                            logging.error(f"Error getting strategy analysis: {e}", exc_info=True)
                    
                    st.session_state.analysis = analysis
        
        # Display the analysis
        if st.session_state.analysis:
            analysis = st.session_state.analysis
            
            # Create columns for the analysis display
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Strengths
                st.markdown("### Teaching Strengths")
                strengths = analysis.get("strengths", [])
                if strengths:
                    for strength in strengths:
                        st.markdown(f"- {strength}")
                else:
                    st.markdown("No specific strengths identified.")
                
                # Areas for improvement
                st.markdown("### Areas for Improvement")
                areas = analysis.get("areas_for_improvement", [])
                if areas:
                    for area in areas:
                        st.markdown(f"- {area}")
                else:
                    st.markdown("No specific areas for improvement identified.")
                
                # Rationale
                if "rationale" in analysis:
                    st.markdown("### Analysis Rationale")
                    st.markdown(analysis["rationale"])
                
                # Strategy Analysis (if available)
                if "strategy_analysis" in analysis and isinstance(analysis["strategy_analysis"], dict):
                    st.markdown("### Classroom Management Strategy Implementation")
                    
                    strategy_analysis = analysis["strategy_analysis"]
                    
                    # Display individual strategy assessments
                    if "strategy_analysis" in strategy_analysis and isinstance(strategy_analysis["strategy_analysis"], list):
                        for strategy in strategy_analysis["strategy_analysis"]:
                            with st.expander(f"{strategy.get('strategy_name', 'Strategy')} - Rating: {strategy.get('rating', 'N/A')}/10"):
                                st.markdown(f"**Evidence:** {strategy.get('evidence', 'No evidence provided')}")
                                st.markdown(f"**Feedback:** {strategy.get('feedback', 'No feedback provided')}")
                    
                    # Display overall assessment
                    if "overall_strategy_implementation" in strategy_analysis:
                        st.markdown("#### Overall Assessment")
                        st.markdown(strategy_analysis["overall_strategy_implementation"])
                    
                    # Display next steps
                    if "next_steps" in strategy_analysis:
                        st.markdown("#### Recommended Next Steps")
                        st.markdown(strategy_analysis["next_steps"])
            
            with col2:
                # Effectiveness score
                if "effectiveness_score" in analysis:
                    score = analysis["effectiveness_score"]
                    st.markdown("### Effectiveness Score")
                    
                    # Create a gauge chart
                    fig = px.bar(
                        x=["Score"], 
                        y=[score], 
                        range_y=[0, 10],
                        color=[score],
                        color_continuous_scale=["red", "yellow", "green"],
                        range_color=[0, 10],
                        labels={"y": "Score", "x": ""}
                    )
                    
                    fig.update_layout(
                        height=300,
                        width=250,
                        xaxis_title=None,
                        yaxis_title="Score (1-10)",
                        coloraxis_showscale=False
                    )
                    
                    st.plotly_chart(fig)
                    
                    # Score interpretation
                    if score >= 8:
                        st.success("Excellent teaching approach!")
                    elif score >= 6:
                        st.info("Good teaching approach with some room for improvement.")
                else:
                    st.warning("Analysis not available. Complete a teaching interaction to see analysis.")

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
                
                resources = st.session_state.llm_interface.get_chat_response([
                    {"role": "system", "content": "You are an educational resource specialist."},
                    {"role": "user", "content": resource_prompt}
                ])
                
                st.session_state.resources = resources
        
        # Display stored resources
        if "resources" in st.session_state and st.session_state.resources:
            st.markdown(st.session_state.resources)
        
        # Additional general resources
        with st.expander("General Teaching Resources"):
            st.markdown("""
            ### Online Platforms
            - **Khan Academy** - Free video lessons and practice exercises
            - **Edutopia** - Research-based strategies for teaching
            - **TeachersPayTeachers** - Lesson plans and teaching materials
            
            ### Teaching Methods
            - **Universal Design for Learning (UDL)** - Framework for flexible learning environments
            - **Differentiated Instruction** - Tailoring instruction to individual needs
            - **Project-Based Learning** - Engaging students through projects
            
            ### Assessment Tools
            - **Formative** - Create and share formative assessments
            - **Kahoot** - Game-based learning platform
            - **Quizlet** - Digital flashcards and study tools
            """)
    
    def chat_interface(self):
        """Display the chat interface for teacher-student interaction."""
        
        # Get scenario from session state, create default if None
        scenario = st.session_state.scenario
        if scenario is None:
            scenario = self.create_default_scenario()
            st.session_state.scenario = scenario
            
        # Now safely use the scenario object, which is guaranteed to be initialized
        student_profile = scenario.get('student_profile', {})
        
        # Initialize the LLM if not already done
        if not st.session_state.llm_interface:
            self.initialize_llm()
        
        # Add app header
        st.markdown("""
        <div class="app-header">
            <h1>Teacher Training Simulator</h1>
            <p>Practice and improve your teaching skills through realistic student interactions</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Layout the interface with a simple two-column design
        col1, col2 = st.columns([7, 3])
        
        with col1:
            # Enhanced header with more scenario information
            st.markdown(f"""
            <div class="scenario-header">
                <div class="scenario-title">
                    <span class="scenario-title-icon">📚</span> {scenario.get('subject', 'Subject')} - {scenario.get('topic', 'Topic')}
                </div>
                <p><strong>Grade Level:</strong> {scenario.get('grade_level', 'Not specified')}</p>
                <p><strong>Scenario Description:</strong> {scenario.get('scenario_description', 'No description available')}</p>
                <p><strong>Current Activity:</strong> {scenario.get('current_activity', 'No activity specified')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Initialize the messages if they don't exist
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Add a welcome message from the student if this is the start of conversation
            if not st.session_state.messages:
                with st.spinner("Student is introducing themselves..."):
                    try:
                        # Generate a more natural introduction from the student
                        initial_prompt = f"""
                        You are a student named {student_profile.get('name', 'Jamie')} in a {scenario.get('grade_level', '7th grade')} classroom. 
                        
                        SCENARIO CONTEXT:
                        - Subject: {scenario.get('subject', 'Mathematics')}
                        - Topic: {scenario.get('topic', 'Introduction to Algebra')}
                        - Current activity: {scenario.get('current_activity', 'Working on problems')}
                        - Your learning style: {', '.join(student_profile.get('learning_style', ['visual']))}
                        - Your challenges: {', '.join(student_profile.get('challenges', ['math anxiety']))}
                        
                        The teacher has just entered the room and you are making eye contact.
                        Give a brief, realistic greeting to the teacher that subtly shows your personality and 
                        relates to the current subject/activity.
                        
                        Don't introduce yourself with a long description - just a natural, brief greeting
                        that a {student_profile.get('grade_level', '7th grade')} student would say, like "Hi" or "Good morning",
                        possibly with a question or comment about the subject.
                        Keep it to 1-2 sentences maximum and make it relevant to the scenario.
                        """
                        
                        # Fallback introduction for any errors
                        student_name = student_profile.get('name', 'Jamie')
                        fallback_intro = f"Hi, teacher! I'm {student_name}."
                        
                        # Try to generate a response
                        try:
                            # Format conversation history for context
                            history = self._format_conversation_history(st.session_state.messages)
                            
                            # Create context with conversation history
                            context_with_history = scenario.copy()
                            context_with_history['conversation_history'] = history
                            context_with_history['session_state'] = {'last_student_response': ''}
                            
                            # Try to generate a student introduction
                            student_intro = st.session_state.llm_interface.simulate_student_response(
                                teacher_input=initial_prompt,
                                student_profile=student_profile,
                                scenario_context=context_with_history
                            )
                            
                            # Clean up the intro to make sure it's not too formal or descriptive
                            if not student_intro or len(student_intro.split()) > 25 or "Error" in student_intro:
                                student_intro = fallback_intro
                        except Exception as e:
                            logging.error(f"Error generating initial student response: {e}", exc_info=True)
                            student_intro = fallback_intro
                        
                        # Add the student introduction to the messages
                        st.session_state.messages.append({"role": "student", "content": student_intro, "timestamp": self._get_timestamp()})
                    except Exception as e:
                        st.error(f"Error setting up student introduction: {str(e)}")
                        logging.error(f"Error setting up student introduction: {e}", exc_info=True)
                        st.session_state.messages.append({
                            "role": "student", 
                            "content": f"Hi teacher! I'm {student_profile.get('name', 'Jamie')}.",
                            "timestamp": self._get_timestamp()
                        })
            
            # Display conversation header
            st.markdown("""
            <div class="conversation-header">
                <h3>Classroom Conversation</h3>
                <p>Interact with the simulated student below.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Use a container with dynamic height for the conversation
            chat_container = st.container()
            with chat_container:
                # Display the conversation
                for message in st.session_state.messages:
                    with st.chat_message(message["role"], avatar="👨‍🎓" if message["role"] == "student" else "👨‍🏫"):
                        st.write(f"{message['timestamp']}")
                        st.markdown(message["content"])
            
            # Get teacher input
            teacher_input = st.chat_input("Type your response as the teacher...")
            
            if teacher_input:
                # Add teacher message to chat history
                st.session_state.messages.append({"role": "teacher", "content": teacher_input, "timestamp": self._get_timestamp()})
                
                # Process student response in the background
                try:
                    # Format conversation history for context
                    history = self._format_conversation_history(st.session_state.messages)
                    
                    # Get the last student response to avoid repetition
                    last_student_response = ""
                    for msg in reversed(st.session_state.messages):
                        if msg["role"] == "student":
                            last_student_response = msg["content"]
                            break
                    
                    # Add classroom management strategies to the context if any are selected
                    strategy_instructions = ""
                    if st.session_state.classroom_management_strategies:
                        strategy_instructions = "\n\nThe teacher is using the following classroom management strategies:\n"
                        for strategy in st.session_state.classroom_management_strategies:
                            strategy_instructions += f"- {strategy['name']}: {strategy['description']}\n"
                        strategy_instructions += "\nThe student should respond appropriately to these teaching strategies. If the teacher effectively uses the strategy, the student should show a slightly more positive, engaged, or improved response. If the strategy is not effectively applied, the student can show confusion or maintain their previous state."
                    
                    # Create an enhanced teacher input with explicit instructions to be conversational
                    enhanced_teacher_input = f"""
                    The teacher just said: "{teacher_input}"
                    
                    You are {student_profile.get('name', 'Jamie')}, a {student_profile.get('grade_level', '7th grade')} student.
                    Respond naturally as yourself in a brief, realistic way.
                    
                    KEY INSTRUCTIONS:
                    1. DO NOT repeat your introduction or describe yourself again
                    2. DO NOT say "As a 7th-grade student who..." or anything like that
                    3. Respond directly to what the teacher just said
                    4. Keep your response short (1-3 sentences)
                    5. Use casual, age-appropriate language
                    6. Show your personality through your response
                    7. If the teacher just said "hello" or greeted you, respond naturally and maybe mention something about the class or ask a question
                    8. NEVER list your traits, challenges or learning style directly - these should only be implied by your response
                    9. IMPORTANT: NEVER repeat any of your previous responses verbatim
                    10. Each of your responses should be unique and move the conversation forward
                    {strategy_instructions}
                    
                    Your previous response was: "{last_student_response}"
                    DO NOT repeat this response - say something completely new and different that makes sense in the context of this conversation.
                    """
                    
                    # Create context with conversation history and last response
                    context_with_history = scenario.copy()
                    context_with_history['conversation_history'] = history
                    context_with_history['previous_response'] = last_student_response
                    context_with_history['session_state'] = {'last_student_response': last_student_response}
                    context_with_history['instruction'] = "Be brief and conversational, like a real student in class"
                    
                    # Add classroom management strategies to the context
                    if st.session_state.classroom_management_strategies:
                        context_with_history['classroom_management_strategies'] = st.session_state.classroom_management_strategies
                    
                    # Try to get the student's response with fallbacks
                    try:
                        response = st.session_state.llm_interface.simulate_student_response(
                            teacher_input=enhanced_teacher_input,
                            student_profile=student_profile,
                            scenario_context=context_with_history
                        )
                        
                        # Check if response is identical to any previous student response
                        previous_responses = [msg["content"] for msg in st.session_state.messages if msg["role"] == "student"]
                        if response in previous_responses:
                            logging.warning("Generated response is identical to a previous one. Using fallback.")
                            
                            # Random conversational responses that can work in various contexts
                            fallback_options = [
                                "I'm not sure I understand. Could you explain that differently?",
                                "That's interesting. Can we talk more about how this works?",
                                "I've been thinking about what you said. Can you give another example?",
                                "OK, I think I'm starting to get it.",
                                "Let me try to understand this from another angle.",
                                "I'm still confused about part of this.",
                                "So what does that mean for our assignment?",
                                "I hadn't thought about it that way before."
                            ]
                            
                            # Choose a random fallback that's not in previous responses
                            filtered_fallbacks = [opt for opt in fallback_options if opt not in previous_responses]
                            if filtered_fallbacks:
                                response = random.choice(filtered_fallbacks)
                            else:
                                # Create a unique response by adding a small random modifier
                                response = f"I see. {random.choice(['Actually, ', 'Well, ', 'Hmm, ', 'So, '])}I'm still working through this."
                    except Exception as e:
                        logging.error(f"Error generating student response: {e}", exc_info=True)
                        # Set up a fallback response
                        response = "I'm not sure how to respond to that."
                    
                    # Check if response is empty or has an error
                    if not response or "Error" in response:
                        # Create a simple fallback response based on the teacher's input
                        if "?" in teacher_input:
                            response = "I'm not really sure about that. Can you explain it differently?"
                        else:
                            response = "Okay, I understand. What should we do next?"
                    
                    # Check if response seems like a self-description rather than a natural response
                    # We'll look for common patterns that indicate the student is describing themselves
                    description_indicators = [
                        "as a", "my learning style", "I am a", "my strengths", 
                        "my challenges", "I struggle with", "my teachers say"
                    ]
                    
                    is_description = any(indicator in response.lower() for indicator in description_indicators)
                    
                    # If it's a description or too long, generate a simpler response
                    if is_description or len(response.split()) > 50:
                        logging.info("Detected self-description in student response, generating simpler response")
                        
                        # Generate a very simple, age-appropriate response
                        if "hello" in teacher_input.lower() or "hi" in teacher_input.lower() or "hey" in teacher_input.lower():
                            fallback_options = [
                                f"Hi! Are we starting algebra today?",
                                f"Hey! Do we need our textbooks for today's lesson?",
                                f"Hello! I was just trying to figure out this math problem.",
                                f"Hi there! Is this going to be on the test?",
                                f"Hey teacher, I was wondering what we're covering today.",
                                f"Hi! I've been stuck on problem number 5, can you help me?"
                            ]
                            # Avoid repeating the last student response
                            last_response = ""
                            for msg in reversed(st.session_state.messages):
                                if msg["role"] == "student":
                                    last_response = msg["content"]
                                    break
                            
                            # Filter out any options that match the previous response
                            filtered_options = [opt for opt in fallback_options if opt != last_response]
                            
                            # If we have options left, choose from them, otherwise use a more generic response
                            if filtered_options:
                                response = random.choice(filtered_options)
                            else:
                                response = "Sorry, I didn't catch what you said. Can you explain what we're doing today?"
                        elif "?" in teacher_input:
                            fallback_options = [
                                "Um, I'm not really sure.",
                                "I don't know, can you explain it again?",
                                "Maybe? I get confused with all these x's and y's.",
                                "I think so... but I'm not 100% sure.",
                                "Could you break that down for me?",
                                "I'm still trying to understand that concept.",
                                "Not completely. Could you show an example?"
                            ]
                            # Avoid repeating the last student response
                            last_response = ""
                            for msg in reversed(st.session_state.messages):
                                if msg["role"] == "student":
                                    last_response = msg["content"]
                                    break
                            
                            # Filter out any options that match the previous response
                            filtered_options = [opt for opt in fallback_options if opt != last_response]
                            
                            # If we have options left, choose from them, otherwise use a more generic response
                            if filtered_options:
                                response = random.choice(filtered_options)
                            else:
                                response = "I'm confused about that question. Can we try a different approach?"
                        else:
                            fallback_options = [
                                "Okay, I'll try that.",
                                "That makes sense, I think.",
                                "So we're supposed to solve for x?",
                                "Can you show me an example first?",
                                "Math is so confusing sometimes.",
                                "I'll give it my best shot.",
                                "Let me see if I understand the process correctly.",
                                "That's clearer now, thanks!"
                            ]
                            # Avoid repeating the last student response
                            last_response = ""
                            for msg in reversed(st.session_state.messages):
                                if msg["role"] == "student":
                                    last_response = msg["content"]
                                    break
                            
                            # Filter out any options that match the previous response
                            filtered_options = [opt for opt in fallback_options if opt != last_response]
                            
                            # If we have options left, choose from them, otherwise use a more generic response
                            if filtered_options:
                                response = random.choice(filtered_options)
                            else:
                                response = "I'll work on that. Is there something specific I should focus on?"
                    
                    # Add the response to messages
                    timestamp = self._get_timestamp()
                    st.session_state.messages.append({"role": "student", "content": response, "timestamp": timestamp})
                    
                except Exception as e:
                    error_message = f"Error generating student response: {str(e)}"
                    logging.error(f"{error_message}", exc_info=True)
                    
                    # Provide a fallback response
                    fallback_response = f"Sorry, what do you mean?"
                    timestamp = self._get_timestamp()
                    st.session_state.messages.append({"role": "student", "content": fallback_response, "timestamp": timestamp})
                
                # Rerun the app to display the updated conversation
                st.rerun()
        
        # Simplified information panel in the side column
        with col2:
            # Wrap all info cards in a scrollable container
            st.markdown('<div class="info-panel">', unsafe_allow_html=True)
            
            # Student Profile Card - Simplified
            st.markdown("""
            <div class="scenario-card">
                <div class="scenario-title">
                    <span class="scenario-title-icon">👨‍🎓</span> Student Profile
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"**Name:** {student_profile.get('name', 'Not specified')}")
            st.markdown(f"**Grade:** {student_profile.get('grade_level', 'Not specified')}")
            st.markdown(f"**Age:** {student_profile.get('age', 'Not specified')}")
            
            # Add personality and interests
            if student_profile.get('personality'):
                st.markdown(f"**Personality:** {student_profile.get('personality', '')}")
            
            if student_profile.get('interests'):
                st.markdown("**Interests:**")
                interests = student_profile.get('interests', [])
                if isinstance(interests, list):
                    st.markdown(', '.join(interests))
                else:
                    st.markdown(interests)
            
            # Simplified Learning Style display
            if student_profile.get('learning_style'):
                st.markdown("**Learning Style:**")
                st.markdown('<ul class="info-list">', unsafe_allow_html=True)
                for style in student_profile.get('learning_style', []):
                    st.markdown(f"<li>{style}</li>", unsafe_allow_html=True)
                st.markdown('</ul>', unsafe_allow_html=True)
                
            # Simplified Challenges display
            if student_profile.get('challenges'):
                st.markdown("**Challenges:**")
                st.markdown('<ul class="info-list">', unsafe_allow_html=True)
                for challenge in student_profile.get('challenges', []):
                    st.markdown(f"<li>{challenge}</li>", unsafe_allow_html=True)
                st.markdown('</ul>', unsafe_allow_html=True)
            
            # Simplified Strengths display
            if student_profile.get('strengths'):
                st.markdown("**Strengths:**")
                st.markdown('<ul class="info-list">', unsafe_allow_html=True)
                for strength in student_profile.get('strengths', []):
                    st.markdown(f"<li>{strength}</li>", unsafe_allow_html=True)
                st.markdown('</ul>', unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Add a Current Scenario card
            st.markdown("""
            <div class="scenario-card">
                <div class="scenario-title">
                    <span class="scenario-title-icon">📝</span> Current Scenario
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"**Subject:** {scenario.get('subject', 'Not specified')}")
            st.markdown(f"**Topic:** {scenario.get('topic', 'Not specified')}")
            
            if scenario.get('classroom_setting'):
                st.markdown(f"**Setting:** {scenario.get('classroom_setting', '')}")
                
            if scenario.get('difficulty'):
                st.markdown(f"**Difficulty:** {scenario.get('difficulty', '').capitalize()}")
                
            # Add scenario description if available
            if scenario.get('scenario_description'):
                with st.expander("Scenario Description"):
                    st.markdown(scenario.get('scenario_description', ''))
                    
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Learning Objectives - If available
            if scenario.get('learning_objectives'):
                st.markdown("""
                <div class="scenario-card">
                    <div class="scenario-title">
                        <span class="scenario-title-icon">🎯</span> Learning Objectives
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown('<ul class="info-list">', unsafe_allow_html=True)
                for objective in scenario.get('learning_objectives', []):
                    st.markdown(f"<li>{objective}</li>", unsafe_allow_html=True)
                st.markdown('</ul>', unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Control panel for buttons - Simplified
            st.markdown("""
            <div class="control-panel">
                <div class="scenario-title">
                    <span class="scenario-title-icon">⚙️</span> Controls
                </div>
            """, unsafe_allow_html=True)
            
            # Add a button to reset the conversation
            if st.button("Reset Conversation", key="reset_conversation"):
                st.session_state.messages = []
                st.rerun()
            
            # Add a button to start a new scenario
            if st.button("Create New Scenario", key="new_scenario"):
                st.session_state.page = "create_scenario"
                st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Close the info panel div
            st.markdown('</div>', unsafe_allow_html=True)

    def _get_timestamp(self):
        """Get the current timestamp for messages."""
        return datetime.now().strftime("%H:%M:%S")

    def create_default_scenario(self):
        """Create a default teaching scenario when none is selected."""
        default_scenario = {
            "title": "Basic Math Lesson",
            "description": "A typical middle school math lesson focusing on basic algebra concepts.",
            "topic_description": "Basic algebra, focusing on solving linear equations.",
                "grade_level": "7th grade",
            "subject": "Math",
            "student_profile": {
                "name": "Jamie",
                "age": 12,
                "grade_level": "7th grade",
                "learning_style": "Visual learner",
                "strengths": "Good at recognizing patterns",
                "challenges": "Sometimes struggles with abstract concepts",
                "interests": "Video games, basketball, music",
                "background": "Generally does well in school but has been struggling recently with math concepts."
            },
                "example_problems": [
                "Solve for x: 3x + 5 = 20",
                "If 2(x + 3) = 16, what is x?",
                "Find the value of y in the equation y - 7 = 15"
            ],
            "teaching_challenge": "The student has difficulty understanding the concept of variables in algebra."
        }
        
        # Set in session state
        st.session_state.scenario = default_scenario
        
        # Also return the scenario object
        return default_scenario

    def show_evaluation_section(self):
        """Display the evaluation section for teacher responses."""
        st.markdown("### Teaching Improvement Center")
        st.markdown("""
        This section allows you to get scenario-specific recommendations and 
        contribute your expertise to improve the system through human feedback.
        """)
        
        # Check if there are any messages to evaluate
        teacher_messages = [m for m in st.session_state.messages if m["role"] == "teacher"]
        if not st.session_state.messages or len(teacher_messages) == 0:
            st.warning("No teaching responses yet. Please have a conversation with the student first.")
            return
        
        # Create tabs for different evaluation views
        tabs = st.tabs(["Scenario Recommendations", "Human Feedback"])
        
        with tabs[0]:
            self._show_scenario_recommendations()
            
        with tabs[1]:
            self._show_human_feedback_section()
            
    def _show_scenario_recommendations(self):
        """Show teaching recommendations based on the current scenario."""
        scenario = st.session_state.scenario
        student_profile = st.session_state.student_profile or {}
        
        st.markdown("### Teaching Recommendations")
        st.markdown("Based on the current scenario and student profile, here are some recommended teaching strategies:")
        
        # Get scenario-specific recommendations
        with st.container():
            st.markdown("#### Scenario Context")
            scenario_description = scenario.get('description', 'No scenario description available.')
            teaching_challenge = scenario.get('teaching_challenge', 'No specific challenge identified.')
            
            st.markdown(f"**Scenario:** {scenario_description}")
            st.markdown(f"**Teaching Challenge:** {teaching_challenge}")
            
            # Display a divider
            st.markdown("---")
            
            # Provide recommendations based on student profile
            st.markdown("#### Student-Specific Strategies")
            
            learning_style = student_profile.get('learning_style', 'Not specified')
            challenges = student_profile.get('challenges', 'Not specified')
            
            if isinstance(learning_style, list):
                learning_style = ", ".join(learning_style)
            if isinstance(challenges, list):
                challenges = ", ".join(challenges)
            
            st.markdown(f"**Learning Style:** {learning_style}")
            st.markdown(f"**Challenges:** {challenges}")
            
            # Generate recommendations based on learning style
            learning_style_recommendations = self._get_learning_style_recommendations(learning_style)
            st.markdown("##### Learning Style Recommendations:")
            for rec in learning_style_recommendations:
                st.markdown(f"- {rec}")
                
            # Generate recommendations based on challenges
            challenge_recommendations = self._get_challenge_recommendations(challenges)
            st.markdown("##### Challenge-Specific Recommendations:")
            for rec in challenge_recommendations:
                st.markdown(f"- {rec}")
            
            # Add general teaching strategies
            st.markdown("#### General Teaching Strategies")
            general_strategies = [
                "Use clear and concise explanations with concrete examples",
                "Check for understanding frequently using a variety of assessment methods",
                "Connect new content to students' prior knowledge and experiences",
                "Provide scaffolded support that gradually releases responsibility",
                "Use visual aids and manipulatives to represent abstract concepts",
                "Incorporate opportunities for active learning and student discussion"
            ]
            
            for strategy in general_strategies:
                st.markdown(f"- {strategy}")
                
    def _get_learning_style_recommendations(self, learning_style):
        """Get teaching recommendations based on learning style."""
        learning_style = learning_style.lower()
        
        recommendations = []
        
        if "visual" in learning_style:
            recommendations.extend([
                "Use diagrams, charts, and visual models to explain concepts",
                "Incorporate color-coding to highlight key information",
                "Provide graphic organizers to structure information",
                "Use videos and animations to demonstrate processes"
            ])
            
        if "auditory" in learning_style or "verbal" in learning_style:
            recommendations.extend([
                "Use clear verbal explanations with varied tone and emphasis",
                "Incorporate discussions and think-aloud strategies",
                "Use rhymes, mnemonics, or songs to aid memory",
                "Provide opportunities for students to explain concepts verbally"
            ])
            
        if "kinesthetic" in learning_style or "hands-on" in learning_style or "tactile" in learning_style:
            recommendations.extend([
                "Incorporate manipulatives and hands-on activities",
                "Use role-play and physical movement to represent concepts",
                "Provide opportunities for students to build models",
                "Include experiments and demonstrations where students actively participate"
            ])
            
        if not recommendations:
            recommendations = [
                "Use multi-modal teaching approaches that combine visual, auditory, and hands-on elements",
                "Vary instructional methods to engage different learning preferences",
                "Provide multiple representations of key concepts",
                "Offer choices in how students demonstrate understanding"
            ]
            
        return recommendations
        
    def _get_challenge_recommendations(self, challenges):
        """Get teaching recommendations based on student challenges."""
        challenges = challenges.lower()
        
        recommendations = []
        
        if "attention" in challenges or "focus" in challenges:
            recommendations.extend([
                "Break longer tasks into smaller, manageable chunks",
                "Use timers and clear transitions between activities",
                "Minimize distractions in the learning environment",
                "Incorporate movement breaks between tasks"
            ])
            
        if "anxiety" in challenges or "confidence" in challenges:
            recommendations.extend([
                "Create a supportive environment where mistakes are viewed as learning opportunities",
                "Provide private feedback and avoid putting the student on the spot",
                "Use specific praise to build confidence",
                "Gradually increase task difficulty to build success"
            ])
            
        if "abstract" in challenges or "conceptual" in challenges:
            recommendations.extend([
                "Connect abstract concepts to concrete, real-world examples",
                "Use analogies and metaphors to explain complex ideas",
                "Provide step-by-step procedures with visual supports",
                "Use manipulatives and models to represent abstract concepts"
            ])
            
        if "math" in challenges or "calculation" in challenges:
            recommendations.extend([
                "Use visual representations of mathematical concepts",
                "Break down multi-step problems into smaller steps",
                "Connect mathematical procedures to conceptual understanding",
                "Provide opportunities for repeated practice with immediate feedback"
            ])
            
        if "reading" in challenges or "literacy" in challenges:
            recommendations.extend([
                "Pre-teach key vocabulary and concepts",
                "Use text-to-speech or read material aloud",
                "Provide graphic organizers to structure reading comprehension",
                "Break text into smaller sections with comprehension checks"
            ])
            
        if not recommendations:
            recommendations = [
                "Use scaffolded instruction with gradual release of responsibility",
                "Provide clear, specific feedback focused on improvement",
                "Check for understanding frequently during instruction",
                "Use multiple modalities to present information"
            ]
            
        return recommendations
        
    def _show_human_feedback_section(self):
        """Show a section for capturing expert knowledge and examples."""
        st.markdown("### Expert Knowledge Capture")
        st.markdown("""
        Share your teaching expertise to improve the system. We learn directly from how experts respond to student questions.
        """)
        
        # Two tabs for different ways to contribute expertise
        feedback_tabs = st.tabs(["Improve Responses", "View Collected Examples"])
        
        with feedback_tabs[0]:
            # Get the last student message as context
            student_messages = [m for m in st.session_state.messages if m["role"] == "student"]
            teacher_messages = [m for m in st.session_state.messages if m["role"] == "teacher"]
            
            if not student_messages:
                st.warning("No student questions available. Please have a conversation with the student first.")
                return
                
            # Show the student's question
            last_student_msg = student_messages[-1]["content"]
            last_teacher_msg = teacher_messages[-1]["content"] if teacher_messages else ""
            
            st.markdown("#### Student Question")
            st.info(last_student_msg)
            
            # Current system response (if any)
            if last_teacher_msg:
                st.markdown("#### Current Response")
                st.info(last_teacher_msg)
            
            # Simple form for expert to provide a better response
            st.markdown("#### Your Expert Response")
            st.markdown("Please provide a better response to this student question:")
            
            with st.form("expert_response_form"):
                # Expert's improved response
                expert_response = st.text_area(
                    "How would you respond to this student?",
                    value=last_teacher_msg,
                    height=150,
                    placeholder="Type your expert response here..."
                )
                
                # Simple rating of the original response
                original_rating = st.slider(
                    "How would you rate the original response? (1-10)",
                    min_value=1,
                    max_value=10,
                    value=5
                )
                
                # Why is the expert's response better?
                improvement_reason = st.text_area(
                    "What makes your response better? (Optional)",
                    placeholder="My response is better because..."
                )
                
                # Teaching techniques used
                techniques = st.multiselect(
                    "What teaching techniques did you use? (Select all that apply)",
                    options=[
                        "Clear explanation",
                        "Real-world example",
                        "Questioning technique",
                        "Scaffolding",
                        "Visual representation",
                        "Analogy/metaphor",
                        "Step-by-step process",
                        "Connecting to prior knowledge",
                        "Addressing misconception",
                        "Encouraging reflection",
                        "Positive reinforcement",
                        "Differentiated approach"
                    ]
                )
                
                submit = st.form_submit_button("Submit Expert Response")
                
            if submit:
                # Save the expert example
                try:
                    # Get context information
                    scenario = st.session_state.scenario if hasattr(st.session_state, 'scenario') else {}
                    student_profile = st.session_state.student_profile if hasattr(st.session_state, 'student_profile') else {}
                    
                    # Create an example
                    import datetime
                    
                    example = {
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "student_question": last_student_msg,
                        "original_response": last_teacher_msg,
                        "expert_response": expert_response,
                        "original_rating": original_rating,
                        "improvement_reason": improvement_reason,
                        "techniques_used": techniques,
                        "context": {
                            "scenario": scenario.get("description", ""),
                            "subject": scenario.get("subject", ""),
                            "student_name": student_profile.get("name", ""),
                            "grade_level": student_profile.get("grade_level", ""),
                            "learning_style": student_profile.get("learning_style", "")
                        }
                    }
                    
                    # Store the example
                    if "expert_examples" not in st.session_state:
                        st.session_state.expert_examples = []
                        
                    st.session_state.expert_examples.append(example)
                    
                    # Save to disk
                    self._save_expert_example(example)
                    
                    st.success("Thank you! Your expert response has been recorded.")
                    
                    # Option to use this response in the conversation
                    if st.button("Use my response in the conversation"):
                        # Add the expert's response to the conversation
                        st.session_state.messages.append({
                            "role": "teacher",
                            "content": expert_response,
                            "source": "expert"
                        })
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error saving example: {str(e)}")
                    
        with feedback_tabs[1]:
            # Show collected expert examples
            st.markdown("### Collected Expert Examples")
            
            if "expert_examples" not in st.session_state or not st.session_state.expert_examples:
                st.info("No expert examples collected yet. Please contribute your expertise in the 'Improve Responses' tab.")
            else:
                st.success(f"{len(st.session_state.expert_examples)} expert examples collected")
                
                # Export option
                if st.button("Export Expert Examples Dataset"):
                    export_path = self._export_expert_examples()
                    if export_path:
                        st.success(f"Expert examples exported to: {export_path}")
                    else:
                        st.error("Failed to export expert examples")
                        
                # View collected examples
                if st.session_state.expert_examples:
                    for i, example in enumerate(st.session_state.expert_examples):
                        with st.expander(f"Example #{i+1}: {example.get('context', {}).get('subject', 'Unknown')}"):
                            st.markdown("**Student Question:**")
                            st.info(example.get("student_question", ""))
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Original Response:**")
                                st.info(example.get("original_response", ""))
                                st.markdown(f"Rating: {example.get('original_rating', 0)}/10")
                                
                            with col2:
                                st.markdown("**Expert Response:**")
                                st.success(example.get("expert_response", ""))
                                
                                if example.get("techniques_used"):
                                    st.markdown("**Techniques Used:**")
                                    for technique in example.get("techniques_used", []):
                                        st.markdown(f"- {technique}")
                            
                            if example.get("improvement_reason"):
                                st.markdown("**Improvement Reason:**")
                                st.write(example.get("improvement_reason", ""))
    
    def _save_expert_example(self, example):
        """Save an expert example to disk."""
        import json
        import os
        from datetime import datetime
        
        # Create the examples directory if it doesn't exist
        examples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'expert_examples')
        os.makedirs(examples_dir, exist_ok=True)
        
        # Create a filename based on timestamp
        timestamp = example["timestamp"].replace(":", "-").replace(" ", "_")
        filename = f"expert_example_{timestamp}.json"
        filepath = os.path.join(examples_dir, filename)
        
        # Write to file
        try:
            with open(filepath, 'w') as f:
                json.dump(example, f, indent=2)
            logging.info(f"Expert example saved to {filepath}")
            return filepath
        except Exception as e:
            logging.error(f"Error saving expert example: {str(e)}")
            return None
            
    def _export_expert_examples(self):
        """Export all expert examples as a dataset file."""
        if "expert_examples" not in st.session_state or not st.session_state.expert_examples:
            logging.warning("No expert examples to export")
            return None
            
        import json
        import os
        from datetime import datetime
        
        try:
            # Create the export directory
            export_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'expert_datasets')
            os.makedirs(export_dir, exist_ok=True)
            
            # Create a filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"expert_teaching_dataset_{timestamp}.jsonl"
            filepath = os.path.join(export_dir, filename)
            
            # Write all examples to a JSONL file (for easy machine reading)
            with open(filepath, 'w') as f:
                for example in st.session_state.expert_examples:
                    f.write(json.dumps(example) + '\n')
                    
            # Also create a human-readable CSV version
            csv_path = os.path.join(export_dir, f"expert_teaching_dataset_{timestamp}.csv")
            try:
                import csv
                with open(csv_path, 'w', newline='') as csvfile:
                    fieldnames = [
                        'timestamp', 'student_question', 'original_response', 
                        'expert_response', 'original_rating', 'improvement_reason', 
                        'techniques_used', 'subject', 'grade_level'
                    ]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    writer.writeheader()
                    for example in st.session_state.expert_examples:
                        writer.writerow({
                            'timestamp': example.get('timestamp', ''),
                            'student_question': example.get('student_question', ''),
                            'original_response': example.get('original_response', ''),
                            'expert_response': example.get('expert_response', ''),
                            'original_rating': example.get('original_rating', ''),
                            'improvement_reason': example.get('improvement_reason', ''),
                            'techniques_used': ','.join(example.get('techniques_used', [])),
                            'subject': example.get('context', {}).get('subject', ''),
                            'grade_level': example.get('context', {}).get('grade_level', '')
                        })
            except Exception as e:
                logging.error(f"Error creating CSV: {str(e)}")
                
            # Create a simple README
            readme_path = os.path.join(export_dir, "README.md")
            with open(readme_path, 'w') as f:
                f.write(f"""# Expert Teaching Examples Dataset

This dataset contains {len(st.session_state.expert_examples)} examples of expert teacher responses to student questions.

## Dataset Format

Each example includes:

- Student question/comment
- Original system response
- Expert improved response
- Rating of original response
- Explanation of improvements
- Teaching techniques used
- Contextual information (subject, grade level, etc.)

## Usage

This dataset can be used for:

1. Fine-tuning language models to better respond to student questions
2. Creating teaching technique classifiers
3. Analyzing patterns in expert teaching responses

## Latest Dataset

- JSONL Format: `{filename}` (machine-readable)
- CSV Format: `{os.path.basename(csv_path)}` (human-readable)
- Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- Examples: {len(st.session_state.expert_examples)}
""")
            
            logging.info(f"Expert examples exported with {len(st.session_state.expert_examples)} examples to {filepath}")
            return filepath
            
        except Exception as e:
            logging.error(f"Error exporting expert examples: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return None

    def _prepare_rlhf_dataset(self):
        """Prepare the collected feedback for RLHF fine-tuning."""
        if 'system_feedback' not in st.session_state or not st.session_state.system_feedback:
            logging.warning("No feedback data available for RLHF preparation")
            return None
        
        import json
        import os
        from datetime import datetime
        
        try:
            # Create both RLHF and DSPy dataset directories
            rlhf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'rlhf_datasets')
            dspy_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'dspy_datasets')
            os.makedirs(rlhf_dir, exist_ok=True)
            os.makedirs(dspy_dir, exist_ok=True)
            
            # Timestamp for the dataset
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rlhf_filename = f"rlhf_dataset_{timestamp}.jsonl"
            dspy_filename = f"dspy_dataset_{timestamp}.jsonl"
            rlhf_filepath = os.path.join(rlhf_dir, rlhf_filename)
            dspy_filepath = os.path.join(dspy_dir, dspy_filename)
            
            # Prepare the data in the format needed for RLHF
            rlhf_ready_entries = []
            
            # Prepare data for DSPy fine-tuning
            dspy_examples = []
            
            for entry in st.session_state.system_feedback:
                # Get key data from the feedback
                context = entry.get('student_response', "")
                teacher_response = entry.get('teacher_response', "")
                alternative = entry.get('alternative_response', "")
                preference = entry.get('preference', "")
                preference_reason = entry.get('preference_reason', "")
                
                # Fetch scenario information for context
                scenario = st.session_state.scenario if hasattr(st.session_state, 'scenario') else {}
                student_profile = st.session_state.student_profile if hasattr(st.session_state, 'student_profile') else {}
                
                # Create full context string for DSPy
                full_context = f"""
                SCENARIO: {scenario.get('description', 'Not specified')}
                STUDENT: {student_profile.get('name', 'Student')}, Grade: {student_profile.get('grade_level', 'Not specified')}
                LEARNING STYLE: {student_profile.get('learning_style', 'Not specified')}
                CHALLENGES: {student_profile.get('challenges', 'Not specified')}
                
                STUDENT QUESTION/COMMENT: {context}
                """
                
                # Create DSPy fine-tuning example - including both responses but marking preferred one
                dspy_example = {
                    "task": "teaching_response",
                    "input": full_context.strip(),
                    "output": teacher_response,
                    "alternatives": [alternative],
                    "preferred": 0 if preference == "Original response is better" else 1,
                    "feedback": preference_reason,
                    "metadata": {
                        "scenario_type": scenario.get('title', 'Unknown'),
                        "subject": scenario.get('subject', 'Not specified'),
                        "effectiveness_rating": entry.get('effectiveness_rating', 0),
                        "strengths": entry.get('strengths', ''),
                        "improvements": entry.get('improvements', '')
                    }
                }
                
                dspy_examples.append(dspy_example)
                
                # Only process entries with clear preferences for RLHF
                if 'preference' in entry and entry['preference'] != "Both are equally effective":
                    # Get the original and alternative responses
                    original = entry['teacher_response']
                    
                    if not alternative:
                        continue  # Skip if no alternative available
                    
                    # Determine which is preferred (chosen vs rejected)
                    if entry['preference'] == "Original response is better":
                        chosen = original
                        rejected = alternative
                    else:  # "Alternative response is better"
                        chosen = alternative
                        rejected = original
                    
                    # Create RLHF entry
                    rlhf_entry = {
                        "prompt": context,
                        "chosen": chosen,
                        "rejected": rejected,
                        "reason": entry.get('preference_reason', ""),
                        "metadata": {
                            "scenario": scenario.get('title', 'Unknown'),
                            "effectiveness_rating": entry.get('effectiveness_rating', 0),
                            "timestamp": entry.get('timestamp', "")
                        }
                    }
                    
                    rlhf_ready_entries.append(rlhf_entry)
            
            # Write to JSONL files
            with open(rlhf_filepath, 'w') as f:
                for entry in rlhf_ready_entries:
                    f.write(json.dumps(entry) + '\n')
                    
            with open(dspy_filepath, 'w') as f:
                for example in dspy_examples:
                    f.write(json.dumps(example) + '\n')
            
            # Create a Python script that demonstrates how to use this data with DSPy
            script_path = os.path.join(dspy_dir, f"dspy_finetuning_example_{timestamp}.py")
            self._create_dspy_finetuning_script(script_path, dspy_filepath)
            
            logging.info(f"RLHF dataset prepared with {len(rlhf_ready_entries)} entries at {rlhf_filepath}")
            logging.info(f"DSPy dataset prepared with {len(dspy_examples)} entries at {dspy_filepath}")
            logging.info(f"DSPy finetuning script created at {script_path}")
            
            return {"rlhf": rlhf_filepath, "dspy": dspy_filepath, "script": script_path}
            
        except Exception as e:
            logging.error(f"Error preparing datasets: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return None
            
    def _create_dspy_finetuning_script(self, script_path, dataset_path):
        """Create a Python script showing how to use the feedback data with DSPy."""
        script_content = f'''"""
DSPy Fine-tuning Example Using Teacher Feedback
==============================================
This script demonstrates how to use the collected teacher feedback
to fine-tune a DSPy teaching model.
"""

import json
import dspy
from dspy.teleprompt import BootstrapFewShot
from typing import List

# Path to the dataset generated from feedback
DATASET_PATH = "{dataset_path}"

class TeachingProgram(dspy.Module):
    """A DSPy program for generating teaching responses."""
    
    def __init__(self):
        super().__init__()
        self.generate_response = dspy.ChainOfThought("context -> teaching_response")
    
    def forward(self, context):
        """Generate a teaching response based on the context."""
        return self.generate_response(context=context)

def load_feedback_data(filepath):
    """Load the teaching feedback data collected from experts."""
    examples = []
    with open(filepath, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    return examples

def create_dspy_examples(feedback_data):
    """Convert feedback data to DSPy examples."""
    dspy_examples = []
    for entry in feedback_data:
        example = dspy.Example(
            context=entry["input"],
            teaching_response=entry["output"]
        ).with_metadata(
            preferred=entry["preferred"],
            feedback=entry["feedback"],
            alternatives=entry["alternatives"],
            scenario_type=entry["metadata"]["scenario_type"],
            subject=entry["metadata"]["subject"]
        )
        dspy_examples.append(example)
    return dspy_examples

def teacher_preference_metric(example, pred):
    """Metric that evaluates predictions based on teacher preferences in examples."""
    # This would be expanded to use more sophisticated evaluation logic
    # based on the collected human feedback
    correct_style = any([style in pred.teaching_response.lower() 
                        for style in example.metadata.feedback.lower().split()])
    return float(correct_style)

def main():
    # Load the model (default or fine-tuned)
    model = dspy.OpenAI(model="gpt-3.5-turbo")
    dspy.settings.configure(lm=model)
    
    # Load feedback data
    feedback_data = load_feedback_data(DATASET_PATH)
    examples = create_dspy_examples(feedback_data)
    
    print(f"Loaded {{len(examples)}} teaching examples from feedback")
    
    # Create and compile the teaching program
    program = TeachingProgram()
    
    # Create a training set and validation set
    train_size = int(len(examples) * 0.8)
    train_set = examples[:train_size]
    val_set = examples[train_size:]
    
    # Bootstrap the model with few-shot examples
    optimizer = BootstrapFewShot(
        metric=teacher_preference_metric,
        max_bootstrapped_demos=5,
        num_candidate_programs=10
    )
    
    # Optimize the program using teacher feedback
    optimized_program = optimizer.optimize(
        program=program,
        trainset=train_set,
        valset=val_set
    )
    
    # Test the optimized program
    if val_set:
        example = val_set[0]
        prediction = optimized_program(example.context)
        print(f"Context: {{example.context}}")
        print(f"Prediction: {{prediction.teaching_response}}")
    
    # The optimized program can now be used in the teaching assistant
    print("Optimization complete! The model has been fine-tuned with teacher feedback.")

if __name__ == "__main__":
    main()
'''
        try:
            with open(script_path, 'w') as f:
                f.write(script_content)
            return True
        except Exception as e:
            logging.error(f"Error creating DSPy script: {str(e)}")
        return False

    def _save_system_feedback(self, feedback_entry):
        """Save feedback to a file for future system fine-tuning."""
        import json
        import os
        
        # Create the feedback directory if it doesn't exist
        feedback_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'feedback')
        os.makedirs(feedback_dir, exist_ok=True)
        
        # Create a filename based on timestamp
        timestamp = feedback_entry["timestamp"].replace(":", "-").replace(" ", "_")
        filename = f"feedback_{timestamp}.json"
        filepath = os.path.join(feedback_dir, filename)
        
        # Additional metadata
        feedback_entry["scenario"] = st.session_state.scenario.get("title", "Unknown")
        feedback_entry["student_profile"] = st.session_state.student_profile
        
        # Write to file
        try:
            with open(filepath, 'w') as f:
                json.dump(feedback_entry, f, indent=4)
            logging.info(f"Feedback saved to {filepath}")
        except Exception as e:
            logging.error(f"Error saving feedback: {str(e)}")
            
    def _show_feedback_statistics(self):
        """Display statistics about the collected feedback."""
        if 'system_feedback' not in st.session_state or not st.session_state.system_feedback:
            st.info("No feedback data collected yet.")
            return
            
        # Calculate average effectiveness rating
        ratings = [entry.get("effectiveness_rating", 0) for entry in st.session_state.system_feedback]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        st.metric("Average Effectiveness Rating", f"{avg_rating:.1f}/10")
        st.metric("Total Feedback Submissions", len(st.session_state.system_feedback))
        
        # Display feedback over time if there are multiple entries
        if len(ratings) > 1:
            import pandas as pd
            import altair as alt
            
            # Create dataframe for visualization
            feedback_data = {
                "Timestamp": [entry.get("timestamp", "") for entry in st.session_state.system_feedback],
                "Rating": ratings
            }
            df = pd.DataFrame(feedback_data)
            
            # Create chart
            chart = alt.Chart(df).mark_line().encode(
                x='Timestamp:O',
                y=alt.Y('Rating:Q', scale=alt.Scale(domain=[1, 10])),
                tooltip=['Timestamp', 'Rating']
            ).properties(
                title='Feedback Ratings Over Time'
            )
            
            st.altair_chart(chart, use_container_width=True)

    def run(self):
        """Main method to run the Streamlit application."""
        # Initialize session state, load data, and set up page
        self.init_session_state()
        self.setup_page()
        
        # Single page application focused on SME feedback
        self.show_sme_feedback_interface()
        
    def show_sme_feedback_interface(self):
        """Main interface focused on getting feedback from Subject Matter Experts."""
        st.markdown("## Subject Matter Expert Feedback Platform")
        st.markdown("Help improve our teaching assistant by providing your expertise and evaluating responses.")
        
        # Create tabs for different feedback activities
        tabs = st.tabs(["Review Teaching Examples", "Submit Your Solutions", "View Collected Data"])
        
        with tabs[0]:
            self._show_example_review()
            
        with tabs[1]:
            self._show_teaching_examples_input()
            
        with tabs[2]:
            self._show_simplified_collected_data()
            
    def _show_example_review(self):
        """Show example teaching scenarios for SMEs to review and provide feedback."""
        st.subheader("Review Teaching Scenarios")
        st.markdown("Compare AI-suggested responses with your expert solutions.")
        
        # Generate or load teaching examples
        examples = self._create_sample_teaching_scenarios()
        
        if not examples:
            st.info("No examples are available for review. Please check back later.")
            return
            
        # Example selection
        selected_example_idx = st.selectbox(
            "Select a teaching scenario to review:",
            options=range(len(examples)),
            format_func=lambda i: f"Example {i+1}: {examples[i]['subject']} - {examples[i]['question'][:50]}...",
            key="review_tab_example_selector_main"  # Use a specific static key
        )
        
        selected_example = examples[selected_example_idx]
        
        # Display the selected example
        with st.expander("Student Question", expanded=True):
            st.markdown(f"**Subject Area:** {selected_example['subject']}")
            st.markdown(f"**Grade Level:** {selected_example['grade_level']}")
            st.markdown(f"**Question:**")
            st.markdown(f"> {selected_example['question']}")
            
            # Display context if available
            if 'context' in selected_example and selected_example['context']:
                st.markdown(f"**Context:**")
                st.markdown(f"> {selected_example['context']}")
        
        # Show AI recommendation from knowledge base
        with st.expander("AI Recommended Response", expanded=True):
            ai_response = selected_example.get('ai_response', 'AI is generating a response...')
            st.markdown(ai_response)
            
            quality_score = st.slider(
                "Rate this AI response (1-10):",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                key=f"ai_rating_{selected_example_idx}"
            )
        
        # Get SME solution
        with st.expander("Your Expert Solution", expanded=True):
            st.markdown("Please provide your expert response to this teaching scenario:")
            
            sme_response = st.text_area(
                "Your response:",
                height=200,
                key=f"sme_response_{selected_example_idx}"
            )
            
            teaching_techniques = st.multiselect(
                "Teaching techniques used in your response:",
                options=[
                    "Scaffolding", "Inquiry-based learning", "Direct instruction",
                    "Visual learning", "Differentiated instruction", "Formative assessment",
                    "Peer learning", "Real-world examples", "Metacognitive strategies"
                ],
                key=f"techniques_{selected_example_idx}"
            )
            
            explanation = st.text_area(
                "Explain why your approach is effective:",
                height=100,
                key=f"explanation_{selected_example_idx}"
            )
        
        # Submit feedback
        if st.button("Submit Feedback", key=f"submit_{selected_example_idx}"):
            if not sme_response:
                st.error("Please provide your expert response before submitting.")
                return
                
            # Save feedback data
            feedback = {
                "timestamp": self._get_timestamp(),
                "example_id": selected_example_idx,
                "subject": selected_example['subject'],
                "grade_level": selected_example['grade_level'],
                "question": selected_example['question'],
                "context": selected_example.get('context', ''),
                "ai_response": ai_response,
                "ai_rating": quality_score,
                "sme_response": sme_response,
                "teaching_techniques": teaching_techniques,
                "explanation": explanation
            }
            
            self._save_sme_feedback(feedback)
            self._update_expert_points("review", feedback)
            
            st.success("Thank you for your valuable feedback! Your expertise helps improve our teaching assistant.")
            
    def _create_sample_teaching_scenarios(self):
        """Create sample teaching scenarios from various subjects and grade levels."""
        # Check if we already have cached scenarios
        if hasattr(self, '_cached_scenarios') and self._cached_scenarios:
            return self._cached_scenarios
            
        # Create sample scenarios
        scenarios = [
            {
                "subject": "Mathematics",
                "grade_level": "3rd Grade",
                "question": "I don't understand why I need to regroup numbers when I subtract. Can you explain it?",
                "context": "Working on three-digit subtraction problems",
                "ai_response": "Regrouping in subtraction is like borrowing. When you need to subtract a bigger number from a smaller one in a place value, you 'borrow' from the next higher place value. For example, in 43-25, you can't subtract 5 from 3, so you borrow 1 from the 4 (which is actually 10), making it 3, and the 3 becomes 13. Now you can subtract 5 from 13 to get 8, and then 2 from 3 to get 1, so 43-25=18. Would you like me to show you with a visual example?"
            },
            {
                "subject": "Science",
                "grade_level": "5th Grade",
                "question": "How do plants make their own food? My friend said they eat dirt!",
                "context": "Learning about photosynthesis",
                "ai_response": "Plants don't eat dirt - they actually make their own food through a process called photosynthesis! They take in water through their roots, carbon dioxide from the air through tiny holes in their leaves, and energy from sunlight. Using these ingredients, they create glucose (sugar) and oxygen. The sugar is their food that gives them energy to grow, and they release the oxygen into the air for us to breathe. The soil provides important minerals and nutrients, but it's not their 'food.' Would you like to try a simple experiment to see photosynthesis in action?"
            },
            {
                "subject": "Literacy",
                "grade_level": "2nd Grade",
                "question": "Why do some words sound the same but are spelled differently?",
                "context": "Encountering homophones in a reading passage",
                "ai_response": "Great question! Words that sound the same but are spelled differently are called homophones. They can be tricky! For example, 'bear' (the animal) and 'bare' (uncovered) sound exactly the same, but they have completely different meanings. English has many homophones because our language has borrowed words from different languages throughout history. When we read, the spelling helps us know which meaning is being used. Shall we make a fun game to practice some common homophones like 'their/there/they're' or 'to/too/two'?"
            },
            {
                "subject": "Social Studies",
                "grade_level": "4th Grade",
                "question": "Why did people want to go west in the olden days? Was it dangerous?",
                "context": "Studying westward expansion in American history",
                "ai_response": "People moved west in the 1800s for several exciting reasons! Many wanted free or cheap land for farming through the Homestead Act. Others were searching for gold and silver after discoveries like the California Gold Rush. Some were seeking religious freedom or a fresh start in life. Yes, it was definitely dangerous - pioneers faced rough terrain, harsh weather, limited supplies, potential conflicts with Native American tribes protecting their lands, and disease. They typically traveled in wagon trains for safety and used routes like the Oregon Trail. What aspects of westward expansion do you find most interesting?"
            }
        ]
        
        # Add AI-generated recommendations from knowledge base
        for scenario in scenarios:
            if 'ai_response' not in scenario:
                # In a real implementation, this would call the knowledge base
                scenario['ai_response'] = "This would be an AI-generated response based on our knowledge base."
        
        # Cache the scenarios
        self._cached_scenarios = scenarios
        return scenarios
        
    def _save_sme_feedback(self, feedback):
        """Save SME feedback to disk."""
        # Create feedback directory if it doesn't exist
        feedback_dir = os.path.join(os.path.dirname(__file__), "../../sme_feedback")
        os.makedirs(feedback_dir, exist_ok=True)
        
        # Generate filename with timestamp
        filename = f"sme_feedback_{self._get_timestamp(for_filename=True)}.json"
        filepath = os.path.join(feedback_dir, filename)
        
        # Save the feedback as JSON
        with open(filepath, 'w') as f:
            json.dump(feedback, f, indent=2)
            
        # Update session state to include the new feedback
        if "sme_feedback" not in st.session_state:
            st.session_state.sme_feedback = []
            
        st.session_state.sme_feedback.append(feedback)
        logging.info(f"SME feedback saved to {filepath}")
        
    def _show_simplified_collected_data(self):
        """Display collected SME feedback in a simplified format."""
        st.subheader("Collected Expert Feedback")
        
        # Initialize or get feedback data
        if "sme_feedback" not in st.session_state:
            st.session_state.sme_feedback = []
            
            # Look for existing feedback files
            feedback_dir = os.path.join(os.path.dirname(__file__), "../../sme_feedback")
            if os.path.exists(feedback_dir):
                feedback_files = [f for f in os.listdir(feedback_dir) if f.endswith('.json')]
                
                for file in feedback_files:
                    try:
                        with open(os.path.join(feedback_dir, file), 'r') as f:
                            st.session_state.sme_feedback.append(json.load(f))
                    except Exception as e:
                        logging.error(f"Error loading feedback file {file}: {str(e)}")
        
        # Display feedback statistics
        total_feedback = len(st.session_state.sme_feedback)
        
        if total_feedback == 0:
            st.info("No feedback data collected yet. Review teaching examples or submit your own solutions to contribute.")
            return
            
        st.write(f"**Total contributions:** {total_feedback}")
        
        # Subject area breakdown
        if total_feedback > 0:
            subjects = {}
            for feedback in st.session_state.sme_feedback:
                subject = feedback.get('subject', 'Unknown')
                subjects[subject] = subjects.get(subject, 0) + 1
                
            st.write("**Contributions by subject area:**")
            for subject, count in subjects.items():
                st.write(f"- {subject}: {count}")
                
        # Export data option
        if st.button("Export All Feedback Data"):
            export_data = json.dumps(st.session_state.sme_feedback, indent=2)
            st.download_button(
                label="Download JSON",
                data=export_data,
                file_name=f"sme_feedback_export_{self._get_timestamp(for_filename=True)}.json",
                mime="application/json"
            )
            
        # Browse feedback entries
        st.subheader("Browse Feedback Entries")
        
        for i, feedback in enumerate(st.session_state.sme_feedback):
            with st.expander(f"Feedback #{i+1}: {feedback.get('subject', 'Unknown')} - {feedback.get('timestamp', 'No date')}"):
                st.write(f"**Subject:** {feedback.get('subject', 'Unknown')}")
                st.write(f"**Grade Level:** {feedback.get('grade_level', 'Unknown')}")
                st.write(f"**Question:** {feedback.get('question', 'No question')}")
                
                st.markdown("**AI Response:**")
                st.markdown(f"> {feedback.get('ai_response', 'No AI response')}")
                st.write(f"**AI Rating:** {feedback.get('ai_rating', 'Not rated')}/10")
                
                st.markdown("**Expert Response:**")
                st.markdown(f"> {feedback.get('sme_response', 'No expert response')}")
                
                if 'teaching_techniques' in feedback and feedback['teaching_techniques']:
                    st.write("**Teaching Techniques Used:**")
                    for technique in feedback['teaching_techniques']:
                        st.write(f"- {technique}")
                        
                if 'explanation' in feedback and feedback['explanation']:
                    st.markdown("**Expert Explanation:**")
                    st.markdown(f"> {feedback['explanation']}")
                    
    def _show_teaching_examples_input(self):
        """Simplified interface for SMEs to submit their own teaching examples."""
        st.subheader("Submit Your Teaching Examples")
        st.markdown("Share your expertise by providing examples of how you would respond to student questions.")
        
        # Form for submitting examples
        with st.form(key="teaching_example_form"):
            # Basic metadata
            subject = st.selectbox(
                "Subject Area:",
                options=["Mathematics", "Science", "Literacy", "Social Studies", "Art", "Music", "Physical Education", "Other"]
            )
            
            grade_level = st.selectbox(
                "Grade Level:",
                options=["Kindergarten", "1st Grade", "2nd Grade", "3rd Grade", "4th Grade", "5th Grade"]
            )
            
            # Student question
            student_question = st.text_area(
                "Student Question:",
                placeholder="What question did the student ask?",
                height=100
            )
            
            # Optional context
            context = st.text_area(
                "Context (Optional):",
                placeholder="Any relevant context about the student or classroom situation",
                height=75
            )
            
            # Expert response
            expert_response = st.text_area(
                "Your Expert Response:",
                placeholder="How would you respond to this student?",
                height=200
            )
            
            # Teaching techniques used
            teaching_techniques = st.multiselect(
                "Teaching Techniques Used:",
                options=[
                    "Scaffolding", "Inquiry-based learning", "Direct instruction",
                    "Visual learning", "Differentiated instruction", "Formative assessment",
                    "Peer learning", "Real-world examples", "Metacognitive strategies"
                ]
            )
            
            # Explanation
            explanation = st.text_area(
                "Explain Your Approach:",
                placeholder="Why is this an effective response? What makes it work well?",
                height=150
            )
            
            # Submit button
            submit_button = st.form_submit_button("Submit Example")
        
        if submit_button:
            # Validate inputs
            if not student_question or not expert_response:
                st.error("Please provide both a student question and your expert response.")
                return
                
            # Create example object
            example = {
                "timestamp": self._get_timestamp(),
                "subject": subject,
                "grade_level": grade_level,
                "question": student_question,
                "context": context,
                "sme_response": expert_response,
                "teaching_techniques": teaching_techniques,
                "explanation": explanation
            }
            
            # Save the example
            self._save_teaching_example(example)
            self._update_expert_points("example", example)
            
            st.success("Thank you for sharing your teaching example! Your expertise is greatly appreciated.")
            
    def _save_teaching_example(self, example):
        """Save a teaching example to disk."""
        # Create examples directory if it doesn't exist
        examples_dir = os.path.join(os.path.dirname(__file__), "../../teaching_examples")
        os.makedirs(examples_dir, exist_ok=True)
        
        # Generate filename with timestamp
        filename = f"teaching_example_{self._get_timestamp(for_filename=True)}.json"
        filepath = os.path.join(examples_dir, filename)
        
        # Save the example as JSON
        with open(filepath, 'w') as f:
            json.dump(example, f, indent=2)
            
        # Update session state to include the new example
        if "teaching_examples" not in st.session_state:
            st.session_state.teaching_examples = []
            
        st.session_state.teaching_examples.append(example)
        logging.info(f"Teaching example saved to {filepath}")
        
    def _get_timestamp(self, for_filename=False):
        """Get a formatted timestamp for the current time."""
        now = datetime.now()
        if for_filename:
            return now.strftime("%Y-%m-%d_%H-%M-%S")
        return now.strftime("%Y-%m-%d %H:%M:%S")
        
    def setup_page(self):
        """Set up the page with header and description."""
        # Header with logo and title side by side
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.image("logo.png", width=150)
            
        with col2:
            st.title("Utah Teacher Training Assistant")
            st.markdown("""
            ### Subject Matter Expert Platform
            Share your teaching expertise to help improve AI-assisted instruction for K-5 education.
            """)
            
        # Display current app version and UVU styling based on color palette
        st.markdown("""
        <style>
            :root {
                --uvu-green: #275D38;
                --uvu-black: #000000;
                --uvu-white: #FFFFFF;
                --uvu-light-green: #E9F2ED;
            }
            
            .stApp {
                background-color: var(--uvu-white);
            }
            
            h1, h2, h3 {
                color: var(--uvu-green);
            }
            
            .stButton button {
                background-color: var(--uvu-green);
                color: var(--uvu-white);
            }
            
            .info-box {
                background-color: var(--uvu-light-green);
                border-left: 5px solid var(--uvu-green);
                padding: 10px;
                margin-bottom: 10px;
            }
        </style>
        <div style="text-align: right; font-size: 0.8em; color: gray;">Version 3.0</div>
        """, unsafe_allow_html=True)
        
    def init_session_state(self):
        """Initialize the Streamlit session state with simplified variables."""
        # Initialize points system
        if "expert_points" not in st.session_state:
            st.session_state.expert_points = 0
            
        if "expert_level" not in st.session_state:
            st.session_state.expert_level = "Novice Teacher"
            
        # Initialize feedback and example collections
        if "sme_feedback" not in st.session_state:
            st.session_state.sme_feedback = []
            
        if "teaching_examples" not in st.session_state:
            st.session_state.teaching_examples = []
            
        # Initialize announcement banner state
        if "show_announcement" not in st.session_state:
            st.session_state.show_announcement = True
            
        # Initialize streak and badges system
        if "streak_days" not in st.session_state:
            st.session_state.streak_days = 0
            
        if "last_contribution_date" not in st.session_state:
            st.session_state.last_contribution_date = datetime.now().strftime("%Y-%m-%d")
            
        if "badges" not in st.session_state:
            st.session_state.badges = []
            
        if "awards" not in st.session_state:
            st.session_state.awards = {
                "literacy_star": False,
                "math_wizard": False,
                "science_explorer": False,
                "feedback_champion": False
            }
            
        if "expert_reviews" not in st.session_state:
            st.session_state.expert_reviews = []
            
        if "expert_examples" not in st.session_state:
            st.session_state.expert_examples = []

    def _update_expert_points(self, action_type, content=None):
        """
        Update expert points based on contributions and check for level-ups.
        
        Args:
            action_type (str): Type of action performed (rating, review, streak)
            content (dict, optional): Additional content for context-specific rewards
        """
        points_earned = 0
        level_up = False
        new_badges = []
        
        # Points for different actions
        if action_type == "quick_rate":
            points_earned = 10
            
            # Bonus points for comprehensive feedback
            if content and "techniques_used" in content and len(content["techniques_used"]) >= 3:
                points_earned += 5
                
            # Subject-specific bonuses
            if content and "subject_area" in content:
                subject = content["subject_area"].lower()
                if "literacy" in subject or "language" in subject:
                    if not st.session_state.awards["literacy_star"] and \
                       len([ex for ex in st.session_state.expert_examples 
                            if "subject_area" in ex and ("literacy" in ex["subject_area"].lower() 
                                                        or "language" in ex["subject_area"].lower())]) >= 3:
                        st.session_state.awards["literacy_star"] = True
                        points_earned += 25
                        new_badges.append("🌟 Literacy Star")
                
                elif "math" in subject:
                    if not st.session_state.awards["math_wizard"] and \
                       len([ex for ex in st.session_state.expert_examples 
                            if "subject_area" in ex and "math" in ex["subject_area"].lower()]) >= 3:
                        st.session_state.awards["math_wizard"] = True
                        points_earned += 25
                        new_badges.append("🧙 Math Wizard")
                        
                elif "science" in subject:
                    if not st.session_state.awards["science_explorer"] and \
                       len([ex for ex in st.session_state.expert_examples 
                            if "subject_area" in ex and "science" in ex["subject_area"].lower()]) >= 3:
                        st.session_state.awards["science_explorer"] = True
                        points_earned += 25
                        new_badges.append("🔬 Science Explorer")
            
        elif action_type == "response_review":
            points_earned = 15
            
            # Bonus for detailed reasons
            if content and "reasons" in content and len(content["reasons"]) >= 3:
                points_earned += 8
                
            if len(st.session_state.expert_reviews) >= 5 and not st.session_state.awards["feedback_champion"]:
                st.session_state.awards["feedback_champion"] = True
                points_earned += 30
                new_badges.append("🏆 Feedback Champion")
                
        elif action_type == "streak":
            # Check if it's a new day compared to last contribution
            today = datetime.now().strftime("%Y-%m-%d")
            if today != st.session_state.last_contribution_date:
                st.session_state.streak_days += 1
                st.session_state.last_contribution_date = today
                
                # Bonus points for streaks
                if st.session_state.streak_days >= 3:
                    points_earned = st.session_state.streak_days * 5
                    new_badges.append(f"🔥 {st.session_state.streak_days}-Day Streak")
        
        # Add points
        st.session_state.expert_points += points_earned
        
        # Add any new badges to collection
        if new_badges:
            for badge in new_badges:
                if badge not in st.session_state.badges:
                    st.session_state.badges.append(badge)
        
        # Check for level-ups
        current_level = st.session_state.expert_level
        new_level = current_level
        
        # Define levels
        levels = {
            "Novice Teacher": 0,
            "Assistant Teacher": 50,
            "Lead Teacher": 100,
            "Master Teacher": 200,
            "Distinguished Educator": 350,
            "Legendary Mentor": 500
        }
        
        # Check if user has leveled up
        for level, threshold in levels.items():
            if st.session_state.expert_points >= threshold:
                new_level = level
                
        # If there was a level up
        if new_level != current_level:
            st.session_state.expert_level = new_level
            level_up = True
            
        return {
            "points_earned": points_earned,
            "level_up": level_up,
            "new_level": new_level if level_up else None,
            "new_badges": new_badges,
            "total_points": st.session_state.expert_points
        }

    def _show_teaching_examples_input(self):
        """Show the interface for experts to provide teaching examples."""
        # Display current points and level
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader(f"Your Level: {st.session_state.expert_level}")
            st.write(f"Points: {st.session_state.expert_points} points")
            if st.session_state.streak_days > 1:
                st.write(f"🔥 Current streak: {st.session_state.streak_days} days")
        
        with col2:
            if st.session_state.badges:
                st.subheader("Your Badges:")
                badges_text = ", ".join(st.session_state.badges)
                st.write(badges_text)
        
        st.markdown("---")
        
        # Input form for teaching examples
        with st.form("teaching_example_form"):
            st.subheader("Share Your Teaching Example")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Allow selection of scenario/context
                scenario_options = [s["title"] for s in self.scenarios]
                selected_scenario = st.selectbox(
                    "Select Teaching Context",
                    options=scenario_options,
                    index=0,
                    key="teaching_input_scenario_selectbox"
                )
                
                # Subject area selection
                subject_area = st.selectbox(
                    "Subject Area",
                    options=["Math", "Literacy/Language Arts", "Science", "Social Studies", "Art/Music", "Physical Education", "Social-Emotional Learning"],
                    index=0,
                    key="teaching_input_subject_selectbox"
                )
                
                # Grade level selection
                grade_level = st.selectbox(
                    "Grade Level",
                    options=["Kindergarten", "1st Grade", "2nd Grade", "3rd Grade", "4th Grade", "5th Grade"],
                    index=2,
                    key="teaching_input_grade_selectbox"
                )
                
            with col2:
                # Student profile selection
                student_options = [s["name"] for s in self.students]
                selected_student = st.selectbox(
                    "Select Student Profile",
                    options=student_options,
                    index=0,
                    key="teaching_input_student_selectbox"
                )
                
                # Difficulty level
                difficulty = st.select_slider(
                    "Challenge Level",
                    options=["Easy", "Medium", "Challenging", "Very Challenging"],
                    value="Medium"
                )
                
                # Learning goal
                learning_goal = st.text_input("Learning Goal/Objective")
            
            # Student question input
            student_question = st.text_area(
                "Student Question or Statement",
                placeholder="Example: 'Teacher, I don't understand how to add fractions...'",
                height=100
            )
            
            # Expert response
            expert_response = st.text_area(
                "Your Expert Response",
                placeholder="Enter your response as an elementary teacher...",
                height=150
            )
            
            # Multi-select for teaching techniques
            techniques_used = st.multiselect(
                "Teaching Techniques Used (select all that apply)",
                options=[
                    "Visual aids/demonstrations", 
                    "Scaffolding", 
                    "Growth mindset encouragement",
                    "Real-world connections",
                    "Personalized feedback",
                    "Breaking down complex concepts",
                    "Think-aloud modeling",
                    "Positive reinforcement",
                    "Questioning strategies",
                    "Hands-on learning",
                    "Student-led discovery",
                    "Collaborative learning"
                ]
            )
            
            # Teaching approach explanation
            approach_explanation = st.text_area(
                "Briefly explain your teaching approach (optional)",
                placeholder="Why did you respond this way? What makes this effective for elementary students?",
                height=100
            )
            
            # Submit button
            submitted = st.form_submit_button("Submit Teaching Example")
            
            if submitted:
                if not student_question or not expert_response:
                    st.error("Please provide both a student question and your expert response.")
                else:
                    # Prepare example data
                    example_data = {
                        "scenario": selected_scenario,
                        "subject_area": subject_area,
                        "grade_level": grade_level,
                        "student_profile": selected_student,
                        "difficulty": difficulty,
                        "learning_goal": learning_goal,
                        "student_question": student_question,
                        "expert_response": expert_response,
                        "techniques_used": techniques_used,
                        "approach_explanation": approach_explanation,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    # Store the example
                    if "expert_examples" not in st.session_state:
                        st.session_state.expert_examples = []
                    
                    st.session_state.expert_examples.append(example_data)
                    
                    # Award points and check for level-up
                    reward_info = self._update_expert_points("quick_rate", example_data)
                    
                    # Show success message with points
                    st.success(f"✅ Example submitted successfully! You earned {reward_info['points_earned']} points.")
                    
                    # Show any new badges
                    if reward_info['new_badges']:
                        badges_text = ", ".join(reward_info['new_badges'])
                        st.balloons()
                        st.success(f"🎉 You earned new badges: {badges_text}")
                    
                    # Show level up notification
                    if reward_info['level_up']:
                        st.balloons()
                        st.success(f"🌟 LEVEL UP! You are now a {reward_info['new_level']}!")

    def _create_sample_responses(self):
        """Create sample AI responses for review."""
        return [
            {
                "subject": "Math",
                "grade": "2nd Grade",
                "student_profile": "Visual learner with attention challenges",
                "student_question": "Why do I need to learn how to subtract? I don't get it.",
                "ai_response": "Subtraction is important because we use it every day. For example, if you have 5 cookies and eat 2, you have 3 left. This is subtraction. It's used for calculating change, figuring out how much time is left, and many other everyday activities. Let's practice with some simple problems."
            },
            {
                "subject": "Literacy",
                "grade": "1st Grade",
                "student_profile": "English language learner",
                "student_question": "I can't remember the sounds that letters make.",
                "ai_response": "Learning letter sounds can be challenging. Let's look at each letter and say its sound together. We can create a chart with pictures for each sound. For example, 'A' makes the 'ah' sound like in 'apple'. Let's practice with a few letters at a time."
            },
            {
                "subject": "Science",
                "grade": "3rd Grade",
                "student_profile": "Gifted student who gets bored easily",
                "student_question": "How do plants drink water?",
                "ai_response": "Plants drink water through their roots. The water travels up through tiny tubes inside the stem to reach the leaves. It's similar to how you drink through a straw. We can do an experiment with celery and food coloring to actually see this process happening! Would you like to try that?"
            },
            {
                "subject": "Social Studies",
                "grade": "4th Grade",
                "student_profile": "Student with anxiety about speaking in groups",
                "student_question": "Why do we need to learn about maps?",
                "ai_response": "Maps help us understand where places are in relation to each other. They're like pictures of the Earth from above. We use maps to find our way around towns, parks, and even inside big buildings like malls. Maps also help us learn about different cultures and environments around the world."
            },
            {
                "subject": "Math",
                "grade": "Kindergarten",
                "student_profile": "Active learner who struggles to sit still",
                "student_question": "I don't understand counting by twos.",
                "ai_response": "Counting by twos means we skip every other number. We can practice by jumping while we count - jump once and say '2', jump again and say '4'. Let's try counting toys or blocks by putting them in pairs and counting each pair as one group of two."
            },
            {
                "subject": "Art",
                "grade": "5th Grade",
                "student_profile": "Creative student with perfectionist tendencies",
                "student_question": "I can't draw people right. They always look wrong.",
                "ai_response": "Drawing people is challenging even for professional artists. Let's break it down into simple shapes first - ovals for heads, rectangles for bodies, and lines for limbs. Remember that practice helps us improve, and even famous artists had to learn step by step."
            },
            {
                "subject": "Social-Emotional Learning",
                "grade": "3rd Grade",
                "student_profile": "Student who struggles with emotional regulation",
                "student_question": "What do I do when I get really mad at school?",
                "ai_response": "It's normal to feel angry sometimes. When you feel mad, you can take deep breaths, count to 10, or ask for a quick break. We can make a 'calm down plan' with steps that help you feel better when you're upset. Would you like to create your own plan?"
            },
            {
                "subject": "Physical Education",
                "grade": "1st Grade",
                "student_profile": "Student with coordination difficulties",
                "student_question": "I can't catch the ball. Everyone laughs at me.",
                "ai_response": "Learning to catch takes practice. Let's start with a bigger, softer ball that moves more slowly. We can practice with just you and me first, before joining the group. Everyone learns at their own pace, and I'm here to help you improve step by step."
            },
            {
                "subject": "Science",
                "grade": "4th Grade",
                "student_profile": "Curious student with many questions",
                "student_question": "Why does the moon change shape?",
                "ai_response": "The moon appears to change shape because of how sunlight hits it as it orbits Earth. The moon is always round, but we only see the parts that are lit by the sun. We can demonstrate this with a ball and a flashlight to show how the light creates different phases."
            },
            {
                "subject": "Literacy",
                "grade": "2nd Grade",
                "student_profile": "Student with dyslexia",
                "student_question": "Reading is too hard. The letters get mixed up.",
                "ai_response": "Many students find reading challenging. We can use special tools like a reading guide to help focus on one line at a time. We'll also practice with games that strengthen your letter recognition. Remember that your brain is learning and getting stronger with each practice session."
            }
        ]

# Run the application
if __name__ == "__main__":
    web_interface = WebInterface()
    web_interface.run()
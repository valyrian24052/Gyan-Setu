import streamlit as st
import os
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

class SimpleSMEInterface:
    """A simplified interface for collecting feedback from Subject Matter Experts (SMEs)."""
    
    def __init__(self):
        """Initialize the application."""
        self.run()
        
    def init_session_state(self):
        """Initialize the Streamlit session state."""
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
            
        # Initialize scenario builder state
        if "current_scenario" not in st.session_state:
            st.session_state.current_scenario = {
                "grade_level": "",
                "student_profile": "",
                "subject": "",
                "challenge_type": "",
                "teaching_style": "",
                "scenario_details": "",
                "question": ""
            }
            
        if "show_kb_solutions" not in st.session_state:
            st.session_state.show_kb_solutions = False
            
        if "approach_feedback" not in st.session_state:
            st.session_state.approach_feedback = []
            
        # Initialize announcement banner state
        if "show_announcement" not in st.session_state:
            st.session_state.show_announcement = True
    
    def setup_page(self):
        """Set up the page with header and description."""
        # Header with logo and title side by side
        col1, col2 = st.columns([1, 3])
        
        with col1:
            try:
                st.image("logo.png", width=150)
            except:
                st.info("Logo not found. Please add logo.png to the project root.")
            
        with col2:
            st.title("Utah Teaching Scenario Builder")
            st.markdown("""
            ### Create, Evaluate & Compare Teaching Approaches
            Design teaching scenarios and explore different instructional strategies for K-5 education.
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
            
            /* Points display */
            .points-display {
                background-color: var(--uvu-light-green);
                border-radius: 5px;
                padding: 10px;
                margin-bottom: 15px;
            }
        </style>
        <div style="text-align: right; font-size: 0.8em; color: gray;">Version 3.0</div>
        """, unsafe_allow_html=True)
        
        # Display points and level if points exist
        if st.session_state.expert_points > 0:
            st.markdown(f"""
            <div class="points-display">
                <h3 style="margin: 0; color: #275D38; font-size: 1.2rem;">{st.session_state.expert_level}</h3>
                <p style="margin: 0; font-weight: bold;">{st.session_state.expert_points} points</p>
                <div style="margin-top: 5px; height: 5px; background-color: #D1D5DB; border-radius: 2px;">
                    <div style="height: 100%; width: {min(100, st.session_state.expert_points/5)}%; background-color: #275D38; border-radius: 2px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Optional announcement banner
        if st.session_state.get("show_announcement", True):
            with st.container():
                cols = st.columns([6, 1])
                with cols[0]:
                    st.info("🎮 NEW! Teaching Assistant is now gamified - earn points and level up as you contribute!")
                with cols[1]:
                    if st.button("×", key="close_announcement"):
                        st.session_state.show_announcement = False
                        st.rerun()
    
    def run(self):
        """Main method to run the Streamlit application."""
        # Initialize session state and set up page
        self.init_session_state()
        self.setup_page()
        
        # Single page application focused on SME feedback
        self.show_sme_feedback_interface()
        
    def show_sme_feedback_interface(self):
        """Main interface focused on getting feedback from Subject Matter Experts."""
        st.markdown("## Teaching Scenario Creation & Evaluation Platform")
        st.markdown("Create custom teaching scenarios, compare different approaches, and share your expertise to improve AI-assisted instruction.")
        
        # Create tabs for different feedback activities with "Create Scenarios" first
        tabs = st.tabs(["Create Scenarios", "Review Teaching Examples", "Submit Your Solutions", "View Collected Data"])
        
        with tabs[0]:
            self._show_scenario_builder()
            
        with tabs[1]:
            self._show_example_review()
            
        with tabs[2]:
            self._show_teaching_examples_input()
            
        with tabs[3]:
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
            format_func=lambda i: f"Example {i+1}: {examples[i]['subject']} - {examples[i]['question'][:50]}..."
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
        
        # Compare your solution with AI recommendations
        if st.checkbox("Compare with AI Recommendations"):
            st.info("When you submit your example, we'll show you what our AI would recommend and highlight the differences.")
        
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
            
            # Compare with AI (in a real implementation this would call the AI model)
            if st.session_state.get("compare_with_ai", False):
                st.subheader("Comparison with AI Recommendation")
                st.markdown("**Your Expert Response:**")
                st.markdown(f"> {expert_response}")
                
                st.markdown("**AI Recommendation:**")
                ai_response = "This would be an AI-generated response based on our knowledge base."
                st.markdown(f"> {ai_response}")
                
                st.markdown("**Key Differences:**")
                st.markdown("- The AI would highlight differences here")
                st.markdown("- For example, your use of real-world examples vs. the AI's theoretical approach")
                
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
        
    def _update_expert_points(self, action_type, content=None):
        """Update expert points based on actions and check for level-ups."""
        # Points for different actions
        points_map = {
            "review": 10,     # Reviewing and rating an AI response
            "example": 20,    # Submitting a new teaching example
            "scenario": 25,   # Evaluating a teaching scenario
        }
        
        # Award bonus points for detailed responses
        bonus_points = 0
        if content:
            if action_type == "example":
                # Bonus for longer, more detailed responses
                response_length = len(content.get("sme_response", ""))
                if response_length > 500:
                    bonus_points += 5
                    
                # Bonus for providing techniques and explanation
                if content.get("teaching_techniques", []) and len(content.get("teaching_techniques", [])) > 2:
                    bonus_points += 3
                    
                if content.get("explanation", "") and len(content.get("explanation", "")) > 200:
                    bonus_points += 5
            
            elif action_type == "scenario":
                # Bonus for thorough evaluation
                strengths_length = len(content.get("strengths", ""))
                weaknesses_length = len(content.get("weaknesses", ""))
                modifications_length = len(content.get("modifications", ""))
                
                if strengths_length > 200 and weaknesses_length > 200:
                    bonus_points += 5
                    
                if modifications_length > 300:
                    bonus_points += 10
                    
                # Extra bonus for high-quality evaluations
                if strengths_length > 100 and weaknesses_length > 100 and modifications_length > 200:
                    bonus_points += 5
                
        # Add points
        base_points = points_map.get(action_type, 0)
        total_points = base_points + bonus_points
        st.session_state.expert_points += total_points
        
        # Check for level-ups
        self._check_level_up()
        
        # Notify user of points earned
        if total_points > 0:
            if bonus_points > 0:
                st.sidebar.success(f"You earned {base_points} points + {bonus_points} bonus points!")
            else:
                st.sidebar.success(f"You earned {total_points} points!")
                
    def _check_level_up(self):
        """Check if the user has earned enough points to level up."""
        # Level thresholds
        level_thresholds = {
            "Novice Teacher": 0,
            "Developing Teacher": 50,
            "Proficient Teacher": 150,
            "Expert Teacher": 300,
            "Master Teacher": 600
        }
        
        # Determine current level based on points
        current_level = "Novice Teacher"
        for level, threshold in sorted(level_thresholds.items(), key=lambda x: x[1], reverse=True):
            if st.session_state.expert_points >= threshold:
                if st.session_state.expert_level != level:
                    # Level up occurred!
                    st.session_state.expert_level = level
                    st.balloons()
                    st.sidebar.success(f"🎉 Congratulations! You've reached the '{level}' level!")
                break
                
    def _get_timestamp(self, for_filename=False):
        """Get a formatted timestamp for the current time."""
        now = datetime.now()
        if for_filename:
            return now.strftime("%Y-%m-%d_%H-%M-%S")
        return now.strftime("%Y-%m-%d %H:%M:%S")

    def _show_scenario_builder(self):
        """Create teaching scenarios with hierarchical option selection and knowledge base suggestions."""
        st.subheader("Scenario Builder")
        st.markdown("Create custom teaching scenarios and evaluate different approaches from the knowledge base.")
        
        # Initialize scenario state if needed
        if "current_scenario" not in st.session_state:
            st.session_state.current_scenario = {
                "grade_level": "",
                "student_profile": "",
                "subject": "",
                "challenge_type": "",
                "teaching_style": "",
                "scenario_details": "",
                "question": ""
            }
            
        if "show_kb_solutions" not in st.session_state:
            st.session_state.show_kb_solutions = False
        
        # Hierarchical selection process
        col1, col2 = st.columns(2)
        
        with col1:
            # Step 1: Grade Level
            grade_level = st.selectbox(
                "Grade Level:",
                options=["", "Kindergarten", "1st Grade", "2nd Grade", "3rd Grade", "4th Grade", "5th Grade"],
                index=0,
                key="scenario_grade"
            )
            
            if grade_level:
                st.session_state.current_scenario["grade_level"] = grade_level
                
                # Step 2: Subject Area
                subject = st.selectbox(
                    "Subject Area:",
                    options=["", "Mathematics", "Science", "Literacy", "Social Studies", "Art", "Music", "Physical Education"],
                    index=0,
                    key="scenario_subject"
                )
                
                if subject:
                    st.session_state.current_scenario["subject"] = subject
                    
                    # Step 3: Challenge Type
                    challenge_types = self._get_challenges_for_subject(subject)
                    
                    challenge_type = st.selectbox(
                        "Challenge Type:",
                        options=[""] + challenge_types,
                        index=0,
                        key="scenario_challenge"
                    )
                    
                    if challenge_type:
                        st.session_state.current_scenario["challenge_type"] = challenge_type
        
        with col2:
            # Step 4: Student Profile
            if grade_level:
                student_profiles = self._get_student_profiles_for_grade(grade_level)
                
                student_profile = st.selectbox(
                    "Student Profile:",
                    options=[""] + student_profiles,
                    index=0,
                    key="scenario_student"
                )
                
                if student_profile:
                    st.session_state.current_scenario["student_profile"] = student_profile
                    
                    # Step 5: Teaching Style
                    teaching_style = st.selectbox(
                        "Teaching Style:",
                        options=["", "Direct Instruction", "Inquiry-Based", "Collaborative Learning", 
                                "Differentiated", "Project-Based", "Technology-Enhanced"],
                        index=0,
                        key="scenario_style"
                    )
                    
                    if teaching_style:
                        st.session_state.current_scenario["teaching_style"] = teaching_style
        
        # Only show these options if we have enough context
        if (st.session_state.current_scenario["grade_level"] and 
            st.session_state.current_scenario["subject"] and 
            st.session_state.current_scenario["challenge_type"]):
            
            # Step 6: Scenario Details
            st.markdown("### Scenario Details")
            
            scenario_details = st.text_area(
                "Describe the classroom context:",
                value=st.session_state.current_scenario.get("scenario_details", ""),
                placeholder="Example: Students are working in small groups on a science experiment about plant growth...",
                height=100,
                key="scenario_details"
            )
            
            st.session_state.current_scenario["scenario_details"] = scenario_details
            
            # Step 7: Student Question
            student_question = st.text_area(
                "Student Question or Challenge:",
                value=st.session_state.current_scenario.get("question", ""),
                placeholder="What specific question or challenge does the student present?",
                height=75,
                key="scenario_question"
            )
            
            st.session_state.current_scenario["question"] = student_question
            
            # Generate scenario button
            if st.button("Generate Scenario Solutions", key="generate_solutions"):
                if not scenario_details or not student_question:
                    st.error("Please provide both scenario details and a student question.")
                else:
                    st.session_state.show_kb_solutions = True
                    st.success("Scenario created! Knowledge base solutions generated below.")
                    
            # Show knowledge base solutions if available
            if st.session_state.show_kb_solutions:
                self._show_knowledge_base_solutions()
                
    def _load_example_scenario(self, example_name):
        """Load a pre-defined example scenario."""
        examples = {
            "3rd Grade Math - Fractions Confusion": {
                "grade_level": "3rd Grade",
                "subject": "Mathematics",
                "challenge_type": "Fractions",
                "student_profile": "Struggling Learner",
                "teaching_style": "Visual and Hands-on Learning",
                "scenario_details": "Your students are learning about comparing fractions for the first time. You've introduced the concept using fraction strips, but some students are still confused. During independent practice time, a student calls you over with a frustrated expression.",
                "question": "Teacher, I don't get it. How can 1/4 be smaller than 1/3? Four is bigger than three, so 1/4 should be bigger!"
            },
            "2nd Grade Literacy - Homophones": {
                "grade_level": "2nd Grade",
                "subject": "Literacy",
                "challenge_type": "Vocabulary",
                "student_profile": "Typically Developing",
                "teaching_style": "Inquiry-Based",
                "scenario_details": "Your class is reading a story that contains several homophones. You've noticed that some students are confused by words that sound the same but have different meanings and spellings. During discussion, a student raises their hand.",
                "question": "Why does 'bear' mean the animal and also mean to carry something? And why does it sound exactly like 'bare'? English is so confusing!"
            },
            "5th Grade Science - Photosynthesis": {
                "grade_level": "5th Grade",
                "subject": "Science",
                "challenge_type": "Life Science",
                "student_profile": "Advanced Learner",
                "teaching_style": "Inquiry-Based",
                "scenario_details": "Your class is studying plant biology, specifically how plants make their own food through photosynthesis. You've just introduced the basic concept that plants use sunlight, water, and carbon dioxide to create glucose. An advanced student appears puzzled after the initial explanation.",
                "question": "If plants make sugar during photosynthesis, why don't they taste sweet? And if they release oxygen, where exactly does it come from? Is it made from scratch or converted from something else?"
            },
            "4th Grade Social Studies - Westward Expansion": {
                "grade_level": "4th Grade",
                "subject": "Social Studies",
                "challenge_type": "History",
                "student_profile": "English Language Learner",
                "teaching_style": "Real-world Application",
                "scenario_details": "Your class is learning about westward expansion in the United States during the 1800s. You've shown pictures of wagon trains and talked about the Oregon Trail. One of your English language learners seems interested but confused about the motivations of the settlers.",
                "question": "Teacher, why people leave homes to go west? Was very dangerous, yes? In my country, people stay with family. Why Americans go so far?"
            },
            "1st Grade Behavior - Sharing Materials": {
                "grade_level": "1st Grade",
                "subject": "Other",
                "challenge_type": "Social Skills",
                "student_profile": "Student with Emotional Needs",
                "teaching_style": "Differentiated",
                "scenario_details": "During center time, you notice a conflict brewing at the art table. Art supplies are limited, and students need to share. One student is becoming visibly upset and refusing to share the glue sticks. You approach to help resolve the situation.",
                "question": "I had it first and I'm not done! Why do I always have to share everything? It's not fair! I need all the glue sticks for my project!"
            }
        }
        
        if example_name in examples:
            st.session_state.current_scenario = examples[example_name]
    
    def _show_knowledge_base_solutions(self):
        """Show multiple solution approaches from the knowledge base for the current scenario."""
        st.markdown("## Knowledge Base Teaching Approaches")
        st.markdown("Below are different ways to approach this teaching scenario. As an SME, please evaluate which approach would be most effective.")
        
        # Create several different approaches based on the scenario components
        approaches = self._generate_teaching_approaches()
        
        # Display the approaches with selection option
        selected_approach = st.radio(
            "Which approach do you believe would be most effective?",
            options=range(len(approaches)),
            format_func=lambda i: f"Approach {i+1}: {approaches[i]['name']}",
            key="kb_approach_selection"
        )
        
        # Display the selected approach details
        st.markdown("### Selected Approach Details")
        selected = approaches[selected_approach]
        
        st.markdown(f"**Approach Name:** {selected['name']}")
        st.markdown(f"**Teaching Style:** {selected['style']}")
        st.markdown("**Description:**")
        st.markdown(f"> {selected['description']}")
        
        st.markdown("**Example Response:**")
        st.markdown(f"```\n{selected['example']}\n```")
        
        # Expert feedback on selected approach
        st.markdown("### Your Expert Feedback")
        
        effectiveness = st.slider(
            "Rate the effectiveness of this approach (1-10):",
            min_value=1,
            max_value=10,
            value=7,
            key="approach_effectiveness"
        )
        
        strengths = st.text_area(
            "What are the strengths of this approach?",
            height=100,
            key="approach_strengths"
        )
        
        weaknesses = st.text_area(
            "What could be improved in this approach?",
            height=100,
            key="approach_weaknesses"
        )
        
        modifications = st.text_area(
            "How would you modify this approach?",
            height=150,
            key="approach_modifications"
        )
        
        # Submit feedback
        if st.button("Submit Approach Feedback", key="submit_approach_feedback"):
            if not strengths or not modifications:
                st.error("Please provide both strengths and your suggested modifications.")
                return
                
            # Save feedback
            feedback = {
                "timestamp": self._get_timestamp(),
                "scenario": st.session_state.current_scenario,
                "selected_approach": selected,
                "approach_index": selected_approach,
                "effectiveness_rating": effectiveness,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "modifications": modifications
            }
            
            self._save_approach_feedback(feedback)
            self._update_expert_points("scenario", feedback)
            
            st.success("Thank you for your expert feedback on this teaching approach!")
    
    def _save_approach_feedback(self, feedback):
        """Save approach feedback to disk."""
        # Create feedback directory if it doesn't exist
        feedback_dir = os.path.join(os.path.dirname(__file__), "../../approach_feedback")
        os.makedirs(feedback_dir, exist_ok=True)
        
        # Generate filename with timestamp
        filename = f"approach_feedback_{self._get_timestamp(for_filename=True)}.json"
        filepath = os.path.join(feedback_dir, filename)
        
        # Save the feedback as JSON
        with open(filepath, 'w') as f:
            json.dump(feedback, f, indent=2)
            
        # Update session state to include the new feedback
        if "approach_feedback" not in st.session_state:
            st.session_state.approach_feedback = []
            
        st.session_state.approach_feedback.append(feedback)
        logging.info(f"Approach feedback saved to {filepath}")
    
    def _get_challenges_for_subject(self, subject):
        """Return appropriate challenges based on the selected subject."""
        challenge_map = {
            "Mathematics": [
                "Number Sense", "Operations", "Fractions", "Geometry", 
                "Measurement", "Problem Solving", "Mathematical Reasoning"
            ],
            "Science": [
                "Scientific Method", "Life Science", "Earth Science", "Physical Science",
                "Engineering Design", "Environmental Science", "Scientific Modeling"
            ],
            "Literacy": [
                "Phonics", "Reading Comprehension", "Vocabulary", "Writing", 
                "Grammar", "Listening Skills", "Speaking Skills"
            ],
            "Social Studies": [
                "History", "Geography", "Civics", "Economics",
                "Cultural Studies", "Current Events", "Community Studies"
            ],
            "Art": [
                "Visual Arts", "Art History", "Art Techniques", "Art Appreciation",
                "Creative Expression", "Design Principles", "Art Criticism"
            ],
            "Music": [
                "Music Theory", "Instrument Skills", "Singing", "Music Appreciation",
                "Music History", "Composition", "Performance"
            ],
            "Physical Education": [
                "Motor Skills", "Team Sports", "Individual Sports", "Fitness",
                "Health & Wellness", "Sportsmanship", "Movement Concepts"
            ]
        }
        
        return challenge_map.get(subject, ["Basic Concepts", "Advanced Concepts", "Application", "Problem Solving"])
    
    def _get_student_profiles_for_grade(self, grade_level):
        """Return student profiles appropriate for the selected grade level."""
        # Common profiles across all grade levels
        common_profiles = [
            "Typically Developing", 
            "Advanced Learner",
            "Struggling Learner", 
            "English Language Learner",
            "Student with ADHD", 
            "Student with Learning Disability",
            "Student with Emotional Needs",
            "Student with Autism Spectrum Disorder"
        ]
        
        # Grade-specific profiles
        grade_specific = {
            "Kindergarten": ["Early Reader", "Developing Fine Motor Skills", "Separation Anxiety", "Highly Active"],
            "1st Grade": ["Emerging Reader", "Developing Writer", "Numeracy Development", "Social Development"],
            "2nd Grade": ["Reading Fluency Development", "Basic Math Operations", "Scientific Curiosity", "Friendship Issues"],
            "3rd Grade": ["Reading Comprehension Challenges", "Math Concept Application", "Independent Worker", "Group Work Challenges"],
            "4th Grade": ["Research Skills Development", "Multi-Step Problems", "Subject-Specific Interests", "Test Anxiety"],
            "5th Grade": ["Pre-Adolescent Social Issues", "Abstract Thinking Development", "School Transition Anxiety", "Leadership Skills"]
        }
        
        # Combine common profiles with grade-specific ones
        specific_profiles = grade_specific.get(grade_level, [])
        return common_profiles + specific_profiles
    
    def _generate_teaching_approaches(self):
        """Generate different teaching approaches based on the scenario details."""
        scenario = st.session_state.current_scenario
        subject = scenario["subject"]
        grade_level = scenario["grade_level"]
        challenge = scenario["challenge_type"]
        teaching_style = scenario["teaching_style"]
        
        # Generate approaches based on scenario components
        approaches = []
        
        # Approach 1: Direct Instruction
        direct_instruction = {
            "name": "Clear Direct Instruction",
            "style": "Direct Instruction",
            "description": "This approach uses clear, step-by-step instruction with explicit modeling. The teacher breaks down the concept into manageable parts, demonstrates each step, and checks for understanding throughout the process.",
            "example": self._generate_direct_instruction_example(subject, challenge, grade_level)
        }
        approaches.append(direct_instruction)
        
        # Approach 2: Inquiry-Based
        inquiry_based = {
            "name": "Guided Inquiry Approach",
            "style": "Inquiry-Based",
            "description": "This approach poses questions and scenarios that encourage students to explore and discover concepts themselves. The teacher facilitates learning through thoughtful questions and prompts that guide students toward understanding.",
            "example": self._generate_inquiry_based_example(subject, challenge, grade_level)
        }
        approaches.append(inquiry_based)
        
        # Approach 3: Visual/Manipulative
        visual_approach = {
            "name": "Visual and Hands-on Learning",
            "style": "Multi-sensory",
            "description": "This approach uses visual aids, manipulatives, and hands-on activities to make abstract concepts concrete. It's especially effective for visual and kinesthetic learners or when teaching complex concepts.",
            "example": self._generate_visual_approach_example(subject, challenge, grade_level)
        }
        approaches.append(visual_approach)
        
        # Approach 4: Real-world Connection
        real_world = {
            "name": "Real-world Application",
            "style": "Contextual Learning",
            "description": "This approach connects the concept to real-world situations and practical applications. It helps students understand why the learning matters and how it applies outside the classroom.",
            "example": self._generate_real_world_example(subject, challenge, grade_level)
        }
        approaches.append(real_world)
        
        # Approach 5: Differentiated
        if teaching_style == "Differentiated":
            differentiated = {
                "name": "Tiered Differentiation",
                "style": "Differentiated Instruction",
                "description": "This approach offers multiple pathways to learning the same content, adjusted for different ability levels. It provides support for struggling learners while challenging advanced students.",
                "example": self._generate_differentiated_example(subject, challenge, grade_level)
            }
            approaches.append(differentiated)
        
        return approaches
    
    def _generate_direct_instruction_example(self, subject, challenge, grade_level):
        """Generate a direct instruction example based on scenario components."""
        examples = {
            "Mathematics": {
                "Fractions": "Let me show you step-by-step how to compare fractions. First, we need to find a common denominator. Let's look at 2/3 and 3/4. The least common multiple of 3 and 4 is 12. So we convert 2/3 to 8/12 by multiplying both top and bottom by 4. Then we convert 3/4 to 9/12 by multiplying both top and bottom by 3. Now we can compare 8/12 and 9/12. Since 9 is greater than 8, 3/4 is greater than 2/3. Let's try another example together...",
                "Operations": "When we subtract with regrouping, we're borrowing from the next place value. Let me show you with 43-25. First, we look at the ones column: 3-5. Can we do this? No, 3 is smaller than 5, so we need to borrow. We take 1 from the tens column, making 4 become 3, and add 10 to the ones column, making 3 become 13. Now we can do 13-5=8. Then we subtract the tens: 3-2=1. So 43-25=18. Let's practice with another example..."
            },
            "Science": {
                "Life Science": "Plants make their own food through a process called photosynthesis. Let me show you exactly how this works. Plants need three ingredients: water, carbon dioxide, and sunlight. The plant takes in water through its roots, carbon dioxide through tiny holes in its leaves called stomata, and captures energy from sunlight with chlorophyll in its leaves. Inside the leaf cells, these ingredients combine to create glucose (sugar) and oxygen. The plant uses the glucose for energy to grow, and releases the oxygen into the air. Let's draw this process step by step..."
            }
        }
        
        # Try to get specific example, otherwise return generic response
        subject_examples = examples.get(subject, {})
        specific_example = subject_examples.get(challenge, "")
        
        if specific_example:
            return specific_example
            
        return "Let me explain this concept step by step. First, I'll demonstrate the key points... [walks through detailed example]. Now let's practice together with a similar problem. I'll guide you through each step... [provides guided practice]. Now can you try one on your own? I'm here to help if you get stuck."
    
    def _generate_inquiry_based_example(self, subject, challenge, grade_level):
        """Generate an inquiry-based example based on scenario components."""
        examples = {
            "Science": {
                "Life Science": "That's an interesting question about how plants get their food! What do you think plants need to grow? [Student responds] You mentioned water - that's right! What else have you noticed plants need? [Student mentions sunlight] Excellent observation! Let's think about this: animals eat food, but have you ever seen a plant eat something? [Student responds] So if plants don't eat food like we do, how do you think they get energy to grow? [Guide student toward photosynthesis concepts] What if we set up an experiment to test what plants need to make food?",
                "Physical Science": "That's a great question about why some objects float and others sink! Before I explain, let's try something. Here are several objects - a wooden block, a metal spoon, a rubber ball, and a stone. Which ones do you think will float and which will sink? [Student predicts] Let's test your predictions in this container of water. [After testing] What patterns do you notice? What might be causing some objects to float and others to sink? [Guide student toward concepts of density]"
            },
            "Mathematics": {
                "Fractions": "I see you're wondering about comparing fractions. Let's explore this together. Here are two chocolate bars - one divided into 3 equal pieces with 2 pieces shaded, and another divided into 4 equal pieces with 3 pieces shaded. Which bar has more chocolate shaded? How could we figure that out? [Student responds] That's interesting thinking! What if we had a way to make the pieces the same size? How might we do that? [Guide student toward finding common denominators]"
            }
        }
        
        # Try to get specific example, otherwise return generic response
        subject_examples = examples.get(subject, {})
        specific_example = subject_examples.get(challenge, "")
        
        if specific_example:
            return specific_example
            
        return "That's a fascinating question! What do you already know about this topic? [Student responds] And what makes you curious about this? [Student explains] Let's investigate this together. What if we tried... [suggests exploration activity]? What do you notice? [Student observes] And what might explain what we're seeing? [Guides student toward discovering the concept] How could we test if our idea is correct?"
    
    def _generate_visual_approach_example(self, subject, challenge, grade_level):
        """Generate a visual/manipulative example based on scenario components."""
        examples = {
            "Mathematics": {
                "Fractions": "Let me show you a way to see fractions. [Takes out fraction tiles] These tiles help us visualize fractions. The whole tile represents 1. This tile is 1/2, this one is 1/3, and so on. Let's place 2/3 next to 3/4 and compare. Do you see how they're different sizes? Now, let's find equivalent fractions using these tiles. If we place 1/2 here, which other tiles equal exactly the same amount? [Student explores with tiles] That's right - 2/4, 3/6, and 4/8 all equal 1/2. What patterns do you notice in these equivalent fractions?",
                "Geometry": "Today we're learning about angles. Instead of just talking about them, let's create angles with our arms. [Demonstrates] A right angle looks like this - like the corner of a book. An acute angle is smaller than a right angle - like this. And an obtuse angle is larger than a right angle - like this. Now you try! Show me an acute angle with your arms... [Student demonstrates] Great! Now let's use these protractors to measure the angles in these shapes..."
            },
            "Science": {
                "Earth Science": "To understand the water cycle, let's create a mini-water cycle in this clear container. [Sets up demonstration with warm water, plastic wrap, and ice cubes on top] Watch what happens to the water as it warms up. See those tiny droplets forming? That's condensation - like clouds forming. As the droplets get bigger, they'll start to fall - that's precipitation, like rain. The water collects at the bottom - like lakes and oceans - and then the cycle starts again. Let's draw this cycle and label each phase."
            }
        }
        
        # Try to get specific example, otherwise return generic response
        subject_examples = examples.get(subject, {})
        specific_example = subject_examples.get(challenge, "")
        
        if specific_example:
            return specific_example
            
        return "Let me show you this in a way we can see and touch. [Brings out visual aids or manipulatives] These materials help us see the concept in action. Watch what happens when I... [demonstrates with materials]. Now you try it. What do you notice? [Student interacts with materials] Excellent observation! Now let's draw what we've learned to help remember the key points."
    
    def _generate_real_world_example(self, subject, challenge, grade_level):
        """Generate a real-world connection example based on scenario components."""
        examples = {
            "Mathematics": {
                "Fractions": "Fractions are everywhere in your daily life! When you and your friends share a pizza, you're using fractions. If four of you share a pizza cut into 8 slices equally, how many slices does each person get? [Student: 2] Right! That's 2/8 or 1/4 of the whole pizza. What if one friend wants more and gets 3 slices? What fraction of the pizza did they eat? [Student: 3/8] Exactly! Next time you share food or see a sale at a store like '25% off' (which is 1/4 off), you're using the fractions we're learning about.",
                "Measurement": "Let's talk about when we use measurement in real life. When you bake cookies with your family, you need to measure ingredients carefully. If the recipe calls for 2 cups of flour but you only have a 1/2 cup measuring cup, how many scoops would you need? [Student responds] That's right! Or think about planning a garden - you need to measure the space and decide how far apart to place each plant. Builders use measurement constantly to make sure buildings are safe and stable."
            },
            "Social Studies": {
                "Geography": "The map skills we're learning aren't just for school - they're tools people use every day. When your family takes a road trip, someone needs to navigate using a map or GPS. Both use the coordinate systems and directions we've been studying. And when you hear weather reports talking about storms moving northeast or a cold front coming from the northwest, they're using the cardinal directions we've learned. Even video games like Minecraft use coordinates to help players navigate their worlds!"
            }
        }
        
        # Try to get specific example, otherwise return generic response
        subject_examples = examples.get(subject, {})
        specific_example = subject_examples.get(challenge, "")
        
        if specific_example:
            return specific_example
            
        return "You know, we actually use this concept in everyday life all the time. For example, [provides relevant real-world example]. And professionals like [mentions career] use this knowledge when they [describes application]. Can you think of a time when you might have encountered this outside of school? [Student responds] That's a great example! This is why understanding this concept is so useful beyond our classroom."
    
    def _generate_differentiated_example(self, subject, challenge, grade_level):
        """Generate a differentiated instruction example based on scenario components."""
        examples = {
            "Mathematics": {
                "Problem Solving": "Today we're all going to work on word problems, but I have different approaches based on what works best for each of you. For students who want additional support, I've prepared these word problems with highlighted key words and a step-by-step guide to help identify what operation to use. For students ready for grade-level work, these word problems include visual supports and space to draw out the problem. For students seeking a challenge, these multi-step word problems require applying several different operations. Everyone will meet at the end to share their strategies for solving their problems."
            },
            "Literacy": {
                "Reading Comprehension": "For our reading activity today, everyone will read about animals adapting to their environments, but you'll have choices based on your reading level and interests. Table 1 has books with more picture supports and simpler text. Table 2 has grade-level texts with important vocabulary highlighted. Table 3 has more advanced texts with challenging vocabulary and concepts. Everyone will answer comprehension questions afterward, but the questions are tailored to your text's complexity. Choose the table that will give you the right balance of comfort and challenge."
            }
        }
        
        # Try to get specific example, otherwise return generic response
        subject_examples = examples.get(subject, {})
        specific_example = subject_examples.get(challenge, "")
        
        if specific_example:
            return specific_example
            
        return "I've prepared three different ways for us to work on this concept today. For students who would like more support, we'll start with [describes scaffolded approach]. For students who feel ready for grade-level work, you'll [describes standard approach]. And for students looking for a challenge, I have [describes advanced approach]. You know your learning needs best - choose the approach that will give you the right amount of challenge. We'll all come together at the end to share what we've learned."

# Run the app
if __name__ == "__main__":
    app = SimpleSMEInterface() 
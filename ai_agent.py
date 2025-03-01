"""
Teacher Training Agent - Conversational AI for Teacher Training Simulation

This module implements a conversational AI agent for simulating teacher-student
interactions in a training environment. The goal is to help teachers
practice and improve their teaching skills through realistic simulations
with scaffolded feedback.

The agent implements several key capabilities:
1. Scenario generation based on educational parameters
2. Student simulation with varying characteristics
3. Analysis of teaching strategies
4. Constructive feedback on teaching approaches

The module includes:
- TeacherTrainingGraph: LangGraph-based implementation for the teacher training simulation
- EnhancedTeacherTrainingGraph: Extended implementation with advanced features

Dependencies:
- langgraph: For structuring the agent as a state machine
- langchain: For language model interactions
- llm_handler: For interactions with the language model

Usage:
    agent = TeacherTrainingGraph(model="gpt-4")
    response = agent.run(user_input="How would you explain fractions to a student?")
"""

import os
import json
import uuid
import logging
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Union, cast
from enum import Enum
from pydantic import Field, BaseModel

# LangChain and LangGraph imports
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Local imports
from llm_handler import EnhancedLLMInterface, PedagogicalLanguageProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define state types
class Sentiment(str, Enum):
    """Sentiment of the student response"""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    CONFUSED = "confused"
    
class AgentState(TypedDict):
    """State for the teacher training agent"""
    messages: List[BaseMessage]  # Conversation history
    student_profile: Dict[str, Any]  # Student characteristics
    scenario: Dict[str, Any]  # Current teaching scenario
    teaching_approach: Optional[str]  # Current teaching approach
    student_responses: List[str]  # History of student responses
    analysis: Optional[Dict[str, Any]]  # Analysis of teaching approaches
    agent_feedback: Optional[str]  # Feedback on teaching approach
    sentiment: Optional[Sentiment]  # Sentiment of the latest student response
    
class TeacherTrainingGraph:
    """
    LangGraph-based implementation of the Teacher Training Simulator
    
    This class uses LangGraph to create a directed graph state machine
    for simulating teacher-student interactions for training purposes.
    
    The graph includes nodes for:
    - Scenario generation
    - Teaching approach analysis
    - Student response generation
    - Feedback provision
    - Sentiment analysis
    
    Attributes:
        model_name (str): Name of the LLM model to use
        graph (StateGraph): The LangGraph state machine
        _memory_storage (dict): Dictionary for preserving conversation state
        llm (EnhancedLLMInterface): Interface to the language model
        processor (PedagogicalLanguageProcessor): For processing educational language
    """
    
    def __init__(self, model_name="gpt-4"):
        """
        Initialize the Teacher Training Graph
        
        Args:
            model_name (str): Name of the LLM model to use
        """
        self.model_name = model_name
        self.llm = EnhancedLLMInterface(model_name=model_name)
        self.processor = PedagogicalLanguageProcessor(model=model_name)
        
        # Initialize checkpointing with InMemoryStorage
        # Changed from MemorySaver to directly use a dictionary for storage
        self._memory_storage = {}
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph state machine
        
        This creates a directed graph for the teacher training simulation
        with nodes for different steps in the simulation process.
        
        Returns:
            StateGraph: The constructed graph
        """
        # Initialize the graph with the state
        builder = StateGraph(AgentState)
        
        # Note: This version of LangGraph doesn't support recursion_limit
        # We'll implement recursion checks in the _should_continue method instead
        
        # Add nodes for the different agent functions
        builder.add_node("scenario_generation", self._generate_scenario)
        builder.add_node("teaching_analysis", self._analyze_teaching)
        builder.add_node("student_response", self._generate_student_response)
        builder.add_node("feedback", self._generate_feedback)
        builder.add_node("sentiment_analysis", self._analyze_sentiment)
        
        # Add conditional edge
        builder.add_conditional_edges(
            "sentiment_analysis",
            self._should_continue,
            {
                "continue": "teaching_analysis",
                "end": END
            }
        )
        
        # Define the main flow
        builder.add_edge("scenario_generation", "teaching_analysis")
        builder.add_edge("teaching_analysis", "student_response")
        builder.add_edge("student_response", "feedback")
        builder.add_edge("feedback", "sentiment_analysis")
        
        # Set the entry point
        builder.set_entry_point("scenario_generation")
        
        # Compile the graph
        return builder.compile()
    
    def _generate_scenario(self, state: AgentState) -> AgentState:
        """
        Generate a teaching scenario based on the current state
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with scenario
        """
        # Extract context parameters from messages if available
        context = {
            "subject": "mathematics",  # Default subject
            "difficulty": "intermediate",  # Default difficulty
            "student_profile": {  # Default student profile
                "grade_level": "5th",
                "learning_style": ["visual", "kinesthetic"],
                "challenges": ["focusing", "abstract concepts"],
                "strengths": ["creativity", "collaboration"]
            }
        }
        
        # Generate the scenario
        scenario = self.processor.create_scenario(context)
        
        # Update the state
        state["scenario"] = scenario
        state["student_profile"] = context["student_profile"]
        
        # Add the scenario to messages
        system_prompt = f"""
        You are a teacher training assistant. You help teachers practice their teaching skills.
        
        CURRENT SCENARIO:
        {json.dumps(scenario, indent=2)}
        
        STUDENT PROFILE:
        {json.dumps(context["student_profile"], indent=2)}
        
        Please provide a teaching approach for this scenario.
        """
        
        state["messages"] = [SystemMessage(content=system_prompt)]
        
        # Initialize other state fields
        state["teaching_approach"] = None
        state["student_responses"] = []
        state["analysis"] = None
        state["agent_feedback"] = None
        state["sentiment"] = None
        
        return state
    
    def _analyze_teaching(self, state: AgentState) -> AgentState:
        """
        Analyze the teaching approach in the current state
        
        This node is triggered when:
        1. A new teaching approach is provided by the user
        2. The conversation continues after feedback
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with teaching analysis
        """
        # Extract the latest message from the user
        messages = state.get("messages", [])
        user_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
        
        if not user_messages:
            # If no user messages yet, return state unchanged
            return state
        
        latest_msg = user_messages[-1].content
        
        # Set the teaching approach
        state["teaching_approach"] = latest_msg
        
        # Analyze the teaching approach
        analysis = self.processor.analyze_teaching_response(
            latest_msg, 
            {
                "scenario": state["scenario"],
                "student_profile": state["student_profile"]
            }
        )
        
        # Update state with analysis
        state["analysis"] = analysis
        
        return state
    
    def _generate_student_response(self, state: AgentState) -> AgentState:
        """
        Generate a simulated student response based on the teaching approach
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with student response
        """
        # Get the teaching approach
        teaching_approach = state.get("teaching_approach", "")
        
        if not teaching_approach:
            # Default response if no teaching approach
            student_response = "I'm not sure what we're talking about today."
        else:
            # Generate a realistic student response
            student_response = self.processor.generate_student_reaction(
                teaching_approach,
                state["student_profile"],
                state["scenario"]
            )
        
        # Add to student responses history
        student_responses = state.get("student_responses", [])
        student_responses.append(student_response)
        state["student_responses"] = student_responses
        
        # Add to messages
        messages = state.get("messages", [])
        messages.append(AIMessage(content=f"Student: {student_response}"))
        state["messages"] = messages
        
        return state
    
    def _generate_feedback(self, state: AgentState) -> AgentState:
        """
        Generate feedback on the teaching approach based on analysis
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with feedback
        """
        # Get analysis and teaching approach
        analysis = state.get("analysis", {})
        teaching_approach = state.get("teaching_approach", "")
        
        # Construct a prompt for feedback
        feedback_prompt = f"""
        Analyze the following teaching approach and provide constructive feedback:
        
        TEACHING APPROACH:
        {teaching_approach}
        
        ANALYSIS:
        {json.dumps(analysis, indent=2)}
        
        STUDENT RESPONSE:
        {state["student_responses"][-1] if state["student_responses"] else "No response yet."}
        
        Please provide specific, actionable feedback that would help improve the teaching approach.
        """
        
        # Get feedback from LLM
        feedback_response = self.llm.get_llm_response([
            {"role": "system", "content": "You are an expert teacher trainer providing feedback."},
            {"role": "user", "content": feedback_prompt}
        ])
        
        # Update state with feedback
        state["agent_feedback"] = feedback_response
        
        # Add to messages
        messages = state.get("messages", [])
        messages.append(AIMessage(content=f"Feedback: {feedback_response}"))
        state["messages"] = messages
        
        return state
    
    def _analyze_sentiment(self, state: AgentState) -> AgentState:
        """
        Analyze the sentiment of the student response
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with sentiment analysis
        """
        # Get the latest student response
        student_responses = state.get("student_responses", [])
        
        if not student_responses:
            # Default to neutral if no responses
            state["sentiment"] = Sentiment.NEUTRAL
            return state
        
        latest_response = student_responses[-1]
        
        # Prompt for sentiment analysis
        sentiment_prompt = f"""
        Analyze the sentiment of the following student response:
        
        STUDENT RESPONSE:
        {latest_response}
        
        Choose one sentiment category:
        - positive: The student is engaged and understanding
        - neutral: The student is neither particularly engaged nor disengaged
        - negative: The student is disengaged, frustrated, or unhappy
        - confused: The student is confused or not understanding
        
        Return ONLY the category name, nothing else.
        """
        
        # Get sentiment from LLM
        sentiment_response = self.llm.get_llm_response([
            {"role": "system", "content": "You are analyzing student sentiment."},
            {"role": "user", "content": sentiment_prompt}
        ])
        
        # Parse sentiment
        sentiment_lower = sentiment_response.strip().lower()
        
        if "positive" in sentiment_lower:
            sentiment = Sentiment.POSITIVE
        elif "negative" in sentiment_lower:
            sentiment = Sentiment.NEGATIVE
        elif "confused" in sentiment_lower:
            sentiment = Sentiment.CONFUSED
        else:
            sentiment = Sentiment.NEUTRAL
        
        # Update state with sentiment
        state["sentiment"] = sentiment
        
        return state
    
    def _should_continue(self, state: AgentState) -> str:
        """
        Determine whether to continue the conversation or end it
        
        Args:
            state: Current agent state
            
        Returns:
            "continue" or "end"
        """
        # Get sentiment
        sentiment = state.get("sentiment", Sentiment.NEUTRAL)
        
        # Get number of responses
        responses = state.get("student_responses", [])
        response_count = len(responses)
        
        # Hard safety check - force end after too many iterations to prevent infinite loops
        if response_count >= 3:
            logger.info("Ending conversation - max turns reached")
            return "end"
        
        # End the conversation if:
        # 1. The student is satisfied (positive sentiment after 1+ turns)
        # 2. The conversation has gone on too long (2+ turns)
        if (sentiment == Sentiment.POSITIVE and response_count >= 1) or response_count >= 2:
            logger.info(f"Ending conversation - positive sentiment after {response_count} turns")
            return "end"
        
        logger.info(f"Continuing conversation after {response_count} turns")
        return "continue"
    
    def run(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the agent with user input
        
        Args:
            user_input: The user's input text
            context: Optional context for the conversation
            
        Returns:
            Dict with the agent's response
        """
        # Generate a unique conversation ID
        conversation_id = str(uuid.uuid4())
        
        # Initialize state if needed
        if not hasattr(self, '_state'):
            self._state = {
                "messages": [],
                "student_profile": {},
                "scenario": {},
                "teaching_approach": None,
                "student_responses": [],
                "analysis": None,
                "agent_feedback": None,
                "sentiment": None
            }
        
        # Add user input to messages
        self._state["messages"].append(HumanMessage(content=user_input))
        
        # Run the graph with error handling
        try:
            # Set a maximum iteration counter since LangGraph version doesn't support recursion_limit
            max_iterations = 5
            current_iteration = 0
            
            # Get the initial state size to track changes
            initial_message_count = len(self._state.get("messages", []))
            initial_response_count = len(self._state.get("student_responses", []))
            
            # Process until we reach stability or max iterations
            while current_iteration < max_iterations:
                prev_state = self._state.copy()
                self._state = self.graph.invoke(self._state)
                current_iteration += 1
                
                # Check if the state has converged (no more changes to key fields)
                message_count = len(self._state.get("messages", []))
                response_count = len(self._state.get("student_responses", []))
                
                # If we've added at least one student response and feedback, and no new messages
                # are being added in this iteration, we can stop
                if response_count > initial_response_count and message_count > initial_message_count + 2:
                    # If no new messages were added in this iteration, we're done
                    if message_count == len(prev_state.get("messages", [])):
                        logger.info(f"Converged after {current_iteration} iterations")
                        break
            
            if current_iteration >= max_iterations:
                logger.warning("Reached maximum number of iterations!")
            
            # Ensure all required state fields exist even after graph execution
            for field in ["student_profile", "scenario", "teaching_approach", 
                          "student_responses", "analysis", "agent_feedback", "sentiment"]:
                if field not in self._state or self._state[field] is None:
                    if field in ["student_profile", "scenario", "analysis"]:
                        self._state[field] = {}
                    elif field == "student_responses":
                        self._state[field] = []
                    else:
                        self._state[field] = None
                        
        except Exception as e:
            logger.error(f"Error running the graph: {str(e)}")
            
            # Create a fallback response
            messages = self._state.get("messages", [])
            messages.append(AIMessage(content="I'm sorry, I encountered an issue processing your input. Let's continue with the simulation."))
            self._state["messages"] = messages
            
            # Add a default student response if none exists
            if not self._state.get("student_responses"):
                self._state["student_responses"] = ["I'm not sure I understand. Can you explain again?"]
            
            # Set a default sentiment
            self._state["sentiment"] = Sentiment.NEUTRAL
        
        # Store state for reference
        self._memory_storage[conversation_id] = self._state
        
        # Extract the response
        messages = self._state.get("messages", [])
        ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
        
        response = ai_messages[-1].content if ai_messages else "I'm ready to help with teacher training."
        
        return {
            "response": response,
            "state": self._state,
            "conversation_id": conversation_id
        }

class EnhancedTeacherTrainingGraph(TeacherTrainingGraph):
    """
    Enhanced Teacher Training Graph with additional advanced features
    
    This extends the basic TeacherTrainingGraph with:
    - Customizable student profiles and scenarios
    - Targeted feedback on specific teaching dimensions
    - Enhanced state tracking for multi-turn conversations
    - Additional analysis nodes
    """
    
    def __init__(self, model_name="gpt-4"):
        """
        Initialize the Enhanced Teacher Training Graph
        
        Args:
            model_name (str): Name of the LLM model to use
        """
        super().__init__(model_name=model_name)
        
        # Override the graph with the enhanced version
        self.graph = self._build_enhanced_graph()
    
    def _build_enhanced_graph(self) -> StateGraph:
        """
        Build the enhanced LangGraph state machine
        
        Returns:
            StateGraph: The constructed graph
        """
        # Initialize the graph with the state
        builder = StateGraph(AgentState)
        
        # Note: This version of LangGraph doesn't support recursion_limit
        # We'll implement recursion checks in the _enhanced_should_continue method instead
        
        # Add nodes for the different agent functions (including base nodes)
        builder.add_node("scenario_generation", self._generate_scenario)
        builder.add_node("teaching_analysis", self._analyze_teaching)
        builder.add_node("student_response", self._generate_student_response)
        builder.add_node("feedback", self._generate_feedback)
        builder.add_node("sentiment_analysis", self._analyze_sentiment)
        
        # Add enhanced nodes
        builder.add_node("detailed_teaching_analysis", self._detailed_teaching_analysis)
        builder.add_node("learning_objective_assessment", self._assess_learning_objectives)
        builder.add_node("reflection_prompt", self._generate_reflection_prompt)
        
        # Add conditional edge for sentiment
        builder.add_conditional_edges(
            "sentiment_analysis",
            self._enhanced_should_continue,
            {
                "continue": "teaching_analysis",
                "reflect": "reflection_prompt",
                "end": END
            }
        )
        
        # Special edge from reflection to END to prevent recursion
        builder.add_edge("reflection_prompt", END)
        
        # Define the enhanced flow
        builder.add_edge("scenario_generation", "teaching_analysis")
        builder.add_edge("teaching_analysis", "detailed_teaching_analysis")
        builder.add_edge("detailed_teaching_analysis", "student_response")
        builder.add_edge("student_response", "learning_objective_assessment")
        builder.add_edge("learning_objective_assessment", "feedback")
        builder.add_edge("feedback", "sentiment_analysis")
        
        # Set the entry point
        builder.set_entry_point("scenario_generation")
        
        # Compile the graph
        return builder.compile()
    
    def _detailed_teaching_analysis(self, state: AgentState) -> AgentState:
        """
        Perform a more detailed analysis of the teaching approach
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with detailed analysis
        """
        # Get the teaching approach
        teaching_approach = state.get("teaching_approach", "")
        
        if not teaching_approach:
            return state
        
        # Prompt for detailed analysis
        analysis_prompt = f"""
        Perform a detailed analysis of the following teaching approach:
        
        TEACHING APPROACH:
        {teaching_approach}
        
        SCENARIO:
        {json.dumps(state["scenario"], indent=2)}
        
        STUDENT PROFILE:
        {json.dumps(state["student_profile"], indent=2)}
        
        Analyze the following dimensions:
        1. Content accuracy and relevance
        2. Pedagogical approach and methodology
        3. Engagement and motivation techniques
        4. Differentiation and inclusivity
        5. Assessment and feedback strategies
        
        For each dimension, provide:
        - A score from 1-10
        - Specific strengths
        - Areas for improvement
        - Suggestions for enhancement
        
        Return your analysis as a structured evaluation.
        """
        
        # Get analysis from LLM
        detailed_analysis = self.llm.get_llm_response([
            {"role": "system", "content": "You are an expert teaching analyst."},
            {"role": "user", "content": analysis_prompt}
        ])
        
        # Update the analysis with detailed information
        if "analysis" not in state or not state["analysis"]:
            state["analysis"] = {}
        
        state["analysis"]["detailed"] = detailed_analysis
        
        return state
    
    def _assess_learning_objectives(self, state: AgentState) -> AgentState:
        """
        Assess how well the approach addresses learning objectives
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with learning objective assessment
        """
        # Get scenario, teaching approach, and student response
        scenario = state.get("scenario", {})
        teaching_approach = state.get("teaching_approach", "")
        student_responses = state.get("student_responses", [])
        
        if not teaching_approach or not student_responses or not scenario:
            return state
        
        # Extract learning objectives
        learning_objectives = scenario.get("learning_objectives", [])
        if not learning_objectives:
            return state
        
        latest_response = student_responses[-1]
        
        # Prompt for assessment
        assessment_prompt = f"""
        Assess how well the teaching approach addresses the learning objectives:
        
        LEARNING OBJECTIVES:
        {json.dumps(learning_objectives, indent=2)}
        
        TEACHING APPROACH:
        {teaching_approach}
        
        STUDENT RESPONSE:
        {latest_response}
        
        For each learning objective, provide:
        - Assessment of whether it was addressed (Yes/Partially/No)
        - Evidence from the teaching approach
        - Evidence from the student response
        - Suggestions for better addressing it
        
        Return your assessment as a structured evaluation.
        """
        
        # Get assessment from LLM
        objectives_assessment = self.llm.get_llm_response([
            {"role": "system", "content": "You are assessing learning objectives."},
            {"role": "user", "content": assessment_prompt}
        ])
        
        # Update the analysis
        if "analysis" not in state or not state["analysis"]:
            state["analysis"] = {}
        
        state["analysis"]["objectives_assessment"] = objectives_assessment
        
        return state
    
    def _generate_reflection_prompt(self, state: AgentState) -> AgentState:
        """
        Generate a reflection prompt for the teacher
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with reflection prompt
        """
        # Get teaching approach and analysis
        teaching_approach = state.get("teaching_approach", "")
        analysis = state.get("analysis", {})
        student_responses = state.get("student_responses", [])
        
        # Default reflection - used if anything goes wrong
        default_reflection = "Reflection: Consider how your teaching approach affected student engagement and understanding. What might you do differently next time?"
        
        if not teaching_approach or not analysis:
            # Return state with a default reflection to avoid recursion issues
            logger.warning("Missing teaching approach or analysis, using default reflection")
            messages = state.get("messages", [])
            messages.append(AIMessage(content=default_reflection))
            state["messages"] = messages
            return state
        
        # Safety check - don't try to generate reflection if too many responses already
        if len(student_responses) >= 3:
            logger.warning("Too many responses, using default reflection")
            messages = state.get("messages", [])
            messages.append(AIMessage(content=default_reflection))
            state["messages"] = messages
            return state
        
        # Prompt for reflection
        reflection_prompt = f"""
        Generate a reflective prompt for the teacher based on:
        
        TEACHING APPROACH:
        {teaching_approach}
        
        STUDENT RESPONSES:
        {json.dumps(student_responses, indent=2)}
        
        ANALYSIS:
        {json.dumps(analysis, indent=2)}
        
        The reflection prompt should:
        1. Highlight key strengths to build upon
        2. Identify areas for growth
        3. Ask thought-provoking questions for self-reflection
        4. Suggest specific strategies to try
        
        Format the reflection as a constructive guidance for professional development.
        Limit your response to 250 words or less.
        """
        
        try:
            # Get reflection from LLM with timeout
            reflection = self.llm.get_llm_response([
                {"role": "system", "content": "You are a reflective coaching specialist."},
                {"role": "user", "content": reflection_prompt}
            ])
            
            # Add to messages
            messages = state.get("messages", [])
            messages.append(AIMessage(content=f"Reflection: {reflection}"))
            state["messages"] = messages
        except Exception as e:
            # Handle errors gracefully to avoid recursion issues
            logger.error(f"Error generating reflection: {str(e)}")
            messages = state.get("messages", [])
            messages.append(AIMessage(content=default_reflection))
            state["messages"] = messages
        
        return state
    
    def _enhanced_should_continue(self, state: AgentState) -> str:
        """
        Enhanced decision logic for continuing the conversation
        
        Args:
            state: Current agent state
            
        Returns:
            "continue", "reflect", or "end"
        """
        # Get sentiment
        sentiment = state.get("sentiment", Sentiment.NEUTRAL)
        
        # Get number of responses
        responses = state.get("student_responses", [])
        response_count = len(responses)
        
        # Always end immediately after a single pass through to prevent unnecessary recursion
        # This is the most important change for performance - force a single complete pass
        if "messages" in state and len(state["messages"]) > 2:
            logger.info("Ending enhanced conversation after first complete pass")
            return "end"
        
        # Hard safety check - force end after too many iterations to prevent infinite loops
        if response_count >= 1:
            logger.info("Ending enhanced conversation - max turns reached")
            return "end"
        
        # Force end if negative sentiment persists or confusion after initial response
        if response_count >= 1 and (sentiment == Sentiment.NEGATIVE or sentiment == Sentiment.CONFUSED):
            logger.info(f"Ending enhanced conversation - negative/confused sentiment after {response_count} turns")
            return "end"
            
        # Provide feedback immediately rather than waiting for multiple turns
        if response_count >= 1 and sentiment == Sentiment.POSITIVE:
            logger.info("Ending with positive sentiment - generating feedback")
            return "reflect"
        
        # Default to end to prevent unnecessary recursion
        return "end"
    
    def create_custom_scenario(self, subject: str, grade_level: str, 
                              learning_objectives: List[str],
                              student_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a custom teaching scenario with specific parameters
        
        Args:
            subject: The subject being taught
            grade_level: The student's grade level
            learning_objectives: List of learning objectives
            student_characteristics: Dictionary of student characteristics
            
        Returns:
            Dict with the created scenario
        """
        # Create context for scenario generation
        context = {
            "subject": subject,
            "difficulty": "custom",
            "student_profile": {
                "grade_level": grade_level,
                **student_characteristics
            }
        }
        
        # Generate scenario
        scenario = self.processor.create_scenario(context)
        
        # Override learning objectives
        scenario["learning_objectives"] = learning_objectives
        
        return scenario
    
    def run_with_custom_scenario(self, user_input: str, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the agent with a custom scenario
        
        Args:
            user_input: The user's input text
            scenario: Custom scenario to use
            
        Returns:
            Dict with the agent's response
        """
        # Generate a unique conversation ID
        conversation_id = str(uuid.uuid4())
        
        # Make sure scenario has all required fields
        if not scenario.get("student_profile"):
            scenario["student_profile"] = {}
            
        # Initialize state with custom scenario
        self._state = {
            "messages": [],
            "student_profile": scenario.get("student_profile", {}),
            "scenario": scenario,
            "teaching_approach": user_input,  # Set teaching approach directly from user input
            "student_responses": [],
            "analysis": None,
            "agent_feedback": None,
            "sentiment": None
        }
        
        # Add scenario to messages - streamlined for faster processing
        system_prompt = f"""
        You are a teacher training assistant. Help teachers practice their teaching skills.
        
        CURRENT SCENARIO:
        {json.dumps(scenario, indent=2)}
        
        STUDENT PROFILE:
        {json.dumps(scenario.get("student_profile", {}), indent=2)}
        """
        
        self._state["messages"] = [SystemMessage(content=system_prompt)]
        
        try:
            # Add user input to messages and set as teaching approach
            self._state["messages"].append(HumanMessage(content=user_input))
            
            # Set a maximum iteration counter since LangGraph version doesn't support recursion_limit
            max_iterations = 1  # REDUCED to just 1 to prevent excessive API calls
            current_iteration = 0
            
            # Track API calls for optimization
            api_call_count = 0
            max_api_calls = 4
            
            # Process until we reach stability or max iterations
            while current_iteration < max_iterations and api_call_count < max_api_calls:
                logger.info(f"Starting iteration {current_iteration + 1} of {max_iterations}")
                
                # Process one iteration - use try/except to handle potential errors
                try:
                    self._state = self.graph.invoke(self._state)
                    api_call_count += 1
                    current_iteration += 1
                    
                    # Early exit condition: if we have a student response, exit early
                    if self._state.get("student_responses") and len(self._state["student_responses"]) > 0:
                        logger.info("Successfully received student response, exiting iterations")
                        break
                    
                    # If sentiment indicates we should end, exit early
                    sentiment = self._state.get("sentiment", "")
                    if sentiment and (sentiment == Sentiment.NEGATIVE or sentiment == Sentiment.CONFUSED):
                        logger.info(f"Ending enhanced conversation - {sentiment} sentiment")
                        break
                        
                    # After 3 API calls, force an early exit to prevent excessive calls
                    if api_call_count >= 3:
                        logger.info("API call limit reached, exiting early")
                        break
                        
                except Exception as e:
                    logger.error(f"Error during iteration: {str(e)}")
                    break
            
            # If we still don't have a student response, create one
            if not self._state.get("student_responses") or len(self._state.get("student_responses", [])) == 0:
                logger.warning("No student response generated, creating fallback")
                self._state["student_responses"] = ["I'm not sure what we're talking about today. Can you explain the topic we're covering?"]
                
                # Add to messages if not already there
                student_msg = f"Student: {self._state['student_responses'][-1]}"
                if not any(student_msg in str(msg) for msg in self._state.get("messages", [])):
                    self._state["messages"].append(AIMessage(content=student_msg))
                
                # Also force feedback generation
                if not self._state.get("agent_feedback"):
                    logger.info("Generating fallback feedback")
                    self._state["agent_feedback"] = "Consider introducing the topic clearly and checking for student understanding. Your approach would benefit from more specific learning objectives."
            
            # Ensure all required state fields exist with meaningful defaults
            required_fields = {
                "student_profile": scenario.get("student_profile", {}),
                "scenario": scenario,
                "teaching_approach": user_input,
                "student_responses": ["I'm here and ready to learn. What will we be covering today?"],
                "analysis": {
                    "overall_assessment": "Your teaching approach is still being evaluated.",
                    "identified_strengths": ["Engaging with the student"],
                    "improvement_areas": ["Be more specific about learning objectives"]
                },
                "agent_feedback": "To improve your teaching approach, try to be more specific about the learning objectives and engage the student with questions.",
                "sentiment": Sentiment.NEUTRAL
            }
            
            for field, default_value in required_fields.items():
                if field not in self._state or self._state[field] is None:
                    self._state[field] = default_value
                    # If adding student response, also update messages
                    if field == "student_responses":
                        self._state["messages"].append(AIMessage(content=f"Student: {self._state['student_responses'][-1]}"))
            
            # Store state for reference
            import copy
            self._memory_storage[conversation_id] = copy.deepcopy(self._state)
            
            # Extract the most relevant response for the user
            response = ""
            if self._state.get("agent_feedback"):
                response = f"Feedback: {self._state['agent_feedback']}"
            else:
                # Find the latest student response
                messages = self._state.get("messages", [])
                student_messages = [msg.content for msg in messages if isinstance(msg, AIMessage) and "Student:" in msg.content]
                if student_messages:
                    response = student_messages[-1]
                else:
                    response = "I'm ready to help with teacher training."
            
            # Always include feedback after student interaction
            if not response.startswith("Feedback:") and self._state.get("agent_feedback"):
                response += f"\n\nFeedback: {self._state['agent_feedback']}"
            
            return {
                "response": response,
                "state": self._state,
                "conversation_id": conversation_id
            }
            
        except Exception as e:
            logger.error(f"Error running with custom scenario: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Create a fallback response with student message for better user experience
            fallback_student_response = "Hello teacher! I'm ready to learn about the topic. Can you tell me what we're covering today?"
            fallback_feedback = "Your teaching approach could be more specific. Try to clearly state the learning objectives and engage the student with questions."
            
            # Make sure we have a student response
            if "student_responses" not in self._state or not self._state["student_responses"]:
                self._state["student_responses"] = [fallback_student_response]
            
            # Add a student message to the conversation
            self._state["messages"].append(AIMessage(content=f"Student: {fallback_student_response}"))
            
            # Always provide feedback in case of error
            self._state["agent_feedback"] = fallback_feedback
            
            return {
                "response": f"Student: {fallback_student_response}\n\nFeedback: {fallback_feedback}",
                "state": self._state,
                "conversation_id": conversation_id
            }

    def initialize_agent(self, model_name="gpt-4", context=None, scenario=None):
        """
        Initialize the LangGraph agent with a specific model and context.
        
        Args:
            model_name (str): Name of the LLM model to use
            context (dict, optional): Context information for the simulation
            scenario (dict, optional): Scenario settings for the simulation
            
        Returns:
            bool: Success status
        """
        try:
            self.model_name = model_name
            
            # Log model initialization
            logging.info(f"Initializing agent with model: {model_name}")
            
            # For Llama models, use the enhanced interface
            if "llama" in model_name.lower():
                logging.info("Using local Llama model - checking for model file or Ollama")
                self.llm_interface = EnhancedLLMInterface(model_name=model_name)
            else:
                # For OpenAI or Claude models, use standard initialization
                self.llm_interface = EnhancedLLMInterface(model_name=model_name)
            
            # If scenario is provided, set it directly
            if scenario:
                self.scenario = scenario
            
            # Set the context (if provided)
            if context:
                self.context = context
            
            # Build the graph with the configured LLM
            self.graph = self._build_enhanced_graph()
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize agent: {str(e)}")
            return False
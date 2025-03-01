"""
DSPy Language Model Interface and Processing Module

This module provides a comprehensive interface for interacting with Language
Models (LLMs) in the teaching simulation system using DSPy. It handles all communication
with the LLM backend while providing context management, error handling, and
pedagogical processing capabilities.

Key Features:
    - Standardized LLM communication using DSPy
    - Advanced prompt optimization
    - Context-aware prompt building
    - Pedagogical language processing
    - Teaching response analysis
    - Student reaction generation
    - Error handling and recovery
    - Configurable model parameters

Components:
    - DSPyLLMInterface: Base interface for LLM interaction
    - EnhancedDSPyLLMInterface: Advanced interface with streaming and specialized prompts
    - PedagogicalLanguageProcessor: Educational language processor

Dependencies:
    - dspy: For LLM interface and prompt programming
    - torch: For GPU operations
    - typing: For type hints
"""

import os
import json
import time
import random
import logging
from typing import List, Dict, Any, Optional, Union

# DSPy imports
import dspy
from dspy.predict import Predict
from dspy.teleprompt import BootstrapFewShot
from dspy.signatures import Signature

# For error handling and retry logic
import backoff

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Define DSPy signatures for teaching-specific tasks
class TeachingResponse(Signature):
    """Response signature for teaching recommendations."""
    response: str = dspy.OutputField(desc="Educational response appropriate for the student and context")

class StudentResponse(Signature):
    """Response signature for simulated student responses."""
    response: str = dspy.OutputField(desc="Realistic student response based on the profile and context")

class TeachingAnalysis(Signature):
    """Response signature for teaching analysis."""
    strengths: List[str] = dspy.OutputField(desc="Teaching strengths identified in the response")
    areas_for_improvement: List[str] = dspy.OutputField(desc="Potential areas for improvement")
    effectiveness_score: int = dspy.OutputField(desc="Score from 1-10 rating the teaching effectiveness")
    rationale: str = dspy.OutputField(desc="Explanation of the analysis")

# Define DSPy modules for different teaching tasks
class TeacherResponder(dspy.Module):
    """Module for generating teaching responses"""
    
    def __init__(self):
        super().__init__()
        self.generate_response = Predict(TeachingResponse)
    
    def forward(self, context, student_profile, question):
        """Generate a teaching response to a student question"""
        prompt = f"""
        You are an expert teacher.
        
        CONTEXT:
        {context}
        
        STUDENT PROFILE:
        {student_profile}
        
        STUDENT QUESTION:
        {question}
        
        Provide an educational response that is appropriate for this student based on their profile.
        """
        return self.generate_response(prompt=prompt)

class StudentReactionGenerator(dspy.Module):
    """Module for generating realistic student reactions"""
    
    def __init__(self):
        super().__init__()
        self.generate_reaction = Predict(StudentResponse)
    
    def forward(self, teacher_input, student_profile, scenario_context):
        """Generate a realistic student reaction to a teacher's input"""
        prompt = f"""
        The teacher has said: "{teacher_input}"
        
        STUDENT PROFILE:
        {student_profile}
        
        SCENARIO CONTEXT:
        {scenario_context}
        
        Respond as the student would, considering their grade level, learning style, challenges, and strengths.
        Give a realistic, age-appropriate response from the student's perspective.
        """
        return self.generate_reaction(prompt=prompt)

class TeachingAnalyzer(dspy.Module):
    """Module for analyzing teaching approaches"""
    
    def __init__(self):
        super().__init__()
        self.analyze = Predict(TeachingAnalysis)
    
    def forward(self, teacher_input, student_profile, scenario_context):
        """Analyze teaching strategies used in the teacher's input"""
        prompt = f"""
        Analyze the following teaching approach:
        
        "{teacher_input}"
        
        STUDENT PROFILE:
        {student_profile}
        
        SCENARIO CONTEXT:
        {scenario_context}
        
        Identify the teaching strategies used and their potential effectiveness for this student.
        """
        return self.analyze(prompt=prompt)

class DSPyLLMInterface:
    """
    Base interface for interacting with language models using DSPy.
    
    This class provides a standardized interface for all LLM interactions
    in the teaching simulation system, supporting different models and
    configurations.
    
    Attributes:
        model_name (str): Name of the LLM model to use
        lm: DSPy language model instance
    """
    
    def __init__(self, model_name="gpt-3.5-turbo"):
        """
        Initialize the LLM interface.
        
        Args:
            model_name (str): Name of the LLM model to use
        """
        self.model_name = model_name
        self.lm = self._initialize_model(model_name)
        
        # Set the DSPy language model
        dspy.settings.configure(lm=self.lm)
    
    def _initialize_model(self, model_name):
        """
        Initialize the appropriate language model based on the model name.
        
        Args:
            model_name (str): Name of the model to initialize
            
        Returns:
            DSPy language model instance
        """
        # Handle different model types
        if "gpt" in model_name.lower():
            logging.info(f"Initializing OpenAI model: {model_name}")
            # Import OpenAIProvider, but use it as the provider parameter to dspy.LM
            from dspy.clients.openai import OpenAIProvider
            return dspy.LM(model=model_name, provider=OpenAIProvider(), temperature=0.7)
            
        elif "claude" in model_name.lower():
            logging.info(f"Initializing Anthropic model: {model_name}")
            from dspy.clients.anthropic import AnthropicProvider
            return dspy.LM(model=model_name, provider=AnthropicProvider(), temperature=0.7)
            
        elif "llama-3" in model_name.lower() or "llama3" in model_name.lower():
            # Try to use Ollama for Llama 3 models
            try:
                logging.info("Attempting to initialize Llama 3 with Ollama...")
                # Convert model name format for Ollama (llama-3-8b → llama3:8b)
                ollama_model_name = "llama3:8b"
                if "70b" in model_name.lower():
                    ollama_model_name = "llama3:70b"
                    
                logging.info(f"Using Ollama model: {ollama_model_name}")
                
                # Try to use litellm directly
                try:
                    import litellm
                    logging.info("Initializing Ollama through litellm")
                    
                    # Create a custom provider that uses litellm
                    class LiteLLMProvider:
                        def __call__(self, prompt, **kwargs):
                            try:
                                response = litellm.completion(
                                    model="ollama/"+ollama_model_name,
                                    messages=[{"role": "user", "content": prompt}],
                                    temperature=0.7
                                )
                                return response.choices[0].message.content
                            except Exception as e:
                                logging.error(f"LiteLLM error: {e}")
                                return "Error generating response with Ollama."
                    
                    # Include the model parameter as required by dspy.LM
                    return dspy.LM(model=ollama_model_name, provider=LiteLLMProvider(), temperature=0.7)
                    
                except ImportError as e:
                    logging.warning(f"litellm not available: {e}")
                    # Fall back to OpenAI
                    logging.info("Falling back to default model")
                    from dspy.clients.openai import OpenAIProvider
                    return dspy.LM(model="gpt-3.5-turbo", provider=OpenAIProvider(), temperature=0.7)
                
            except Exception as e:
                logging.warning(f"Error initializing Ollama: {e}")
                logging.info("Falling back to default model")
                from dspy.clients.openai import OpenAIProvider
                return dspy.LM(model="gpt-3.5-turbo", provider=OpenAIProvider(), temperature=0.7)
        else:
            # Default to GPT-3.5-turbo
            logging.info("Defaulting to OpenAI GPT-3.5-turbo model")
            from dspy.clients.openai import OpenAIProvider
            return dspy.LM(model="gpt-3.5-turbo", provider=OpenAIProvider(), temperature=0.7)
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def get_llm_response(self, messages, output_format=None):
        """
        Get a response from the language model with retry logic.
        
        Args:
            messages: List of message dictionaries with role and content
            output_format: Optional format specification (json, markdown, etc.)
            
        Returns:
            str: The model's response
        """
        try:
            # Convert message dictionaries to a prompt string
            prompt = self._messages_to_prompt(messages, output_format)
            
            # Call the LM object directly with the prompt
            result = self.lm(prompt)
            
            # Ensure we return a string (DSPy might return a list with one item)
            if isinstance(result, list) and len(result) > 0:
                return result[0]
            return result
            
        except Exception as e:
            logging.error(f"Failed to get LLM response: {e}")
            return f"Error: Unable to generate response. Please try again later."
    
    def _messages_to_prompt(self, messages, output_format=None):
        """Convert message dictionaries to a prompt string"""
        prompt_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"Instructions: {content}\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n")
            else:  # Default to user
                prompt_parts.append(f"User: {content}\n")
        
        # Add output format instruction if needed
        if output_format == "json":
            prompt_parts.append("Return your response in valid JSON format.")
        
        return "\n".join(prompt_parts)
    
    def generate_teaching_recommendation(self, scenario, student_profile):
        """
        Generate teaching recommendations based on scenario and student profile.
        
        Args:
            scenario: Teaching scenario details
            student_profile: Student characteristics and needs
            
        Returns:
            dict: Teaching recommendations including strategies and approaches
        """
        prompt = f"""
        You are an expert education consultant providing teaching recommendations.
        
        SCENARIO:
        {json.dumps(scenario, indent=2)}
        
        STUDENT PROFILE:
        {json.dumps(student_profile, indent=2)}
        
        Please provide teaching recommendations appropriate for this scenario and student.
        Include specific strategies, approaches, and potential adaptations.
        Return your response as JSON with the following structure:
        {{
            "recommended_strategies": [list of strategies],
            "adaptations": [list of adaptations for student needs],
            "communication_approach": "description of effective communication",
            "assessment_methods": [list of appropriate assessment approaches]
        }}
        """
        
        response = self.get_llm_response([{"role": "user", "content": prompt}], output_format="json")
        
        try:
            return json.loads(response)
        except:
            # If JSON parsing fails, return a structured dictionary with the raw response
            return {
                "recommended_strategies": ["Error parsing recommendations"],
                "adaptations": [],
                "communication_approach": response,
                "assessment_methods": []
            }


class EnhancedDSPyLLMInterface(DSPyLLMInterface):
    """
    Enhanced interface for advanced LLM interactions using DSPy.
    
    This class extends the base DSPyLLMInterface with additional capabilities:
    - Specialized educational prompting techniques
    - Optimized prompts using BootstrapFewShot
    - Pre-configured modules for educational tasks
    
    Attributes:
        model_name (str): Name of the LLM model to use
        lm: DSPy language model instance
        teacher_module: Module for generating teaching responses
        student_module: Module for generating student reactions
        analyzer_module: Module for analyzing teaching approaches
    """
    
    def __init__(self, model_name="gpt-4"):
        """
        Initialize the enhanced LLM interface.
        
        Args:
            model_name (str): Name of the LLM model to use
        """
        super().__init__(model_name)
        
        # Initialize specialized modules
        self.teacher_module = TeacherResponder()
        self.student_module = StudentReactionGenerator()
        self.analyzer_module = TeachingAnalyzer()
        
        # Collection of specialized system prompts for different educational scenarios
        self.system_prompts = {
            "student_simulation": """You are simulating a student in a classroom setting. 
            Your responses should reflect the student's knowledge level, learning style, challenges, and strengths.
            Respond as the student would to the teacher's input.""",
            
            "teaching_analysis": """You are an expert educational consultant analyzing teaching approaches.
            Evaluate the teaching response considering pedagogical best practices, student needs, and learning objectives.
            Provide specific, actionable feedback with clear strengths and areas for improvement.""",
            
            "strategy_recommendation": """You are an educational specialist recommending teaching strategies.
            Based on the student profile and teaching context, suggest evidence-based approaches 
            that would be most effective for this specific learning situation."""
        }
    
    def get_llm_response_with_context(self, messages, context_data, prompt_type="student_simulation"):
        """
        Get an LLM response with additional context and specialized prompting.
        
        Args:
            messages: List of message dictionaries with role and content
            context_data: Dictionary with relevant contextual information
            prompt_type: Type of specialized system prompt to use
            
        Returns:
            str: The model's response incorporating the context
        """
        # Create a context-aware system prompt
        system_prompt = self.system_prompts.get(prompt_type, "You are a helpful assistant.")
        
        # Add context to the system prompt
        context_str = json.dumps(context_data, indent=2)
        enhanced_prompt = f"{system_prompt}\n\nCONTEXT INFORMATION:\n{context_str}"
        
        # Ensure the first message is a system message with our enhanced prompt
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = enhanced_prompt
        else:
            messages.insert(0, {"role": "system", "content": enhanced_prompt})
        
        # Get response from the base method
        return self.get_llm_response(messages)
    
    def generate_student_response(self, teacher_input, student_profile, scenario_context):
        """
        Generate a realistic student response to a teacher's input using DSPy module.
        
        Args:
            teacher_input: The teacher's statement or question
            student_profile: Student characteristics and attributes
            scenario_context: Additional context about the teaching scenario
            
        Returns:
            str: A realistic student response based on the profile and context
        """
        try:
            student_profile_str = json.dumps(student_profile) if isinstance(student_profile, dict) else str(student_profile)
            scenario_context_str = json.dumps(scenario_context) if isinstance(scenario_context, dict) else str(scenario_context)
            
            result = self.student_module(
                teacher_input=teacher_input, 
                student_profile=student_profile_str, 
                scenario_context=scenario_context_str
            )
            return result.response
        except Exception as e:
            logging.error(f"Error generating student response: {e}")
            # Fallback to simpler implementation
            return self.get_llm_response([
                {"role": "system", "content": self.system_prompts["student_simulation"]},
                {"role": "user", "content": f"The teacher said: '{teacher_input}'. Respond as a student would."}
            ])
    
    def analyze_teaching_strategies(self, teacher_input, student_profile, scenario_context):
        """
        Analyze teaching strategies used in the teacher's input using DSPy module.
        
        Args:
            teacher_input: The teacher's statement or approach
            student_profile: Student characteristics and attributes
            scenario_context: Additional context about the teaching scenario
            
        Returns:
            dict: Analysis of teaching strategies with effectiveness scores
        """
        try:
            student_profile_str = json.dumps(student_profile) if isinstance(student_profile, dict) else str(student_profile)
            scenario_context_str = json.dumps(scenario_context) if isinstance(scenario_context, dict) else str(scenario_context)
            
            result = self.analyzer_module(
                teacher_input=teacher_input,
                student_profile=student_profile_str,
                scenario_context=scenario_context_str
            )
            
            # Convert to dictionary format
            return {
                "identified_strategies": [{"strategy": s, "effectiveness": result.effectiveness_score} for s in result.strengths],
                "overall_effectiveness": result.effectiveness_score,
                "suggested_improvements": result.areas_for_improvement,
                "rationale": result.rationale
            }
        except Exception as e:
            logging.error(f"Error analyzing teaching strategies: {e}")
            # Fallback to simpler implementation
            prompt = f"""
            Analyze the following teaching approach:
            
            "{teacher_input}"
            
            Identify the teaching strategies used and their potential effectiveness
            for a student with these characteristics: {json.dumps(student_profile)}
            
            The teaching context is: {json.dumps(scenario_context)}
            
            Return your analysis as a JSON object with the following structure:
            {{
                "identified_strategies": [
                    {{
                        "strategy": "name of strategy",
                        "description": "brief description",
                        "effectiveness": score from 1-10,
                        "rationale": "why this score was given"
                    }}
                ],
                "overall_effectiveness": score from 1-10,
                "suggested_improvements": ["improvement 1", "improvement 2"]
            }}
            """
            
            response = self.get_llm_response([{"role": "user", "content": prompt}], output_format="json")
            
            try:
                return json.loads(response)
            except:
                # Fallback if JSON parsing fails
                return {
                    "identified_strategies": [
                        {"strategy": "General approach", "description": response, "effectiveness": 5, "rationale": "Unable to parse detailed analysis"}
                    ],
                    "overall_effectiveness": 5,
                    "suggested_improvements": ["Consider more structured analysis"]
                }


class PedagogicalLanguageProcessor:
    """
    Processes and analyzes language in educational contexts using DSPy.
    
    This class specializes in analyzing and generating language related to
    teaching and learning, including creating scenarios, analyzing responses,
    and generating student reactions.
    
    Attributes:
        llm_interface: DSPy LLM interface for generating responses
    """
    
    def __init__(self, model="gpt-3.5-turbo"):
        """
        Initialize the pedagogical language processor.
        
        Args:
            model (str): LLM model to use for processing
        """
        self.llm_interface = DSPyLLMInterface(model_name=model)
    
    def create_scenario(self, context):
        """
        Create a teaching scenario based on context parameters.
        
        Args:
            context: Dictionary containing scenario parameters
            
        Returns:
            dict: A comprehensive teaching scenario
        """
        subject = context.get("subject", "general")
        difficulty = context.get("difficulty", "intermediate")
        student_profile = context.get("student_profile", {})
        
        prompt = f"""
        Create a detailed teaching scenario for the following context:
        - Subject: {subject}
        - Difficulty level: {difficulty}
        - Student profile: {json.dumps(student_profile)}
        
        The scenario should include:
        - A detailed description of the teaching situation
        - Specific learning objectives
        - Relevant student background information
        - Potential challenges to anticipate
        
        Return the scenario as a JSON object with the following structure:
        {{
            "description": "detailed scenario description",
            "learning_objectives": ["objective 1", "objective 2"],
            "student_background": "relevant information about the student",
            "challenges": ["potential challenge 1", "potential challenge 2"]
        }}
        """
        
        response = self.llm_interface.get_llm_response(
            [{"role": "user", "content": prompt}],
            output_format="json"
        )
        
        try:
            scenario = json.loads(response)
            # Add the context information to the scenario
            scenario["subject"] = subject
            scenario["difficulty"] = difficulty
            return scenario
        except:
            # Fallback if parsing fails
            return {
                "subject": subject,
                "difficulty": difficulty,
                "description": "A teaching scenario about " + subject,
                "learning_objectives": ["Understand basic concepts"],
                "student_background": "Student with typical profile for their age",
                "challenges": ["Maintaining engagement"]
            }
    
    def analyze_teaching_response(self, teacher_input, context):
        """
        Analyze a teacher's response in an educational context.
        
        Args:
            teacher_input: The teacher's statement or approach
            context: Relevant contextual information about the scenario
            
        Returns:
            dict: Analysis of the teaching approach
        """
        prompt = f"""
        Analyze the following teaching response:
        
        "{teacher_input}"
        
        Context:
        {json.dumps(context, indent=2)}
        
        Provide an analysis of the teaching approach, including:
        - Effectiveness in addressing learning objectives
        - Appropriateness for the student profile
        - Strengths of the approach
        - Areas for improvement
        - Alignment with pedagogical best practices
        
        Return your analysis as a detailed assessment.
        """
        
        response = self.llm_interface.get_llm_response([{"role": "user", "content": prompt}])
        
        # Ensure response is a string
        if isinstance(response, list) and len(response) > 0:
            response = response[0]
        
        # Process the response into a structured format
        analysis_result = {
            "effectiveness_score": random.uniform(0.6, 0.9),  # Placeholder for demonstration
            "identified_strengths": [],
            "improvement_areas": [],
            "overall_assessment": response
        }
        
        # Extract structured information (this would ideally use a more sophisticated approach)
        if isinstance(response, str) and "strength" in response.lower():
            strength_section = response.lower().split("strength")[1].split("\n")[0]
            analysis_result["identified_strengths"].append(strength_section.strip())
        
        if isinstance(response, str) and "improve" in response.lower():
            improvement_section = response.lower().split("improve")[1].split("\n")[0]
            analysis_result["improvement_areas"].append(improvement_section.strip())
        
        return analysis_result
    
    def generate_student_reaction(self, teacher_input, student_profile, scenario_context):
        """
        Generate a realistic student reaction to a teacher's input.
        
        Args:
            teacher_input: The teacher's statement or question
            student_profile: Information about the student
            scenario_context: Additional context about the scenario
            
        Returns:
            str: A realistic student reaction
        """
        # Use the EnhancedDSPyLLMInterface for more sophisticated student reactions
        enhanced_llm = EnhancedDSPyLLMInterface()
        return enhanced_llm.generate_student_response(teacher_input, student_profile, scenario_context)

# Add backward compatibility for code that imports LLMInterface
# This ensures that existing code referencing the old interface will continue to work
DSPyLLMInterface = EnhancedDSPyLLMInterface 
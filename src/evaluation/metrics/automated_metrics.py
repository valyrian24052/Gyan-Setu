"""
Enhanced Automated Metrics for Teacher Response Evaluation

This module provides comprehensive automated metrics for evaluating
teaching responses in educational scenarios.
"""

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import textstat

@dataclass
class TeachingMetrics:
    """Comprehensive metrics for teaching response evaluation."""
    clarity_score: float
    engagement_score: float
    pedagogical_score: float
    emotional_support_score: float
    content_accuracy_score: float
    age_appropriateness_score: float
    overall_score: float
    detailed_feedback: Dict[str, Any]

class AutomatedMetricsEvaluator:
    """Enhanced automated metrics evaluator for teaching responses."""
    
    def __init__(self):
        """Initialize the evaluator with necessary models."""
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.nlp = spacy.load('en_core_web_sm')
        
        # Pedagogical keywords and phrases
        self.pedagogical_markers = {
            'scaffolding': ['let\'s break this down', 'step by step', 'first we\'ll', 'then we can'],
            'engagement': ['what do you think?', 'can you explain', 'try this', 'let\'s explore'],
            'reinforcement': ['great job', 'excellent work', 'you\'re getting it', 'well done'],
            'support': ['don\'t worry', 'it\'s okay to make mistakes', 'let\'s try again', 'you can do this']
        }
        
    def evaluate_response(self, 
                         teacher_response: str,
                         student_profile: Dict[str, Any],
                         context: Dict[str, Any]) -> TeachingMetrics:
        """
        Evaluate a teaching response comprehensively.
        
        Args:
            teacher_response: The teacher's response to evaluate
            student_profile: Student information including grade level, needs, etc.
            context: Teaching context including subject, topic, etc.
            
        Returns:
            TeachingMetrics: Comprehensive evaluation metrics
        """
        # Calculate individual metrics
        clarity = self._evaluate_clarity(teacher_response)
        engagement = self._evaluate_engagement(teacher_response)
        pedagogical = self._evaluate_pedagogical_approach(teacher_response, context)
        emotional = self._evaluate_emotional_support(teacher_response)
        accuracy = self._evaluate_content_accuracy(teacher_response, context)
        age_appropriate = self._evaluate_age_appropriateness(teacher_response, student_profile)
        
        # Calculate overall score with weighted components
        overall = np.mean([
            clarity * 0.2,
            engagement * 0.2,
            pedagogical * 0.25,
            emotional * 0.15,
            accuracy * 0.1,
            age_appropriate * 0.1
        ])
        
        # Generate detailed feedback
        detailed_feedback = self._generate_detailed_feedback(
            clarity, engagement, pedagogical, emotional, accuracy, age_appropriate
        )
        
        return TeachingMetrics(
            clarity_score=clarity,
            engagement_score=engagement,
            pedagogical_score=pedagogical,
            emotional_support_score=emotional,
            content_accuracy_score=accuracy,
            age_appropriateness_score=age_appropriate,
            overall_score=overall,
            detailed_feedback=detailed_feedback
        )
    
    def _evaluate_clarity(self, response: str) -> float:
        """Evaluate the clarity of the response."""
        # Use textstat for readability metrics
        readability = textstat.flesch_reading_ease(response) / 100
        
        # Analyze sentence structure
        doc = self.nlp(response)
        avg_sentence_length = np.mean([len(sent) for sent in doc.sents])
        sentence_complexity = min(1.0, avg_sentence_length / 20)  # Normalize
        
        # Combined clarity score
        clarity = np.mean([readability, 1 - sentence_complexity])
        return float(clarity)
    
    def _evaluate_engagement(self, response: str) -> float:
        """Evaluate the engagement level of the response."""
        doc = self.nlp(response)
        
        # Check for questions and interactive elements
        question_count = sum(1 for sent in doc.sents if sent.text.strip().endswith('?'))
        
        # Check for engagement markers
        engagement_phrases = sum(
            1 for phrase in self.pedagogical_markers['engagement']
            if phrase.lower() in response.lower()
        )
        
        # Normalize and combine scores
        engagement = min(1.0, (question_count * 0.3 + engagement_phrases * 0.2))
        return float(engagement)
    
    def _evaluate_pedagogical_approach(self, response: str, context: Dict[str, Any]) -> float:
        """Evaluate the pedagogical approach used."""
        # Check for scaffolding elements
        scaffolding_score = sum(
            1 for phrase in self.pedagogical_markers['scaffolding']
            if phrase.lower() in response.lower()
        ) * 0.25
        
        # Check if approach matches context
        context_alignment = self._calculate_context_alignment(response, context)
        
        # Combined score
        pedagogical = min(1.0, scaffolding_score + context_alignment)
        return float(pedagogical)
    
    def _evaluate_emotional_support(self, response: str) -> float:
        """Evaluate the emotional support in the response."""
        support_phrases = sum(
            1 for phrase in self.pedagogical_markers['support']
            if phrase.lower() in response.lower()
        )
        
        reinforcement_phrases = sum(
            1 for phrase in self.pedagogical_markers['reinforcement']
            if phrase.lower() in response.lower()
        )
        
        emotional = min(1.0, (support_phrases * 0.3 + reinforcement_phrases * 0.2))
        return float(emotional)
    
    def _evaluate_content_accuracy(self, response: str, context: Dict[str, Any]) -> float:
        """Evaluate the accuracy of the content."""
        # Compare response embedding with context embedding
        response_embedding = self.embedding_model.encode([response])
        context_text = context.get('topic_description', '')
        context_embedding = self.embedding_model.encode([context_text])
        
        similarity = cosine_similarity(response_embedding, context_embedding)[0][0]
        return float(max(0.0, min(1.0, similarity)))
    
    def _evaluate_age_appropriateness(self, response: str, student_profile: Dict[str, Any]) -> float:
        """Evaluate if the response is age-appropriate."""
        grade_level = student_profile.get('grade_level', 5)  # Default to 5th grade
        
        # Calculate reading level
        reading_level = textstat.coleman_liau_index(response) / 12  # Normalize to 0-1
        
        # Compare with expected grade level
        level_difference = abs(reading_level - (grade_level / 12))
        appropriateness = max(0.0, 1.0 - level_difference)
        
        return float(appropriateness)
    
    def _calculate_context_alignment(self, response: str, context: Dict[str, Any]) -> float:
        """Calculate how well the response aligns with the teaching context."""
        context_embedding = self.embedding_model.encode([str(context)])
        response_embedding = self.embedding_model.encode([response])
        
        alignment = cosine_similarity(context_embedding, response_embedding)[0][0]
        return float(max(0.0, min(1.0, alignment)))
    
    def _generate_detailed_feedback(self, *scores) -> Dict[str, Any]:
        """Generate detailed feedback based on scores."""
        clarity, engagement, pedagogical, emotional, accuracy, age = scores
        
        feedback = {
            "strengths": [],
            "areas_for_improvement": [],
            "recommendations": []
        }
        
        # Add feedback based on scores
        if clarity > 0.8:
            feedback["strengths"].append("Excellent clarity in explanation")
        elif clarity < 0.6:
            feedback["areas_for_improvement"].append("Could be clearer")
            feedback["recommendations"].append("Try using simpler sentences")
            
        if engagement > 0.8:
            feedback["strengths"].append("Highly engaging approach")
        elif engagement < 0.6:
            feedback["areas_for_improvement"].append("Could be more interactive")
            feedback["recommendations"].append("Add more questions and activities")
            
        # Add more feedback for other metrics...
        
        return feedback 
import os
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
import torch

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize the sentence transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class TeacherBot:
    def __init__(self):
        self.model = "gpt-3.5-turbo"  # Using a smaller model for cost-effectiveness
        
    def generate_student_query(self, scenario, personality, tone):
        """Generate a student query based on the given parameters."""
        prompt = f"""
        You are simulating an elementary school student.
        Scenario: {scenario}
        Personality: {personality}
        Emotional Tone: {tone}
        Generate a realistic question as the student.
        """
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a student asking questions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    
    def evaluate_response(self, teacher_response, expected_response):
        """Evaluate the teacher's response using semantic similarity."""
        if not teacher_response or not expected_response:
            return 0.0
            
        # Generate embeddings
        teacher_embedding = embedding_model.encode(teacher_response, convert_to_tensor=True)
        expected_embedding = embedding_model.encode(expected_response, convert_to_tensor=True)
        
        # Calculate cosine similarity
        similarity = util.cos_sim(teacher_embedding, expected_embedding)
        
        return float(similarity[0][0]) * 100
    
    def get_feedback(self, similarity_score, teacher_response, expected_response):
        """Generate feedback based on the similarity score."""
        prompt = f"""
        Teacher's Response: {teacher_response}
        Expected Response: {expected_response}
        Similarity Score: {similarity_score:.2f}%
        
        Provide brief, constructive feedback on the teacher's response.
        """
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an educational coach providing feedback."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()

# Example personalities and tones for variety
PERSONALITIES = [
    "Curious",
    "Shy",
    "Energetic",
    "Analytical",
    "Creative"
]

TONES = [
    "Excited",
    "Confused",
    "Worried",
    "Interested",
    "Frustrated"
] 
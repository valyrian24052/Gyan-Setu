import os
from llama_cpp import Llama
from config import MODEL_PATH, MODEL_N_CTX, MODEL_N_THREADS, MODEL_N_GPU_LAYERS

# Initialize Llama model
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=MODEL_N_CTX,
    n_threads=MODEL_N_THREADS,
    n_gpu_layers=MODEL_N_GPU_LAYERS
)

def get_completion(prompt, max_tokens=100, temperature=0.7):
    """
    Get a completion from the Llama model.
    
    Args:
        prompt (str): The input prompt
        max_tokens (int): Maximum number of tokens to generate
        temperature (float): Controls randomness in generation
        
    Returns:
        str: The generated response
    """
    try:
        response = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["</s>"],  # Llama's end of sequence token
            echo=False
        )
        return response['choices'][0]['text']
    except Exception as e:
        print(f"Error in model completion: {e}")
        return None

def generate_response(scenario, personality="helpful", tone="professional"):
    """
    Generate a teacher's response to a student scenario.
    
    Args:
        scenario (str): The student's question or situation
        personality (str): The desired personality trait
        tone (str): The desired tone of response
        
    Returns:
        str: The generated teacher response
    """
    prompt = f"""<s>[INST]You are a teacher with a {personality} personality.
    Respond to this student scenario in a {tone} tone:
    
    {scenario}[/INST]"""
    
    return get_completion(prompt)

def evaluate_response(actual_response, expected_response, threshold=0.7):
    """
    Evaluate a teacher's response against expected response.
    
    Args:
        actual_response (str): The actual response given
        expected_response (str): The ideal response
        threshold (float): Minimum similarity score
        
    Returns:
        tuple: (bool, float) - (passed threshold, similarity score)
    """
    # Implementation using sentence-transformers for similarity
    # This part remains unchanged as it doesn't use OpenAI
    pass

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
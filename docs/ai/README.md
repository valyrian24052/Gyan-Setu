# AI Integration Documentation

## Overview

The AI component of the Teacher Training Chatbot uses OpenAI's GPT models for natural language processing and sentence transformers for semantic similarity calculations.

## Components

### 1. Language Model Integration

#### OpenAI GPT Setup
```python
from openai import OpenAI
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a student asking questions."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=100,
    temperature=0.7
)
```

#### Prompt Engineering
```python
def generate_student_query(scenario, personality, tone):
    prompt = f"""
    You are simulating an elementary school student.
    Scenario: {scenario}
    Personality: {personality}
    Emotional Tone: {tone}
    Generate a realistic question as the student.
    """
    return get_completion(prompt)
```

### 2. Semantic Similarity

#### Sentence Transformer Setup
```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_similarity(text1, text2):
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)
    return float(util.cos_sim(embedding1, embedding2)[0][0])
```

### 3. Response Generation

#### Student Query Generation
```python
def generate_student_query(scenario, personality, tone):
    messages = [
        {"role": "system", "content": "You are a student with specific traits."},
        {"role": "user", "content": f"Generate a question about {scenario} ..."}
    ]
    return get_completion(messages)
```

#### Response Evaluation
```python
def evaluate_response(teacher_response, expected_response):
    similarity = calculate_similarity(teacher_response, expected_response)
    feedback = generate_feedback(similarity, teacher_response, expected_response)
    return similarity, feedback
```

## Implementation Guidelines

### 1. Error Handling

```python
def safe_completion(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return get_completion(prompt)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(1)
```

### 2. Rate Limiting

```python
from ratelimit import limits, sleep_and_retry

@sleep_and_retry
@limits(calls=60, period=60)
def rate_limited_completion(prompt):
    return get_completion(prompt)
```

### 3. Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_embedding(text):
    return model.encode(text, convert_to_tensor=True)
```

## Best Practices

### 1. Prompt Engineering

- Be specific and clear
- Include context
- Use consistent formatting
- Handle edge cases
- Validate outputs

### 2. Model Selection

- GPT-3.5-turbo for general responses
- Sentence transformers for embeddings
- Consider cost vs. performance
- Monitor token usage

### 3. Performance Optimization

- Batch processing
- Response caching
- Embedding precomputation
- Asynchronous processing

## Security Considerations

### 1. Input Validation

```python
def sanitize_input(text):
    # Remove potentially harmful content
    return clean_text(text)
```

### 2. Output Filtering

```python
def filter_response(response):
    # Ensure response meets guidelines
    return validate_content(response)
```

### 3. API Key Management

- Use environment variables
- Rotate keys regularly
- Monitor usage
- Set up alerts

## Monitoring and Logging

### 1. Performance Metrics

```python
def log_completion_metrics(start_time, end_time, tokens):
    duration = end_time - start_time
    log_metric("completion_duration", duration)
    log_metric("tokens_used", tokens)
```

### 2. Error Tracking

```python
def log_ai_error(error, context):
    logger.error(f"AI Error: {error}", extra={
        "context": context,
        "timestamp": datetime.utcnow()
    })
```

## Testing

### 1. Unit Tests

```python
def test_response_generation():
    response = generate_student_query(
        scenario="Math problem",
        personality="Curious",
        tone="Confused"
    )
    assert len(response) > 0
```

### 2. Integration Tests

```python
def test_end_to_end_interaction():
    query = generate_student_query(scenario)
    response = get_teacher_response(query)
    similarity = evaluate_response(response)
    assert similarity > 0.5
```

## Deployment

### 1. Environment Setup

```bash
export OPENAI_API_KEY="your-key-here"
export MODEL_CACHE_DIR="/path/to/cache"
```

### 2. Model Management

```python
def load_models():
    global embedding_model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
```

## Troubleshooting

### Common Issues

1. API Rate Limits
2. Token Usage
3. Response Quality
4. Performance Issues

### Solutions

1. Implement retries
2. Monitor usage
3. Adjust prompts
4. Optimize caching 
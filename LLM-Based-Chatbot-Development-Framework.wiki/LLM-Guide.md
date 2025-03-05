# LLM Integration Guide

[![Status](https://img.shields.io/badge/status-complete-green.svg)]() 
[![Last Updated](https://img.shields.io/badge/last%20updated-April%202024-blue.svg)]()

## Introduction to LLMs in the Framework

This guide covers how to integrate various Large Language Models (LLMs) into your chatbot applications using our framework. The framework supports both cloud-based LLMs (like OpenAI and Anthropic) and local LLMs that can be run on your own hardware.

### What are LLMs?
Large Language Models (LLMs) are advanced AI systems trained on vast amounts of text data. They represent a breakthrough in natural language processing, enabling sophisticated language understanding and generation capabilities.

### Understanding Tokens and Context Windows

#### What are Tokens?
Tokens are the basic units that LLMs use to process text. They are not exactly words, but rather pieces of words that the model understands. Here's how tokenization works:

- A word might be split into multiple tokens
- Common words might be single tokens
- Rare words might be split into several tokens
- Special characters and spaces are also tokens

Examples:
```text
"Hello"          → 1 token
"Supercalifragilisticexpialidocious" → 5-7 tokens (depending on the tokenizer)
"The quick brown fox jumps over the lazy dog" → ~8-10 tokens
```

#### Context Windows
The context window is the total amount of text an LLM can "see" at once:

- It includes both the input (prompt) and the generated output
- Limited by model architecture (ranges from 4K to 200K+ tokens)
- Critical for knowledge retrieval applications

## LLM Integration Options

The framework provides multiple ways to integrate LLMs into your application:

### 1. Cloud-Based LLM Providers

The framework includes built-in support for major LLM providers:

```python
from src.llm.handlers.openai_handler import OpenAIHandler
from src.llm.handlers.anthropic_handler import AnthropicHandler

# OpenAI integration
openai_llm = OpenAIHandler(
    model="gpt-4",
    temperature=0.7,
    max_tokens=1000,
    api_key="your_api_key"  # Or use environment variable
)

# Anthropic integration
anthropic_llm = AnthropicHandler(
    model="claude-3-opus-20240229",
    temperature=0.7,
    max_tokens=1000,
    api_key="your_api_key"  # Or use environment variable
)
```

### 2. Local LLM Integration

For privacy, reduced costs, or offline usage, the framework supports local LLMs:

```python
from src.llm.handlers.local_llm_handler import LocalLLMHandler

# Local LLM integration using llama.cpp
local_llm = LocalLLMHandler(
    model_path="/path/to/model.gguf",
    context_size=4096,
    gpu_layers=0  # Set higher for GPU acceleration
)

# Inference with local model
response = local_llm.generate(
    prompt="Explain the concept of vector databases",
    max_tokens=500
)
```

### 3. LLM Provider Abstraction

The framework offers a unified interface for working with different LLM providers:

```python
from src.llm.handler import get_llm_handler

# Get the appropriate handler based on configuration
llm = get_llm_handler(
    provider="openai",  # or "anthropic", "local", etc.
    model="gpt-3.5-turbo",
    temperature=0.7
)

# Use the handler with a consistent API
response = llm.generate(
    prompt="Explain knowledge retrieval in chatbots",
    system_message="You are an expert in AI systems"
)
```

## Advanced LLM Techniques

### 1. DSPy Integration

The framework integrates Stanford's DSPy for declarative and self-improving prompting:

```python
from src.llm.dspy.handler import DSPyLLMHandler
import dspy

# Initialize DSPy with the framework
dspy_handler = DSPyLLMHandler(
    model_name="gpt-3.5-turbo",
    temperature=0.7
)

# Define a DSPy module
class KnowledgeQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=3)
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")
    
    def forward(self, question):
        context = self.retrieve(question).passages
        answer = self.generate_answer(context=context, question=question)
        return answer

# Use DSPy with the framework
qa_module = KnowledgeQA()
answer = dspy_handler.predict_with_module(
    qa_module,
    question="How do vector databases improve chatbot responses?"
)
```

### 2. Prompt Engineering

The framework includes a comprehensive prompt management system:

```python
from src.llm.prompts.prompt_template import PromptTemplate
from src.llm.handlers.openai_handler import OpenAIHandler

# Define a prompt template
qa_template = PromptTemplate(
    template="""
    Answer the following question based on the context provided.
    
    Context: {context}
    
    Question: {question}
    
    Answer: 
    """,
    input_variables=["context", "question"]
)

# Use the template with an LLM
llm = OpenAIHandler(model="gpt-4")
response = llm.generate(
    prompt=qa_template.format(
        context="Vector databases store text as numerical representations...",
        question="How do vector databases work?"
    )
)
```

### 3. Streaming Responses

For real-time interactions, the framework supports streaming responses:

```python
from src.llm.handlers.openai_handler import OpenAIHandler

# Initialize with streaming enabled
llm = OpenAIHandler(
    model="gpt-4",
    streaming=True
)

# Stream the response
for chunk in llm.generate_stream(
    prompt="Explain the benefits of retrieval augmented generation",
    max_tokens=1000
):
    print(chunk, end="", flush=True)
```

## LLM Performance Optimization

### 1. Prompt Caching

The framework implements efficient caching to reduce API calls:

```python
from src.llm.handlers.cached_handler import CachedLLMHandler
from src.llm.handlers.openai_handler import OpenAIHandler

# Create a cached version of any LLM handler
cached_llm = CachedLLMHandler(
    base_handler=OpenAIHandler(model="gpt-3.5-turbo"),
    cache_dir="./cache",
    ttl=3600  # Cache for 1 hour
)

# Use like any other handler
response = cached_llm.generate(prompt="What is retrieval augmented generation?")
```

### 2. Batch Processing

For processing multiple items efficiently:

```python
from src.llm.handlers.openai_handler import OpenAIHandler

llm = OpenAIHandler(model="gpt-3.5-turbo")

questions = [
    "What are vector embeddings?",
    "How do transformers work?",
    "What is attention in neural networks?"
]

# Process in a batch
responses = llm.generate_batch(
    prompts=questions,
    max_tokens=200,
    batch_size=3  # Number of concurrent requests
)
```

### 3. Model Selection and Fallbacks

The framework supports automatic model fallbacks:

```python
from src.llm.handlers.fallback_handler import FallbackLLMHandler
from src.llm.handlers.openai_handler import OpenAIHandler
from src.llm.handlers.anthropic_handler import AnthropicHandler

# Create a fallback chain of LLMs
fallback_llm = FallbackLLMHandler(
    handlers=[
        OpenAIHandler(model="gpt-4"),  # Primary model
        OpenAIHandler(model="gpt-3.5-turbo"),  # Fallback #1
        AnthropicHandler(model="claude-instant-1.2")  # Fallback #2
    ],
    max_retries=3
)

# Use with automatic fallback on failure
response = fallback_llm.generate(
    prompt="Explain knowledge retrieval systems"
)
```

## Case Study: UTTA LLM Integration

The Utah Teacher Training Assistant (UTTA) project demonstrates advanced LLM integration:

```python
from src.llm.handlers.openai_handler import OpenAIHandler
from src.llm.prompts.prompt_template import PromptTemplate
from src.retrieval.llama_index.integration import LlamaIndexRetriever

# Initialize components
llm = OpenAIHandler(
    model="gpt-4",
    temperature=0.7,
    system_message="You are an expert teacher mentor."
)

retriever = LlamaIndexRetriever(vector_db_path="data/teaching_knowledge")

# Define teaching scenario generation template
scenario_template = PromptTemplate(
    template="""
    Create a realistic classroom scenario for a grade {grade} teacher.
    The scenario should involve {topic} and be at a {difficulty} difficulty level.
    
    Include:
    1. A detailed description of the classroom situation
    2. Student behavior and characteristics
    3. Educational objectives relevant to the scenario
    4. Key challenges the teacher needs to address
    """,
    input_variables=["grade", "topic", "difficulty"]
)

# Generate a teaching scenario
scenario = llm.generate(
    prompt=scenario_template.format(
        grade=5,
        topic="classroom disruption",
        difficulty="challenging"
    )
)

# Evaluate a teacher's response using knowledge retrieval
def evaluate_teaching_response(teacher_response, scenario):
    # Retrieve relevant teaching principles
    knowledge = retriever.retrieve(
        query=f"teaching strategies for {scenario}",
        limit=3
    )
    
    # Create evaluation prompt with retrieved knowledge
    evaluation_prompt = f"""
    Scenario: {scenario}
    
    Teacher's Response: {teacher_response}
    
    Relevant Teaching Principles:
    {knowledge}
    
    Based on educational best practices, evaluate this teacher's response on:
    1. Effectiveness (1-10)
    2. Student-centeredness (1-10)
    3. Alignment with teaching principles (1-10)
    
    Provide specific feedback for improvement.
    """
    
    # Generate evaluation
    return llm.generate(evaluation_prompt)
```

## Best Practices for LLM Integration

1. **API Key Security**
   - Store keys in environment variables
   - Use the framework's secure key management
   - Avoid hardcoding keys in source code

2. **Error Handling**
   - Implement robust retry mechanisms for API failures
   - Set appropriate timeouts
   - Use the framework's fallback handlers

3. **Cost Optimization**
   - Use caching for repeated queries
   - Implement token counting to optimize prompts
   - Choose appropriate models for different tasks

4. **Performance Monitoring**
   - Track latency and reliability
   - Monitor token usage
   - Analyze response quality

5. **Prompt Engineering**
   - Use the framework's template system
   - Implement systematic prompt testing
   - Maintain a library of effective prompts

By following this guide, you can effectively integrate various LLMs into your applications using the framework's comprehensive support for both cloud and local models. 
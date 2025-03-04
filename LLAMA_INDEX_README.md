# LlamaIndex Integration for UTTA

This document provides an overview of the LlamaIndex integration with the UTTA (Universal Teacher's Teaching Assistant) platform.

## Overview

LlamaIndex is a data framework for LLM-based applications to ingest, structure, and access private or domain-specific data. This integration allows UTTA to leverage educational documents and resources to enhance its knowledge base and provide more accurate and contextually relevant responses to educational queries.

## Integration with UTTA

The LlamaIndex integration serves as a complementary knowledge retrieval system to UTTA's existing vector database system described in the main README. Key integration points include:

1. **Enhanced Knowledge Retrieval**: Provides a modern, flexible retrieval system using LlamaIndex's advanced capabilities.
2. **Multiple LLM Support**: Supports OpenAI, Anthropic, and local models through a unified interface.
3. **Configurable Parameters**: Centralized configuration system allows easy tuning for different use cases.
4. **Web Application Ready**: Drop-in utilities for Flask or FastAPI integration.
5. **Production Testing Tools**: Comprehensive testing and benchmarking tools for production deployment.

This integration can work alongside the existing vector database system (`vector_database.py`) or as a complete replacement, providing more advanced retrieval capabilities and LLM integration.

## Architecture

The integration consists of several components:

1. **LlamaIndexKnowledgeManager** (`llama_index_integration.py`): Core class managing the knowledge base, document indexing, and query processing.
2. **Configuration Management** (`llama_index_config.py`): Centralized configuration for all LlamaIndex settings.
3. **Web Application Integration** (`web_app_llama_integration.py`): Utilities to integrate the knowledge base with web frameworks.
4. **Production Testing** (`test_production_llama_index.py`): Comprehensive tools for testing and benchmarking.
5. **Environment Configuration** (`.env.example`): Template for setting up environment variables.

## Setup Instructions

### 1. Install Dependencies

```bash
pip install llama-index openai anthropic sentence-transformers python-dotenv
```

For GPU acceleration (recommended for production):
```bash
pip install torch torchvision
```

### 2. Configure Environment

Copy the example environment file and fill in your values:

```bash
cp .env.example .env
```

Edit the `.env` file with your API keys and configuration preferences.

### 3. Prepare Knowledge Base

Place educational documents (PDF, TXT, DOCX, etc.) in the knowledge base directory:

```bash
mkdir -p knowledge_base/books
# Copy your documents to knowledge_base/books/
```

### 4. Basic Usage

```python
from llama_index_integration import LlamaIndexKnowledgeManager

# Initialize the knowledge manager
knowledge_manager = LlamaIndexKnowledgeManager(
    documents_dir="knowledge_base/books",
    llm_provider="openai"  # or "anthropic" or "local"
)

# Load or create the index
knowledge_manager.load_or_create_index()

# Query the knowledge base
result = knowledge_manager.query_knowledge(
    "What are effective strategies for teaching fractions?",
    top_k=5
)

# Print the response
print(result["response"])

# Print the sources
for source in result["sources"]:
    print(f"Source: {source['source']}")
    print(f"Text: {source['text'][:200]}...")
```

### 5. Web Application Integration

For Flask:

```python
from flask import Flask, request, jsonify
from web_app_llama_integration import init_knowledge_manager, query_educational_knowledge

app = Flask(__name__)

@app.route('/api/knowledge/query', methods=['POST'])
def api_knowledge_query():
    data = request.json
    query = data.get('query', '')
    top_k = data.get('top_k', None)
    
    if not query:
        return jsonify({
            "success": False,
            "error": "No query provided",
            "message": "Please provide a query"
        }), 400
    
    result = query_educational_knowledge(query, top_k)
    return jsonify(result)

if __name__ == '__main__':
    # Initialize the knowledge manager on startup
    init_knowledge_manager()
    app.run(debug=True)
```

For FastAPI:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from web_app_llama_integration import init_knowledge_manager, query_educational_knowledge

app = FastAPI()

class KnowledgeQuery(BaseModel):
    query: str
    top_k: Optional[int] = None

@app.post("/api/knowledge/query")
async def api_knowledge_query(query_data: KnowledgeQuery):
    if not query_data.query:
        raise HTTPException(status_code=400, detail="No query provided")
    
    result = query_educational_knowledge(query_data.query, query_data.top_k)
    
    if not result.get("success", False):
        raise HTTPException(status_code=500, detail=result.get("message", "Unknown error"))
    
    return result

@app.on_event("startup")
async def startup_event():
    # Initialize the knowledge manager on startup
    init_knowledge_manager()
```

## Key Components

### 1. LlamaIndexKnowledgeManager

The core component that manages the knowledge base:

```python
# Main methods:
knowledge_manager = LlamaIndexKnowledgeManager(documents_dir="knowledge_base/books")
knowledge_manager.load_or_create_index()  # Load or create the index
result = knowledge_manager.query_knowledge("How do I teach fractions?")  # Query the knowledge base
knowledge_manager.clear_cache()  # Clear the query cache
```

### 2. Configuration System

Centralized configuration through environment variables and the `llama_index_config.py` file:

```python
# Access configuration:
from llama_index_config import get_llm_settings, get_document_settings

# Get LLM-specific settings
llm_settings = get_llm_settings(provider="openai")

# Get document processing settings
doc_settings = get_document_settings()

# Get all settings
all_settings = get_all_settings()
```

### 3. Web Application Integration

Utilities for integrating with web frameworks:

```python
# Initialize the knowledge manager:
from web_app_llama_integration import init_knowledge_manager, query_educational_knowledge

# Initialize once at application startup
init_knowledge_manager()

# Use the decorator to inject the knowledge manager
@with_knowledge_manager
def my_handler(query, knowledge_manager=None):
    result = knowledge_manager.query_knowledge(query)
    return result
```

### 4. Production Testing

The `test_production_llama_index.py` script provides comprehensive testing tools:

```bash
# Print current configuration
python test_production_llama_index.py --config

# Interactive testing mode
python test_production_llama_index.py --interactive

# Benchmark different providers
python test_production_llama_index.py --benchmark --query "How do I teach fractions?"

# Test a specific query with a specific provider
python test_production_llama_index.py --query "How do I manage a classroom?" --provider openai
```

## Configuration Options

The integration is highly configurable through the `llama_index_config.py` file and environment variables:

| Setting | Environment Variable | Default | Description |
|---------|---------------------|---------|-------------|
| LLM Provider | `LLAMA_INDEX_LLM_PROVIDER` | `openai` | The LLM provider to use (openai, anthropic, local) |
| OpenAI Model | `OPENAI_MODEL` | `gpt-3.5-turbo` | OpenAI model to use |
| Anthropic Model | `ANTHROPIC_MODEL` | `claude-3-sonnet-20240229` | Anthropic model to use |
| Embedding Model | `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Model for generating embeddings |
| Chunk Size | `LLAMA_INDEX_CHUNK_SIZE` | `500` | Size of text chunks for indexing |
| Chunk Overlap | `LLAMA_INDEX_CHUNK_OVERLAP` | `50` | Overlap between text chunks |
| Enable Caching | `LLAMA_INDEX_ENABLE_CACHE` | `true` | Whether to enable query caching |
| Cache TTL | `LLAMA_INDEX_CACHE_TTL` | `3600` | Cache time-to-live in seconds |
| Top K | `LLAMA_INDEX_TOP_K` | `5` | Number of top results to retrieve |
| Similarity Cutoff | `LLAMA_INDEX_SIMILARITY_CUTOFF` | `0.7` | Minimum similarity score for results |

## Integration with Existing UTTA Components

The LlamaIndex integration can be combined with UTTA's existing knowledge management system in several ways:

### 1. Complementary Knowledge Retrieval

The LlamaIndex integration can work alongside the existing vector database (`vector_database.py`):

```python
# Example of using both systems
from llama_index_integration import LlamaIndexKnowledgeManager
from vector_database import VectorDatabase

# Initialize both knowledge systems
llama_manager = LlamaIndexKnowledgeManager()
vector_db = VectorDatabase()

# Query both systems and combine results
llama_results = llama_manager.query_knowledge("How do I teach fractions?")
vector_results = vector_db.search("How do I teach fractions?")

# Combine or compare results as needed
```

### 2. Integration with AI Agent

The LlamaIndex integration can be used with the existing AI agent:

```python
# Example integration with AI agent
from ai_agent import AI_Agent
from llama_index_integration import LlamaIndexKnowledgeManager

# Initialize the knowledge manager
knowledge_manager = LlamaIndexKnowledgeManager()
knowledge_manager.load_or_create_index()

# Create AI agent with LlamaIndex knowledge
agent = AI_Agent(knowledge_source="llama_index", 
                 knowledge_manager=knowledge_manager)

# Process queries with LlamaIndex-enhanced knowledge
response = agent.process_query("How do I improve classroom management?")
```

### 3. DSPy Integration

The LlamaIndex integration can be combined with DSPy for enhanced capabilities:

```python
# Example DSPy integration
from dspy_llm_handler import DSPyHandler
from llama_index_integration import LlamaIndexKnowledgeManager

# Initialize components
knowledge_manager = LlamaIndexKnowledgeManager()
dspy_handler = DSPyHandler()

# Use LlamaIndex to retrieve knowledge
knowledge = knowledge_manager.query_knowledge("How do I teach fractions?")

# Process with DSPy
enhanced_response = dspy_handler.process_with_knowledge(
    query="How do I teach fractions?",
    knowledge=knowledge["sources"]
)
```

## Performance Considerations

For best performance:

1. **Use GPU acceleration** for embedding generation:
   - Ensure you have a CUDA-compatible GPU and PyTorch installed with CUDA support
   - Set appropriate batch sizes for your GPU memory

2. **Enable caching** to speed up repeated queries:
   - The system includes a built-in caching mechanism with TTL (time-to-live)
   - Configure cache settings in your `.env` file
   - Use `clear_knowledge_cache()` to reset when needed

3. **Use appropriate chunk sizes** for your documents:
   - Smaller chunks (300-500 words) for precise retrieval
   - Larger chunks (500-1000 words) for more context
   - Adjust overlap to maintain coherence between chunks

4. **Optimize retrieval parameters**:
   - Adjust the top_k value based on your needs (higher for more diverse sources)
   - Tune the similarity cutoff based on your document collection
   - Consider filtering by category for more targeted results

5. **LLM Provider selection**:
   - OpenAI typically provides faster responses
   - Claude (Anthropic) often has better comprehension of educational concepts
   - Local models offer privacy but may require optimized parameters

## Troubleshooting

Common issues:

1. **API Key Errors**: 
   - Ensure your API keys are correctly set in the `.env` file
   - Check for any whitespace or invalid characters
   - Verify your API key subscription is active

2. **Document Loading Errors**: 
   - Check file formats and permissions
   - Ensure the paths to your documents are correct
   - For PDFs, try pre-converting to text if extraction fails

3. **GPU Acceleration**: 
   - If using GPU, ensure PyTorch is properly installed with CUDA support
   - Run `python -c "import torch; print(torch.cuda.is_available())"` to verify

4. **Memory Issues**: 
   - For large document collections, consider increasing chunk size and reducing overlap
   - Use streaming response features when available
   - Process documents in batches if necessary

5. **Index Corruption**:
   - If you encounter index errors, try deleting the index directory and rebuilding
   - Use `force_reload=True` to recreate the index from scratch

## Security Considerations

1. **API Keys**: 
   - Never commit API keys to version control
   - Use environment variables or secure secrets management
   - Consider using key rotation for production systems

2. **User Data**: 
   - Be cautious about what user data is sent to LLM providers
   - Consider using local models for sensitive information
   - Implement data filtering to prevent PII transmission

3. **Rate Limiting**: 
   - Implement rate limiting to prevent abuse and control costs
   - Monitor API usage to avoid unexpected charges
   - Consider implementing request throttling in production

4. **Model Output Safety**:
   - Implement content filtering for potentially harmful responses
   - Consider a human-in-the-loop approach for educational content
   - Log and review model outputs regularly

## Testing

You can test the LlamaIndex integration with the MockLLM provider:

```python
from llama_index_integration import demonstrate_llama_index

# This will use MockLLM instead of a real provider
demonstrate_llama_index()
```

For more comprehensive testing, use the provided test script:

```bash
# Run interactive test mode
python test_production_llama_index.py --interactive

# Benchmark multiple providers
python test_production_llama_index.py --benchmark
```

## Further Resources

- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [Anthropic API Documentation](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [UTTA Main Documentation](README.md) 
# LLM-Based Chatbot Development

## Overview

This repository provides a comprehensive framework for developing advanced LLM-based chatbots with sophisticated knowledge retrieval capabilities. It demonstrates how to build domain-specific AI assistants that leverage both general LLM capabilities and specialized knowledge bases.

The repository includes a complete case study of an **Enhanced Teacher Training System** - a practical implementation of the framework for educational purposes. This system provides realistic classroom management scenarios for elementary education teachers, showcasing how domain-specific knowledge can be integrated with LLMs to create powerful, context-aware applications.

## Key Features

- **Document Processing Pipeline**: Process various document formats (PDF, EPUB, TXT, JSON) to extract domain knowledge
- **Vector Knowledge Base**: Store and retrieve information using semantic search in a SQLite vector database
- **Multi-Provider LLM Support**: Integrate with OpenAI, Anthropic, and local LLM models
- **DSPy Integration**: Leverage Stanford's DSPy framework for self-improving prompts and declarative programming
- **LlamaIndex Integration**: Utilize modern knowledge retrieval systems with configurable parameters
- **Performance Tracking**: Monitor which knowledge is most effective in different situations
- **Web Application Integration**: Ready-to-use utilities for Flask and FastAPI
- **Production Testing Tools**: Comprehensive testing and benchmarking tools

## Directory Structure

```
llm-chatbot-framework/
├── data/                      # Data storage
│   └── sample_documents/      # Sample educational documents
├── examples/                  # Example applications
│   ├── simple_chatbot.py      # Simple chatbot example
│   └── teacher_training/      # Teacher training system
├── src/                       # Core framework code
│   ├── core/                  # Core components
│   │   ├── document_processor.py # Document processing utilities
│   │   └── vector_database.py # Vector database implementation
│   ├── llm/                   # LLM integration
│   │   ├── dspy/              # DSPy integration
│   │   ├── handlers/          # LLM handlers
│   │   └── prompts/           # Prompt templates
│   ├── retrieval/             # Retrieval components
│   │   ├── basic/             # Basic retrieval methods
│   │   └── llama_index/       # LlamaIndex integration
│   ├── utils/                 # Utility functions
│   └── web/                   # Web application components
│       └── static/            # Static web assets
├── tests/                     # Unit and integration tests
│   ├── core/                  # Tests for core components
│   ├── examples/              # Tests for examples
│   ├── llm/                   # Tests for LLM components
│   └── retrieval/             # Tests for retrieval components
├── tools/                     # Utility tools
│   ├── knowledge_analyzer.py  # Knowledge base analysis tool
│   ├── performance_testing/   # Performance testing tools
│   └── setup/                 # Setup scripts
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/llm-chatbot-framework.git
   cd llm-chatbot-framework
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the root directory with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   ```

## Usage

### Teacher Training System Example

The Teacher Training System is an example application that demonstrates how to use the framework to create a system that helps teachers practice classroom scenarios.

To run the Teacher Training System:

```bash
python -m examples.teacher_training.run
```

This will:
1. Process educational documents in `data/sample_documents/`
2. Build a knowledge base
3. Start an interactive session where you can practice responding to classroom scenarios

### Creating Your Own Chatbot

To create your own chatbot using this framework:

1. Create a new directory in the `examples` folder
2. Import the necessary components from the framework
3. Customize the behavior for your specific use case

Example:

```python
from src.core.vector_database import VectorDatabase
from src.llm.dspy.handler import DSPyLLMHandler
from src.core.document_processor import DocumentProcessor

# Initialize components
vector_db = VectorDatabase()
llm_handler = DSPyLLMHandler(model_name="gpt-3.5-turbo")
document_processor = DocumentProcessor()

# Process your documents
# ...

# Create your chatbot logic
# ...
```

## Components

### Vector Database

The `VectorDatabase` class provides a way to store and retrieve documents using vector embeddings. It supports multiple embedding providers and persistence options.

```python
from src.core.vector_database import VectorDatabase

# Initialize with default settings
db = VectorDatabase()

# Add documents
db.add_document("Document text", {"source": "example.pdf"})

# Search for similar documents
results = db.search("Query text")
```

### DSPy LLM Handler

The `DSPyLLMHandler` class provides a unified interface for interacting with language models using DSPy.

```python
from src.llm.dspy.handler import DSPyLLMHandler

# Initialize with default settings
handler = DSPyLLMHandler()

# Generate text
response = handler.generate("Explain the concept of differentiated instruction.")

# Answer a question
answer = handler.answer_question("What are effective classroom management strategies?")
```

### Document Processor

The `DocumentProcessor` class provides utilities for processing text from various sources.

```python
from src.core.document_processor import DocumentProcessor

# Initialize
processor = DocumentProcessor()

# Extract text from a PDF
chunks = processor.extract_text_from_pdf("document.pdf")

# Extract keywords
keywords = processor.extract_keywords(text)
```

### LlamaIndex Integration

The framework includes integration with LlamaIndex for advanced document retrieval:

```python
from src.retrieval.llama_index.integration import LlamaIndexRetriever

# Initialize
retriever = LlamaIndexRetriever()

# Index documents
retriever.index_documents("path/to/documents")

# Query the index
results = retriever.query("What are effective teaching strategies?")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the [MIT License](./LICENSE).
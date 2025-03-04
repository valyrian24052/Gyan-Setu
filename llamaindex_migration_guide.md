# LlamaIndex Migration Guide

This guide provides instructions for transitioning from the custom vector database and document processing components to LlamaIndex.

## Why Transition to LlamaIndex?

LlamaIndex provides several advantages over the custom implementation:

1. **More advanced retrieval methods**: Beyond basic vector similarity, LlamaIndex offers hybrid search, reranking, and structured data queries.
2. **Better integration with LLMs**: Built-in methods for routing queries, summarization, and integration with various LLM providers.
3. **Enhanced document processing**: Better handling of various document types and advanced chunking strategies.
4. **Long-term maintenance**: Continually updated by a large community rather than requiring in-house maintenance.
5. **Ecosystem integration**: Easily integrates with other tools like LangChain.

## Migration Steps

### 1. Install LlamaIndex

Update your environment with LlamaIndex:

```bash
# If using conda with environment.yml (already updated)
conda env update -f environment.yml

# Or with pip
pip install llama-index
```

### 2. Update Document Processing

Replace the `document_processor.py` usage with LlamaIndex's document loaders:

**Before:**
```python
from document_processor import DocumentProcessor

processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
processor.process_books("knowledge_base/books")
chunks = processor.get_chunks()
```

**After:**
```python
from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser

# Load documents
documents = SimpleDirectoryReader("knowledge_base/books").load_data()

# Parse documents into nodes (similar to chunks)
node_parser = SimpleNodeParser.from_defaults(
    chunk_size=500, 
    chunk_overlap=100
)
nodes = node_parser.get_nodes_from_documents(documents)
```

### 3. Update Vector Database Usage

Replace the `vector_database.py` usage with LlamaIndex's vector store:

**Before:**
```python
from vector_database import VectorDatabase

db = VectorDatabase()
db.add_chunks(chunks, category="general")
results = db.search("teaching fractions", top_k=5)
```

**After:**
```python
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.embeddings import HuggingFaceEmbedding

# Setup embedding model (similar to what was used before)
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
service_context = ServiceContext.from_defaults(embed_model=embed_model)

# Create vector index
index = VectorStoreIndex.from_documents(
    documents, 
    service_context=service_context
)

# Create query engine and search
query_engine = index.as_query_engine()
response = query_engine.query("teaching fractions")
```

### 4. Convert Existing Data (Optional)

If you have existing data in the custom vector database, you can migrate it:

```python
from llama_index.schema import Document
from vector_database import VectorDatabase
from llama_index import VectorStoreIndex

# Load existing chunks
vector_db = VectorDatabase()
chunks = vector_db.get_chunks_by_category("general", limit=10000)

# Convert to LlamaIndex documents
documents = []
for chunk in chunks:
    doc = Document(
        text=chunk["text"],
        metadata=chunk.get("metadata", {})
    )
    documents.append(doc)

# Create new index with existing documents
index = VectorStoreIndex.from_documents(documents)
```

### 5. Update Query Integration

Update how queries are processed:

**Before:**
```python
def get_relevant_knowledge(query):
    results = db.search(query, top_k=5)
    knowledge = "\n".join([r["text"] for r in results])
    return knowledge
```

**After:**
```python
def get_relevant_knowledge(query):
    response = query_engine.query(query)
    # The response object contains both the answer and source nodes
    return str(response), response.source_nodes
```

### 6. Advanced Features

Once migrated, you can take advantage of more advanced LlamaIndex features:

#### Structured Data Queries

```python
from llama_index.indices.struct_store import SimpleStructStoreIndex
from llama_index.schema import TextNode

# Create structured data
nodes = [
    TextNode(text="2nd grade math: fractions", metadata={"grade": "2nd", "subject": "math"}),
    TextNode(text="High school: algebra", metadata={"grade": "high school", "subject": "math"})
]

# Query by metadata
query_engine = index.as_query_engine(
    filters={"grade": "2nd"}
)
```

#### Multi-Modal Support

```python
from llama_index.multi_modal_llms import OpenAIMultiModal

# Load and use multi-modal data (text + images)
multi_modal_llm = OpenAIMultiModal(model="gpt-4-vision-preview")
# Now you can process queries that reference images
```

## Full Integration Example

See the `llama_index_integration.py` and `test_llama_index_agent.py` files for complete examples of how to integrate LlamaIndex with the existing codebase.

## Comparison Metrics

Here's a comparison between the custom implementation and LlamaIndex for a typical educational retrieval task:

| Metric | Custom Vector DB | LlamaIndex |
|--------|-----------------|------------|
| Setup complexity | Medium (custom code) | Low (standardized API) |
| Query speed | Fast | Fast |
| Retrieval quality | Good | Excellent (more methods) |
| Document format support | Limited | Extensive |
| Advanced features | Basic | Extensive |
| Maintenance burden | High | Low |

## Troubleshooting

### Common Issues:

1. **ImportError**: Make sure you have installed LlamaIndex and its dependencies.

2. **Embedding model errors**: If you get errors with the embedding model, try:
   ```python
   # Fallback to OpenAI embeddings
   from llama_index.embeddings import OpenAIEmbedding
   embed_model = OpenAIEmbedding()
   ```

3. **Performance issues**: If performance is slow, consider using a more efficient vector store:
   ```python
   from llama_index.vector_stores import FaissVectorStore
   vector_store = FaissVectorStore()
   storage_context = StorageContext.from_defaults(vector_store=vector_store)
   index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
   ``` 
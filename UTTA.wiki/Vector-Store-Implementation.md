# Chapter 5: Vector Store Implementation

[![Status](https://img.shields.io/badge/status-complete-green.svg)]() 
[![Last Updated](https://img.shields.io/badge/last%20updated-February%202024-blue.svg)]()

## Learning Objectives

By the end of this chapter, you'll be able to:
- Evaluate different vector database options for knowledge-enhanced applications
- Implement efficient vector storage and retrieval mechanisms
- Optimize vector search for performance and accuracy
- Scale vector databases for production applications

## 5.1 Vector Database Options

When implementing a knowledge base, choosing the right vector storage solution is crucial for performance and scalability.

### 5.1.1 SQLite-Based Vector Storage

SQLite offers a lightweight, file-based approach to vector storage:

```python
import sqlite3
import numpy as np

def initialize_sqlite_vector_db(db_path):
    """
    Initialize a SQLite database for vector storage
    
    Args:
        db_path (str): Path to the database file
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY,
        text TEXT,
        metadata TEXT,
        category TEXT
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS embeddings (
        chunk_id INTEGER PRIMARY KEY,
        vector BLOB,
        FOREIGN KEY (chunk_id) REFERENCES chunks(id)
    )
    """)
    
    conn.commit()
    conn.close()
```

#### Key Advantages

- **Simplified architecture**: Single database for vectors and metadata
- **Transactional safety**: ACID compliance for reliable operations
- **SQL query language**: Familiar syntax for filtering and joining
- **Portability**: Self-contained file that's easy to back up and distribute
- **No separate service**: Runs in-process without additional infrastructure

#### Limitations

- **Performance at scale**: Less efficient with millions of vectors
- **No specialized indexing**: Missing advanced vector search optimizations
- **Memory constraints**: Limited by available RAM for large operations

### 5.1.2 Dedicated Vector Databases

For larger applications, dedicated vector databases may be more appropriate:

```python
import chromadb

def initialize_chroma_vector_db(persist_directory):
    """
    Initialize a Chroma vector database
    
    Args:
        persist_directory (str): Directory for persistent storage
    """
    client = chromadb.PersistentClient(path=persist_directory)
    
    # Create collection
    collection = client.create_collection(
        name="knowledge_base",
        metadata={"hnsw:space": "cosine"}
    )
    
    return collection

def add_documents_to_chroma(collection, documents, embeddings, metadatas, ids):
    """
    Add documents to a Chroma collection
    
    Args:
        collection: Chroma collection
        documents (list): List of text documents
        embeddings (list): List of vector embeddings
        metadatas (list): List of metadata dictionaries
        ids (list): List of unique identifiers
    """
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
```

#### Popular Vector Database Options

| Database | Best For | Key Features |
|----------|----------|--------------|
| Chroma | Development, small/medium datasets | Easy setup, Python-native |
| FAISS | High-performance, read-heavy | Advanced indexing, optimized search |
| Pinecone | Managed service, scalability | Serverless, production-ready |
| Weaviate | Rich schema, GraphQL | Object-based storage, filterable search |
| Qdrant | Complex filtering, hybrid search | Rich filtering, multi-vector search |

## 5.2 Vector Search Implementation

Once vectors are stored, efficient search is critical for knowledge retrieval.

### 5.2.1 Basic SQLite Vector Search

```python
def vector_search_sqlite(db_path, query_vector, top_k=5, category=None):
    """
    Search for similar vectors in SQLite
    
    Args:
        db_path (str): Path to the database file
        query_vector (np.ndarray): Query vector
        top_k (int): Number of results to return
        category (str): Optional category filter
        
    Returns:
        list: Top matching chunks with metadata
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Build query conditions
    conditions = ""
    params = []
    
    if category:
        conditions = "WHERE c.category = ?"
        params.append(category)
    
    # Retrieve all embeddings
    cursor.execute(f"""
    SELECT c.id, c.text, c.metadata, c.category, e.vector
    FROM chunks c
    JOIN embeddings e ON c.id = e.chunk_id
    {conditions}
    """, params)
    
    results = []
    
    # Calculate similarities
    for chunk_id, text, metadata_json, category, vector_blob in cursor.fetchall():
        vector = np.frombuffer(vector_blob, dtype=np.float32)
        
        # Compute cosine similarity
        dot_product = np.dot(query_vector, vector)
        query_norm = np.linalg.norm(query_vector)
        vector_norm = np.linalg.norm(vector)
        
        if query_norm * vector_norm == 0:
            similarity = 0
        else:
            similarity = dot_product / (query_norm * vector_norm)
        
        results.append({
            "id": chunk_id,
            "text": text,
            "metadata": json.loads(metadata_json),
            "category": category,
            "similarity": float(similarity)
        })
    
    # Sort by similarity (highest first)
    results.sort(key=lambda x: x["similarity"], reverse=True)
    
    return results[:top_k]
```

### 5.2.2 Approximate Nearest Neighbor Search

For larger vector collections, approximate nearest neighbor (ANN) search improves performance:

```python
import faiss
import numpy as np

def create_faiss_index(vectors, index_type="Flat"):
    """
    Create a FAISS index for efficient vector search
    
    Args:
        vectors (np.ndarray): Vector embeddings (n_vectors x dimension)
        index_type (str): Type of FAISS index
        
    Returns:
        faiss.Index: FAISS index for vector search
    """
    dimension = vectors.shape[1]
    
    if index_type == "Flat":
        # Exact search, slower but accurate
        index = faiss.IndexFlatIP(dimension)
    elif index_type == "HNSW":
        # Hierarchical Navigable Small World, faster approximate search
        index = faiss.IndexHNSWFlat(dimension, 32)  # 32 neighbors per node
    elif index_type == "IVF":
        # Inverted File Index, good balance of speed and accuracy
        nlist = int(np.sqrt(vectors.shape[0]))  # Rule of thumb
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        # Requires training
        index.train(vectors)
    else:
        raise ValueError(f"Unknown index type: {index_type}")
    
    # Add vectors to the index
    index.add(vectors)
    return index

def search_faiss_index(index, query_vector, top_k=5):
    """
    Search the FAISS index for similar vectors
    
    Args:
        index (faiss.Index): FAISS index
        query_vector (np.ndarray): Query vector
        top_k (int): Number of results to return
        
    Returns:
        tuple: (similarities, indices)
    """
    # Reshape to expected format
    if len(query_vector.shape) == 1:
        query_vector = query_vector.reshape(1, -1)
    
    # Search the index
    similarities, indices = index.search(query_vector, top_k)
    
    return similarities[0], indices[0]
```

## 5.3 Performance Optimization

Vector search performance can be significantly improved with these techniques:

### 5.3.1 Batch Processing

Process vectors in batches to improve efficiency:

```python
def batch_add_vectors(db_path, chunks, vectors, batch_size=100):
    """
    Add vectors to the database in batches
    
    Args:
        db_path (str): Path to the database
        chunks (list): List of text chunks with metadata
        vectors (list): List of vector embeddings
        batch_size (int): Size of each batch
        
    Returns:
        int: Number of vectors added
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    total_added = 0
    
    for i in range(0, len(chunks), batch_size):
        # Get the current batch
        batch_chunks = chunks[i:i+batch_size]
        batch_vectors = vectors[i:i+batch_size]
        
        # Begin transaction
        conn.execute("BEGIN TRANSACTION")
        
        try:
            # Insert chunks
            chunk_ids = []
            for chunk in batch_chunks:
                cursor.execute(
                    "INSERT INTO chunks (text, metadata, category) VALUES (?, ?, ?)",
                    (chunk["text"], json.dumps(chunk["metadata"]), chunk["category"])
                )
                chunk_ids.append(cursor.lastrowid)
            
            # Insert vectors
            for chunk_id, vector in zip(chunk_ids, batch_vectors):
                binary_vector = vector.astype(np.float32).tobytes()
                cursor.execute(
                    "INSERT INTO embeddings (chunk_id, vector) VALUES (?, ?)",
                    (chunk_id, binary_vector)
                )
            
            # Commit the batch
            conn.commit()
            total_added += len(chunk_ids)
            
        except Exception as e:
            # Rollback on error
            conn.rollback()
            print(f"Error in batch {i//batch_size}: {e}")
    
    conn.close()
    return total_added
```

### 5.3.2 Database Indexing

Create indices to speed up common queries:

```sql
-- Optimize category-based filtering
CREATE INDEX idx_chunks_category ON chunks(category);

-- Optimize metadata-based filtering (if using a JSON1 extension)
CREATE INDEX idx_chunks_metadata ON chunks(json_extract(metadata, '$.source'));
```

### 5.3.3 Memory-Mapped Vectors

For read-heavy operations, memory-mapping improves performance:

```python
import mmap
import numpy as np
import os

class MemoryMappedVectors:
    """Memory-mapped vector storage for efficient reading"""
    
    def __init__(self, filename, vector_dim, dtype=np.float32):
        self.filename = filename
        self.vector_dim = vector_dim
        self.dtype = dtype
        self.dtype_size = np.dtype(dtype).itemsize
        self.vector_size = vector_dim * self.dtype_size
        
        # Create file if it doesn't exist
        if not os.path.exists(filename):
            with open(filename, 'wb') as f:
                # Write an empty header (num_vectors = 0)
                f.write(np.array([0], dtype=np.int32).tobytes())
        
        # Open the file and memory-map it
        self.file = open(filename, 'r+b')
        self.mmap = mmap.mmap(self.file.fileno(), 0)
        
        # Read the number of vectors
        self.num_vectors = np.frombuffer(self.mmap[:4], dtype=np.int32)[0]
    
    def add_vector(self, vector):
        """Add a vector to the storage"""
        # Ensure correct shape
        if len(vector) != self.vector_dim:
            raise ValueError(f"Vector dimension mismatch: {len(vector)} != {self.vector_dim}")
        
        # Update the header
        self.num_vectors += 1
        self.mmap[:4] = np.array([self.num_vectors], dtype=np.int32).tobytes()
        
        # Calculate position to write
        pos = 4 + (self.num_vectors - 1) * self.vector_size
        
        # Extend file if needed
        if pos + self.vector_size > self.mmap.size():
            self.mmap.resize(pos + self.vector_size)
        
        # Write the vector
        self.mmap[pos:pos+self.vector_size] = vector.astype(self.dtype).tobytes()
        
        return self.num_vectors - 1  # Return the index
    
    def get_vector(self, index):
        """Get a vector by index"""
        if index < 0 or index >= self.num_vectors:
            raise IndexError(f"Vector index out of range: {index}")
        
        pos = 4 + index * self.vector_size
        vector_bytes = self.mmap[pos:pos+self.vector_size]
        
        return np.frombuffer(vector_bytes, dtype=self.dtype)
    
    def close(self):
        """Close the memory-mapped file"""
        if self.mmap:
            self.mmap.close()
        if self.file:
            self.file.close()
```

## 5.4 Scaling for Production

As your knowledge base grows, consider these scaling approaches:

### 5.4.1 Sharding Strategies

Divide your vector database into multiple shards:

```python
def create_sharded_db(base_path, num_shards):
    """
    Create a set of sharded vector databases
    
    Args:
        base_path (str): Base path for database files
        num_shards (int): Number of shards to create
        
    Returns:
        list: Paths to created database files
    """
    db_paths = []
    
    for i in range(num_shards):
        shard_path = f"{base_path}_shard_{i}.sqlite"
        initialize_sqlite_vector_db(shard_path)
        db_paths.append(shard_path)
    
    return db_paths

def add_to_shard(chunk, vector, db_paths):
    """
    Add a chunk to the appropriate shard based on hash
    
    Args:
        chunk (dict): Text chunk with metadata
        vector (np.ndarray): Vector embedding
        db_paths (list): Paths to sharded databases
        
    Returns:
        str: Path to the database where the chunk was added
    """
    # Determine shard based on hash of text
    chunk_hash = hash(chunk["text"])
    shard_index = chunk_hash % len(db_paths)
    shard_path = db_paths[shard_index]
    
    # Add to the selected shard
    conn = sqlite3.connect(shard_path)
    cursor = conn.cursor()
    
    cursor.execute(
        "INSERT INTO chunks (text, metadata, category) VALUES (?, ?, ?)",
        (chunk["text"], json.dumps(chunk["metadata"]), chunk["category"])
    )
    chunk_id = cursor.lastrowid
    
    binary_vector = vector.astype(np.float32).tobytes()
    cursor.execute(
        "INSERT INTO embeddings (chunk_id, vector) VALUES (?, ?)",
        (chunk_id, binary_vector)
    )
    
    conn.commit()
    conn.close()
    
    return shard_path
```

### 5.4.2 Caching Layer

Implement a cache for frequently accessed vectors:

```python
import lru

class VectorCache:
    """LRU cache for vector lookups"""
    
    def __init__(self, max_size=1000):
        self.cache = lru.LRU(max_size)
    
    def get(self, key):
        """Get a vector from the cache"""
        if key in self.cache:
            return self.cache[key]
        return None
    
    def put(self, key, vector):
        """Add a vector to the cache"""
        self.cache[key] = vector
    
    def clear(self):
        """Clear the cache"""
        self.cache.clear()
```

## 5.5 Hands-On Exercise: Building a Vector Search System

Let's implement a complete vector search system:

```python
import argparse
import sqlite3
import numpy as np
import json
from sentence_transformers import SentenceTransformer

class VectorSearchSystem:
    """Complete vector search system"""
    
    def __init__(self, db_path, model_name="all-MiniLM-L6-v2"):
        """Initialize the search system"""
        self.db_path = db_path
        self.model = SentenceTransformer(model_name)
        
        # Connect to the database
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        
        # Ensure tables exist
        self._create_tables_if_needed()
    
    def _create_tables_if_needed(self):
        """Create database tables if they don't exist"""
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            text TEXT,
            metadata TEXT,
            category TEXT
        )
        """)
        
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            chunk_id INTEGER PRIMARY KEY,
            vector BLOB,
            FOREIGN KEY (chunk_id) REFERENCES chunks(id)
        )
        """)
        
        self.conn.commit()
    
    def add_chunk(self, text, metadata, category):
        """Add a text chunk to the database"""
        # Insert the chunk
        self.cursor.execute(
            "INSERT INTO chunks (text, metadata, category) VALUES (?, ?, ?)",
            (text, json.dumps(metadata), category)
        )
        chunk_id = self.cursor.lastrowid
        
        # Generate and store the embedding
        embedding = self.model.encode(text)
        binary_embedding = embedding.astype(np.float32).tobytes()
        
        self.cursor.execute(
            "INSERT INTO embeddings (chunk_id, vector) VALUES (?, ?)",
            (chunk_id, binary_embedding)
        )
        
        self.conn.commit()
        return chunk_id
    
    def search(self, query, top_k=5, category=None):
        """Search for similar chunks"""
        # Generate query embedding
        query_embedding = self.model.encode(query)
        
        # Build query conditions
        conditions = ""
        params = []
        
        if category:
            conditions = "WHERE c.category = ?"
            params.append(category)
        
        # Retrieve all chunks and embeddings
        self.cursor.execute(f"""
        SELECT c.id, c.text, c.metadata, c.category, e.vector
        FROM chunks c
        JOIN embeddings e ON c.id = e.chunk_id
        {conditions}
        """, params)
        
        results = []
        
        # Calculate similarities
        for chunk_id, text, metadata_json, category, vector_blob in self.cursor.fetchall():
            vector = np.frombuffer(vector_blob, dtype=np.float32)
            
            # Compute cosine similarity
            similarity = np.dot(query_embedding, vector) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(vector)
            )
            
            results.append({
                "id": chunk_id,
                "text": text,
                "metadata": json.loads(metadata_json),
                "category": category,
                "similarity": float(similarity)
            })
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return results[:top_k]
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vector search system")
    parser.add_argument("--db", default="vector_search.sqlite", help="Database path")
    parser.add_argument("--add", action="store_true", help="Add a chunk")
    parser.add_argument("--search", action="store_true", help="Search for chunks")
    parser.add_argument("--text", help="Text to add or search query")
    parser.add_argument("--category", help="Category for filtering")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    
    args = parser.parse_args()
    
    # Initialize the system
    system = VectorSearchSystem(args.db)
    
    try:
        if args.add and args.text:
            # Add a chunk
            metadata = {"source": "example", "timestamp": "2023-01-01"}
            chunk_id = system.add_chunk(args.text, metadata, args.category or "general")
            print(f"Added chunk with ID: {chunk_id}")
        
        elif args.search and args.text:
            # Search for chunks
            results = system.search(args.text, args.top_k, args.category)
            
            print(f"Top {len(results)} results for query: {args.text}")
            for i, result in enumerate(results):
                print(f"\n[{i+1}] Similarity: {result['similarity']:.4f}")
                print(f"Category: {result['category']}")
                print(f"Text: {result['text'][:100]}...")
                print(f"Source: {result['metadata'].get('source', 'Unknown')}")
    
    finally:
        system.close()
```

## 5.6 Key Takeaways

- Vector databases are essential for semantic search and knowledge retrieval
- SQLite offers a simple approach for small to medium-sized applications
- Dedicated vector databases provide better performance at scale
- Performance optimization techniques can significantly improve search speed
- The choice of vector storage solution depends on your specific requirements

## 5.7 Chapter Project: Vector Search Implementation

For this chapter's project, you'll implement a vector search system:

1. Choose an appropriate vector database for your application
2. Implement vector storage and retrieval functions
3. Add performance optimizations (indexing, batching, caching)
4. Create a simple API for searching the vector database
5. Test with a small collection of documents
6. Benchmark search performance with different techniques

## References

- Johnson, J., et al. (2019). *Billion-scale similarity search with GPUs*
- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*
- FAISS Documentation: [https://faiss.ai/](https://faiss.ai/)
- SQLite Documentation: [https://www.sqlite.org/docs.html](https://www.sqlite.org/docs.html)

## Further Reading

- [Chapter 3: Knowledge Base Structure](Knowledge-Base-Structure)
- [Chapter 4: Knowledge Processing Pipeline](Knowledge-Processing-Pipeline)
- [Chapter 6: Practical Applications of Knowledge Bases](Knowledge-Applications)
- [Chapter 7: LLM Integration Techniques](Knowledge-LLM-Integration) 
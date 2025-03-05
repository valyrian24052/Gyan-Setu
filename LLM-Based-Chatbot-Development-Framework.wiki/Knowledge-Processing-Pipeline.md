# Chapter 4: Knowledge Processing Pipeline

[![Status](https://img.shields.io/badge/status-complete-green.svg)]() 
[![Last Updated](https://img.shields.io/badge/last%20updated-February%202024-blue.svg)]()

## Learning Objectives

By the end of this chapter, you'll be able to:
- Design a complete pipeline for processing knowledge base content
- Implement text extraction techniques for various document formats
- Create effective chunking strategies for optimal knowledge retrieval
- Generate and store vector embeddings for semantic search

## 4.1 Content Acquisition

Building a knowledge base begins with acquiring high-quality content. This section explores methods for collecting and organizing educational materials.

### 4.1.1 Document Types and Formats

Knowledge bases can incorporate various document types:

- **PDF textbooks**: Academic textbooks and educational guides
- **DOCX documents**: Teaching resources and lesson plans
- **TXT files**: Research papers and articles
- **JSON/JSONL**: Structured behavioral data
- **HTML/Web content**: Online educational resources

### 4.1.2 Content Collection Strategy

Implementing a systematic approach to content acquisition:

```python
import os
import shutil
import hashlib
from datetime import datetime

def process_new_document(source_path, target_directory):
    """
    Process a new document for the knowledge base
    
    Args:
        source_path (str): Path to the source document
        target_directory (str): Target directory for processed files
        
    Returns:
        str: Path to the processed document
    """
    # Create file hash to prevent duplicates
    with open(source_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    
    # Extract file information
    filename = os.path.basename(source_path)
    file_ext = os.path.splitext(filename)[1].lower()
    
    # Create organized directory structure
    category_dir = os.path.join(
        target_directory, 
        determine_category(filename, file_ext)
    )
    os.makedirs(category_dir, exist_ok=True)
    
    # Create target path with timestamp and hash
    timestamp = datetime.now().strftime("%Y%m%d")
    target_path = os.path.join(
        category_dir,
        f"{timestamp}_{file_hash[:8]}_{filename}"
    )
    
    # Copy the file
    shutil.copy2(source_path, target_path)
    
    # Log the acquisition
    log_document_acquisition(source_path, target_path)
    
    return target_path
```

## 4.2 Document Processing

Once documents are acquired, they need to be processed to extract usable text.

### 4.2.1 Text Extraction

Different document formats require specific extraction methods:

```python
import PyPDF2
import docx
import json
import re
from bs4 import BeautifulSoup
import requests

def extract_text(file_path):
    """
    Extract text from a document based on its file type
    
    Args:
        file_path (str): Path to the document
        
    Returns:
        str: Extracted text content
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.pdf':
        return extract_from_pdf(file_path)
    elif file_ext == '.docx':
        return extract_from_docx(file_path)
    elif file_ext == '.txt':
        return extract_from_txt(file_path)
    elif file_ext in ['.json', '.jsonl']:
        return extract_from_json(file_path)
    elif file_ext in ['.html', '.htm']:
        return extract_from_html(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

def extract_from_pdf(file_path):
    """Extract text from PDF files"""
    text = ""
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() + "\n\n"
    return text

def extract_from_docx(file_path):
    """Extract text from DOCX files"""
    doc = docx.Document(file_path)
    return "\n\n".join([para.text for para in doc.paragraphs])
```

### 4.2.2 Text Cleaning and Normalization

Raw extracted text requires cleaning and normalization:

```python
def clean_text(text):
    """
    Clean and normalize extracted text
    
    Args:
        text (str): Raw extracted text
        
    Returns:
        str: Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page numbers
    text = re.sub(r'\bPage \d+\b', '', text)
    
    # Remove headers/footers (customize for your documents)
    text = re.sub(r'Chapter \d+:? .*?\n', '', text)
    
    # Fix hyphenated words at line breaks
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    
    # Normalize quotes
    text = text.replace(''', "'").replace('"', '"').replace('"', '"')
    
    return text.strip()
```

### 4.2.3 Metadata Extraction

Capturing document metadata enhances knowledge retrieval:

```python
def extract_metadata(file_path, text_content):
    """
    Extract metadata from document content and properties
    
    Args:
        file_path (str): Path to the document
        text_content (str): Extracted text
        
    Returns:
        dict: Document metadata
    """
    filename = os.path.basename(file_path)
    file_ext = os.path.splitext(filename)[1].lower()
    file_stats = os.stat(file_path)
    
    # Basic metadata
    metadata = {
        "filename": filename,
        "file_type": file_ext[1:],  # Remove the dot
        "file_size_bytes": file_stats.st_size,
        "created_date": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
        "modified_date": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
        "processed_date": datetime.now().isoformat()
    }
    
    # Format-specific metadata extraction
    if file_ext == '.pdf':
        pdf_metadata = extract_pdf_metadata(file_path)
        metadata.update(pdf_metadata)
    
    # Extract title from content
    title_match = re.search(r'^# (.*?)$', text_content, re.MULTILINE)
    if title_match:
        metadata["title"] = title_match.group(1)
    
    return metadata
```

## 4.3 Chunking Strategies

For effective knowledge retrieval, documents must be split into appropriate chunks.

### 4.3.1 Text Chunking Approaches

Different chunking strategies serve different purposes:

```python
def chunk_text(text, method="paragraph", max_chunk_size=500):
    """
    Split text into chunks using various methods
    
    Args:
        text (str): Text to chunk
        method (str): Chunking method (paragraph, fixed_size, semantic)
        max_chunk_size (int): Maximum chunk size in characters
        
    Returns:
        list: Text chunks
    """
    if method == "paragraph":
        # Split by paragraphs
        chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    elif method == "fixed_size":
        # Split into fixed-size chunks
        chunks = []
        words = text.split()
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_size = len(word) + 1  # +1 for space
            if current_size + word_size > max_chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = word_size
            else:
                current_chunk.append(word)
                current_size += word_size
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
    
    elif method == "semantic":
        # Split by semantic units (requires NLP)
        import nltk
        nltk.download('punkt')
        sentences = nltk.sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence) + 1  # +1 for space
            
            if current_size + sentence_size > max_chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
    
    else:
        raise ValueError(f"Unknown chunking method: {method}")
    
    # Filter out chunks that are too small
    min_chunk_size = 50  # Minimum chunk size in characters
    chunks = [chunk for chunk in chunks if len(chunk) >= min_chunk_size]
    
    return chunks
```

### 4.3.2 Optimizing Chunk Size

The ideal chunk size depends on your specific use case:

| Chunk Size | Advantages | Disadvantages |
|------------|------------|---------------|
| Small (100-300 chars) | Precise retrieval<br>Better for specific facts | May lose context<br>Can fragment concepts |
| Medium (300-800 chars) | Balance of context and specificity<br>Works well for most applications | May include irrelevant information | 
| Large (800+ chars) | Preserves context<br>Captures complete concepts | Less precise retrieval<br>May exceed context windows |

For educational content, chunks around 500 characters typically perform well, as they balance contextual information with relevance.

## 4.4 Vector Embedding Generation

Converting text chunks to vector embeddings is essential for semantic search.

### 4.4.1 Embedding Model Selection

Choose an embedding model based on your requirements:

```python
from sentence_transformers import SentenceTransformer

def initialize_embedding_model(model_name="all-MiniLM-L6-v2"):
    """
    Initialize a sentence transformer model for embeddings
    
    Args:
        model_name (str): Name of the model to use
        
    Returns:
        SentenceTransformer: Initialized model
    """
    return SentenceTransformer(model_name)

def generate_embeddings(chunks, model):
    """
    Generate embeddings for a list of text chunks
    
    Args:
        chunks (list): List of text chunks
        model (SentenceTransformer): Embedding model
        
    Returns:
        list: Embeddings for each chunk
    """
    # Generate embeddings in batches
    batch_size = 32
    embeddings = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_embeddings = model.encode(batch)
        embeddings.extend(batch_embeddings)
    
    return embeddings
```

### 4.4.2 Vector Storage

Store vectors efficiently for retrieval:

```python
import sqlite3
import numpy as np
import json

def store_chunks_and_embeddings(chunks, embeddings, metadata_list, db_path):
    """
    Store text chunks and their embeddings in the database
    
    Args:
        chunks (list): List of text chunks
        embeddings (list): List of embeddings
        metadata_list (list): List of metadata dictionaries
        db_path (str): Path to the database
        
    Returns:
        int: Number of chunks stored
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Ensure tables exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY,
        text TEXT,
        metadata TEXT,
        category TEXT,
        usage_count INTEGER DEFAULT 0,
        effectiveness_score REAL DEFAULT 0
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS embeddings (
        chunk_id INTEGER PRIMARY KEY,
        vector BLOB,
        FOREIGN KEY (chunk_id) REFERENCES chunks(id)
    )
    """)
    
    # Store chunks and embeddings
    for i, (chunk, embedding, metadata) in enumerate(zip(chunks, embeddings, metadata_list)):
        # Determine category from metadata
        category = determine_category_from_metadata(metadata)
        
        # Insert the chunk
        cursor.execute(
            "INSERT INTO chunks (text, metadata, category) VALUES (?, ?, ?)",
            (chunk, json.dumps(metadata), category)
        )
        chunk_id = cursor.lastrowid
        
        # Insert the embedding
        binary_embedding = embedding.astype(np.float32).tobytes()
        cursor.execute(
            "INSERT INTO embeddings (chunk_id, vector) VALUES (?, ?)",
            (chunk_id, binary_embedding)
        )
    
    conn.commit()
    conn.close()
    
    return len(chunks)
```

## 4.5 Hands-On Exercise: Processing Pipeline

Let's implement a simple processing pipeline:

```python
import os
import argparse
from pathlib import Path

def process_document_pipeline(input_file, output_db):
    """
    Process a document through the entire pipeline
    
    Args:
        input_file (str): Path to input document
        output_db (str): Path to output database
        
    Returns:
        int: Number of chunks processed
    """
    print(f"Processing {input_file}...")
    
    # Extract text from document
    text = extract_text(input_file)
    print(f"Extracted {len(text)} characters of text")
    
    # Clean the text
    text = clean_text(text)
    print(f"Cleaned text: {len(text)} characters")
    
    # Extract metadata
    metadata = extract_metadata(input_file, text)
    print(f"Extracted metadata: {list(metadata.keys())}")
    
    # Chunk the text
    chunks = chunk_text(text, method="semantic", max_chunk_size=500)
    print(f"Created {len(chunks)} chunks")
    
    # Initialize the embedding model
    model = initialize_embedding_model()
    print(f"Initialized embedding model: {model}")
    
    # Generate embeddings
    embeddings = generate_embeddings(chunks, model)
    print(f"Generated {len(embeddings)} embeddings")
    
    # Create metadata list (one per chunk)
    metadata_list = []
    for i, chunk in enumerate(chunks):
        chunk_metadata = metadata.copy()
        chunk_metadata["chunk_index"] = i
        chunk_metadata["chunk_text_length"] = len(chunk)
        metadata_list.append(chunk_metadata)
    
    # Store chunks and embeddings
    num_stored = store_chunks_and_embeddings(
        chunks, embeddings, metadata_list, output_db
    )
    print(f"Stored {num_stored} chunks in the database")
    
    return num_stored

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process documents for knowledge base")
    parser.add_argument("input_file", help="Path to input document")
    parser.add_argument("--output_db", default="knowledge_base.sqlite", 
                        help="Path to output database")
    args = parser.parse_args()
    
    process_document_pipeline(args.input_file, args.output_db)
```

## 4.6 Key Takeaways

- A well-designed processing pipeline ensures high-quality knowledge base content
- Document processing requires format-specific extraction techniques
- Chunking strategies directly impact knowledge retrieval quality
- Vector embeddings enable semantic search capabilities
- Metadata enrichment improves search relevance and attribution

## 4.7 Chapter Project: Knowledge Processing System

For this chapter's project, you'll build a complete knowledge processing system:

1. Create a document acquisition module that handles multiple file formats
2. Implement text extraction and cleaning for these formats
3. Develop a chunking module with multiple strategies
4. Generate vector embeddings using a pre-trained model
5. Store both chunks and embeddings in a database
6. Test your pipeline with sample documents

## References

- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*
- Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
- Sentence Transformers Documentation: [https://www.sbert.net/](https://www.sbert.net/)

## Further Reading

- [Chapter 1: Introduction to Knowledge Bases for GenAI](Knowledge-Base-Overview)
- [Chapter 2: Educational Content](Educational-Content)
- [Chapter 3: Knowledge Base Structure](Knowledge-Base-Structure)
- [Chapter 5: Vector Store Implementation](Vector-Store-Implementation) 
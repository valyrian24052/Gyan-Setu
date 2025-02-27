"""
Educational Knowledge Management Module

This module handles all aspects of educational knowledge management, from static 
reference data to document processing and knowledge base management. It serves as 
a comprehensive repository for teaching strategies, student characteristics, and 
pedagogical approaches, while also providing functionality to build and maintain 
knowledge bases from educational content.

Key Components:
    - Static Knowledge Data: Pre-defined educational references
    - Knowledge Manager: Processes and indexes educational content 
    - Document Processing: Handles various document formats
    - Vector Database Integration: Creates searchable knowledge embeddings
    - Performance Tracking: Monitors knowledge effectiveness

Usage:
    This module is used by the Teacher Training Agent to:
    - Access standardized teaching strategies and student profiles
    - Process educational books and materials
    - Build searchable knowledge bases
    - Track knowledge effectiveness in scenarios

Example:
    manager = PedagogicalKnowledgeManager("./knowledge_base/books")
    manager.process_documents()
    manager.build_knowledge_base()
"""

import os
import json
import argparse
import sqlite3
import numpy as np
import faiss
from tqdm import tqdm

# Comprehensive characteristics of second-grade students including
# academic abilities, behavioral patterns, and social-emotional traits
SECOND_GRADE_CHARACTERISTICS = {
    # Cognitive and academic capabilities typical for second grade
    "cognitive": {
        # Subject-specific academic skills organized by difficulty level
        "academic_skills": {
            # Mathematics skills progression
            "math": {
                "basic": ["counting to 100", "basic addition", "basic subtraction"],
                "intermediate": ["skip counting", "place value", "mental math"],
                "challenging": ["word problems", "beginning multiplication", "money math"]
            },
            # Reading skills progression
            "reading": {
                "basic": ["sight words", "phonics", "basic comprehension"],
                "intermediate": ["fluency", "vocabulary", "main idea"],
                "challenging": ["inference", "context clues", "summarizing"]
            }
        }
    },
    # Common behavioral scenarios with triggers and appropriate responses
    "behavioral_scenarios": {
        # Attention-related behaviors and interventions
        "attention": {
            # Situations that may cause attention issues
            "triggers": [
                "long tasks",
                "complex instructions",
                "distracting environment",
                "abstract concepts"
            ],
            # How attention issues typically manifest
            "manifestations": [
                "fidgeting",
                "looking around",
                "doodling",
                "asking off-topic questions"
            ],
            # Research-based intervention strategies
            "effective_interventions": [
                "break tasks into smaller parts",
                "use visual aids",
                "incorporate movement",
                "provide frequent breaks"
            ]
        },
        # Frustration-related behaviors and interventions
        "frustration": {
            # Situations that may cause frustration
            "triggers": [
                "difficult problems",
                "repeated mistakes",
                "peer comparison",
                "time pressure"
            ],
            # How frustration typically manifests
            "manifestations": [
                "giving up quickly",
                "saying 'I can't do it'",
                "becoming withdrawn",
                "acting out"
            ],
            # Effective strategies for managing frustration
            "effective_interventions": [
                "provide encouragement",
                "scaffold learning",
                "celebrate small successes",
                "offer alternative approaches"
            ]
        }
    },
    # Typical social and emotional expressions
    "social_emotional": {
        # Common phrases used by second-grade students
        "common_expressions": [
            "This is too hard!",
            "I don't get it...",
            "Can you help me?",
            "I'm trying my best"
        ]
    }
}

# Comprehensive collection of effective teaching strategies
# organized by pedagogical purpose
TEACHING_STRATEGIES = {
    # Strategies for maintaining student engagement
    "engagement": [
        "Use hands-on materials",
        "Incorporate movement",
        "Connect to real life",
        "Use visual aids"
    ],
    # Approaches for differentiating instruction
    "differentiation": [
        "Provide multiple examples",
        "Offer choice in activities",
        "Adjust difficulty levels",
        "Use varied approaches"
    ],
    # Methods for providing student support
    "support": [
        "Give specific praise",
        "Break tasks into steps",
        "Model thinking process",
        "Provide scaffolding"
    ]
}

class PedagogicalKnowledgeManager:
    """
    Manages educational knowledge and teaching strategies by loading files
    from a specified directory and creating a knowledge base.
    
    The knowledge base can be stored as either a vector database (using FAISS)
    or as a relational database (using SQLite), as selected by the configuration.
    
    Features:
    - Optimized document chunking
    - Batch processing for large datasets
    - Efficient vector indexing with FAISS
    - Support for multiple document formats (PDF, TXT, JSON, EPUB)
    """
    def __init__(self, kb_dir, db_type="vector", index_output="vector_index.index",
                 metadata_output="vector_metadata.json", relational_db_output="knowledge.db", 
                 chunk_size=500, chunk_overlap=50):
        self.kb_dir = kb_dir
        self.db_type = db_type.lower()  # "vector" or "relational"
        self.index_output = index_output
        self.metadata_output = metadata_output
        self.relational_db_output = relational_db_output
        self.documents = []
        self.metadata = []
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Check for PyMuPDF (better PDF extraction)
        try:
            import fitz
            self.pdf_extractor = "pymupdf"
        except ImportError:
            # Fall back to PyPDF2
            try:
                import PyPDF2
                self.pdf_extractor = "pypdf2"
            except ImportError:
                print("Warning: Neither PyMuPDF nor PyPDF2 is installed. PDF processing will be disabled.")
                self.pdf_extractor = None
        
        # Check for EPUB support
        try:
            import ebooklib
            from ebooklib import epub
            self.epub_support = True
        except ImportError:
            print("Warning: ebooklib not installed. EPUB processing will be disabled.")
            self.epub_support = False
            
        # Load sentence transformer model
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Loaded sentence transformer model")
        except ImportError:
            print("Warning: sentence_transformers not installed. Vector embedding will be disabled.")
            self.model = None
        
    def process_documents(self):
        """Process all documents in the knowledge base directory."""
        if not os.path.exists(self.kb_dir):
            os.makedirs(self.kb_dir, exist_ok=True)
            print(f"Created knowledge base directory: {self.kb_dir}")
            return
        
        files = [f for f in os.listdir(self.kb_dir) if os.path.isfile(os.path.join(self.kb_dir, f))]
        if not files:
            print(f"No files found in {self.kb_dir}")
            return
            
        # Process each file based on type
        for filename in tqdm(files, desc="Processing files"):
            filepath = os.path.join(self.kb_dir, filename)
            try:
                if filename.lower().endswith('.pdf') and self.pdf_extractor:
                    self._process_pdf(filepath)
                elif filename.lower().endswith('.txt'):
                    self._process_txt(filepath)
                elif filename.lower().endswith(('.json', '.jsonl')):
                    self._process_json(filepath)
                elif filename.lower().endswith('.epub') and self.epub_support:
                    self._process_epub(filepath)
                elif filename.lower().endswith('.docx'):
                    self._process_docx(filepath)
                else:
                    print(f"Skipping unsupported file: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                
        print(f"Processed {len(self.documents)} documents with {sum(len(doc['chunks']) for doc in self.documents)} chunks")
    
    def _process_pdf(self, filepath):
        """Process PDF file based on available extractor."""
        if self.pdf_extractor == "pymupdf":
            self._process_pdf_pymupdf(filepath)
        elif self.pdf_extractor == "pypdf2":
            self._process_pdf_pypdf2(filepath)
            
    def _process_pdf_pymupdf(self, filepath):
        """Process PDF using PyMuPDF (better quality)."""
        import fitz
        document = {"source": os.path.basename(filepath), "chunks": [], "metadata": {}}
        
        try:
            with fitz.open(filepath) as pdf:
                text = ""
                for page_num, page in enumerate(pdf):
                    text += page.get_text()
                    # Extract metadata from first page if available
                    if page_num == 0:
                        document["metadata"]["title"] = pdf.metadata.get("title", "")
                        document["metadata"]["author"] = pdf.metadata.get("author", "")
                        document["metadata"]["subject"] = pdf.metadata.get("subject", "")
                
                # Chunk the text
                chunks = self._chunk_text(text)
                document["chunks"] = chunks
                self.documents.append(document)
        except Exception as e:
            print(f"Error processing PDF with PyMuPDF: {str(e)}")
            
    def _process_pdf_pypdf2(self, filepath):
        """Process PDF using PyPDF2 (fallback)."""
        import PyPDF2
        document = {"source": os.path.basename(filepath), "chunks": [], "metadata": {}}
        
        try:
            with open(filepath, 'rb') as file:
                reader = PyPDF2.PdfFileReader(file)
                text = ""
                # Get document info if available
                if reader.getDocumentInfo():
                    info = reader.getDocumentInfo()
                    document["metadata"]["title"] = info.get("/Title", "")
                    document["metadata"]["author"] = info.get("/Author", "")
                    document["metadata"]["subject"] = info.get("/Subject", "")
                
                # Extract text from pages
                for page_num in range(reader.numPages):
                    page = reader.getPage(page_num)
                    text += page.extractText()
                
                # Chunk the text
                chunks = self._chunk_text(text)
                document["chunks"] = chunks
                self.documents.append(document)
        except Exception as e:
            print(f"Error processing PDF with PyPDF2: {str(e)}")
    
    def _process_txt(self, filepath):
        """Process plain text file."""
        document = {"source": os.path.basename(filepath), "chunks": [], "metadata": {}}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                text = file.read()
                
                # Extract basic metadata from filename
                base_name = os.path.basename(filepath)
                document["metadata"]["title"] = os.path.splitext(base_name)[0]
                
                # Chunk the text
                chunks = self._chunk_text(text)
                document["chunks"] = chunks
                self.documents.append(document)
        except Exception as e:
            print(f"Error processing TXT file: {str(e)}")
    
    def _process_json(self, filepath):
        """Process JSON file with educational content."""
        document = {"source": os.path.basename(filepath), "chunks": [], "metadata": {}}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                # Handle both single JSON object and JSONL format
                if filepath.lower().endswith('.jsonl'):
                    # JSONL format (one JSON per line)
                    lines = file.readlines()
                    for line in lines:
                        if line.strip():
                            data = json.loads(line)
                            self._extract_json_content(data, document)
                else:
                    # Single JSON object
                    data = json.load(file)
                    self._extract_json_content(data, document)
                
                self.documents.append(document)
        except Exception as e:
            print(f"Error processing JSON file: {str(e)}")
    
    def _extract_json_content(self, data, document):
        """Extract and process content from JSON data."""
        # Extract content based on common JSON structures
        if isinstance(data, dict):
            # If there's a "content" field, use it directly
            if "content" in data:
                content = data["content"]
                chunks = self._chunk_text(content)
                document["chunks"].extend(chunks)
                
                # Extract metadata if available
                if "metadata" in data:
                    document["metadata"].update(data["metadata"])
                elif "title" in data:
                    document["metadata"]["title"] = data["title"]
                
            # Otherwise, check for text content in various fields
            elif any(key in data for key in ["text", "body", "description"]):
                content = data.get("text", data.get("body", data.get("description", "")))
                chunks = self._chunk_text(content)
                document["chunks"].extend(chunks)
                
            # Process nested objects
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    self._extract_json_content(value, document)
        
        # Process lists of content
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    self._extract_json_content(item, document)
                elif isinstance(item, str) and len(item) > 100:  # Only process substantial text
                    chunks = self._chunk_text(item)
                    document["chunks"].extend(chunks)
    
    def _process_epub(self, filepath):
        """Process EPUB ebook file."""
        document = {"source": os.path.basename(filepath), "chunks": [], "metadata": {}}
        
        try:
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup
            
            book = epub.read_epub(filepath)
            
            # Extract metadata
            document["metadata"]["title"] = book.get_metadata('DC', 'title')[0][0] if book.get_metadata('DC', 'title') else ""
            document["metadata"]["author"] = book.get_metadata('DC', 'creator')[0][0] if book.get_metadata('DC', 'creator') else ""
            
            # Extract content from each chapter
            all_text = ""
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    # Parse HTML content
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text = soup.get_text()
                    all_text += text + "\n\n"
            
            # Chunk the text
            chunks = self._chunk_text(all_text)
            document["chunks"] = chunks
            self.documents.append(document)
        except Exception as e:
            print(f"Error processing EPUB file: {str(e)}")
    
    def _process_docx(self, filepath):
        """Process Microsoft Word DOCX file."""
        document = {"source": os.path.basename(filepath), "chunks": [], "metadata": {}}
        
        try:
            import docx
            
            doc = docx.Document(filepath)
            
            # Extract text from paragraphs
            text = "\n".join([para.text for para in doc.paragraphs])
            
            # Extract basic metadata from filename
            base_name = os.path.basename(filepath)
            document["metadata"]["title"] = os.path.splitext(base_name)[0]
            
            # Chunk the text
            chunks = self._chunk_text(text)
            document["chunks"] = chunks
            self.documents.append(document)
        except Exception as e:
            print(f"Error processing DOCX file: {str(e)}")
    
    def _chunk_text(self, text):
        """
        Chunk text into smaller pieces using word-based approach with overlap.
        This performs better than the NLTK sentence-based approach.
        """
        # Clean text
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split())  # Normalize whitespace
        
        # Skip if text is too short
        if len(text) < 50:
            return []
            
        # Split into words
        words = text.split()
        word_count = len(words)
        
        # Convert chunk_size and overlap from characters to word counts (approx)
        # Assuming average word length of 5 characters + 1 space
        words_per_chunk = max(1, self.chunk_size // 6)
        overlap_words = max(1, self.chunk_overlap // 6)
        
        # Create chunks with overlap
        chunks = []
        for i in range(0, word_count, words_per_chunk - overlap_words):
            # Get chunk of words
            end_idx = min(i + words_per_chunk, word_count)
            if end_idx - i < words_per_chunk // 2:  # Skip chunks that are too small
                continue
                
            chunk = ' '.join(words[i:end_idx])
            
            # Add chunk with positional metadata
            chunks.append({
                "text": chunk,
                "start_word": i,
                "end_word": end_idx,
                "position": len(chunks)
            })
            
            # Stop if we've reached the end
            if end_idx == word_count:
                break
                
        return chunks
    
    def build_knowledge_base(self):
        """Build the knowledge base based on configured type."""
        if not self.documents:
            print("No documents processed. Please process documents first.")
            return
            
        if self.db_type == "vector":
            self._build_vector_database()
        elif self.db_type == "relational":
            self._build_relational_database()
        else:
            print(f"Unsupported database type: {self.db_type}")
    
    def _build_vector_database(self):
        """Build vector database using FAISS."""
        if not self.model:
            print("Sentence transformer model not available. Cannot build vector database.")
            return
            
        # Check if GPU FAISS is available
        try:
            res = faiss.StandardGpuResources()
            gpu_available = True
            print("GPU acceleration enabled for vector database. Using FAISS-GPU.")
        except Exception as e:
            gpu_available = False
            print("GPU acceleration not available. Using CPU-only mode for vector database.")
        
        # Prepare chunks for embedding
        all_chunks = []
        metadata = []
        
        for doc_idx, doc in enumerate(self.documents):
            for chunk_idx, chunk in enumerate(doc["chunks"]):
                all_chunks.append(chunk["text"])
                
                # Create metadata for this chunk
                meta = {
                    "id": f"doc{doc_idx}_chunk{chunk_idx}",
                    "source": doc["source"],
                    "position": chunk["position"],
                    "document_index": doc_idx
                }
                
                # Add document metadata
                if "metadata" in doc:
                    meta["title"] = doc["metadata"].get("title", "")
                    meta["author"] = doc["metadata"].get("author", "")
                    meta["subject"] = doc["metadata"].get("subject", "")
                    
                metadata.append(meta)
        
        # Create embeddings in batches to avoid memory issues
        batch_size = 32
        all_embeddings = []
        
        for i in tqdm(range(0, len(all_chunks), batch_size), desc="Creating embeddings"):
            batch = all_chunks[i:i+batch_size]
            embeddings = self.model.encode(batch)
            all_embeddings.append(embeddings)
            
        # Combine all batches
        embeddings = np.vstack(all_embeddings)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index based on dataset size
        index_factory_string = "IDMap,Flat" if len(all_chunks) < 10000 else "IVF100,PQ8"
        dimension = embeddings.shape[1]
        
        if index_factory_string == "IDMap,Flat":
            # Simple index for smaller datasets (higher quality but slower for large datasets)
            index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))
        else:
            # IVF with Product Quantization for larger datasets (faster but approximation)
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFPQ(quantizer, dimension, 100, 8, 8)
            index.train(embeddings)
        
        # Convert to GPU index if available
        if gpu_available:
            try:
                index = faiss.index_cpu_to_gpu(res, 0, index)
                print("Successfully moved FAISS index to GPU.")
            except Exception as e:
                print(f"Could not use GPU for index: {str(e)}. Using CPU index.")
        
        # Add vectors to index with IDs
        ids = np.arange(len(all_chunks), dtype=np.int64)
        index.add_with_ids(embeddings, ids)
        
        # Save index and metadata
        faiss.write_index(index, self.index_output)
        with open(self.metadata_output, 'w') as f:
            json.dump({"chunks": metadata}, f)
            
        print(f"Vector database built with {len(all_chunks)} chunks. Saved to {self.index_output}")
    
    def _build_relational_database(self):
        """Build relational database using SQLite."""
        # Create SQLite database
        conn = sqlite3.connect(self.relational_db_output)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            source TEXT,
            title TEXT,
            author TEXT,
            subject TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            document_id INTEGER,
            text TEXT,
            position INTEGER,
            metadata TEXT,
            FOREIGN KEY (document_id) REFERENCES documents (id)
        )
        ''')
        
        # Create search index
        cursor.execute('CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, content=chunks, content_rowid=id)')
        
        # Insert documents and chunks
        for doc_idx, doc in enumerate(self.documents):
            # Insert document
            source = doc["source"]
            title = doc["metadata"].get("title", "")
            author = doc["metadata"].get("author", "")
            subject = doc["metadata"].get("subject", "")
            
            cursor.execute(
                'INSERT INTO documents (source, title, author, subject) VALUES (?, ?, ?, ?)',
                (source, title, author, subject)
            )
            document_id = cursor.lastrowid
            
            # Insert chunks
            for chunk in doc["chunks"]:
                text = chunk["text"]
                position = chunk["position"]
                metadata = json.dumps({
                    "start_word": chunk.get("start_word", 0),
                    "end_word": chunk.get("end_word", 0)
                })
                
                cursor.execute(
                    'INSERT INTO chunks (document_id, text, position, metadata) VALUES (?, ?, ?, ?)',
                    (document_id, text, position, metadata)
                )
                chunk_id = cursor.lastrowid
                
                # Add to search index
                cursor.execute('INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)', (chunk_id, text))
        
        # Commit and close
        conn.commit()
        conn.close()
        
        print(f"Relational database built with {len(self.documents)} documents. Saved to {self.relational_db_output}")
    
    def search_knowledge_base(self, query, top_k=5):
        """Search the knowledge base for relevant information."""
        if self.db_type == "vector":
            return self._search_vector(query, top_k)
        elif self.db_type == "relational":
            return self._search_relational(query, top_k)
        else:
            print(f"Unsupported database type: {self.db_type}")
            return []
    
    def _search_vector(self, query, top_k=5):
        """Search the vector database using semantic similarity."""
        if not os.path.exists(self.index_output) or not os.path.exists(self.metadata_output):
            print("Vector database files not found. Please build the knowledge base first.")
            return []
            
        if not self.model:
            print("Sentence transformer model not available. Cannot search vector database.")
            return []
            
        # Load index and metadata
        index = faiss.read_index(self.index_output)
        with open(self.metadata_output, 'r') as f:
            metadata = json.load(f)
            
        # Convert query to embedding
        query_embedding = self.model.encode([query])
        
        # Normalize embedding for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search the index
        distances, indices = index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:  # No more results
                continue
                
            chunk_meta = metadata["chunks"][idx]
            results.append({
                "text": self.documents[chunk_meta["document_index"]]["chunks"][chunk_meta["position"]]["text"],
                "metadata": chunk_meta,
                "source": chunk_meta["source"],
                "similarity": float(distances[0][i])
            })
            
        return results
    
    def _search_relational(self, query, top_k=5):
        """Search the relational database using full-text search."""
        if not os.path.exists(self.relational_db_output):
            print("Relational database file not found. Please build the knowledge base first.")
            return []
            
        # Connect to database
        conn = sqlite3.connect(self.relational_db_output)
        cursor = conn.cursor()
        
        # Search using FTS
        cursor.execute('''
        SELECT c.id, c.text, c.metadata, d.source, d.title, d.author, d.subject
        FROM chunks_fts as fts
        JOIN chunks as c ON fts.rowid = c.id
        JOIN documents as d ON c.document_id = d.id
        WHERE fts.text MATCH ?
        ORDER BY rank
        LIMIT ?
        ''', (query, top_k))
        
        results = []
        for chunk_id, text, metadata_json, source, title, author, subject in cursor.fetchall():
            metadata = json.loads(metadata_json) if metadata_json else {}
            metadata.update({
                "source": source,
                "title": title,
                "author": author,
                "subject": subject
            })
            
            results.append({
                "text": text,
                "metadata": metadata,
                "source": source,
                "id": chunk_id
            })
            
        conn.close()
        return results
    
    def get_teaching_strategies(self):
        """Get the predefined teaching strategies."""
        return TEACHING_STRATEGIES
        
    def get_student_characteristics(self, grade="second"):
        """Get student characteristics for a specific grade level."""
        if grade.lower() == "second":
            return SECOND_GRADE_CHARACTERISTICS
        else:
            # Default to second grade if others not implemented
            return SECOND_GRADE_CHARACTERISTICS


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process educational documents into a knowledge base")
    parser.add_argument("--kb-dir", default="./knowledge_base/books", 
                        help="Directory containing educational documents")
    parser.add_argument("--db-type", choices=["vector", "relational"], default="vector",
                        help="Type of database to create")
    parser.add_argument("--index-output", default="vector_index.index",
                        help="Output path for vector index")
    parser.add_argument("--metadata-output", default="vector_metadata.json",
                        help="Output path for vector metadata")
    parser.add_argument("--db-file", default="knowledge.db",
                        help="Output path for relational database")
    parser.add_argument("--chunk-size", type=int, default=500,
                        help="Size of text chunks in characters")
    parser.add_argument("--chunk-overlap", type=int, default=50,
                        help="Overlap between text chunks in characters")
                        
    args = parser.parse_args()
    
    # Create and run knowledge manager
    manager = PedagogicalKnowledgeManager(
        kb_dir=args.kb_dir,
        db_type=args.db_type,
        index_output=args.index_output,
        metadata_output=args.metadata_output,
        relational_db_output=args.db_file,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    print(f"Processing documents from {args.kb_dir}...")
    manager.process_documents()
    
    print(f"Building {args.db_type} knowledge base...")
    manager.build_knowledge_base()
    
    print("Knowledge base creation complete!")
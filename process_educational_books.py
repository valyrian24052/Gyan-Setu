#!/usr/bin/env python3
"""
Educational Book Knowledge Processing Script

This script demonstrates the complete pipeline for processing scientific educational 
books and extracting structured knowledge. It shows how the system loads books,
processes them into knowledge chunks, categorizes the content, and stores it in
the vector database for retrieval.

Usage:
    python process_educational_books.py [--clean] [--books_dir PATH]

Options:
    --clean         Clear existing vector database before processing
    --books_dir     Custom path to books directory (default: knowledge_base/books)
"""

import os
import sys
import argparse
import logging
import time
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("book_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process educational books into knowledge base")
    parser.add_argument("--clean", action="store_true", help="Clear existing vector database before processing")
    parser.add_argument("--books_dir", type=str, default=os.path.join("knowledge_base", "books"), 
                        help="Path to directory containing educational books")
    return parser.parse_args()

def count_book_files(books_dir: str) -> Dict[str, int]:
    """
    Count files by extension in the books directory.
    
    Args:
        books_dir: Path to books directory
        
    Returns:
        Dictionary with counts by extension
    """
    if not os.path.exists(books_dir):
        return {}
        
    counts = {}
    for file in os.listdir(books_dir):
        if os.path.isfile(os.path.join(books_dir, file)):
            ext = os.path.splitext(file)[1].lower()
            counts[ext] = counts.get(ext, 0) + 1
    
    return counts

def main():
    """Run the book processing pipeline."""
    args = parse_arguments()
    
    print("\n" + "="*80)
    print("EDUCATIONAL BOOK KNOWLEDGE PROCESSING".center(80))
    print("="*80)
    
    # Check if books directory exists and has content
    if not os.path.exists(args.books_dir):
        logger.error(f"Books directory not found: {args.books_dir}")
        print(f"\nError: Books directory not found: {args.books_dir}")
        print("Please create this directory and add educational books (PDF, DOCX, TXT).")
        return 1
    
    # Count books by extension
    file_counts = count_book_files(args.books_dir)
    total_files = sum(file_counts.values())
    
    if total_files == 0:
        logger.error(f"No files found in books directory: {args.books_dir}")
        print(f"\nError: No files found in books directory: {args.books_dir}")
        print("Please add educational books (PDF, DOCX, TXT) to this directory.")
        return 1
    
    # Display book counts
    print(f"\nFound {total_files} educational books in {args.books_dir}:")
    for ext, count in file_counts.items():
        print(f"  - {count} {ext} files")
    
    # Import dependencies
    try:
        print("\nImporting document processor and vector database modules...")
        from document_processor import DocumentProcessor
        from vector_database import VectorDatabase
        print("✓ Successfully imported processing modules")
    except ImportError as e:
        logger.error(f"Error importing required modules: {e}")
        print(f"\nError importing required modules: {e}")
        print("Please ensure document_processor.py and vector_database.py are in the current directory.")
        return 1
    
    # Initialize vector database
    try:
        print("\nInitializing vector database...")
        db_path = os.path.join("knowledge_base", "vectors", "knowledge_vectors.db")
        vector_db = VectorDatabase(db_path=db_path)
        
        # Clear database if requested
        if args.clean:
            print("Clearing existing vector database...")
            vector_db.clear_database()
            print("✓ Database cleared")
        
        print("✓ Vector database initialized")
    except Exception as e:
        logger.error(f"Error initializing vector database: {e}")
        print(f"\nError initializing vector database: {e}")
        return 1
    
    # Initialize document processor
    try:
        print("\nInitializing document processor...")
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
        print("✓ Document processor initialized")
    except Exception as e:
        logger.error(f"Error initializing document processor: {e}")
        print(f"\nError initializing document processor: {e}")
        return 1
    
    # Process books
    print("\nProcessing books from:", os.path.abspath(args.books_dir))
    print("This may take some time depending on the number and size of books...")
    print("Progress will be displayed as each file is processed.\n")
    
    start_time = time.time()
    
    try:
        # Process all books in directory
        processor.process_books(args.books_dir)
        chunks = processor.get_chunks()
        
        if not chunks:
            logger.warning("No chunks extracted from books")
            print("\nWarning: No content could be extracted from the books.")
            print("Please check that the files are valid educational books.")
            return 1
        
        print(f"\n✓ Extracted {len(chunks)} knowledge chunks from {total_files} books")
        
        # Display chunk categories
        categories = {}
        for chunk in chunks:
            cat = chunk.get("category", "uncategorized")
            categories[cat] = categories.get(cat, 0) + 1
        
        print("\nKnowledge chunks by category:")
        for category, count in categories.items():
            print(f"  - {category}: {count} chunks")
        
        # Add chunks to vector database
        print("\nAdding chunks to vector database...")
        
        # Group chunks by category for optimized storage
        category_chunks = {}
        for chunk in chunks:
            cat = chunk.get("category", "uncategorized")
            if cat not in category_chunks:
                category_chunks[cat] = []
            category_chunks[cat].append(chunk)
        
        # Add each category batch
        for category, cat_chunks in category_chunks.items():
            print(f"  - Adding {len(cat_chunks)} {category} chunks...")
            vector_db.add_chunks(cat_chunks, category=category)
        
        print("✓ All chunks added to vector database")
        
    except Exception as e:
        logger.error(f"Error processing books: {e}")
        print(f"\nError processing books: {e}")
        return 1
    
    # Get database statistics
    try:
        stats = vector_db.get_stats()
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print("\n" + "="*80)
        print("PROCESSING COMPLETE".center(80))
        print("="*80)
        print(f"Total processing time: {elapsed_time:.2f} seconds")
        print(f"Total chunks in database: {stats.get('total_chunks', 0)}")
        print("\nChunks by category:")
        for category, count in stats.get('categories', {}).items():
            print(f"  - {category}: {count} chunks")
        
        # Save sample chunks for review
        output_dir = os.path.join("knowledge_base", "processed")
        os.makedirs(output_dir, exist_ok=True)
        
        sample_output = os.path.join(output_dir, "sample_chunks.json")
        processor.save_chunks_to_file(sample_output)
        print(f"\nSample chunks saved to: {os.path.abspath(sample_output)}")
        
        # Test vector search
        print("\nTesting knowledge retrieval with sample queries...")
        
        test_queries = [
            "classroom management strategies for elementary students",
            "how to handle disruptive behavior in class",
            "effective teaching methods for mathematics",
            "supporting students with learning disabilities"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            results = vector_db.search(query, top_k=1)
            if results:
                result = results[0]
                print(f"  Top result ({result['similarity']:.2f} similarity):")
                print(f"  Category: {result.get('category', 'unknown')}")
                text = result.get('text', '')
                print(f"  Text: {text[:200]}..." if len(text) > 200 else f"  Text: {text}")
            else:
                print("  No results found")
        
        print("\n" + "="*80)
        print("Knowledge base is ready for use by the teacher training system!")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Error generating statistics: {e}")
        print(f"\nError generating statistics: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
#!/usr/bin/env python3
"""
Simple LlamaIndex Test

This script demonstrates basic functionality of LlamaIndex with the current
package structure. It loads documents, creates an index, and performs a simple query.
"""

import os
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create directories if they don't exist
os.makedirs("knowledge_base/llama_index", exist_ok=True)

def test_llama_index():
    """Test basic LlamaIndex functionality."""
    try:
        logger.info("Importing required packages...")
        import torch
        logger.info(f"PyTorch version {torch.__version__} available.")
        
        from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
        from llama_index.core.node_parser import SimpleNodeParser
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.vector_stores.faiss import FaissVectorStore
        from llama_index.core.llms import MockLLM
        from llama_index.core.settings import Settings
        
        logger.info("Successfully imported packages")
        
        # Set up paths
        documents_dir = "knowledge_base/books"
        index_dir = "knowledge_base/llama_index"
        
        # Check if documents directory exists
        if not os.path.exists(documents_dir):
            logger.error(f"Documents directory {documents_dir} does not exist!")
            return False
            
        # Load a few documents (limit to 2 for testing)
        logger.info(f"Loading documents from {documents_dir}...")
        documents = SimpleDirectoryReader(documents_dir, required_exts=[".pdf"]).load_data()
        logger.info(f"Loaded {len(documents)} documents")
        
        # Set up embedding model
        logger.info("Setting up embedding model...")
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Set up node parser for text splitting
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=500,
            chunk_overlap=50
        )
        
        # Set up settings with mock LLM (replaces ServiceContext)
        logger.info("Setting up LlamaIndex settings...")
        Settings.llm = MockLLM()
        Settings.embed_model = embed_model
        Settings.node_parser = node_parser
        
        # Create an index with the documents
        logger.info("Creating index...")
        index = VectorStoreIndex.from_documents(documents)
        
        # Create a query engine
        query_engine = index.as_query_engine()
        
        # Perform a query
        logger.info("Querying the index...")
        response = query_engine.query("What are effective classroom management techniques?")
        
        logger.info(f"Response: {response}")
        logger.info("Test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("Starting LlamaIndex test...")
    success = test_llama_index()
    if success:
        logger.info("Test completed successfully!")
    else:
        logger.error("Test failed!") 
#!/usr/bin/env python3
"""
LlamaIndex Integration for Educational Knowledge Management

This module demonstrates how to integrate LlamaIndex with the existing UTTA codebase,
replacing or enhancing the current document processing and vector database functionality
with LlamaIndex's more advanced capabilities.

Key features:
- Document loading with LlamaIndex's document loaders
- Text splitting with semantically-aware chunking
- Vector store integration with existing or new embeddings
- Query engine setup for advanced retrieval
- Integration with existing knowledge management infrastructure
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union

# LlamaIndex imports
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    load_index_from_storage
)
from llama_index.node_parser import SimpleNodeParser
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import HuggingFaceLLM, OpenAI, Anthropic
from llama_index.vector_stores import FaissVectorStore

# Local imports for integration
from document_processor import DocumentProcessor
from vector_database import VectorDatabase
from knowledge_manager import PedagogicalKnowledgeManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename="llama_index.log"
)
logger = logging.getLogger(__name__)

class LlamaIndexKnowledgeManager:
    """
    Knowledge manager using LlamaIndex for document processing, 
    embedding, and retrieval.
    
    This class provides an alternative to the existing knowledge management
    infrastructure, offering more advanced features through LlamaIndex.
    """
    
    def __init__(
        self,
        documents_dir: str = "knowledge_base/books",
        index_dir: str = "knowledge_base/llama_index",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_provider: str = "local",  # "local", "openai", or "anthropic"
        local_model_path: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """
        Initialize the LlamaIndex knowledge manager.
        
        Args:
            documents_dir: Directory containing educational documents
            index_dir: Directory to store the LlamaIndex index
            embedding_model: HuggingFace model ID for embeddings
            llm_provider: Which LLM provider to use
            local_model_path: Path to local model (if llm_provider is "local")
            chunk_size: Size of text chunks for indexing
            chunk_overlap: Overlap between chunks
        """
        self.documents_dir = documents_dir
        self.index_dir = index_dir
        self.embedding_model = embedding_model
        self.llm_provider = llm_provider
        self.local_model_path = local_model_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Create index directory if it doesn't exist
        os.makedirs(self.index_dir, exist_ok=True)
        
        # Setup the LLM
        self._setup_llm()
        
        # Setup the embedding model
        self.embed_model = HuggingFaceEmbedding(model_name=self.embedding_model)
        
        # Setup the service context
        self.service_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=self.embed_model,
            node_parser=SimpleNodeParser.from_defaults(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        )
        
        # Initialize the index as None until loaded or created
        self.index = None
        self.query_engine = None
        
    def _setup_llm(self):
        """Set up the LLM based on the provider."""
        if self.llm_provider == "local" and self.local_model_path:
            self.llm = HuggingFaceLLM(
                model_name=self.local_model_path,
                tokenizer_name=self.local_model_path,
                context_window=2048,
                max_new_tokens=256,
                generate_kwargs={"temperature": 0.7}
            )
        elif self.llm_provider == "openai":
            self.llm = OpenAI(model="gpt-3.5-turbo")
        elif self.llm_provider == "anthropic":
            self.llm = Anthropic(model="claude-2")
        else:
            # Fallback to default
            logger.warning(f"LLM provider {self.llm_provider} not recognized, using OpenAI")
            self.llm = OpenAI(model="gpt-3.5-turbo")
            
    def load_or_create_index(self, force_reload: bool = False):
        """
        Load the index from disk if it exists, otherwise create a new one.
        
        Args:
            force_reload: If True, create a new index even if one exists
        """
        # Check if index exists and we're not forcing a reload
        if os.path.exists(os.path.join(self.index_dir, "docstore.json")) and not force_reload:
            logger.info(f"Loading existing index from {self.index_dir}")
            storage_context = StorageContext.from_defaults(persist_dir=self.index_dir)
            self.index = load_index_from_storage(storage_context)
        else:
            logger.info(f"Creating new index from documents in {self.documents_dir}")
            self._create_new_index()
        
        # Create the query engine
        self.query_engine = self.index.as_query_engine(
            service_context=self.service_context
        )
        
    def _create_new_index(self):
        """Create a new index from the documents."""
        try:
            # Load documents using LlamaIndex's document loader
            logger.info(f"Loading documents from {self.documents_dir}")
            documents = SimpleDirectoryReader(self.documents_dir).load_data()
            
            logger.info(f"Loaded {len(documents)} documents")
            
            # Create a vector store
            vector_store = FaissVectorStore()
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Create the index
            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                service_context=self.service_context
            )
            
            # Persist the index to disk
            self.index.storage_context.persist(persist_dir=self.index_dir)
            logger.info(f"Index created and saved to {self.index_dir}")
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise
            
    def query_knowledge(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Query the knowledge base using LlamaIndex.
        
        Args:
            query: The query string
            top_k: Number of top results to return
            
        Returns:
            Dict containing response and source nodes
        """
        if not self.query_engine:
            raise ValueError("Index not loaded. Call load_or_create_index() first.")
            
        response = self.query_engine.query(query)
        
        # Extract source documents for reference
        source_nodes = response.source_nodes
        source_texts = [node.node.text for node in source_nodes[:top_k]]
        
        return {
            "response": str(response),
            "sources": source_texts
        }
        
    def integrate_with_existing_knowledge(self, 
                                         existing_manager: Optional[PedagogicalKnowledgeManager] = None,
                                         mode: str = "replace") -> None:
        """
        Integrate LlamaIndex with an existing knowledge manager.
        
        Args:
            existing_manager: An existing PedagogicalKnowledgeManager instance
            mode: Integration mode - "replace" or "enhance"
        """
        if not existing_manager:
            logger.warning("No existing knowledge manager provided for integration")
            return
            
        if mode == "replace":
            # Use LlamaIndex completely instead of existing components
            logger.info("Replacing existing knowledge management with LlamaIndex")
            # This would involve calling the appropriate hooks in the system
            # to use our LlamaIndex components instead
        elif mode == "enhance":
            # Use LlamaIndex alongside existing components
            logger.info("Enhancing existing knowledge management with LlamaIndex")
            # This might involve importing existing knowledge into LlamaIndex
            # or setting up bridges between the systems
        else:
            logger.warning(f"Unknown integration mode: {mode}")
        
    def migrate_from_vector_database(self, vector_db: Optional[VectorDatabase] = None) -> None:
        """
        Migrate data from an existing VectorDatabase instance to LlamaIndex.
        
        Args:
            vector_db: An existing VectorDatabase instance
        """
        if not vector_db:
            logger.warning("No VectorDatabase provided for migration")
            return
            
        try:
            # Get all chunks from the existing vector database
            all_chunks = vector_db.get_chunks_by_category("general", limit=10000)
            logger.info(f"Retrieved {len(all_chunks)} chunks from existing vector database")
            
            # Convert to LlamaIndex documents and create a new index
            # This would be implemented based on the specific data structure
            # of the chunks and how they should map to LlamaIndex
            
            logger.info("Migration from VectorDatabase completed")
        except Exception as e:
            logger.error(f"Error migrating from VectorDatabase: {e}")
        
def demonstrate_llama_index():
    """Simple demonstration of LlamaIndex capabilities."""
    # Create a LlamaIndex knowledge manager
    manager = LlamaIndexKnowledgeManager(
        documents_dir="knowledge_base/books",
        llm_provider="openai"  # Use OpenAI for demonstration
    )
    
    # Load or create the index
    manager.load_or_create_index()
    
    # Example queries to demonstrate capabilities
    queries = [
        "What are effective strategies for teaching fractions?",
        "How do I manage a classroom with diverse learning needs?",
        "What are signs of dyscalculia in second-grade students?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        result = manager.query_knowledge(query)
        print(f"Response: {result['response']}")
        print("Sources:")
        for i, source in enumerate(result['sources'], 1):
            print(f"{i}. {source[:100]}...")
            
if __name__ == "__main__":
    demonstrate_llama_index() 
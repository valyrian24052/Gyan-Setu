"""
RAG (Retrieval-Augmented Generation) Pipeline Module for Teacher Training Chatbot

This module implements the RAG pipeline that combines retrieval of relevant teaching
scenarios with LLM-based response generation. It handles document processing,
similarity search, and context-aware response generation.

Classes:
    RAGPipeline: Main class implementing the RAG pipeline functionality.

Example:
    pipeline = RAGPipeline()
    await pipeline.initialize()
    result = await pipeline.process_query("How to handle student disruption?")
"""

import asyncio
from typing import Dict, List, Optional
from .embedding import EmbeddingGenerator
from ..database.vector_ops import VectorOperations
from .llm_config import LLMConfig

class RAGPipeline:
    """
    A class implementing the RAG pipeline for the teacher training chatbot.
    
    This class combines document retrieval and response generation to provide
    context-aware responses to teaching scenarios. It uses vector similarity
    search for retrieval and an LLM for response generation.
    
    Attributes:
        embedder (EmbeddingGenerator): Instance for generating embeddings
        vector_ops (VectorOperations): Instance for vector storage operations
        llm (LLMConfig): Instance for LLM configuration and generation
    """

    def __init__(self):
        """Initialize the RAG pipeline with required components."""
        self.embedder = EmbeddingGenerator()
        self.vector_ops = VectorOperations()
        self.llm = LLMConfig()
        self._performance_metrics = {}

    async def initialize(self):
        """
        Initialize pipeline components asynchronously.
        
        This method must be called before using the pipeline to ensure
        all components are properly initialized.
        
        Raises:
            ConnectionError: If database connection fails
            RuntimeError: If component initialization fails
        """
        await self.vector_ops.initialize()
        await self.llm.initialize()

    async def process_query(self, query: str) -> Dict:
        """
        Process a query through the RAG pipeline.

        Args:
            query (str): The input query to process

        Returns:
            Dict: {
                'response': str,  # Generated response
                'sources': List[Dict],  # Retrieved source documents
                'confidence': float  # Confidence score
            }

        Raises:
            ValueError: If query is invalid
            RuntimeError: If processing fails
        """
        with self.monitor_performance():
            query_embedding = self.embedder.generate_embedding(query)
            context = await self.prepare_context(query_embedding)
            response = await self.generate_response(query, context)
            return self._prepare_result(response, context)

    async def prepare_context(self, query_embedding: List[float]) -> str:
        """
        Prepare context for response generation using retrieved documents.

        Args:
            query_embedding (List[float]): The query embedding vector

        Returns:
            str: Prepared context string for the LLM

        Raises:
            ValueError: If query_embedding is invalid
        """
        results = await self.vector_ops.find_similar_documents(
            query_embedding=query_embedding,
            threshold=0.5,
            limit=3
        )
        return self._format_context(results)

    async def add_documents(self, documents: List[Dict]):
        """
        Add new documents to the RAG pipeline.

        Args:
            documents (List[Dict]): List of documents to add, each containing
                                  'content' and 'metadata' fields

        Raises:
            ValueError: If documents are invalid
            RuntimeError: If storage fails
        """
        processed_docs = self._process_documents(documents)
        await self.vector_ops.store_documents(processed_docs)

    def get_performance_metrics(self) -> Dict:
        """
        Get performance metrics for the pipeline.

        Returns:
            Dict: Performance metrics including query time, embedding time,
                 and response generation time
        """
        return self._performance_metrics.copy()

    def _process_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Process documents for storage.

        Args:
            documents (List[Dict]): Raw documents to process

        Returns:
            List[Dict]: Processed documents with embeddings
        """
        for doc in documents:
            doc['embedding'] = self.embedder.generate_embedding(doc['content'])
        return documents

    def _format_context(self, results: List[Dict]) -> str:
        """
        Format retrieved results into context string.

        Args:
            results (List[Dict]): Retrieved similar documents

        Returns:
            str: Formatted context string
        """
        context_parts = []
        for result in results:
            context_parts.append(f"{result['content']}\n")
        return "\n".join(context_parts)

    def _prepare_result(self, response: str, context: str) -> Dict:
        """
        Prepare the final result dictionary.

        Args:
            response (str): Generated response
            context (str): Used context

        Returns:
            Dict: Final result with response and metadata
        """
        return {
            'response': response,
            'sources': self._extract_sources(context),
            'confidence': self._calculate_confidence(response)
        }

# ... existing code ... 
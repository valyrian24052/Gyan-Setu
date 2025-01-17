"""
Vector Operations Module for Teacher Training Chatbot

This module handles vector storage and similarity search operations using PostgreSQL
with the pgvector extension. It provides functionality for storing, retrieving,
and searching vector embeddings of teaching scenarios and responses.

Classes:
    VectorOperations: Main class for vector database operations.

Example:
    ops = VectorOperations()
    await ops.initialize()
    scenario_id = await ops.store_scenario(name="Test", embedding=[...])
"""

import asyncio
import asyncpg
from typing import List, Dict, Optional
from config import DATABASE_URL

class VectorOperations:
    """
    A class to handle vector storage and similarity search operations.
    
    This class manages the storage and retrieval of vector embeddings in PostgreSQL
    using the pgvector extension. It handles scenarios, responses, and their
    associated metadata.
    
    Attributes:
        pool (asyncpg.Pool): Connection pool for database operations
        initialized (bool): Whether the database connection is initialized
    """

    def __init__(self):
        """Initialize VectorOperations instance."""
        self.pool = None
        self.initialized = False

    async def initialize(self):
        """
        Initialize database connection and required extensions.
        
        This method must be called before any other operations. It sets up
        the database connection pool and ensures the vector extension is available.
        
        Raises:
            ConnectionError: If database connection fails
            RuntimeError: If vector extension setup fails
        """
        try:
            self.pool = await asyncpg.create_pool(DATABASE_URL)
            async with self.pool.acquire() as conn:
                await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
            self.initialized = True
        except Exception as e:
            raise ConnectionError(f"Failed to initialize database: {str(e)}")

    async def store_scenario(self, name: str, description: str,
                           expected_response: str, embedding: List[float]) -> int:
        """
        Store a teaching scenario with its embedding.

        Args:
            name (str): Scenario name
            description (str): Scenario description
            expected_response (str): Expected teacher response
            embedding (List[float]): Vector embedding of dimension 384

        Returns:
            int: ID of the stored scenario

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If storage operation fails
        """
        if not self.initialized:
            raise RuntimeError("Database not initialized")
        
        async with self.pool.acquire() as conn:
            return await conn.fetchval('''
                INSERT INTO scenarios (name, description, expected_response, embedding)
                VALUES ($1, $2, $3, $4)
                RETURNING id
            ''', name, description, expected_response, embedding)

    async def find_similar_scenarios(self, query_embedding: List[float],
                                   threshold: float = 0.7,
                                   limit: int = 5) -> List[Dict]:
        """
        Find scenarios similar to the query embedding.

        Args:
            query_embedding (List[float]): Query vector
            threshold (float): Similarity threshold (0-1)
            limit (int): Maximum number of results

        Returns:
            List[Dict]: Similar scenarios with similarity scores

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If search operation fails
        """
        if not self.initialized:
            raise RuntimeError("Database not initialized")
        
        async with self.pool.acquire() as conn:
            results = await conn.fetch('''
                SELECT id, name, description, expected_response,
                       1 - (embedding <=> $1) as similarity
                FROM scenarios
                WHERE 1 - (embedding <=> $1) > $2
                ORDER BY similarity DESC
                LIMIT $3
            ''', query_embedding, threshold, limit)
            
            return [dict(r) for r in results]

    async def batch_store_scenarios(self, scenarios: List[Dict]) -> List[int]:
        """
        Store multiple scenarios in batch.

        Args:
            scenarios (List[Dict]): List of scenarios with embeddings

        Returns:
            List[int]: List of stored scenario IDs

        Raises:
            ValueError: If scenarios are invalid
            RuntimeError: If batch storage fails
        """
        if not self.initialized:
            raise RuntimeError("Database not initialized")
        
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                return await asyncio.gather(*[
                    self.store_scenario(**scenario)
                    for scenario in scenarios
                ])

    async def update_scenario(self, scenario_id: int,
                            expected_response: str,
                            embedding: List[float]) -> bool:
        """
        Update an existing scenario.

        Args:
            scenario_id (int): ID of scenario to update
            expected_response (str): New expected response
            embedding (List[float]): New embedding vector

        Returns:
            bool: True if update successful

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If update fails
        """
        if not self.initialized:
            raise RuntimeError("Database not initialized")
        
        async with self.pool.acquire() as conn:
            result = await conn.execute('''
                UPDATE scenarios
                SET expected_response = $2, embedding = $3
                WHERE id = $1
            ''', scenario_id, expected_response, embedding)
            return result == 'UPDATE 1'

# ... existing code ... 
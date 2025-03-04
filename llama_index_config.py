#!/usr/bin/env python3
"""
LlamaIndex Configuration

This module centralizes all configuration options for the LlamaIndex integration.
It allows for easy modification of settings in one place.
"""

import os
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Base paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge_base")
BOOKS_DIR = os.path.join(KNOWLEDGE_DIR, "books")
INDEX_DIR = os.path.join(KNOWLEDGE_DIR, "llama_index")
CACHE_DIR = os.path.join(KNOWLEDGE_DIR, "query_cache")

# Make sure directories exist
for directory in [KNOWLEDGE_DIR, BOOKS_DIR, INDEX_DIR, CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Default LLM settings
DEFAULT_LLM_PROVIDER = os.environ.get("LLAMA_INDEX_LLM_PROVIDER", "openai")
DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
DEFAULT_ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
DEFAULT_EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Embedding settings
EMBEDDING_DIMENSIONS = 384  # Dimensions for all-MiniLM-L6-v2

# Document processing settings
CHUNK_SIZE = int(os.environ.get("LLAMA_INDEX_CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.environ.get("LLAMA_INDEX_CHUNK_OVERLAP", "50"))

# Caching settings
ENABLE_CACHING = os.environ.get("LLAMA_INDEX_ENABLE_CACHE", "true").lower() == "true"
CACHE_TTL = int(os.environ.get("LLAMA_INDEX_CACHE_TTL", "3600"))  # Default 1 hour

# Retrieval settings
DEFAULT_TOP_K = int(os.environ.get("LLAMA_INDEX_TOP_K", "5"))
SIMILARITY_CUTOFF = float(os.environ.get("LLAMA_INDEX_SIMILARITY_CUTOFF", "0.7"))

def get_llm_settings(provider: str = None) -> Dict[str, Any]:
    """
    Get LLM-specific settings based on the provider.
    
    Args:
        provider: LLM provider ("openai", "anthropic", or "local")
        
    Returns:
        Dictionary of LLM settings
    """
    if provider is None:
        provider = DEFAULT_LLM_PROVIDER
    
    provider = provider.lower()
    
    if provider == "openai":
        return {
            "model": DEFAULT_OPENAI_MODEL,
            "temperature": float(os.environ.get("OPENAI_TEMPERATURE", "0.7")),
            "api_key": os.environ.get("OPENAI_API_KEY", ""),
            "streaming": True
        }
    elif provider == "anthropic":
        return {
            "model": DEFAULT_ANTHROPIC_MODEL,
            "temperature": float(os.environ.get("ANTHROPIC_TEMPERATURE", "0.7")),
            "api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
            "streaming": True
        }
    elif provider == "local":
        # For a local model like LlamaCpp
        model_path = os.environ.get("LOCAL_MODEL_PATH", "")
        return {
            "model_path": model_path,
            "temperature": float(os.environ.get("LOCAL_TEMPERATURE", "0.7")),
            "max_tokens": int(os.environ.get("LOCAL_MAX_TOKENS", "2048")),
            "streaming": False
        }
    else:
        # Default to MockLLM for testing
        return {
            "model": "mock",
            "temperature": 0.7,
            "streaming": False
        }

def get_embed_model_settings() -> Dict[str, Any]:
    """
    Get embedding model settings.
    
    Returns:
        Dictionary of embedding model settings
    """
    return {
        "model_name": DEFAULT_EMBED_MODEL,
        "dimensions": EMBEDDING_DIMENSIONS,
        "cache_folder": os.path.join(BASE_DIR, "model_cache", "embedding")
    }

def get_document_settings() -> Dict[str, Any]:
    """
    Get document processing settings.
    
    Returns:
        Dictionary of document processing settings
    """
    return {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "documents_dir": BOOKS_DIR,
        "index_dir": INDEX_DIR
    }

def get_cache_settings() -> Dict[str, Any]:
    """
    Get caching settings.
    
    Returns:
        Dictionary of cache settings
    """
    return {
        "enable_caching": ENABLE_CACHING,
        "cache_dir": CACHE_DIR,
        "cache_ttl": CACHE_TTL
    }

def get_retrieval_settings() -> Dict[str, Any]:
    """
    Get retrieval settings.
    
    Returns:
        Dictionary of retrieval settings
    """
    return {
        "top_k": DEFAULT_TOP_K,
        "similarity_cutoff": SIMILARITY_CUTOFF
    }

def get_all_settings() -> Dict[str, Dict[str, Any]]:
    """
    Get all settings in a nested dictionary.
    
    Returns:
        Nested dictionary of all settings
    """
    return {
        "llm": get_llm_settings(),
        "embedding": get_embed_model_settings(),
        "document": get_document_settings(),
        "cache": get_cache_settings(),
        "retrieval": get_retrieval_settings(),
        "paths": {
            "base_dir": str(BASE_DIR),
            "knowledge_dir": KNOWLEDGE_DIR,
            "books_dir": BOOKS_DIR,
            "index_dir": INDEX_DIR,
            "cache_dir": CACHE_DIR
        }
    }

if __name__ == "__main__":
    """Print current configuration when run directly."""
    import json
    print(json.dumps(get_all_settings(), indent=2)) 
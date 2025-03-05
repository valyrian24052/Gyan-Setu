#!/usr/bin/env python3
"""
Quick script to fix imports in llama_index_integration.py
"""
import re

# Read the file
with open('llama_index_integration.py', 'r') as f:
    content = f.read()

# Apply import fixes
content = re.sub(
    r'from llama_index import \(',
    'from llama_index.core import (',
    content
)

content = re.sub(
    r'from llama_index\.core\.node_parser',
    'from llama_index.core.node_parser',
    content
)

content = re.sub(
    r'from llama_index\.core\.indices\.vector_store import VectorStoreIndex',
    'from llama_index.core.indices.vector_store import VectorStoreIndex',
    content
)

content = re.sub(
    r'from llama_index\.llms\.openai import OpenAI',
    'from llama_index.llms.openai import OpenAI',
    content
)

content = re.sub(
    r'from llama_index\.llms\.anthropic import Anthropic',
    'from llama_index.llms.anthropic import Anthropic',
    content
)

content = re.sub(
    r'from llama_index\.llms import MockLLM',
    'from llama_index.llms.mock import MockLLM',
    content
)

content = re.sub(
    r'from llama_index\.embeddings\.openai import OpenAIEmbedding',
    'from llama_index.embeddings.openai import OpenAIEmbedding',
    content
)

# Write back to file
with open('llama_index_integration.py', 'w') as f:
    f.write(content)

print("Imports fixed in llama_index_integration.py") 
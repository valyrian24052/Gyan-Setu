#!/usr/bin/env python3
"""
Book Knowledge Configuration

This module defines configuration parameters for the book-based knowledge system,
controlling document processing, vector database, and knowledge categorization.

These settings can be modified to customize the behavior of the knowledge extraction
and retrieval process without changing the core code.
"""

import os
from typing import Dict, List, Any, Union, Optional

# Define the base directory for knowledge storage
KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_base")

##############################################################################
# Document Processing Configuration
##############################################################################

# Document source directories
BOOKS_DIR = os.path.join(KNOWLEDGE_BASE_DIR, "books")
PROCESSED_DIR = os.path.join(KNOWLEDGE_BASE_DIR, "processed")

# Supported file formats for book processing
SUPPORTED_FORMATS = {
    "pdf": {"enabled": True, "priority": 1},
    "epub": {"enabled": True, "priority": 2},
    "txt": {"enabled": True, "priority": 3},
    "docx": {"enabled": False, "priority": 4},  # Not implemented yet
    "html": {"enabled": False, "priority": 5},  # Not implemented yet
}

# Document chunking settings
CHUNK_CONFIG = {
    "min_chunk_size": 150,            # Minimum words per chunk
    "max_chunk_size": 300,            # Maximum words per chunk
    "chunk_overlap": 50,              # Number of words to overlap between chunks
    "respect_paragraphs": True,       # Try to keep paragraphs intact
    "respect_sections": True,         # Try to keep sections intact
    "include_metadata": True,         # Include page numbers and other metadata
}

# Document preprocessing options
PREPROCESSING = {
    "remove_headers_footers": True,   # Attempt to remove headers and footers
    "remove_page_numbers": True,      # Remove page numbers
    "normalize_whitespace": True,     # Convert all whitespace to single spaces
    "remove_urls": False,             # Remove URLs from text
    "remove_citations": False,        # Remove citation markers [1], [2], etc.
    "remove_tables": False,           # Remove table content
    "trim_line_breaks": True,         # Trim excessive line breaks
}

##############################################################################
# Vector Database Configuration
##############################################################################

# Vector database settings
VECTOR_DB_CONFIG = {
    "db_path": os.path.join(KNOWLEDGE_BASE_DIR, "vector_db.sqlite"),
    "embedding_model": "all-MiniLM-L6-v2",  # SentenceTransformers model name
    "embedding_dimension": 384,             # Dimension of vectors (model-dependent)
    "use_sqlite": True,                     # Use SQLite for storage
    "in_memory_search": False,              # Load vectors into memory for faster search
    "max_database_size_mb": 1000,           # Maximum database size in MB
}

# Search configuration
SEARCH_CONFIG = {
    "default_limit": 5,                    # Default number of results to return
    "min_relevance_score": 0.25,           # Minimum relevance score (0-1)
    "category_boost": {                    # Boost certain categories during search
        "classroom_management": 1.2,
        "teaching_strategies": 1.1,
        "learning_theories": 1.0,
        "assessment": 0.9,
    },
    "recency_boost": True,                 # Boost more recent publications
}

##############################################################################
# Knowledge Categorization Configuration
##############################################################################

# Categories for knowledge chunks
CATEGORIES = [
    "classroom_management",
    "teaching_strategies",
    "learning_theories",
    "assessment",
    "student_development",
    "curriculum_design",
    "educational_technology",
    "special_education",
    "literacy",
    "mathematics",
    "science",
    "social_studies",
    "educational_psychology",
    "educational_policy",
    "diversity_and_inclusion",
    "professional_development",
    "other",
]

# Keywords associated with each category (for rule-based categorization)
CATEGORY_KEYWORDS = {
    "classroom_management": [
        "discipline", "behavior", "classroom management", "rules", "procedures",
        "routines", "disruption", "misbehavior", "order", "environment",
        "conduct", "expectations", "monitoring", "classroom climate"
    ],
    
    "teaching_strategies": [
        "instructional strategies", "teaching methods", "pedagogy", "instruction",
        "demonstration", "modeling", "scaffolding", "differentiation", "grouping",
        "inquiry", "project-based", "direct instruction", "lecture", "discussion"
    ],
    
    "learning_theories": [
        "constructivism", "behaviorism", "cognitivism", "social learning",
        "cognitive development", "schema", "zone of proximal development",
        "multiple intelligences", "learning styles", "growth mindset"
    ],
    
    "assessment": [
        "assessment", "evaluation", "grading", "test", "quiz", "rubric",
        "formative", "summative", "diagnostic", "feedback", "performance",
        "standards", "measurement", "data", "achievement"
    ],
    
    "student_development": [
        "development", "cognitive", "social", "emotional", "physical",
        "adolescent", "child", "stages", "milestones", "psychology",
        "motivation", "engagement", "self-regulation", "identity"
    ],
    
    "curriculum_design": [
        "curriculum", "standards", "objectives", "goals", "planning",
        "content", "scope and sequence", "unit", "lesson", "alignment",
        "integration", "design", "backward design", "outcomes"
    ],
    
    "educational_technology": [
        "technology", "digital", "online", "computers", "software",
        "apps", "internet", "multimedia", "interactive", "virtual",
        "blended", "hybrid", "e-learning", "educational technology"
    ],
    
    "special_education": [
        "special education", "disabilities", "IEP", "individualized",
        "accommodation", "modification", "inclusion", "exceptional",
        "intervention", "RTI", "504", "learning disability", "gifted"
    ],
    
    "literacy": [
        "literacy", "reading", "writing", "language arts", "vocabulary",
        "comprehension", "phonics", "phonemic awareness", "fluency",
        "spelling", "grammar", "composition", "text", "literature"
    ],
    
    "mathematics": [
        "mathematics", "math", "numeracy", "arithmetic", "algebra",
        "geometry", "calculus", "statistics", "problem solving",
        "computation", "mathematical", "number sense", "reasoning"
    ],
    
    "science": [
        "science", "scientific", "biology", "chemistry", "physics",
        "earth science", "inquiry", "experiment", "hypothesis",
        "observation", "investigation", "STEM", "laboratory"
    ],
    
    "social_studies": [
        "social studies", "history", "geography", "civics",
        "economics", "sociology", "anthropology", "political science",
        "cultural", "global", "citizenship", "society", "government"
    ],
    
    "educational_psychology": [
        "educational psychology", "cognition", "information processing",
        "memory", "attention", "perception", "learning theory", "transfer",
        "metacognition", "self-efficacy", "attribution", "executive function"
    ],
    
    "educational_policy": [
        "policy", "reform", "legislation", "standards", "accountability",
        "funding", "governance", "administration", "leadership", "law",
        "regulation", "mandate", "compliance", "initiative"
    ],
    
    "diversity_and_inclusion": [
        "diversity", "inclusion", "equity", "cultural", "multicultural",
        "social justice", "bias", "discrimination", "identity", "race",
        "ethnicity", "gender", "socioeconomic", "linguistic diversity"
    ],
    
    "professional_development": [
        "professional development", "teacher training", "in-service",
        "workshop", "continuing education", "mentoring", "coaching",
        "professional learning community", "collaboration", "reflection"
    ],
}

# Configuration for ML-based categorization (if used)
ML_CATEGORIZATION = {
    "enabled": False,              # Use ML for categorization
    "model_path": os.path.join(KNOWLEDGE_BASE_DIR, "models", "categorization_model.pkl"),
    "confidence_threshold": 0.6,   # Minimum confidence for category assignment
    "fallback_to_rules": True,     # Use rule-based as fallback
}

##############################################################################
# Scenario Generation Configuration
##############################################################################

# Knowledge integration settings for scenario generation
SCENARIO_KNOWLEDGE_CONFIG = {
    "max_knowledge_chunks": 7,     # Maximum number of knowledge chunks to use per scenario
    "min_knowledge_chunks": 2,     # Minimum number of knowledge chunks to use per scenario
    "diversity_factor": 0.7,       # 0-1 factor for diverse knowledge sources (1=max diversity)
    "include_book_references": True,  # Include references to source materials
    "include_rationales": True,    # Include rationales from research
}

# Subject-specific boosting for knowledge retrieval
SUBJECT_BOOST = {
    "math": {
        "boost_categories": ["mathematics", "teaching_strategies", "assessment"],
        "boost_factor": 1.5
    },
    "reading": {
        "boost_categories": ["literacy", "teaching_strategies", "assessment"],
        "boost_factor": 1.5
    },
    "science": {
        "boost_categories": ["science", "teaching_strategies", "assessment"],
        "boost_factor": 1.5
    },
    "social_studies": {
        "boost_categories": ["social_studies", "teaching_strategies", "assessment"],
        "boost_factor": 1.5
    },
    "classroom_management": {
        "boost_categories": ["classroom_management", "student_development"],
        "boost_factor": 2.0
    }
}

# Grade level emphasis
GRADE_LEVEL_CONFIG = {
    "elementary": {
        "emphasized_categories": ["classroom_management", "literacy", "mathematics"],
        "deemphasized_categories": ["educational_policy", "curriculum_design"]
    },
    "middle": {
        "emphasized_categories": ["student_development", "assessment", "content_areas"],
        "deemphasized_categories": ["early_literacy"]
    },
    "high": {
        "emphasized_categories": ["curriculum_design", "content_areas", "assessment"],
        "deemphasized_categories": ["early_literacy", "elementary_methods"]
    }
}

##############################################################################
# Logging and Debugging Configuration
##############################################################################

# Logging configuration
LOGGING_CONFIG = {
    "log_level": "INFO",           # DEBUG, INFO, WARNING, ERROR, CRITICAL
    "log_file": os.path.join(KNOWLEDGE_BASE_DIR, "book_knowledge.log"),
    "console_output": True,        # Output logs to console
    "file_output": True,           # Output logs to file
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
}

# Debug options
DEBUG_OPTIONS = {
    "save_chunks_to_json": True,   # Save processed chunks to JSON for inspection
    "save_vector_samples": True,   # Save sample vectors for inspection
    "track_processing_time": True, # Track and report processing time
    "validation_checks": True,     # Perform validation checks during processing
} 
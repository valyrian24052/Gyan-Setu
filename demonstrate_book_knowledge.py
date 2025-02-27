#!/usr/bin/env python3
"""
Book-Based Knowledge Demonstration

This script demonstrates how the teacher training system uses knowledge extracted
from scientific educational books to generate realistic classroom scenarios and
evidence-based teaching strategies.

Usage:
    python demonstrate_book_knowledge.py [--scenarios NUM] [--subject SUBJECT]

Options:
    --scenarios    Number of scenarios to generate (default: 3)
    --subject      Subject area to focus on (math, reading, general)
"""

import os
import sys
import argparse
import json
import random
from typing import List, Dict, Any, Optional
import time

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Demonstrate book-based knowledge in teaching scenarios")
    parser.add_argument("--scenarios", type=int, default=3, help="Number of scenarios to generate")
    parser.add_argument("--subject", type=str, default="general", 
                       choices=["math", "reading", "general"], 
                       help="Subject area to focus on")
    return parser.parse_args()

def format_text_box(title: str, content: str, width: int = 80, padding: int = 1) -> str:
    """
    Format text within a box.
    
    Args:
        title: Box title
        content: Content text
        width: Box width
        padding: Padding size
        
    Returns:
        Formatted text box as string
    """
    lines = []
    lines.append("┌" + "─" * (width - 2) + "┐")
    
    # Title
    title_line = "│" + title.center(width - 2) + "│"
    lines.append(title_line)
    lines.append("├" + "─" * (width - 2) + "┤")
    
    # Content
    content_lines = content.split("\n")
    for line in content_lines:
        # Handle long lines by wrapping
        if len(line) > width - (2 + padding * 2):
            current_line = ""
            words = line.split()
            for word in words:
                if len(current_line) + len(word) + 1 <= width - (2 + padding * 2):
                    current_line += word + " "
                else:
                    padded_line = "│" + " " * padding + current_line.ljust(width - 2 - padding * 2) + " " * padding + "│"
                    lines.append(padded_line)
                    current_line = word + " "
            
            if current_line:
                padded_line = "│" + " " * padding + current_line.ljust(width - 2 - padding * 2) + " " * padding + "│"
                lines.append(padded_line)
        else:
            padded_line = "│" + " " * padding + line.ljust(width - 2 - padding * 2) + " " * padding + "│"
            lines.append(padded_line)
            
    lines.append("└" + "─" * (width - 2) + "┘")
    return "\n".join(lines)

def main():
    """Run the demonstration."""
    args = parse_arguments()
    
    print("\n" + "="*80)
    print("SCIENTIFIC BOOK-BASED TEACHING SCENARIOS".center(80))
    print("="*80)
    print("\nThis demonstration shows how the teacher training system uses knowledge")
    print("extracted from educational books to generate realistic classroom scenarios")
    print("and evidence-based teaching strategies.\n")
    
    # Import dependencies
    try:
        print("Importing required modules...")
        from document_processor import DocumentProcessor
        from vector_database import VectorDatabase
        from scenario_generator import ClassroomScenarioGenerator
        print("✓ Successfully imported modules")
    except ImportError as e:
        print(f"\nError importing required modules: {e}")
        print("Please ensure all necessary modules are in the current directory.")
        return 1
    
    # Initialize vector database
    vector_db_path = os.path.join("knowledge_base", "vectors", "knowledge_vectors.db")
    default_db_path = os.path.join("knowledge_base", "vector_db.sqlite")
    
    # Check if vector database exists
    if os.path.exists(vector_db_path):
        db_path = vector_db_path
    elif os.path.exists(default_db_path):
        db_path = default_db_path
    else:
        print("\nNo vector database found. Using fallback knowledge only.")
        db_path = vector_db_path
    
    print(f"\nUsing vector database: {db_path}")
    vector_db = VectorDatabase(db_path=db_path)
    
    # Get database statistics
    stats = vector_db.get_stats()
    total_chunks = stats.get('total_chunks', 0)
    
    if total_chunks > 0:
        print(f"✓ Found {total_chunks} knowledge chunks from educational books")
        print("\nKnowledge by category:")
        for category, count in stats.get('categories', {}).items():
            print(f"  - {category}: {count} chunks")
    else:
        print("! No knowledge chunks found in database.")
        print("  Using fallback knowledge from default strategies files.")
    
    # Initialize scenario generator (without LLM for this demo)
    generator = ClassroomScenarioGenerator(vector_db)
    
    # Generate scenarios
    print(f"\nGenerating {args.scenarios} teaching scenarios based on educational research...")
    print(f"Subject focus: {args.subject}")
    
    scenarios = []
    
    for i in range(args.scenarios):
        # Select subject based on user choice
        if args.subject == "general":
            subject = random.choice(["math", "reading", "science", "social studies"])
        else:
            subject = args.subject
            
        # Select grade level
        grade_level = random.choice(["elementary", "middle", "high"])
        
        # Generate scenario
        scenario = generator.generate_scenario(
            grade_level=grade_level,
            subject=subject
        )
        
        scenarios.append(scenario)
        print(f"✓ Generated scenario {i+1}: {scenario['title']}")
    
    # Display scenarios with knowledge attribution
    print("\nDisplaying scenarios with their scientific knowledge sources:\n")
    
    for i, scenario in enumerate(scenarios, 1):
        print("\n" + "="*80)
        print(f"SCENARIO {i}: {scenario['title']}".center(80))
        print("="*80 + "\n")
        
        # Display scenario content
        print(format_text_box("CLASSROOM SCENARIO", scenario['scenario_content']))
        
        # Display knowledge sources
        print("\n" + "-"*40)
        print("EVIDENCE-BASED STRATEGIES".center(40))
        print("-"*40 + "\n")
        
        print("Based on these educational research concepts:")
        
        for j, source in enumerate(scenario.get('knowledge_sources', []), 1):
            category = source.get('category', 'unknown')
            source_name = source.get('source', 'unknown')
            
            # Format the content based on type
            content = source.get('content', {})
            
            if isinstance(content, dict) and 'strategies' in content:
                # It's a structured content
                strategies = content.get('strategies', '')
                examples = content.get('examples', '')
                
                print(f"\n{j}. From {category} research ({source_name}):")
                print(f"   {strategies}")
                
                if examples:
                    print(f"   Example: {examples}")
            elif isinstance(content, dict) and 'manifestations' in content:
                # It's behavior content
                print(f"\n{j}. From {category} research ({source_name}):")
                if content.get('strategies'):
                    for strategy in content.get('strategies', [])[:2]:
                        print(f"   - {strategy}")
                if content.get('rationale'):
                    print(f"   Rationale: {content.get('rationale', '')[:100]}...")
            elif isinstance(content, str) and len(content) > 0:
                # It's a text chunk from vector DB
                print(f"\n{j}. From educational literature ({source_name}):")
                # Show first 150 chars
                print(f"   \"{content[:150]}...\"")
                if 'relevance' in source:
                    print(f"   Relevance: {source['relevance']:.2f}")
        
        # Display recommended strategies
        print("\n" + "-"*40)
        print("RECOMMENDED TEACHER ACTIONS".center(40))
        print("-"*40 + "\n")
        
        strategies = scenario.get('recommended_strategies', [])
        if strategies:
            for strategy in strategies:
                print(f"• {strategy}")
        else:
            print("No specific strategies available for this scenario.")
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE".center(80))
    print("="*80)
    print("\nThis demonstration shows how scientific educational books inform the")
    print("teaching scenarios and recommendations in the training system.")
    print("All strategies and suggestions are based on evidence from educational")
    print("research literature processed by the system.\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
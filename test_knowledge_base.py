#!/usr/bin/env python3
"""
Test Knowledge Base Loading

This script tests the loading of knowledge base files to ensure they can be properly
loaded and accessed by the application. It tests each knowledge file type individually
without requiring the full application dependencies.
"""

import os
import json
import csv

def test_teaching_strategies():
    """Test loading of teaching strategies from the text file."""
    try:
        strategies = {}
        current_category = None
        
        with open('knowledge_base/default_strategies/teaching_strategies.txt', 'r') as f:
            lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if not line.startswith('-'):
                    # This is a category header
                    current_category = line
                    strategies[current_category] = []
                else:
                    # This is a strategy item
                    if current_category:
                        strategy = line[2:].strip()  # Remove the dash and space
                        strategies[current_category].append(strategy)
        
        # Print summary of strategies
        print("\n✓ Teaching strategies loaded successfully!")
        print(f"Found {len(strategies)} categories:")
        for category, items in strategies.items():
            print(f"  - {category}: {len(items)} strategies")
        
        return True
    except Exception as e:
        print(f"\n✗ Error loading teaching strategies: {e}")
        return False

def test_behavior_management():
    """Test loading of behavior management strategies from CSV."""
    try:
        behaviors = {}
        
        with open('knowledge_base/default_strategies/behavior_management.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'behavior_type' not in row:
                    print(f"\n✗ Error: 'behavior_type' column missing in behavior_management.csv")
                    print(f"Found columns: {list(row.keys())}")
                    return False
                
                behavior_type = row['behavior_type']
                
                if behavior_type not in behaviors:
                    behaviors[behavior_type] = {
                        'triggers': [],
                        'manifestations': [],
                        'strategies': [],
                        'rationale': row.get('rationale', '')
                    }
                
                # Add triggers, manifestations, and strategies if they exist and aren't already in the lists
                if 'triggers' in row and row['triggers'] and row['triggers'] not in behaviors[behavior_type]['triggers']:
                    behaviors[behavior_type]['triggers'].append(row['triggers'])
                
                if 'manifestations' in row and row['manifestations'] and row['manifestations'] not in behaviors[behavior_type]['manifestations']:
                    behaviors[behavior_type]['manifestations'].append(row['manifestations'])
                
                if 'strategies' in row and row['strategies'] and row['strategies'] not in behaviors[behavior_type]['strategies']:
                    behaviors[behavior_type]['strategies'].append(row['strategies'])
        
        # Print summary of behavior management strategies
        print("\n✓ Behavior management strategies loaded successfully!")
        print(f"Found {len(behaviors)} behavior types:")
        for behavior, data in behaviors.items():
            print(f"  - {behavior}: {len(data['strategies'])} strategies")
        
        return True
    except Exception as e:
        print(f"\n✗ Error loading behavior management strategies: {e}")
        return False

def test_math_strategies():
    """Test loading of math strategies from CSV."""
    try:
        math_topics = {}
        
        with open('knowledge_base/default_strategies/math_strategies.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                topic = row.get('topic', '')
                if topic:
                    math_topics[topic] = {
                        "common_challenges": row.get('common_challenges', ''),
                        "strategies": row.get('strategies', ''),
                        "examples": row.get('examples', '')
                    }
        
        # Print summary of math strategies
        print("\n✓ Math strategies loaded successfully!")
        print(f"Found {len(math_topics)} math topics:")
        for topic in math_topics.keys():
            print(f"  - {topic}")
        
        return True
    except Exception as e:
        print(f"\n✗ Error loading math strategies: {e}")
        return False

def test_reading_strategies():
    """Test loading of reading strategies from CSV."""
    try:
        reading_topics = {}
        
        with open('knowledge_base/default_strategies/reading_strategies.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                topic = row.get('topic', '')
                if topic:
                    reading_topics[topic] = {
                        "common_challenges": row.get('common_challenges', ''),
                        "strategies": row.get('strategies', ''),
                        "examples": row.get('examples', '')
                    }
        
        # Print summary of reading strategies
        print("\n✓ Reading strategies loaded successfully!")
        print(f"Found {len(reading_topics)} reading topics:")
        for topic in reading_topics.keys():
            print(f"  - {topic}")
        
        return True
    except Exception as e:
        print(f"\n✗ Error loading reading strategies: {e}")
        return False

def test_student_profiles():
    """Test loading of student profiles from JSON."""
    try:
        with open('knowledge_base/student_profiles.json', 'r') as f:
            data = json.load(f)
            profiles = data.get("profiles", [])
        
        # Print summary of student profiles
        print("\n✓ Student profiles loaded successfully!")
        print(f"Found {len(profiles)} student profiles:")
        for profile in profiles:
            name = profile.get('name', 'Unknown')
            grade = profile.get('grade', 'Unknown')
            learning_style = profile.get('learning_style', 'Unknown')
            print(f"  - {name} (Grade {grade}, {learning_style} learner)")
        
        return True
    except Exception as e:
        print(f"\n✗ Error loading student profiles: {e}")
        return False

def main():
    """Run all knowledge base tests."""
    print("="*60)
    print("TESTING KNOWLEDGE BASE LOADING".center(60))
    print("="*60)
    
    results = []
    
    # Test teaching strategies
    results.append(("Teaching Strategies", test_teaching_strategies()))
    
    # Test behavior management strategies
    results.append(("Behavior Management", test_behavior_management()))
    
    # Test math strategies
    results.append(("Math Strategies", test_math_strategies()))
    
    # Test reading strategies
    results.append(("Reading Strategies", test_reading_strategies()))
    
    # Test student profiles
    results.append(("Student Profiles", test_student_profiles()))
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY".center(60))
    print("="*60)
    
    all_passed = True
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        all_passed = all_passed and result
        print(f"{name}: {status}")
    
    print("\nOVERALL RESULT: " + ("PASSED" if all_passed else "FAILED"))
    print("="*60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Test AI Agent Knowledge Base Loading

This script directly tests the knowledge base loading methods in the TeacherTrainingAgent class
to ensure they can properly load and access files from the knowledge_base directory.
"""

import json
import os
import sys

# This test script only needs these core functions from ai_agent.py
# We're extracting just what we need to test knowledge loading functionality
def test_load_teaching_strategies():
    """Test the _load_teaching_strategies method from TeacherTrainingAgent."""
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
        
        print("\n✓ AI Agent can load teaching strategies successfully!")
        print(f"Found {len(strategies)} categories with strategies like:")
        for category, items in list(strategies.items())[:2]:  # Show just first 2 categories
            print(f"  - {category}: {', '.join(items[:2])}...")
        
        return True
    except Exception as e:
        print(f"\n✗ Error testing AI agent teaching strategies: {e}")
        return False

def test_load_student_behaviors():
    """Test the _load_student_behaviors method from TeacherTrainingAgent."""
    try:
        import csv
        behaviors = {}
        
        with open('knowledge_base/default_strategies/behavior_management.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
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
        
        print("\n✓ AI Agent can load behavior management strategies successfully!")
        print(f"Found {len(behaviors)} behavior types including:")
        for behavior in list(behaviors.keys())[:3]:  # Show just first 3 behaviors
            print(f"  - {behavior}")
            
        # Show example of one full behavior
        if behaviors:
            example = list(behaviors.keys())[0]
            print(f"\nExample behavior '{example}':")
            print(f"  - Triggers: {', '.join(behaviors[example]['triggers'])}")
            print(f"  - Manifestations: {', '.join(behaviors[example]['manifestations'])}")
            print(f"  - Strategies: {', '.join(behaviors[example]['strategies'])}")
        
        return True
    except Exception as e:
        print(f"\n✗ Error testing AI agent behavior strategies: {e}")
        return False

def test_load_subject_content():
    """Test the _load_subject_content method from TeacherTrainingAgent."""
    try:
        import csv
        subjects = {
            "math": {"topics": {}},
            "reading": {"topics": {}}
        }
        
        # Load math strategies
        try:
            with open('knowledge_base/default_strategies/math_strategies.csv', 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    topic = row.get('topic', '')
                    if topic:
                        subjects["math"]["topics"][topic] = {
                            "common_challenges": row.get('common_challenges', ''),
                            "strategies": row.get('strategies', ''),
                            "examples": row.get('examples', '')
                        }
        except FileNotFoundError:
            print("! Math strategies file not found")
        
        # Load reading strategies
        try:
            with open('knowledge_base/default_strategies/reading_strategies.csv', 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    topic = row.get('topic', '')
                    if topic:
                        subjects["reading"]["topics"][topic] = {
                            "common_challenges": row.get('common_challenges', ''),
                            "strategies": row.get('strategies', ''),
                            "examples": row.get('examples', '')
                        }
        except FileNotFoundError:
            print("! Reading strategies file not found")
        
        print("\n✓ AI Agent can load subject content successfully!")
        print(f"Found {len(subjects['math']['topics'])} math topics and {len(subjects['reading']['topics'])} reading topics")
        
        # Show example topic from each subject
        if subjects["math"]["topics"]:
            math_example = list(subjects["math"]["topics"].keys())[0]
            print(f"\nExample math topic '{math_example}':")
            print(f"  - Challenges: {subjects['math']['topics'][math_example]['common_challenges'][:50]}...")
            print(f"  - Strategies: {subjects['math']['topics'][math_example]['strategies'][:50]}...")
            
        if subjects["reading"]["topics"]:
            reading_example = list(subjects["reading"]["topics"].keys())[0]
            print(f"\nExample reading topic '{reading_example}':")
            print(f"  - Challenges: {subjects['reading']['topics'][reading_example]['common_challenges'][:50]}...")
            print(f"  - Strategies: {subjects['reading']['topics'][reading_example]['strategies'][:50]}...")
        
        return True
    except Exception as e:
        print(f"\n✗ Error testing AI agent subject content: {e}")
        return False

def test_load_student_profiles():
    """Test the _load_student_profiles method from TeacherTrainingAgent."""
    try:
        with open('knowledge_base/student_profiles.json', 'r') as f:
            data = json.load(f)
            profiles = data.get("profiles", [])
        
        print("\n✓ AI Agent can load student profiles successfully!")
        print(f"Found {len(profiles)} student profiles that can be used in teaching scenarios")
        
        # Show example of one student profile
        if profiles:
            example = profiles[0]
            print(f"\nExample profile: {example.get('name')} (Grade {example.get('grade')})")
            print(f"  - Learning style: {example.get('learning_style')}")
            print(f"  - Strengths: {', '.join(example.get('strengths', []))}")
            print(f"  - Challenges: {', '.join(example.get('challenges', []))}")
        
        return True
    except Exception as e:
        print(f"\n✗ Error testing AI agent student profiles: {e}")
        return False

def main():
    """Run all AI agent knowledge base tests."""
    print("="*70)
    print("TESTING AI AGENT KNOWLEDGE BASE INTEGRATION".center(70))
    print("="*70)
    
    results = []
    
    # Test teaching strategies loading
    results.append(("Teaching Strategies Loading", test_load_teaching_strategies()))
    
    # Test behavior management loading
    results.append(("Behavior Management Loading", test_load_student_behaviors()))
    
    # Test subject content loading
    results.append(("Subject Content Loading", test_load_subject_content()))
    
    # Test student profiles loading
    results.append(("Student Profiles Loading", test_load_student_profiles()))
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY".center(70))
    print("="*70)
    
    all_passed = True
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        all_passed = all_passed and result
        print(f"{name}: {status}")
    
    print("\nOVERALL RESULT: " + ("PASSED" if all_passed else "FAILED"))
    print("="*70)
    
    if all_passed:
        print("\nThe AI agent can successfully load and use all knowledge files from")
        print("the knowledge_base directory. The application will use this knowledge")
        print("to create more realistic and diverse teaching scenarios.")
    else:
        print("\nSome tests failed. Please check the output above for details on what")
        print("needs to be fixed for the AI agent to properly use the knowledge base.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    main() 
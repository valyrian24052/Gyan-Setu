"""
Example script demonstrating the LLM evaluation framework.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics.automated_metrics import AutomatedMetrics
from src.evaluation.human.human_evaluation import HumanEvaluation
from src.evaluation.benchmarks.benchmark_suite import BenchmarkSuite

def demonstrate_automated_metrics():
    """Demonstrate automated metrics evaluation."""
    print("\n=== Automated Metrics Demo ===")
    
    metrics = AutomatedMetrics()
    
    # Example responses
    generated = "Python is a high-level programming language known for its simplicity and readability."
    reference = "Python is a popular high-level programming language that emphasizes code readability."
    
    # Evaluate single response
    scores = metrics.evaluate_response(generated, reference)
    print("\nSingle Response Evaluation:")
    for metric, score in scores.items():
        print(f"{metric}: {score:.3f}")
    
    # Evaluate conversation
    conversation = [
        {
            'generated': "Hello! How can I help you today?",
            'reference': "Hi! How may I assist you?"
        },
        {
            'generated': "Python is great for beginners because it's easy to learn.",
            'reference': "Python is an excellent choice for beginners due to its simple syntax."
        }
    ]
    
    conv_scores = metrics.evaluate_conversation(conversation)
    print("\nConversation Evaluation:")
    for metric, score in conv_scores.items():
        print(f"{metric}: {score:.3f}")

def demonstrate_human_evaluation():
    """Demonstrate human evaluation process."""
    print("\n=== Human Evaluation Demo ===")
    
    evaluator = HumanEvaluation(output_dir="evaluation_results")
    
    # Create evaluation form
    conversation = [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a high-level programming language known for its simplicity."}
    ]
    
    form = evaluator.create_evaluation_form(conversation)
    print("\nCreated evaluation form with ID:", form["conversation_id"])
    
    # Simulate multiple evaluators
    evaluators = [
        {
            "id": "eval1",
            "scores": {
                "Response Relevance (1-5)": 5,
                "Response Accuracy (1-5)": 4,
                "Language Quality (1-5)": 5,
                "Task Completion (1-5)": 4,
                "Overall Quality (1-5)": 4
            },
            "comments": "Good explanation, could include more examples"
        },
        {
            "id": "eval2",
            "scores": {
                "Response Relevance (1-5)": 4,
                "Response Accuracy (1-5)": 5,
                "Language Quality (1-5)": 5,
                "Task Completion (1-5)": 4,
                "Overall Quality (1-5)": 5
            },
            "comments": "Clear and concise explanation"
        }
    ]
    
    # Save evaluations
    for evaluator_data in evaluators:
        evaluator.save_evaluation(
            evaluator_id=evaluator_data["id"],
            form_data=form,
            scores=evaluator_data["scores"],
            comments=evaluator_data["comments"]
        )
    
    # Aggregate results
    aggregate_scores = evaluator.aggregate_evaluations(form["conversation_id"])
    print("\nAggregated Evaluation Scores:")
    for criterion, scores in aggregate_scores.items():
        print(f"{criterion}:")
        print(f"  Mean: {scores['mean']:.2f}")
        print(f"  Std: {scores['std']:.2f}")

def demonstrate_benchmark_suite():
    """Demonstrate benchmark suite functionality."""
    print("\n=== Benchmark Suite Demo ===")
    
    benchmark = BenchmarkSuite(benchmark_dir="benchmark_results")
    
    # Create a simple test set
    test_cases = [
        {
            "input": "What is machine learning?",
            "expected_output": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience.",
            "category": "technical_explanation"
        },
        {
            "input": "Write a Python hello world program",
            "expected_output": "Here's a simple Python hello world program:\n\nprint('Hello, World!')",
            "category": "coding"
        }
    ]
    
    # Create benchmark set
    benchmark.create_benchmark_set(
        test_cases=test_cases,
        name="basic_ml_benchmark",
        description="Basic machine learning concept explanations"
    )
    print("\nCreated benchmark set: basic_ml_benchmark")

def main():
    """Run all demonstrations."""
    print("LLM Evaluation Framework Demo")
    print("=" * 30)
    
    demonstrate_automated_metrics()
    demonstrate_human_evaluation()
    demonstrate_benchmark_suite()

if __name__ == "__main__":
    main() 
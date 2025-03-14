"""
Comprehensive Benchmark Suite for Teacher Training Chatbot

This module provides a suite of benchmarks to evaluate the performance
of the teacher training chatbot across various scenarios and metrics.
"""

import yaml
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from tqdm import tqdm

from ..metrics.automated_metrics import AutomatedMetricsEvaluator
from ...llm.llm_handler import EnhancedLLMInterface
from ...core.vector_database import VectorDatabase

@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    scenario_name: str
    metrics: Dict[str, float]
    response_times: List[float]
    memory_usage: List[float]
    detailed_feedback: Dict[str, Any]

class BenchmarkSuite:
    """Comprehensive benchmark suite for the teacher training chatbot."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the benchmark suite.
        
        Args:
            config_path: Path to benchmark configuration file
        """
        self.config = self._load_config(config_path)
        self.llm = EnhancedLLMInterface()
        self.evaluator = AutomatedMetricsEvaluator()
        self.vector_db = VectorDatabase()
        
        # Ensure results directory exists
        self.results_dir = Path("benchmark_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load benchmark configuration."""
        default_config = {
            "scenarios": {
                "basic_math": {
                    "grade_level": 2,
                    "subject": "math",
                    "topic": "addition",
                    "student_profile": {
                        "learning_style": "visual",
                        "challenges": ["number sense"],
                        "strengths": ["pattern recognition"]
                    }
                },
                "reading_comprehension": {
                    "grade_level": 3,
                    "subject": "reading",
                    "topic": "main idea",
                    "student_profile": {
                        "learning_style": "auditory",
                        "challenges": ["vocabulary"],
                        "strengths": ["critical thinking"]
                    }
                }
            },
            "metrics": {
                "response_time_threshold": 5.0,  # seconds
                "memory_limit": 1024,  # MB
                "min_clarity_score": 0.7,
                "min_engagement_score": 0.6
            }
        }
        
        if config_path:
            with open(config_path) as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
                
        return default_config
    
    def run_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmarks and collect results."""
        results = []
        
        for scenario_name, scenario_config in tqdm(self.config["scenarios"].items()):
            # Run single scenario benchmark
            result = self.benchmark_scenario(scenario_name, scenario_config)
            results.append(result)
            
            # Save intermediate results
            self._save_result(result)
            
        # Generate and save summary report
        self._generate_summary_report(results)
        
        return results
    
    def benchmark_scenario(self, scenario_name: str, config: Dict[str, Any]) -> BenchmarkResult:
        """Run benchmark for a single scenario."""
        response_times = []
        memory_usage = []
        all_metrics = []
        
        # Generate test cases for the scenario
        test_cases = self._generate_test_cases(config)
        
        for test_case in test_cases:
            # Measure response time
            start_time = time.time()
            response = self.llm.get_llm_response_with_context(
                messages=[{"role": "user", "content": test_case["input"]}],
                context_data=test_case["context"]
            )
            response_time = time.time() - start_time
            response_times.append(response_time)
            
            # Measure memory usage
            memory_usage.append(self._get_memory_usage())
            
            # Evaluate response
            metrics = self.evaluator.evaluate_response(
                response,
                test_case["student_profile"],
                test_case["context"]
            )
            all_metrics.append(metrics)
        
        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(all_metrics)
        
        # Generate detailed feedback
        detailed_feedback = self._generate_detailed_feedback(
            aggregated_metrics,
            response_times,
            memory_usage,
            self.config["metrics"]
        )
        
        return BenchmarkResult(
            scenario_name=scenario_name,
            metrics=aggregated_metrics,
            response_times=response_times,
            memory_usage=memory_usage,
            detailed_feedback=detailed_feedback
        )
    
    def _generate_test_cases(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate test cases for a scenario."""
        test_cases = []
        
        # Basic test case
        test_cases.append({
            "input": f"How would you explain {config['topic']} to a {config['grade_level']}th grade student?",
            "context": {
                "subject": config["subject"],
                "topic": config["topic"],
                "grade_level": config["grade_level"]
            },
            "student_profile": config["student_profile"]
        })
        
        # Challenge test case
        test_cases.append({
            "input": f"The student is struggling with {config['topic']}. How would you help?",
            "context": {
                "subject": config["subject"],
                "topic": config["topic"],
                "grade_level": config["grade_level"],
                "difficulty": "challenging"
            },
            "student_profile": config["student_profile"]
        })
        
        return test_cases
    
    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics from multiple test cases."""
        aggregated = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list]
            aggregated[key] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
        return aggregated
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _generate_detailed_feedback(self,
                                  metrics: Dict[str, float],
                                  response_times: List[float],
                                  memory_usage: List[float],
                                  thresholds: Dict[str, float]) -> Dict[str, Any]:
        """Generate detailed feedback based on benchmark results."""
        feedback = {
            "performance_summary": {
                "average_response_time": np.mean(response_times),
                "max_response_time": max(response_times),
                "average_memory_usage": np.mean(memory_usage),
                "peak_memory_usage": max(memory_usage)
            },
            "metrics_summary": {
                metric: value
                for metric, value in metrics.items()
                if not metric.endswith("_std")
            },
            "warnings": [],
            "recommendations": []
        }
        
        # Check response time
        if max(response_times) > thresholds["response_time_threshold"]:
            feedback["warnings"].append(
                f"Response time exceeded threshold: {max(response_times):.2f}s"
            )
            feedback["recommendations"].append(
                "Consider optimizing prompt length or using response caching"
            )
        
        # Check memory usage
        if max(memory_usage) > thresholds["memory_limit"]:
            feedback["warnings"].append(
                f"Memory usage exceeded limit: {max(memory_usage):.2f}MB"
            )
            feedback["recommendations"].append(
                "Consider implementing memory-efficient processing"
            )
        
        # Check quality metrics
        if metrics["clarity_score"] < thresholds["min_clarity_score"]:
            feedback["warnings"].append(
                f"Clarity score below threshold: {metrics['clarity_score']:.2f}"
            )
            feedback["recommendations"].append(
                "Review and optimize response templates for clarity"
            )
        
        return feedback
    
    def _save_result(self, result: BenchmarkResult):
        """Save benchmark result to file."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        result_file = self.results_dir / f"benchmark_{result.scenario_name}_{timestamp}.json"
        
        with open(result_file, "w") as f:
            json.dump({
                "scenario_name": result.scenario_name,
                "metrics": result.metrics,
                "response_times": result.response_times,
                "memory_usage": result.memory_usage,
                "detailed_feedback": result.detailed_feedback
            }, f, indent=2)
    
    def _generate_summary_report(self, results: List[BenchmarkResult]):
        """Generate and save a summary report of all benchmark results."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        report_file = self.results_dir / f"benchmark_summary_{timestamp}.md"
        
        with open(report_file, "w") as f:
            f.write("# Benchmark Summary Report\n\n")
            
            for result in results:
                f.write(f"## Scenario: {result.scenario_name}\n\n")
                
                # Performance metrics
                f.write("### Performance Metrics\n")
                f.write("```\n")
                f.write(f"Average Response Time: {np.mean(result.response_times):.2f}s\n")
                f.write(f"Peak Memory Usage: {max(result.memory_usage):.2f}MB\n")
                f.write("```\n\n")
                
                # Quality metrics
                f.write("### Quality Metrics\n")
                f.write("```\n")
                for metric, value in result.metrics.items():
                    if not metric.endswith("_std"):
                        f.write(f"{metric}: {value:.2f}\n")
                f.write("```\n\n")
                
                # Recommendations
                if result.detailed_feedback["recommendations"]:
                    f.write("### Recommendations\n")
                    for rec in result.detailed_feedback["recommendations"]:
                        f.write(f"- {rec}\n")
                    f.write("\n")
                
                f.write("---\n\n") 
#!/usr/bin/env python3
"""
Advanced Testing and Benchmarking Tools for LlamaIndex Integration

This script provides advanced testing and production monitoring tools for the LlamaIndex 
integration, including response quality evaluation, cost estimation, long-running benchmarks, 
automated regression testing, and performance visualization.
"""

import os
import sys
import time
import json
import argparse
import logging
import datetime
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path

# Import plotting libraries (if available)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

# Import LlamaIndex components - updated imports for compatibility with newer versions
try:
    from llama_index_integration import LlamaIndexKnowledgeManager
    from llama_index_config import get_llm_settings, get_all_settings
except ImportError as e:
    print(f"Error importing LlamaIndex components: {str(e)}")
    print("Please check that llama_index_integration.py has been updated for your LlamaIndex version.")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger("advanced_testing")

# Constants
DEFAULT_RESULTS_DIR = Path("test_results")
DEFAULT_DB_PATH = DEFAULT_RESULTS_DIR / "benchmark_history.sqlite"
OPENAI_COSTS = {
    "gpt-3.5-turbo": 0.0015,  # per 1K tokens
    "gpt-4": 0.03,  # per 1K tokens
    "gpt-4-turbo": 0.01,  # per 1K tokens
    "gpt-4o": 0.005,  # per 1K tokens
    "text-embedding-ada-002": 0.0001  # per 1K tokens
}
ANTHROPIC_COSTS = {
    "claude-3-sonnet-20240229": 0.015,  # per 1K tokens
    "claude-3-opus-20240229": 0.045,  # per 1K tokens
    "claude-3-haiku-20240307": 0.0025,  # per 1K tokens
}

# Initialize results directory
DEFAULT_RESULTS_DIR.mkdir(exist_ok=True, parents=True)

def estimate_token_count(text: str) -> int:
    """
    Estimate the number of tokens in a text string.
    This is a rough estimate based on words/characters.
    
    Args:
        text: The text to estimate tokens for
        
    Returns:
        Estimated token count
    """
    # Simple estimation: ~4 chars per token on average
    return max(1, len(text) // 4)

def estimate_query_cost(
    provider: str,
    query: str,
    response: str,
    source_texts: List[str],
    model_name: Optional[str] = None
) -> Dict[str, float]:
    """
    Estimate the cost of a query based on token counts and provider rates.
    
    Args:
        provider: The LLM provider used
        query: The query text
        response: The response text
        source_texts: The texts of retrieved sources
        model_name: The specific model name (if None, use default)
        
    Returns:
        Dictionary with cost breakdown
    """
    # Skip cost estimate for local models
    if provider == "local":
        return {"total_cost": 0.0, "details": "Local models have no API costs"}
    
    # Get provider-specific settings
    if not model_name:
        settings = get_llm_settings(provider)
        if provider == "openai":
            model_name = settings.get("model", "gpt-3.5-turbo")
        elif provider == "anthropic":
            model_name = settings.get("model", "claude-3-sonnet-20240229")
    
    # Get token rates
    if provider == "openai":
        rate = OPENAI_COSTS.get(model_name, 0.002)  # Default if unknown
        embedding_rate = OPENAI_COSTS.get("text-embedding-ada-002", 0.0001)
    elif provider == "anthropic":
        rate = ANTHROPIC_COSTS.get(model_name, 0.015)  # Default if unknown
        embedding_rate = OPENAI_COSTS.get("text-embedding-ada-002", 0.0001)  # Using OpenAI embeddings
    else:
        return {"total_cost": 0.0, "details": f"Unknown provider: {provider}"}
    
    # Estimate token counts
    query_tokens = estimate_token_count(query)
    response_tokens = estimate_token_count(response)
    source_tokens = sum(estimate_token_count(text) for text in source_texts)
    
    # Calculate costs
    query_cost = (query_tokens / 1000) * rate
    response_cost = (response_tokens / 1000) * rate
    embedding_cost = (source_tokens / 1000) * embedding_rate
    total_cost = query_cost + response_cost + embedding_cost
    
    return {
        "total_cost": total_cost,
        "query_cost": query_cost,
        "response_cost": response_cost,
        "embedding_cost": embedding_cost,
        "query_tokens": query_tokens,
        "response_tokens": response_tokens,
        "source_tokens": source_tokens,
        "rate_per_1k": rate,
        "embedding_rate_per_1k": embedding_rate,
        "model": model_name
    }

def evaluate_response_quality(
    query: str,
    response: str,
    sources: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Evaluate the quality of a response using heuristic metrics.
    
    Args:
        query: The original query
        response: The response text
        sources: The source information (list of dictionaries)
        
    Returns:
        Dictionary with quality metrics
    """
    # Extract source texts for analysis
    source_texts = [s.get("text", "") if isinstance(s, dict) else str(s) for s in sources]
    
    # Basic metrics
    metrics = {}
    
    # 1. Response length (normalized to a 0-1 score, assuming ideal length 100-1000 chars)
    resp_len = len(response)
    metrics["length_score"] = min(1.0, max(0.0, (resp_len - 50) / 950)) if resp_len < 1000 else 1.0
    
    # 2. Source diversity - average similarity between sources (lower is better)
    if len(source_texts) > 1:
        diversity_scores = []
        for i in range(len(source_texts)):
            for j in range(i+1, len(source_texts)):
                # Simple overlap calculation
                s1 = set(source_texts[i].lower().split())
                s2 = set(source_texts[j].lower().split())
                if not s1 or not s2:
                    continue
                overlap = len(s1.intersection(s2)) / len(s1.union(s2))
                diversity_scores.append(1.0 - overlap)  # Higher score = more diverse
        
        metrics["diversity_score"] = sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0.5
    else:
        metrics["diversity_score"] = 0.5  # Neutral score for single source
    
    # 3. Query relevance - check if query terms appear in response
    query_terms = set(query.lower().split())
    query_terms = {term for term in query_terms if len(term) > 3}  # Filter out short words
    if query_terms:
        response_terms = set(response.lower().split())
        overlap = len(query_terms.intersection(response_terms)) / len(query_terms)
        metrics["relevance_score"] = overlap
    else:
        metrics["relevance_score"] = 0.5
    
    # 4. Source utilization - check if response contains content from sources
    source_utilization_scores = []
    for source_text in source_texts:
        if not source_text:
            continue
            
        # Extract meaningful terms from source (excluding stopwords)
        source_terms = set(source_text.lower().split())
        source_terms = {term for term in source_terms if len(term) > 4}  # Focus on meaningful terms
        
        if not source_terms:
            continue
            
        # Sample a subset of terms to check (for efficiency)
        sample_size = min(20, len(source_terms))
        sample_terms = set(list(source_terms)[:sample_size])
        
        # Check what percentage of sampled terms appear in response
        if sample_terms:
            response_terms = set(response.lower().split())
            term_overlap = len(sample_terms.intersection(response_terms)) / len(sample_terms)
            source_utilization_scores.append(term_overlap)
    
    metrics["source_utilization"] = sum(source_utilization_scores) / len(source_utilization_scores) if source_utilization_scores else 0.0
    
    # 5. Overall quality score (weighted average of other metrics)
    metrics["overall_score"] = (
        metrics["length_score"] * 0.1 +
        metrics["diversity_score"] * 0.2 +
        metrics["relevance_score"] * 0.4 +
        metrics["source_utilization"] * 0.3
    )
    
    return metrics

def create_benchmark_database(db_path: Union[str, Path] = DEFAULT_DB_PATH):
    """
    Create or ensure the benchmark history database exists.
    
    Args:
        db_path: Path to the SQLite database file
    """
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS benchmark_runs (
        id INTEGER PRIMARY KEY,
        timestamp TEXT,
        query TEXT,
        provider TEXT,
        model TEXT,
        execution_time REAL,
        total_cost REAL,
        overall_quality REAL,
        query_tokens INTEGER,
        response_tokens INTEGER,
        source_count INTEGER
    )
    ''')
    
    # Create metrics table for detailed quality metrics
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS quality_metrics (
        benchmark_id INTEGER,
        metric_name TEXT,
        metric_value REAL,
        FOREIGN KEY (benchmark_id) REFERENCES benchmark_runs(id)
    )
    ''')
    
    conn.commit()
    conn.close()
    
    logger.info(f"Benchmark database initialized at: {db_path}")

def save_benchmark_result(
    result: Dict[str, Any],
    query: str,
    provider: str,
    db_path: Union[str, Path] = DEFAULT_DB_PATH
):
    """
    Save a benchmark result to the database.
    
    Args:
        result: The result dictionary from test_knowledge_query
        query: The query that was tested
        provider: The provider that was used
        db_path: Path to the SQLite database
    """
    # Create database if it doesn't exist
    if not Path(db_path).exists():
        create_benchmark_database(db_path)
    
    # Extract data from result
    timestamp = datetime.datetime.now().isoformat()
    execution_time = result.get("execution_time", 0.0)
    response = result.get("response", "")
    sources = result.get("sources", [])
    source_count = len(sources)
    
    # Extract source texts
    source_texts = []
    for source in sources:
        if isinstance(source, dict):
            source_texts.append(source.get("text", ""))
        else:
            source_texts.append(str(source))
    
    # Evaluate quality
    quality_metrics = evaluate_response_quality(query, response, sources)
    overall_quality = quality_metrics.get("overall_score", 0.0)
    
    # Estimate cost
    model_name = result.get("model_name", None)
    cost_estimate = estimate_query_cost(provider, query, response, source_texts, model_name)
    total_cost = cost_estimate.get("total_cost", 0.0)
    query_tokens = cost_estimate.get("query_tokens", 0)
    response_tokens = cost_estimate.get("response_tokens", 0)
    
    # Save to database
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Insert main benchmark record
    cursor.execute('''
    INSERT INTO benchmark_runs (
        timestamp, query, provider, model, execution_time, 
        total_cost, overall_quality, query_tokens, response_tokens, source_count
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        timestamp, query, provider, model_name, execution_time,
        total_cost, overall_quality, query_tokens, response_tokens, source_count
    ))
    
    # Get the ID of the inserted record
    benchmark_id = cursor.lastrowid
    
    # Insert quality metrics
    for metric_name, metric_value in quality_metrics.items():
        cursor.execute('''
        INSERT INTO quality_metrics (benchmark_id, metric_name, metric_value)
        VALUES (?, ?, ?)
        ''', (benchmark_id, metric_name, metric_value))
    
    conn.commit()
    conn.close()
    
    logger.info(f"Benchmark result saved to database: {db_path}")

def run_comprehensive_benchmark(
    queries: Optional[List[str]] = None,
    providers: Optional[List[str]] = None,
    documents_dir: Optional[str] = None,
    top_k: int = 5,
    verbose: bool = False,
    save_results: bool = True,
    db_path: Union[str, Path] = DEFAULT_DB_PATH
) -> Dict[str, Dict[str, Any]]:
    """
    Run a comprehensive benchmark across multiple queries and providers.
    
    Args:
        queries: List of queries to test (if None, use example queries)
        providers: List of providers to test (if None, use all available)
        documents_dir: Directory containing documents
        top_k: Number of results to retrieve
        verbose: Whether to print detailed output
        save_results: Whether to save results to database
        db_path: Path to the benchmark database
        
    Returns:
        Nested dictionary of results by provider and query
    """
    # Use default queries if none provided
    if queries is None:
        queries = example_queries()
    
    # Use all available providers if none specified
    if providers is None:
        providers = list_available_providers()
    
    results = {}
    
    # Initialize results structure
    for provider in providers:
        results[provider] = {}
    
    # Run benchmarks
    for provider in providers:
        logger.info(f"Benchmarking provider: {provider}")
        
        for query in queries:
            logger.info(f"Testing query: {query}")
            
            # Run the query
            result = test_knowledge_query(
                provider=provider,
                query=query,
                documents_dir=documents_dir,
                top_k=top_k,
                verbose=verbose
            )
            
            # Save to results dictionary
            results[provider][query] = result
            
            # Save to database if requested
            if save_results:
                save_benchmark_result(result, query, provider, db_path)
    
    # Print summary
    print("\n" + "=" * 70)
    print("COMPREHENSIVE BENCHMARK SUMMARY")
    print("=" * 70)
    
    for provider in providers:
        print(f"\nProvider: {provider}")
        print("-" * 50)
        
        times = []
        qualities = []
        costs = []
        
        for query in queries:
            result = results[provider][query]
            time_taken = result.get("execution_time", 0)
            times.append(time_taken)
            
            # Evaluate quality
            if "response" in result:
                quality = evaluate_response_quality(
                    query, 
                    result["response"], 
                    result.get("sources", [])
                )
                overall_quality = quality.get("overall_score", 0.0)
                qualities.append(overall_quality)
            
            # Estimate cost
            if "response" in result:
                source_texts = []
                for source in result.get("sources", []):
                    if isinstance(source, dict):
                        source_texts.append(source.get("text", ""))
                    else:
                        source_texts.append(str(source))
                
                cost = estimate_query_cost(
                    provider, 
                    query, 
                    result["response"], 
                    source_texts
                )
                total_cost = cost.get("total_cost", 0.0)
                costs.append(total_cost)
        
        if times:
            avg_time = sum(times) / len(times)
            print(f"Average Time: {avg_time:.2f}s")
        
        if qualities:
            avg_quality = sum(qualities) / len(qualities)
            print(f"Average Quality Score: {avg_quality:.2f}")
        
        if costs:
            avg_cost = sum(costs) / len(costs)
            total_cost = sum(costs)
            print(f"Average Cost: ${avg_cost:.5f}")
            print(f"Total Cost: ${total_cost:.5f}")
    
    print("=" * 70)
    
    return results

def run_automated_regression_test(
    baseline_db_path: Union[str, Path] = DEFAULT_DB_PATH,
    documents_dir: Optional[str] = None,
    provider: str = "openai",
    sample_size: int = 3,
    threshold: float = 0.8,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run an automated regression test comparing current performance against historical baseline.
    
    Args:
        baseline_db_path: Path to the database with baseline results
        documents_dir: Directory containing documents
        provider: Provider to test
        sample_size: Number of queries to sample from history
        threshold: Quality threshold ratio (current/baseline) to pass
        verbose: Whether to print detailed output
        
    Returns:
        Dictionary with regression test results
    """
    # Check if database exists
    if not Path(baseline_db_path).exists():
        logger.error(f"Baseline database not found at: {baseline_db_path}")
        return {"error": "Baseline database not found"}
    
    # Query baseline data
    conn = sqlite3.connect(str(baseline_db_path))
    df = pd.read_sql(
        f"SELECT * FROM benchmark_runs WHERE provider='{provider}' ORDER BY timestamp DESC",
        conn
    )
    conn.close()
    
    if len(df) == 0:
        logger.error(f"No baseline data found for provider: {provider}")
        return {"error": f"No baseline data found for provider: {provider}"}
    
    # Sample queries from history
    if len(df) <= sample_size:
        sampled_df = df
    else:
        sampled_df = df.sample(sample_size)
    
    queries = sampled_df['query'].tolist()
    
    # Run the queries again
    current_results = {}
    regression_detected = False
    
    for query in queries:
        logger.info(f"Running regression test for query: {query}")
        
        # Get baseline result
        baseline = sampled_df[sampled_df['query'] == query].iloc[0]
        baseline_quality = baseline['overall_quality']
        baseline_time = baseline['execution_time']
        
        # Run current test
        current = test_knowledge_query(
            provider=provider,
            query=query,
            documents_dir=documents_dir,
            verbose=verbose
        )
        
        # Evaluate current quality
        if "response" in current:
            quality = evaluate_response_quality(
                query, 
                current["response"], 
                current.get("sources", [])
            )
            current_quality = quality.get("overall_score", 0.0)
        else:
            current_quality = 0.0
        
        current_time = current.get("execution_time", 0.0)
        
        # Calculate quality and time ratio
        quality_ratio = current_quality / baseline_quality if baseline_quality > 0 else 0.0
        time_ratio = baseline_time / current_time if current_time > 0 else 0.0
        
        passed = quality_ratio >= threshold
        if not passed:
            regression_detected = True
        
        # Store result
        current_results[query] = {
            "baseline_quality": baseline_quality,
            "current_quality": current_quality,
            "quality_ratio": quality_ratio,
            "baseline_time": baseline_time,
            "current_time": current_time,
            "time_ratio": time_ratio,
            "passed": passed
        }
        
        # Print result
        if verbose:
            print(f"\nQuery: {query}")
            print(f"Baseline Quality: {baseline_quality:.2f}")
            print(f"Current Quality: {current_quality:.2f}")
            print(f"Quality Ratio: {quality_ratio:.2f} (threshold: {threshold})")
            print(f"Passed: {passed}")
    
    # Overall result
    overall_result = {
        "regression_detected": regression_detected,
        "queries_tested": len(queries),
        "queries_passed": sum(1 for r in current_results.values() if r["passed"]),
        "details": current_results
    }
    
    # Print summary
    print("\n" + "=" * 70)
    print("REGRESSION TEST SUMMARY")
    print("=" * 70)
    print(f"Provider: {provider}")
    print(f"Queries Tested: {len(queries)}")
    print(f"Queries Passed: {sum(1 for r in current_results.values() if r['passed'])}")
    print(f"Regression Detected: {regression_detected}")
    print("=" * 70)
    
    return overall_result

def generate_performance_report(
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
    output_dir: Union[str, Path] = DEFAULT_RESULTS_DIR,
    days_back: int = 30,
    format: str = "md"
) -> str:
    """
    Generate a performance report from benchmark history.
    
    Args:
        db_path: Path to the benchmark database
        output_dir: Directory to save report and visualizations
        days_back: Number of days to look back
        format: Report format ('md', 'html', 'json')
        
    Returns:
        Path to the generated report
    """
    # Check if database exists
    if not Path(db_path).exists():
        logger.error(f"Database not found at: {db_path}")
        return f"Error: Database not found at: {db_path}"
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Calculate date cutoff
    cutoff_date = (datetime.datetime.now() - datetime.timedelta(days=days_back)).isoformat()
    
    # Query data
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql(
        f"SELECT * FROM benchmark_runs WHERE timestamp > '{cutoff_date}'",
        conn
    )
    
    # Get quality metrics
    metrics_df = pd.read_sql(
        f"""
        SELECT bm.id, bm.provider, bm.timestamp, qm.metric_name, qm.metric_value
        FROM benchmark_runs bm
        JOIN quality_metrics qm ON bm.id = qm.benchmark_id
        WHERE bm.timestamp > '{cutoff_date}'
        """,
        conn
    )
    conn.close()
    
    if len(df) == 0:
        logger.warning(f"No benchmark data found for the last {days_back} days")
        return f"Warning: No benchmark data found for the last {days_back} days"
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    
    # Generate provider summary
    provider_summary = df.groupby('provider').agg({
        'execution_time': ['mean', 'min', 'max'],
        'overall_quality': ['mean', 'min', 'max'],
        'total_cost': ['mean', 'sum'],
        'id': 'count'
    }).reset_index()
    
    provider_summary.columns = [
        'provider', 'avg_time', 'min_time', 'max_time',
        'avg_quality', 'min_quality', 'max_quality',
        'avg_cost', 'total_cost', 'query_count'
    ]
    
    # Generate time series data
    time_series = df.groupby(['date', 'provider']).agg({
        'execution_time': 'mean',
        'overall_quality': 'mean',
        'total_cost': 'sum'
    }).reset_index()
    
    # Create visualizations if matplotlib is available
    plots_created = False
    if HAS_PLOTTING:
        try:
            # Set style
            sns.set_style("whitegrid")
            
            # Plot 1: Average execution time by provider
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=time_series, x='date', y='execution_time', hue='provider')
            plt.title('Average Query Execution Time by Provider')
            plt.ylabel('Execution Time (seconds)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / 'execution_time.png')
            
            # Plot 2: Average quality by provider
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=time_series, x='date', y='overall_quality', hue='provider')
            plt.title('Average Response Quality by Provider')
            plt.ylabel('Quality Score (0-1)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / 'quality.png')
            
            # Plot 3: Daily cost by provider
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=time_series, x='date', y='total_cost', hue='provider')
            plt.title('Daily API Cost by Provider')
            plt.ylabel('Cost (USD)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / 'cost.png')
            
            # Plot 4: Quality metrics comparison
            metrics_pivot = metrics_df.pivot_table(
                index='provider',
                columns='metric_name',
                values='metric_value',
                aggfunc='mean'
            )
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(metrics_pivot, annot=True, cmap='viridis', fmt='.2f')
            plt.title('Average Quality Metrics by Provider')
            plt.tight_layout()
            plt.savefig(output_dir / 'quality_metrics.png')
            
            plots_created = True
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            plots_created = False
    
    # Generate report
    if format == 'json':
        report = {
            "generated_at": datetime.datetime.now().isoformat(),
            "period": f"Last {days_back} days",
            "provider_summary": provider_summary.to_dict(orient='records'),
            "time_series": time_series.to_dict(orient='records')
        }
        
        report_path = output_dir / 'performance_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
    elif format == 'html':
        html_content = f"""
        <html>
        <head>
            <title>LlamaIndex Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                h1, h2, h3 {{ color: #333; }}
                .summary {{ margin-bottom: 30px; }}
                .visualization {{ margin-bottom: 30px; text-align: center; }}
                .visualization img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>LlamaIndex Performance Report</h1>
            <p>Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Period: Last {days_back} days</p>
            
            <div class="summary">
                <h2>Provider Summary</h2>
                <table>
                    <tr>
                        <th>Provider</th>
                        <th>Queries</th>
                        <th>Avg Time (s)</th>
                        <th>Avg Quality</th>
                        <th>Avg Cost ($)</th>
                        <th>Total Cost ($)</th>
                    </tr>
        """
        
        for _, row in provider_summary.iterrows():
            html_content += f"""
                    <tr>
                        <td>{row['provider']}</td>
                        <td>{row['query_count']}</td>
                        <td>{row['avg_time']:.2f}</td>
                        <td>{row['avg_quality']:.2f}</td>
                        <td>${row['avg_cost']:.5f}</td>
                        <td>${row['total_cost']:.5f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
        """
        
        if plots_created:
            html_content += """
            <div class="visualizations">
                <h2>Performance Visualizations</h2>
                
                <div class="visualization">
                    <h3>Average Query Execution Time</h3>
                    <img src="execution_time.png" alt="Execution Time Chart">
                </div>
                
                <div class="visualization">
                    <h3>Response Quality</h3>
                    <img src="quality.png" alt="Quality Chart">
                </div>
                
                <div class="visualization">
                    <h3>API Costs</h3>
                    <img src="cost.png" alt="Cost Chart">
                </div>
                
                <div class="visualization">
                    <h3>Quality Metrics Comparison</h3>
                    <img src="quality_metrics.png" alt="Quality Metrics Chart">
                </div>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        report_path = output_dir / 'performance_report.html'
        with open(report_path, 'w') as f:
            f.write(html_content)
    
    else:  # Default to markdown
        md_content = f"""
# LlamaIndex Performance Report

Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
Period: Last {days_back} days

## Provider Summary

| Provider | Queries | Avg Time (s) | Avg Quality | Avg Cost ($) | Total Cost ($) |
|----------|---------|--------------|-------------|--------------|----------------|
"""
        
        for _, row in provider_summary.iterrows():
            md_content += f"| {row['provider']} | {row['query_count']} | {row['avg_time']:.2f} | {row['avg_quality']:.2f} | ${row['avg_cost']:.5f} | ${row['total_cost']:.5f} |\n"
        
        if plots_created:
            md_content += """
## Performance Visualizations

### Average Query Execution Time
![Execution Time Chart](execution_time.png)

### Response Quality
![Quality Chart](quality.png)

### API Costs
![Cost Chart](cost.png)

### Quality Metrics Comparison
![Quality Metrics Chart](quality_metrics.png)
"""
        
        report_path = output_dir / 'performance_report.md'
        with open(report_path, 'w') as f:
            f.write(md_content)
    
    logger.info(f"Performance report generated at: {report_path}")
    return str(report_path)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Advanced testing and benchmarking for LlamaIndex")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--comprehensive-benchmark", "-cb", action="store_true", 
                      help="Run comprehensive benchmark across multiple queries and providers")
    group.add_argument("--regression-test", "-rt", action="store_true",
                      help="Run regression test against baseline")
    group.add_argument("--performance-report", "-pr", action="store_true",
                      help="Generate performance report from benchmark history")
    
    parser.add_argument("--provider", "-p", type=str, choices=["openai", "anthropic", "local"],
                        help="LLM provider to use")
    parser.add_argument("--documents-dir", "-d", type=str, 
                        help="Directory containing documents")
    parser.add_argument("--db-path", type=str, default=str(DEFAULT_DB_PATH),
                        help="Path to benchmark database")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_RESULTS_DIR),
                        help="Directory for output files")
    parser.add_argument("--format", type=str, choices=["md", "html", "json"], default="md",
                        help="Format for performance report")
    parser.add_argument("--days", type=int, default=30,
                        help="Number of days to look back for performance report")
    parser.add_argument("--verbose", "-v", action="store_true", 
                        help="Print detailed output")
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_arguments()
    
    if args.comprehensive_benchmark:
        # Determine providers to test
        if args.provider:
            providers = [args.provider]
        else:
            providers = list_available_providers()
        
        run_comprehensive_benchmark(
            providers=providers,
            documents_dir=args.documents_dir,
            verbose=args.verbose,
            db_path=args.db_path
        )
        return
    
    if args.regression_test:
        provider = args.provider or os.environ.get("LLAMA_INDEX_LLM_PROVIDER") or "openai"
        
        run_automated_regression_test(
            baseline_db_path=args.db_path,
            documents_dir=args.documents_dir,
            provider=provider,
            verbose=args.verbose
        )
        return
    
    if args.performance_report:
        generate_performance_report(
            db_path=args.db_path,
            output_dir=args.output_dir,
            days_back=args.days,
            format=args.format
        )
        return
    
    # If no action specified, print help
    print("No action specified. Use one of: --comprehensive-benchmark, --regression-test, or --performance-report")
    print("Use -h or --help for more information")

if __name__ == "__main__":
    main() 
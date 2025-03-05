# LlamaIndex Performance Testing Tools

This document provides comprehensive instructions for using the LlamaIndex Performance Testing Tools with the UTTA platform. These tools help you evaluate, benchmark, and monitor the performance of your LlamaIndex integration, providing valuable insights for production deployment.

## Table of Contents

- [Overview](#overview)
- [Setup Instructions](#setup-instructions)
- [Running the Performance Dashboard](#running-the-performance-dashboard)
- [Command-Line Tools](#command-line-tools)
- [Benchmarking Features](#benchmarking-features)
- [Analyzing Results](#analyzing-results)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

## Overview

The LlamaIndex Performance Testing Tools include:

1. **Advanced Testing Tools** (`advanced_testing_tools.py`): A command-line utility for:
   - Running comprehensive benchmarks across multiple LLM providers
   - Conducting automated regression tests
   - Generating performance reports
   - Estimating API costs
   - Evaluating response quality

2. **Performance Dashboard** (`benchmark_dashboard.py`): A web-based interactive dashboard for:
   - Visualizing performance metrics
   - Comparing providers
   - Analyzing costs and token usage
   - Getting insights and recommendations
   - Running new benchmarks directly from the UI

3. **Setup Script** (`setup_performance_tools.sh`): Automates environment configuration and dependency installation.

## Setup Instructions

### Prerequisites

- Python 3.8 or newer
- Conda environment (we use the 'utta' environment)
- API keys for providers you want to test (OpenAI, Anthropic, etc.)
- SQLite (recommended for database operations)

### Installation

1. Make sure you have activated the 'utta' Conda environment:

   ```bash
   conda activate utta
   ```

2. Make the setup script executable:

   ```bash
   chmod +x setup_performance_tools.sh
   ```

3. Run the setup script:

   ```bash
   ./setup_performance_tools.sh
   ```

4. The script will:
   - Verify your Python version
   - Check you're using the correct Conda environment
   - Install required dependencies
   - Initialize the benchmark database
   - Make testing scripts executable

### Environment Configuration

Ensure your API keys are set in your environment variables:

```bash
# For OpenAI
export OPENAI_API_KEY=your_openai_key

# For Anthropic
export ANTHROPIC_API_KEY=your_anthropic_key

# Optional: Set knowledge directory
export KNOWLEDGE_DIR=/path/to/your/documents
```

You can add these to your `.env` file for persistent configuration.

## Running the Performance Dashboard

The dashboard provides an intuitive interface for all performance testing features:

```bash
# Make sure you're in the utta environment
conda activate utta

# Run the dashboard
python benchmark_dashboard.py
```

The dashboard will open in your web browser (typically at http://localhost:8501) with the following features:

### Dashboard Sections

1. **Overview**: Summary metrics and performance timeline charts
2. **Provider Comparison**: Detailed comparison of different LLM providers
3. **Cost Analysis**: Token usage and cost breakdowns
4. **Insights**: Automatic recommendations based on benchmark data
5. **Run New Benchmark**: Interface for setting up and running new benchmarks
6. **Regression Testing**: Tools for detecting performance regressions

## Command-Line Tools

For automated testing or CI/CD integration, use the command-line tools:

### Running a Comprehensive Benchmark

```bash
# Basic benchmark with all available providers
python advanced_testing_tools.py --comprehensive-benchmark

# Specify a provider
python advanced_testing_tools.py --comprehensive-benchmark --provider openai

# Specify documents directory and enable verbose output
python advanced_testing_tools.py --comprehensive-benchmark --documents-dir ./knowledge --verbose
```

### Running Regression Tests

```bash
# Basic regression test with default settings
python advanced_testing_tools.py --regression-test

# With specific provider and sample size
python advanced_testing_tools.py --regression-test --provider anthropic --sample-size 5
```

### Generating Performance Reports

```bash
# Generate markdown report (default)
python advanced_testing_tools.py --performance-report

# Generate HTML report for the last 60 days
python advanced_testing_tools.py --performance-report --format html --days 60
```

### Getting Help

```bash
python advanced_testing_tools.py --help
```

## Benchmarking Features

The performance testing tools include several sophisticated evaluation mechanisms:

### Response Quality Evaluation

Responses are evaluated based on multiple criteria:
- Relevance to the query
- Source utilization
- Content diversity
- Response length and completeness

### Cost Estimation

The tools track and estimate costs based on:
- Token counts (query and response)
- Provider-specific pricing
- Embedding costs
- Total and per-query costs

### Regression Detection

Benchmarks are stored in a SQLite database, allowing for:
- Historical performance tracking
- Automated regression detection
- Quality and speed comparisons over time

## Analyzing Results

### Quality Metrics

- **Overall Quality Score**: Weighted average of all quality metrics (0-1)
- **Relevance Score**: How well the response addresses the query
- **Source Utilization**: How effectively source content is used
- **Diversity Score**: Variety of information provided

### Performance Metrics

- **Execution Time**: Time taken to process queries
- **Token Usage**: Number of tokens used in queries and responses
- **Cost Efficiency**: Quality delivered per dollar spent

## Troubleshooting

### Import Errors

If you encounter import errors with LlamaIndex, ensure you're using a compatible version:

```bash
# Check your current version
pip show llama-index

# If needed, install a specific version
pip install llama-index==0.7.0
```

For newer LlamaIndex versions, you may need to update import statements:

```python
# Old import
from llama_index import VectorStoreIndex

# New import (for recent versions)
from llama_index.indices.vector_store import VectorStoreIndex
```

### Database Issues

If you encounter database errors:

```bash
# Recreate the benchmark database
python -c "from advanced_testing_tools import create_benchmark_database; create_benchmark_database()"
```

### Dashboard Doesn't Load

If the dashboard doesn't load:

1. Check if Streamlit is installed:
   ```bash
   pip install streamlit
   ```

2. Verify that you're using the correct Python environment:
   ```bash
   conda activate utta
   ```

## Advanced Usage

### Custom Evaluation Metrics

You can extend the quality evaluation metrics by modifying the `evaluate_response_quality` function in `advanced_testing_tools.py`.

### Integration with CI/CD

For continuous integration:

```bash
# Basic integration test with exit code based on regression
python advanced_testing_tools.py --regression-test --exit-on-regression
```

### Custom Queries

You can provide custom queries for benchmarks:

```bash
# Create a file with one query per line
echo "What are effective teaching strategies for ESL students?" > custom_queries.txt
echo "How can teachers support students with ADHD?" >> custom_queries.txt

# Use these queries in a benchmark
python advanced_testing_tools.py --comprehensive-benchmark --queries-file custom_queries.txt
```

---

## Further Assistance

If you encounter any issues or have questions about the performance testing tools, please refer to the code documentation or reach out to the development team. 
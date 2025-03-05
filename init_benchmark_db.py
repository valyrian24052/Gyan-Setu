#!/usr/bin/env python3
"""
Initialize the benchmark database
"""
from pathlib import Path
import sys

# Create test results directory if it doesn't exist
Path("test_results").mkdir(exist_ok=True, parents=True)

try:
    from advanced_testing_tools import create_benchmark_database
    create_benchmark_database()
    print("Benchmark database initialized successfully!")
except Exception as e:
    print(f"Error initializing benchmark database: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 
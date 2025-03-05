#!/bin/bash
# Setup script for LlamaIndex Performance Testing Tools
# This script installs necessary dependencies and configures the performance testing environment

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}   LlamaIndex Performance Testing Tools Setup     ${NC}"
echo -e "${GREEN}==================================================${NC}"

# Check if python is installed
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found. Please ensure Python is installed.${NC}"
    exit 1
fi

# Check python version - fixed comparison logic
python_version=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "${GREEN}Found Python version: ${python_version}${NC}"

python_major=$(echo $python_version | cut -d. -f1)
python_minor=$(echo $python_version | cut -d. -f2)

if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 8 ]); then
    echo -e "${RED}Error: Python 3.8 or newer is required. Found ${python_version}${NC}"
    exit 1
fi

echo -e "${GREEN}Python version check passed.${NC}"

# Check if conda is activated and 'utta' environment is active
if [[ -z "${CONDA_PREFIX}" ]]; then
    echo -e "${YELLOW}Conda environment not detected. Please run 'conda activate utta' first.${NC}"
    echo -e "${YELLOW}Then run this script again.${NC}"
    exit 1
fi

ENV_NAME=$(basename "${CONDA_PREFIX}")
if [[ "${ENV_NAME}" != "utta" ]]; then
    echo -e "${YELLOW}You are using conda environment '${ENV_NAME}', not 'utta'.${NC}"
    echo -e "${YELLOW}Please run 'conda activate utta' first, then run this script again.${NC}"
    echo -e "${YELLOW}Or confirm you want to install in the current environment.${NC}"
    
    read -p "Continue with environment '${ENV_NAME}'? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install or upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install required packages for performance testing
echo -e "${YELLOW}Installing required packages in conda environment...${NC}"
pip install pandas numpy matplotlib seaborn plotly streamlit

# Check if llama-index is installed, if not prompt to install
if ! pip show llama-index &> /dev/null; then
    echo -e "${YELLOW}LlamaIndex not found. Installing llama-index...${NC}"
    pip install llama-index
else
    echo -e "${GREEN}LlamaIndex already installed.${NC}"
    pip show llama-index | grep Version
fi

# Create test_results directory if it doesn't exist
if [ ! -d "test_results" ]; then
    echo -e "${YELLOW}Creating test_results directory...${NC}"
    mkdir -p test_results
fi

# Install SQLite if not available
if ! command -v sqlite3 &> /dev/null; then
    echo -e "${YELLOW}SQLite not found. It's recommended to install SQLite for database operations.${NC}"
    if [ "$(uname)" == "Darwin" ]; then
        echo -e "${YELLOW}On macOS, you can install SQLite with: brew install sqlite${NC}"
    elif [ "$(uname)" == "Linux" ]; then
        echo -e "${YELLOW}On Linux, you can install SQLite with your package manager (e.g., apt install sqlite3)${NC}"
    fi
else
    echo -e "${GREEN}SQLite is available.${NC}"
fi

# Initialize the benchmark database
echo -e "${YELLOW}Initializing benchmark database...${NC}"
python -c "from advanced_testing_tools import create_benchmark_database; create_benchmark_database()"

echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}   Installation Complete!                         ${NC}"
echo -e "${GREEN}==================================================${NC}"
echo -e "To run the performance dashboard: ${YELLOW}python benchmark_dashboard.py${NC}"
echo -e "To run benchmarks from the command line: ${YELLOW}python advanced_testing_tools.py --comprehensive-benchmark${NC}"
echo -e "To run regression tests: ${YELLOW}python advanced_testing_tools.py --regression-test${NC}"
echo -e "To generate a performance report: ${YELLOW}python advanced_testing_tools.py --performance-report${NC}"
echo
echo -e "For more options, run: ${YELLOW}python advanced_testing_tools.py --help${NC}"
echo -e "${GREEN}==================================================${NC}"

# Make the testing scripts executable
chmod +x advanced_testing_tools.py
chmod +x benchmark_dashboard.py

echo -e "${YELLOW}Would you like to run a sample benchmark now? (y/n)${NC}"
read -r run_benchmark

if [[ $run_benchmark == "y" || $run_benchmark == "Y" ]]; then
    echo -e "${GREEN}Running a sample benchmark...${NC}"
    python advanced_testing_tools.py --comprehensive-benchmark --verbose
    
    echo -e "${GREEN}Starting the dashboard...${NC}"
    python benchmark_dashboard.py
fi

echo -e "${GREEN}Setup completed successfully!${NC}" 
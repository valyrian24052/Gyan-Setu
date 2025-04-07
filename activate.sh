#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Activate conda base environment first (in case conda isn't initialized)
eval "$(conda shell.bash hook)"

# Activate the utta environment
conda activate utta

# Change to the project directory
cd "$SCRIPT_DIR"

# Print confirmation message
echo "Activated utta environment in $SCRIPT_DIR"
echo "You can now run your commands" 
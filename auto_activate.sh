#!/bin/bash

# Add this to your ~/.bashrc or ~/.bash_profile:

function conda_auto_env() {
  if [ -e ".env" ]; then
    # Get the CONDA_ENV_NAME from .env file
    ENV_NAME=$(grep "CONDA_ENV_NAME" .env | cut -d'=' -f2)
    
    # If environment name is found and we're not already in it
    if [ ! -z "$ENV_NAME" ] && [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
      conda activate $ENV_NAME
    fi
  fi
}

# Execute the function above when changing directories
export PROMPT_COMMAND=conda_auto_env 
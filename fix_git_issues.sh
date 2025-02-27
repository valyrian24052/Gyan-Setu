#!/bin/bash

echo "==============================================="
echo "UTTA Git Repository Fix Script"
echo "==============================================="
echo "This script will help fix two issues:"
echo "1. Remove the exposed Hugging Face token from git history"
echo "2. Set up Git LFS for the large knowledge_base/vector_db.sqlite file"
echo ""
echo "WARNING: This will rewrite git history!"
echo "If others have pulled this repository, they will need to re-clone after this fix."
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Operation cancelled."
    exit 1
fi

# Install git-filter-repo if not already installed
if ! command -v git-filter-repo &> /dev/null
then
    echo "Installing git-filter-repo..."
    pip install git-filter-repo
fi

# Install Git LFS if not already installed
if ! command -v git lfs &> /dev/null
then
    echo "Installing Git LFS..."
    # For Ubuntu/Debian
    sudo apt-get install git-lfs
    
    # For macOS (uncomment if needed)
    # brew install git-lfs
    
    # For Windows, manual installation is required
    
    git lfs install
fi

# First, configure Git LFS for the large file
echo ""
echo "Setting up Git LFS for large files..."
git lfs install
git lfs track "knowledge_base/vector_db.sqlite"

# Add .gitattributes to the repo
git add .gitattributes

# Next, clean the history to remove the token
echo ""
echo "Cleaning git history to remove the exposed token..."
# Use git-filter-repo to replace the token references
git-filter-repo --path run_enhanced.py --replace-text <(echo "Bearer hf_**** ==> Bearer <TOKEN>")

echo ""
echo "Adding knowledge_base/vector_db.sqlite to Git LFS..."
git lfs migrate import --include="knowledge_base/vector_db.sqlite" --everything

echo ""
echo "Success! The repository should now be fixed."
echo ""
echo "Next steps:"
echo "1. Force push the changes with: git push --force origin phase-1-chatbot:phase-1-chatbot"
echo "2. Ask collaborators to re-clone the repository if needed"
echo ""
echo "IMPORTANT: Make sure to update your Hugging Face token:"
echo "1. Revoke the exposed token from your Hugging Face account"
echo "2. Generate a new token"
echo "3. Set it as an environment variable with: export HF_TOKEN=your_new_token"
echo "" 
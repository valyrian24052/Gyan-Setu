# LLM Chatbot Framework

This repository contains the LLM Chatbot Framework, a flexible framework for building and evaluating educational LLM-powered chatbots with fine-tuning capabilities.

## Repository Structure

The main project is located in the `llm-chatbot-framework` directory. Please navigate there for the full project:

```cd llm-chatbot-framework
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-chatbot-framework.git

# Navigate to the project directory
cd llm-chatbot-framework

# Set up the environment
conda create -n utta python=3.10
conda activate utta

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Model Access and Authentication

The framework supports various LLM models, including:
- LLaMA 2 models (7B, 13B, 70B variants)
- Mistral models
- Other Hugging Face models

To use these models:

1. Create a Hugging Face account at https://huggingface.co
2. Generate an access token from your Hugging Face account settings
3. Set up your credentials using one of these secure methods:
   ```bash
   # Method 1: Use the CLI tool (recommended)
   huggingface-cli login
   
   # Method 2: The framework will prompt for your token when needed
   # and store it securely
   ```
4. Request access to specific models (e.g., LLaMA 2) through their respective model pages on Hugging Face

## LLM Evaluation Framework

The project includes a comprehensive evaluation framework for assessing chatbot performance:

### Supported Models
- LLaMA 2 series (7B, 13B, 70B)
- Mistral-7B
- Other Hugging Face compatible models

### Automated Metrics
- ROUGE scores for text overlap
- BERTScore for semantic similarity
- BLEU and METEOR for translation quality
- Support for both single response and conversation evaluation

### Human Evaluation
- Structured evaluation forms
- Multiple criteria assessment (relevance, accuracy, quality)
- Aggregation of evaluator scores
- Detailed feedback collection

### Benchmark Suite
- YAML-based test case management
- Automated benchmark running
- Result comparison across versions
- Predefined test sets for common scenarios
- GPU resource monitoring and optimization

### Running Evaluations

```bash
# Example of running an evaluation with LLaMA 2 13B
python evaluate.py --model="meta-llama/Llama-2-13b-hf" --dataset="your_dataset"
```

For detailed documentation and examples, see the [LLM Evaluation Wiki](UTTA.wiki/LLM-Evaluation.md).

## License

This project is licensed under the MIT License - see the LICENSE file for details.
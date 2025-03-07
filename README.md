# LLM Chatbot Framework

This repository contains the LLM Chatbot Framework, a flexible framework for building educational LLM-powered chatbots with fine-tuning capabilities.

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

For detailed documentation, please see the [documentation](llm-chatbot-framework/docs/Home.md).

## LLM Evaluation Framework

The project includes a comprehensive evaluation framework for assessing chatbot performance:

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

For detailed documentation and examples, see the [LLM Evaluation Wiki](UTTA.wiki/LLM-Evaluation.md).

## License

This project is licensed under the MIT License - see the LICENSE file for details.